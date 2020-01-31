import argparse

import skimage.io as skio
import tensorflow as tf
from tensorflow import keras

from examples.mnist_olivetti.architectures import construct_dgcspn_model, construct_ratspn_model
from examples.mnist_olivetti.data import load_data
from libspn_keras.backprop_mode import BackpropMode
from libspn_keras.losses.negative_log_joint import NegativeLogJoint
from libspn_keras.losses.negative_log_marginal import NegativeLogMarginal
from libspn_keras.metrics.log_likelihood import LogMarginal
import numpy as np


def main():
    if args.eager:
        tf.config.experimental_run_functions_eagerly(True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    accumulator_regularizer = keras.regularizers.l1_l2(l1=args.l1, l2=args.l2)
    if args.mode == "generative-hard-em":
        logspace_accumulators = False
        backprop_mode = BackpropMode.HARD_EM
        loss = NegativeLogMarginal(name="NegativeLogMarginal")
        metrics = [LogMarginal(name="LogMarginal")]
        optimizer = tf.keras.optimizers.SGD(lr=args.lr)
        return_weighted_child_logits = False
    elif args.mode == "generative-hard-em-unweighted":
        logspace_accumulators = False
        backprop_mode = BackpropMode.HARD_EM_UNWEIGHTED
        loss = NegativeLogMarginal(name="NegativeLogMarginal")
        metrics = [LogMarginal(name="LogMarginal")]
        optimizer = tf.keras.optimizers.SGD(lr=args.lr)
        return_weighted_child_logits = False
    elif args.mode == "generative-soft-em":
        logspace_accumulators = False
        backprop_mode = BackpropMode.EM
        loss = NegativeLogMarginal(name="NegativeLogMarginal")
        metrics = [LogMarginal(name="LogMarginal")]
        optimizer = tf.keras.optimizers.SGD(lr=args.lr)
        return_weighted_child_logits = False
    elif args.mode == "generative-hard-em-supervised":
        logspace_accumulators = False
        backprop_mode = BackpropMode.HARD_EM
        loss = NegativeLogJoint()
        metrics = [
            LogMarginal(name="LogMarginal"),
            keras.metrics.SparseCategoricalAccuracy(name="Accuracy")
        ]
        optimizer = tf.keras.optimizers.SGD(lr=args.lr)
        return_weighted_child_logits = True
    elif args.mode == "generative-hard-em-unweighted-supervised":
        logspace_accumulators = False
        backprop_mode = BackpropMode.HARD_EM_UNWEIGHTED
        loss = NegativeLogJoint()
        metrics = [
            LogMarginal(name="LogMarginal"),
            keras.metrics.SparseCategoricalAccuracy(name="Accuracy")
        ]
        optimizer = tf.keras.optimizers.SGD(lr=args.lr)
        return_weighted_child_logits = True
    elif args.mode == "generative-gd":
        logspace_accumulators = True
        backprop_mode = BackpropMode.GRADIENT
        loss = NegativeLogMarginal(name="NegativeLogLikelihood")
        metrics = [LogMarginal(name="LogMarginal")]
        optimizer = tf.keras.optimizers.Adam()
        return_weighted_child_logits = False
    elif args.mode == "discriminative-gd":
        logspace_accumulators = True
        backprop_mode = BackpropMode.GRADIENT
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [keras.metrics.SparseCategoricalAccuracy(name="Accuracy")]
        optimizer = tf.keras.optimizers.Adam(7e-3)
        return_weighted_child_logits = True
    else:
        raise ValueError("Unknown mode: {}".format(args.mode))

    x_train, y_train, x_test, y_test = load_data(
        spatial=args.model == 'dgcspn', dataset=args.dataset)

    spn = construct_spn(accumulator_regularizer, backprop_mode, logspace_accumulators,
                        return_weighted_child_logits, x_train)

    # Important to use from_logits=True with the cross-entropy loss
    spn.compile(optimizer=optimizer, loss=loss,  metrics=metrics)
    show_summary(spn, x_train)
    spn.evaluate(x_test, y_test, verbose=2)

    if args.completion and args.model == "dgcspn":
        model_completion = construct_dgcspn_model(
            x_train.shape[1:], logspace_accumulators, BackpropMode.GRADIENT,
            return_weighted_child_logits,
            completion_by_posterior_marginal=True, initialization_data=x_train,
            dropout_rate=args.dropout_rate, input_dropout_rate=args.input_dropout_rate,
            cdf_rate=args.cdf_rate, config_name=args.dataset,
            discriminative=args.mode == 'discriminative-gd',
            normalization_epsilon=args.normalization_epsilon,
            accumulator_regularizer=accumulator_regularizer,
            with_evidence_mask=True
        )
        model_completion.compile(loss=loss)
        model_completion.predict([x_test[:1], np.ones_like(x_test[:1])])
        model_completion.set_weights(spn.get_weights())

        evaluate_completion(model_completion, x_test)

    # Train
    # Note that y_train and y_test are effectively ignored if the learning mode is anything
    # different from 'discriminative-gd'
    spn.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    spn.evaluate(x_test, y_test, verbose=2)

    if args.completion:
        model_completion = construct_dgcspn_model(
            x_train.shape[1:], logspace_accumulators, BackpropMode.GRADIENT,
            return_weighted_child_logits,
            completion_by_posterior_marginal=True, initialization_data=x_train,
            dropout_rate=args.dropout_rate, input_dropout_rate=args.input_dropout_rate,
            cdf_rate=args.cdf_rate, config_name=args.dataset,
            discriminative=args.mode == 'discriminative',
            normalization_epsilon=args.normalization_epsilon,
            accumulator_regularizer=accumulator_regularizer,
            with_evidence_mask=True
        )
        model_completion.compile(loss=loss)
        model_completion.predict([x_test[:1], np.ones_like(x_test[:1])])
        model_completion.set_weights(spn.get_weights())

        evaluate_completion(model_completion, x_test)
    else:
        # Evaluate
        spn.evaluate(x_test,  y_test, verbose=2)


def construct_spn(accumulator_regularizer, backprop_mode, logspace_accumulators,
                  return_weighted_child_logits, x_train):
    if args.model == 'ratspn':
        num_vars = x_train.shape[1]
        model = construct_ratspn_model(
            num_vars, logspace_accumulators, backprop_mode, return_weighted_child_logits,
            args.weight_stddev
        )
    else:
        model = construct_dgcspn_model(
            x_train.shape[1:], logspace_accumulators, backprop_mode, return_weighted_child_logits,
            completion_by_posterior_marginal=False, initialization_data=x_train,
            weight_stddev=args.weight_stddev,
            dropout_rate=args.dropout_rate, input_dropout_rate=args.input_dropout_rate,
            cdf_rate=args.cdf_rate, config_name=args.dataset,
            normalization_epsilon=args.normalization_epsilon,
            accumulator_regularizer=accumulator_regularizer,
            accumulator_init_epsilon=args.accumulator_init_epsilon,
            discriminative=args.mode == 'discriminative-gd',
            with_evidence_mask=False
        )
    return model


def show_summary(model, x_train):
    model.build(x_train.shape)
    model.call(keras.layers.Input(shape=x_train.shape[1:]))
    model.summary()


def evaluate_completion(model_completion, x_test):
    # Build masks
    mask_bottom, mask_left, mask_right, mask_top = build_completion_masks(x_test)

    kwargs = dict(verbose=2, batch_size=args.batch_size)
    print("Bottom completion")
    model_completion.evaluate([x_test, mask_bottom], x_test, **kwargs)
    print("Left completion")
    model_completion.evaluate([x_test, mask_left], x_test, **kwargs)
    print("Right completion")
    model_completion.evaluate([x_test, mask_right], x_test, **kwargs)
    print("Top completion")
    model_completion.evaluate([x_test, mask_top], x_test, **kwargs)

    if args.model == 'dgcspn' and args.saveimg:
        print("Exporting completion images")
        im_grid = make_image_grid(x_test, num_rows=5)
        skio.imsave('test_data.png', arr=im_grid.astype(np.uint8))

        kwargs = dict(batch_size=args.batch_size)
        bottom_completion = model_completion.predict([x_test, mask_bottom], **kwargs)
        left_completion = model_completion.predict([x_test, mask_left], **kwargs)
        right_completion = model_completion.predict([x_test, mask_right], **kwargs)
        top_completion = model_completion.predict([x_test, mask_top], **kwargs)

        skio.imsave('comp_left.png', make_image_grid(left_completion, 5).astype(np.uint8))
        skio.imsave('comp_right.png', make_image_grid(right_completion, 5).astype(np.uint8))
        skio.imsave('comp_top.png', make_image_grid(top_completion, 5).astype(np.uint8))
        skio.imsave('comp_bottom.png', make_image_grid(bottom_completion, 5).astype(np.uint8))


def build_completion_masks(x_test):
    mask = np.ones_like(x_test).astype(bool)
    mask_left, mask_right, mask_top, mask_bottom = [mask.copy() for _ in range(4)]
    mid = int(np.sqrt(np.prod(x_test.shape[1:3]))) // 2
    mask_left[:, :, :mid, :] = False
    mask_right[:, :, mid:, :] = False
    mask_top[:, :mid, :, :] = False
    mask_bottom[:, mid:, :, :] = False
    return mask_bottom, mask_left, mask_right, mask_top


def make_image_grid(images, num_rows):
    images_per_row = np.split(images, axis=0, indices_or_sections=num_rows)
    rows = [np.concatenate(imgs, axis=1) for imgs in images_per_row]
    full_grid = np.concatenate(rows, axis=0)
    return full_grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=[
            'generative-gd',
            'generative-soft-em',
            'generative-hard-em',
            'generative-hard-em-supervised',
            'generative-hard-em-unweighted',
            'generative-hard-em-supervised-unweighted',
            'discriminative-gd'
        ],
        required=True
    )
    parser.add_argument("--input-dropout-rate", default=None, type=float)
    parser.add_argument("--dropout-rate", default=None, type=float)
    parser.add_argument("--cdf-rate", default=None, type=float)
    parser.add_argument("--weight-stddev", default=0.1, type=float)
    parser.add_argument("--model", default='dgcspn', choices=['ratspn', 'dgcspn'])
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--completion", action='store_true', dest='completion')
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--dataset", default='olivetti', type=str, choices=['mnist', 'olivetti'])
    parser.add_argument("--saveimg", action='store_true', dest='saveimg')
    parser.add_argument("--eager", action='store_true', dest='eager')
    parser.add_argument("--normalization-epsilon", type=float, default=1e-8)
    parser.add_argument("--accumulator-init-epsilon", type=float, default=1e-8)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.set_defaults(completion=False, saveimg=False, eager=False)
    args = parser.parse_args()
    main()

