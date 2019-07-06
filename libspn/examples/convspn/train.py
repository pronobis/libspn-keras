from argparse import ArgumentParser

from libspn.examples.convspn.architecture import wicker_convspn_two_non_overlapping, full_wicker
from libspn.examples.convspn.utils import DataIterator, ImageIterator, ExperimentLogger, GroupedMetrics
from libspn.examples.convspn.amsgrad import AMSGrad
import tensorflow as tf
import libspn as spn
import numpy as np
import pprint
from collections import deque
from libspn.log import get_logger
from sklearn.utils import shuffle
from libspn.utils.initializers import Equidistant
from sklearn.datasets import olivetti_faces
import itertools
import os.path as opth
import os

LearningType = spn.LearningMethodType

from libspn.graph.leaf.normal import NormalLeaf
from libspn.graph.leaf.multivariate_normal_diag import MultivariateNormalDiagLeaf
from libspn.graph.leaf.multivariate_cauchy_diag import MultivariateCauchyDiagLeaf
from libspn.graph.leaf.laplace import LaplaceLeaf
from libspn.graph.leaf.cauchy import CauchyLeaf
from libspn.graph.leaf.truncated_normal import TruncatedNormalLeaf
from libspn.graph.leaf.continuous_base import ContinuousLeafBase
from libspn.graph.leaf.student_t import StudentTLeaf

logger = get_logger()

tfk = tf.keras


def train(args):
    reporter = ExperimentLogger(args.name, args.log_base_path)
    reporter.write_hyperparameter_dict(vars(args))
    test_x, test_y, train_x, train_y, num_classes = load_data(args)

    num_rows, num_cols = train_x.shape[1:3]
    num_vars = train_x.shape[1] * train_x.shape[2]
    num_dims = train_x.shape[-1]

    train_x = np.squeeze(train_x.reshape(-1, num_vars, num_dims))
    test_x = np.squeeze(test_x.reshape(-1, num_vars, num_dims))

    train_augmented_iterator = ImageIterator(
        [train_x, train_y], batch_size=args.batch_size, shuffle=True,
        width_shift_range=args.width_shift_range, height_shift_range=args.height_shift_range,
        shear_range=args.shear_range, rotation_range=args.rotation_range,
        zoom_range=args.zoom_range, horizontal_flip=args.horizontal_flip,
        image_dims=(num_rows, num_cols))
    train_iterator = DataIterator(
        [train_x, train_y], batch_size=args.batch_size, shuffle=True)
    test_iterator = DataIterator(
        [test_x, test_y], batch_size=args.eval_batch_size, shuffle=False)

    in_var, root = build_spn(args, num_dims, num_vars, train_x, train_y)
    spn.generate_weights(
        root, log=args.log_weights,
        initializer=tf.initializers.random_uniform(args.weight_init_min, args.weight_init_max))

    init_weights = spn.initialize_weights(root)

    correct, labels_node, loss, likelihood, update_op, pred_op, reg_loss, loss_per_sample, \
    mpe_in_var = setup_learning(args, in_var, root)
    
    # Set up the evaluation tasks
    def evaluate_classification(image_batch, labels_batch, epoch, step):
        feed_dict = {in_var: image_batch}
        if args.supervised:
            feed_dict[labels_node] = labels_batch
        loss_out, correct_out, likelihood_out, reg_loss_out, loss_per_sample_out = sess.run(
            [loss, correct, likelihood, reg_loss, loss_per_sample], feed_dict=feed_dict)
        return [loss_out, reg_loss_out, correct_out * 100, likelihood_out]

    # Set up the evaluation tasks
    def evaluate_likelihood(image_batch, labels_batch, epoch, step):
        feed_dict = {in_var: image_batch}
        if args.supervised:
            feed_dict[labels_node] = labels_batch
        return sess.run(likelihood, feed_dict=feed_dict)
        
    # These are default evaluation metrics to be measured at the end of each epoch
    metrics = ["loss", "reg_loss", "accuracy", 'likelihood']
    gm_default = GroupedMetrics(reporter=reporter,
                                reduce_fun=np.mean if not args.novelty_detection else 'roc')

    if args.supervised:

        gm_default.add_task('test_epoch.csv', fun=evaluate_classification, iterator=test_iterator,
                            metric_names=metrics, desc="Evaluate test ",
                            batch_size=args.eval_batch_size)
        gm_default.add_task('train_epoch.csv', fun=evaluate_classification, iterator=train_iterator,
                            metric_names=metrics, desc="Evaluate train",
                            batch_size=args.eval_batch_size, return_val=True)
    else:
        gm_default.add_task('test_epoch.csv', fun=evaluate_likelihood, iterator=test_iterator,
                            metric_names=['likelihood'], desc="Likelihood test ",
                            batch_size=args.eval_batch_size)
        gm_default.add_task('train_epoch.csv', fun=evaluate_likelihood, iterator=train_iterator,
                            metric_names=['likelihood'], desc="Likelihood train",
                            batch_size=args.eval_batch_size, return_val=True)
    if args.completion:

        with tf.name_scope("CompletionSummary"):
            truth = in_var.feed if not args.discrete else tf.placeholder(
                tf.float32, [None, num_vars])
            completion_indices = tf.equal(in_var.feed, -1) if args.discrete else tf.logical_not(
                in_var.evidence)
            shape = (-1, num_rows, num_cols, num_dims)
            mosaic = impainting_mosaic(
                reconstruction=tf.reshape(mpe_in_var, shape), truth=tf.reshape(truth, shape),
                completion_indices=tf.reshape(completion_indices, shape), num_rows=4,
                batch_size=args.completion_batch_size, invert=args.dataset == "mnist")
            mosaic_summary = tf.summary.image("Completion", mosaic)

        def completion_left(image_batch, labels_batch, epoch, step):
            shape = [len(image_batch), num_rows, num_cols // 2]
            if np.prod(shape) > image_batch.size:
                shape.append(3)
            completion_ind = np.concatenate([np.ones(shape), np.zeros(shape)], axis=2) \
                .astype(np.bool)
            evidence_ind = np.logical_not(completion_ind)
            evidence_ind = np.reshape(evidence_ind, image_batch.shape[:2])
            completion_ind = np.reshape(completion_ind, image_batch.shape[:2])
            return _measure_completion(completion_ind, epoch, evidence_ind, image_batch,
                                       labels_batch, step, tag='left', writer=test_writer_left)

        def completion_bottom(image_batch, labels_batch, epoch, step):
            shape = [len(image_batch), num_rows // 2, num_cols]
            if np.prod(shape) > image_batch.size:
                shape.append(3)
            completion_ind = np.concatenate([np.zeros(shape), np.ones(shape)], axis=1) \
                .astype(np.bool)
            evidence_ind = np.logical_not(completion_ind)
            evidence_ind = np.reshape(evidence_ind, image_batch.shape[:2])
            completion_ind = np.reshape(completion_ind, image_batch.shape[:2])

            return _measure_completion(completion_ind, epoch, evidence_ind, image_batch,
                                       labels_batch, step, tag='bottom', writer=test_writer_bottom)

        def _measure_completion(completion_ind, epoch, evidence_ind, image_batch, labels_batch,
                                step, writer, tag="comp"):
            if args.discrete:
                im = image_batch.copy()
                im[completion_ind] = -1
                feed_dict = {in_var: im}
            else:
                feed_dict = {in_var: image_batch, in_var.evidence: evidence_ind}
            if args.supervised:
                feed_dict[labels_node] = labels_batch
            if step == 0:
                if args.discrete:
                    feed_dict[truth] = image_batch
                mpe_in_var_out, mosaic_summary_out, mosaic_out = sess.run(
                    [mpe_in_var, mosaic_summary, mosaic], feed_dict=feed_dict)
                writer.add_summary(mosaic_summary_out, epoch)
                reporter.write_image(
                    np.squeeze(mosaic_out, axis=0), 'completion/epoch_{}_{}.png'.format(
                        epoch, tag))
            else:
                mpe_in_var_out = sess.run(mpe_in_var, feed_dict=feed_dict)

            if not args.normalize_data:
                mpe_in_var_out *= 255
                orig = image_batch.copy() * 255
            else:
                orig = image_batch

            hamming = np.equal(orig, mpe_in_var_out)[completion_ind]
            max_fluctuation = args.num_vals ** 2 if args.discrete else 1.0
            l2 = np.square(orig - mpe_in_var_out)[completion_ind]
            l1 = np.abs(orig - mpe_in_var_out)[completion_ind]
            hamming = np.mean(hamming.reshape(len(orig), -1), axis=-1)
            l2 = np.mean(l2.reshape(len(orig), -1), axis=-1)
            l1 = np.mean(l1.reshape(len(orig), -1), axis=-1)
            psnr = 10 * np.log10(max_fluctuation / l2)
            tv = tv_norm(
                mpe_in_var_out.reshape(-1, num_rows, num_cols, num_dims),
                completion_ind.reshape(-1, num_rows, num_cols, num_dims))
            return l1, l2, hamming, psnr, tv

        gm_default.add_task(
            "test_epoch.csv", fun=completion_bottom, iterator=test_iterator,
            metric_names=['l1_b', 'l2_b', 'hamming_b', 'psnr_b', 'tv_b'],
            desc='Completion bottom', batch_size=args.completion_batch_size)
        gm_default.add_task(
            "test_epoch.csv", fun=completion_left, iterator=test_iterator,
            metric_names=['l1_l', 'l2_l', 'hamming_l', 'psnr_l', 'tv_l'],
            desc='Completion left  ', batch_size=args.completion_batch_size)

    # Reporting total number of trainable variables
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    total_trainable_parameters = sum([np.prod(v.shape.as_list()) for v in trainable_vars])
    reporter.write_line(
        'num_trainable_parameters.csv', total_trainable_parameters=total_trainable_parameters)
    logger.info("Num trainable parameters = {}".format(total_trainable_parameters))

    # Remember five last metrics for determining stop criterion
    progress_history = deque(maxlen=5)
    progress_metric = 'loss' if args.supervised else 'likelihood'
    with tf.Session() as sess:
        train_writer, test_writer = initialize_graph(init_weights, reporter, sess)
        test_writer_left = reporter.tfwriter('test', 'completion', 'left', exist_ok=True)
        test_writer_bottom = reporter.tfwriter('test', 'completion', 'bottom', exist_ok=True)

        progress_prev = gm_default.evaluate_one_epoch(0)[progress_metric]
        progress_history.append(progress_prev)
        for epoch in range(args.num_epochs):
            # Train, nothing more nothing less
            for image_batch, labels_batch in train_augmented_iterator.iter_epoch("Train"):
                if args.input_dropout:
                    dropout_mask = np.less(
                        np.random.rand(*image_batch.shape[:2]), args.input_dropout)
                    if args.discrete:
                        image_batch_copy = image_batch.copy()
                        image_batch_copy[dropout_mask] = -1
                        feed_dict = {in_var: image_batch_copy}
                    else:
                        feed_dict = {
                            in_var: image_batch, in_var.evidence: np.logical_not(dropout_mask)}
                else:
                    feed_dict = {in_var: image_batch}

                if args.supervised:
                    feed_dict[labels_node] = labels_batch
                sess.run(update_op, feed_dict=feed_dict)
                        
            # Check stopping criterion
            progress_epoch = gm_default.evaluate_one_epoch(epoch + 1)[progress_metric]
            progress_history.append(progress_epoch)
            if len(progress_history) == 5 and np.std(progress_history) < args.stop_epsilon or \
                np.isnan(progress_epoch) or progress_epoch == 0.0:
                print("Stopping criterion reached!")
                break

        # Store locations and scales
        if not args.discrete:
            loc, scale = sess.run([in_var.loc_variable, in_var.scale_variable])
            reporter.write_numpy(loc, "dist/loc")
            reporter.write_numpy(scale, "dist/scale")
            print("Locations\n", np.unique(loc))
            print("\nScales:\n", np.unique(scale))


def initialize_graph(init_weights, reporter, sess):
    accumulator_vars = tf.get_collection('em_accumulators')
    init_accumulators = tf.group(*[v.initializer for v in accumulator_vars])
    train_writer = reporter.tfwriter("train", graph=sess.graph)
    test_writer = reporter.tfwriter("test")
    sess.run([init_accumulators, init_weights, tf.global_variables_initializer()])
    return train_writer, test_writer


def build_spn(args, num_dims, num_vars, train_x, train_y):
    softplus_scale = args.learning_algo != 'em'
    if args.discrete:
        in_var = spn.IndicatorLeaf(num_vars=num_vars, num_vals=2, name="IndicatorLeaf")
    else:
        if num_dims == 1:
            LeafDist = {
                "laplace": LaplaceLeaf,
                'normal': NormalLeaf,
                'cauchy': CauchyLeaf,
                'truncate': TruncatedNormalLeaf,
            }[args.dist]
            if args.dist == "normal" and not args.equidistant_means:
                kwargs = dict(initialization_data=train_x,
                              estimate_scale=args.estimate_scale)
            else:
                kwargs = dict()
            in_var = LeafDist(
                num_vars=num_vars, softplus_scale=softplus_scale,
                num_components=args.num_components, trainable_scale=not args.fixed_variance,
                scale_init=args.variance_init, trainable_loc=not args.fixed_mean,
                share_scales=args.share_scales, share_locs_across_vars=args.share_locs,
                loc_init=Equidistant(-2.0, 2.0) if args.normalize_data else Equidistant(),
                samplewise_normalization=args.normalize_data,
                **kwargs)
        else:
            LeafDist = {
                'cauchy': MultivariateCauchyDiagLeaf,
                'normal': MultivariateNormalDiagLeaf
            }[args.dist]
            minval = -2 if args.normalize_data else 0
            maxval = 2 if args.normalize_data else 1
            in_var = LeafDist(
                num_vars=num_vars, softplus_scale=softplus_scale,
                num_components=args.num_components, trainable_scale=not args.fixed_variance,
                scale_init=args.variance_init, trainable_loc=not args.fixed_mean,
                share_scales=args.share_scales, share_locs_across_vars=args.share_locs,
                dimensionality=num_dims, samplewise_normalization=args.normalize_data,
                loc_init=tf.initializers.random_uniform(minval=minval, maxval=maxval))
    prod_num_channels_head = [args.prod_num_c0 if args.num_components > 4 
                              else args.num_components ** 4]
    prod_num_channels_tail = [args.prod_num_c1, args.prod_num_c2, args.prod_num_c3,
                              args.prod_num_c4]
    sum_num_channels = [args.sum_num_c0, args.sum_num_c1, args.sum_num_c2,
                        args.sum_num_c3, args.sum_num_c4]
    prod_num_channels = prod_num_channels_head + prod_num_channels_tail

    if args.dataset in ['olivetti', 'caltech']:
        prod_num_channels.insert(1, prod_num_channels[1])
        sum_num_channels.insert(1, sum_num_channels[1])

    if args.dataset == 'caltech':
        prod_num_channels.insert(1, prod_num_channels[1])
        sum_num_channels.insert(1, sum_num_channels[1])

    prod_num_channels = tuple(prod_num_channels)
    sum_num_channels = tuple(sum_num_channels)
    num_classes = args.class_subset if args.class_subset else 10
    edge_size = {
        'mnist': 28,
        'fashion_mnist': 28,
        'cifar10': 32,
        'olivetti': 4,
        'caltech': 100
    }[args.dataset]
    if args.dataset == 'cifar10':
        in_var_ = spn.LocalSums(in_var, num_channels=args.sum_num_c0, spatial_dim_sizes=[32, 32])
    else:
        in_var_ = in_var

    if args.supervised:
        root, class_roots = wicker_convspn_two_non_overlapping(
            in_var_, prod_num_channels, sum_num_channels, num_classes=num_classes, edge_size=edge_size,
            first_depthwise=args.first_depthwise, supervised=args.supervised)
    else:
        root, class_roots = full_wicker(
            in_var_, prod_num_channels, sum_num_channels, num_classes=num_classes,
            edge_size=edge_size, first_depthwise=args.first_depthwise, supervised=args.supervised)
    return in_var, root


def setup_learning(args, in_var, root):
    no_op = tf.constant(0)
    inference_type = spn.InferenceType.MARGINAL if args.value_inf_type == 'marginal' \
        else spn.InferenceType.MPE
    mpe_state = spn.MPEState(value_inference_type=inference_type, matmul_or_conv=True)
    if args.supervised:
        # Root is provided with labels, p(x,y)
        labels_node = root.generate_latent_indicators(name="LabelIndicators")

        # Marginalized root, so without filling in labels, so p(x) = \sum_y p(x,y)
        root_marginalized = spn.Sum(*root.values, name="RootMarginalized", weights=root.weights)
        # A dummy node to get MPE state
        labels_no_evidence_node = root_marginalized.generate_latent_indicators(
            name="LabesNoEvidenceIndicators", feed=-tf.ones([tf.shape(in_var.feed)[0], 1], dtype=tf.int32))

        # Get prediction from dummy node
        with tf.name_scope("Prediction"):
            logger.info("Setting up MPE state")
            if args.completion_by_marginal and isinstance(in_var, ContinuousLeafBase):
                in_var_mpe = in_var.impute_by_posterior_marginal(labels_no_evidence_node)
                class_mpe, = mpe_state.get_state(
                    root_marginalized, labels_no_evidence_node)
            else:
                class_mpe, in_var_mpe = mpe_state.get_state(
                    root_marginalized, labels_no_evidence_node, in_var)
            correct = tf.squeeze(tf.equal(class_mpe, tf.to_int64(labels_node.feed)))
    else:
        with tf.name_scope("Prediction"):
            class_mpe = correct = no_op
            labels_node = root_marginalized = None
            if args.completion_by_marginal and isinstance(in_var, ContinuousLeafBase):
                in_var_mpe = in_var.impute_by_posterior_marginal(root)
            else:
                in_var_mpe, = mpe_state.get_state(root, in_var)

    # Get the log likelihood
    with tf.name_scope("LogLikelihoods"):
        logger.info("Setting up log-likelihood")
        val_gen = spn.LogValue(inference_type=inference_type)
        labels_llh = val_gen.get_value(root)
        no_labels_llh = val_gen.get_value(root_marginalized) if args.supervised else labels_llh

    if args.learning_algo == "em":
        em_learning = spn.HardEMLearning(
            root, value_inference_type=inference_type,
            initial_accum_value=args.initial_accum_value, sample_winner=args.sample_path,
            sample_prob=args.sample_prob, use_unweighted=args.use_unweighted)
        accumulate = em_learning.accumulate_updates()
        with tf.control_dependencies([accumulate]):
            update_op = em_learning.update_spn()

        return correct, labels_node, labels_llh, no_labels_llh, update_op, class_mpe, no_op, \
               no_op, in_var_mpe

    logger.info("Setting up GD learning")
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        args.learning_rate, global_step, args.lr_decay_steps, args.lr_decay_rate, staircase=True)
    learning_method = spn.LearningMethodType.DISCRIMINATIVE if args.learning_type == 'discriminative' else \
        spn.LearningMethodType.GENERATIVE
    learning = spn.GDLearning(
        root, learning_task_type=spn.LearningTaskType.SUPERVISED if args.supervised else \
            spn.LearningTaskType.UNSUPERVISED,
        learning_method=learning_method, learning_rate=learning_rate,
        marginalizing_root=root_marginalized, global_step=global_step)

    optimizer = {
        'adam': tf.train.AdamOptimizer,
        'rmsprop': tf.train.RMSPropOptimizer,
        'amsgrad': AMSGrad,
    }[args.learning_algo]()
    minimize_op, _ = learning.learn(optimizer=optimizer)

    logger.info("Settting up test loss")
    with tf.name_scope("DeterministicLoss"):
        main_loss = learning.loss()
        regularization_loss = learning.regularization_loss()
        loss_per_sample = learning.loss(reduce_fn=lambda x: tf.reshape(x, (-1,)))

    return correct, labels_node, main_loss, no_labels_llh, minimize_op, class_mpe, \
           regularization_loss, loss_per_sample, in_var_mpe


def load_data(args):
    if args.dataset == "cifar10":
        test_x, test_y, train_x, train_y = _read_cifar10()
    elif args.dataset == "fashion_mnist":
        test_x, test_y, train_x, train_y = _read_fashion_mnist()
    elif args.dataset == "mnist":
        test_x, test_y, train_x, train_y = _read_mnist()
    elif args.dataset == "olivetti":
        test_x, test_y, train_x, train_y = _read_olivetti()
    elif args.dataset == "caltech":
        test_x, test_y, train_x, train_y = _read_caltech()
    else:
        raise ValueError("Unknown dataset...")
    if args.class_subset:
        train_x = np.concatenate([train_x[train_y == c] for c in range(args.class_subset)])
        test_x = np.concatenate([test_x[test_y == c] for c in range(args.class_subset)])
        train_y = np.concatenate([train_y[train_y == c] for c in range(args.class_subset)])
        test_y = np.concatenate([test_y[test_y == c] for c in range(args.class_subset)])
    train_y = np.expand_dims(train_y, 1)
    test_y = np.expand_dims(test_y, 1)

    num_classes = train_y.max() + 1
    if args.novelty_detection:
        train_y = np.where(np.isin(train_y, args.novel_classes), 0, 1)
        test_y = np.where(np.isin(test_y, args.novel_classes), 0, 1)

        # Novel classes in the train set can be moved to the test set
        test_x = np.concatenate([test_x, train_x[train_y.ravel() == 0]], axis=0)
        test_y = np.concatenate([test_y, train_y[train_y.ravel() == 0]], axis=0)
        train_x = train_x[train_y.ravel() == 1]
        train_y = train_y[train_y.ravel() == 1]

    train_x, train_y = shuffle(train_x, train_y, random_state=1234)
    test_x, test_y = shuffle(test_x, test_y, random_state=1234)

    if args.dist == "beta":
        train_x = np.clip(train_x, 0.01, 0.99)
        test_x = np.clip(test_x, 0.01, 0.99)

    return test_x, test_y, train_x, train_y, num_classes


def _read_fashion_mnist():
    (train_x, train_y), (test_x, test_y) = tfk.datasets.fashion_mnist.load_data()
    train_x = np.reshape(train_x / 255., (-1, 28, 28, 1))
    test_x = np.reshape(test_x / 255., (-1, 28, 28, 1))
    train_y = train_y.squeeze()
    test_y = test_y.squeeze()
    return test_x, test_y, train_x, train_y


def _read_cifar10():
    (train_x, train_y), (test_x, test_y) = tfk.datasets.cifar10.load_data()
    train_x = train_x / 255.
    test_x = test_x / 255.
    train_y = train_y.squeeze()
    test_y = test_y.squeeze()
    return test_x, test_y, train_x, train_y


def _read_mnist():
    (train_x, train_y), (test_x, test_y) = tfk.datasets.mnist.load_data()
    train_x = np.reshape(train_x / 255., (-1, 28, 28, 1))
    test_x = np.reshape(test_x / 255., (-1, 28, 28, 1))
    train_y = train_y.squeeze()
    test_y = test_y.squeeze()

    if args.discrete:
        train_x = np.greater(train_x, 20 / 256).astype(np.int32)
        test_x = np.greater(test_x, 20 / 256).astype(np.int32)

    return test_x, test_y, train_x, train_y


def _read_olivetti():
    bunch = olivetti_faces.fetch_olivetti_faces()
    x, y = np.expand_dims(bunch.images, axis=-1), bunch.target
    train_x, train_y, test_x, test_y = [], [], [], []
    for label in range(max(y) + 1):
        x_class = x[y == label]
        y_class = [label] * len(x_class)
        # print(label, len(x_class))
        test_size = min(30, len(x_class) // 3)
        train_x.extend(x_class[:-test_size])
        train_y.extend(y_class[:-test_size])
        test_x.extend(x_class[-test_size:])
        test_y.extend(y_class[-test_size:])
    train_x, test_x, train_y, test_y = np.asarray(train_x), np.asarray(test_x), \
                                       np.asarray(train_y), np.asarray(test_y)
    if args.normalize_data:
        return test_x * 255, test_y, train_x * 255, train_y
    return test_x, test_y, train_x, train_y


def _read_caltech():
    base = opth.expanduser("~/datasets/caltech")
    class_dirs = [d for d in os.listdir(base) if opth.isdir(opth.join(base, d))]
    train_x, train_y, test_x, test_y = [], [], [], []
    for label, cd in enumerate(class_dirs):
        dirpath = opth.join(base, cd)
        fnms = os.listdir(dirpath)
        x = [np.loadtxt(opth.join(dirpath, fnm)) for fnm in fnms]
        y = [label] * len(fnms)
        test_size = min(50, len(fnms) // 3)
        train_x.extend(x[:-test_size])
        train_y.extend(y[:-test_size])
        test_x.extend(x[-test_size:])
        test_y.extend(y[-test_size:])

    train_x = np.expand_dims(np.asarray(train_x), -1)
    test_x = np.expand_dims(np.asarray(test_x), -1)
    train_y = np.asarray(train_y)
    test_y = np.asarray(test_y)

    if not args.normalize_data:
        train_x /= 255.
        test_x /= 255.

    return test_x, test_y, train_x, train_y


def impainting_mosaic(reconstruction, truth, completion_indices, num_rows, batch_size,
                      pad_constant=0.0, invert=False):

    truth = tf.to_float(truth)
    reconstruction = tf.to_float(reconstruction)
    if invert:
        reconstruction = 1.0 - reconstruction
        truth = 1.0 - truth

    if reconstruction.shape.as_list()[-1] != 3:
        reconstruction = tf.image.grayscale_to_rgb(reconstruction)
        truth = tf.image.grayscale_to_rgb(truth)
        completion_indices = tf.tile(completion_indices, (1, 1, 1, 3))

    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

    def normalize_01(im):
        rec_min = tf.reduce_min(im, axis=[1, 2, 3], keepdims=True)
        rec_max = tf.reduce_max(im, axis=[1, 2, 3], keepdims=True)
        im = (im - rec_min) / (rec_max - rec_min)
        return im

    reconstruction = normalize_01(reconstruction)
    truth = normalize_01(truth)

    rgb_red = tf.reshape(tf.constant([1.0, 0.3, 0.3]), (1, 1, 1, 3))
    truth = tf.where(completion_indices, truth * rgb_red, truth)

    truth = tf.pad(truth, paddings, constant_values=pad_constant)
    reconstruction = tf.pad(reconstruction, paddings, constant_values=pad_constant)

    reconstruction_rows = [tf.concat(tf.unstack(r, num=batch_size // num_rows), axis=1)
                           for r in tf.split(reconstruction, num_rows, axis=0)]
    truth_rows = [tf.concat(tf.unstack(r, num=batch_size // num_rows), axis=1)
                  for r in tf.split(truth, num_rows, axis=0)]
    alternating_rows = list(itertools.chain(*zip(truth_rows, reconstruction_rows)))
    return tf.expand_dims(tf.concat(alternating_rows, axis=0), 0)


def tv_norm(x, completion_ind):
    """Computes the total variation norm and its gradient. From jcjohnson/cnn-vis."""
    x_diff = x - np.roll(x, -1, axis=2)
    y_diff = x - np.roll(x, -1, axis=1)
    grad_norm2 = x_diff ** 2 + y_diff ** 2
    shape = (-1, int((x.size - np.sum(completion_ind)) / x.shape[0]))
    norm = np.mean(np.sqrt(grad_norm2[completion_ind]).reshape(shape), axis=1)
    return norm


if __name__ == "__main__":
    params = ArgumentParser()
    params.add_argument("--log_base_path", default='logs')
    params.add_argument("--name", default="convspn")
    params.add_argument("--batch_size", default=32, type=int)
    params.add_argument(
        "--learning_algo", default='amsgrad', choices=['amsgrad', 'adam', 'rmsprop', 'em'])

    params.add_argument("--num_components", default=4, type=int)
    params.add_argument("--fixed_variance", action='store_true', dest='fixed_variance')
    params.add_argument("--fixed_mean", action="store_true", dest='fixed_mean')
    params.add_argument("--variance_init", type=float, default=1.0)
    params.add_argument("--log_weights", action='store_true', dest="log_weights")
    params.add_argument("--weight_init_min", type=float, default=1.0)
    params.add_argument("--weight_init_max", type=float, default=10.0)
    params.add_argument("--value_inf_type", default='marginal', choices=['marginal', 'mpe'])
    params.add_argument("--learning_rate", type=float, default=1e-2)
    params.add_argument("--dropconnect_keep_prob", type=float, default=None)
    params.add_argument("--learning_type", default='discriminative', choices=['discriminative', 'generative'])
    params.add_argument("--completion", action='store_true', dest='completion')
    params.add_argument("--completion_batch_size", type=int, default=32)
    params.add_argument("--equidistant_means", action='store_true', dest='equidistant_means')
    params.add_argument("--num_epochs", default=500, type=int)
    params.add_argument("--input_dropout", default=None, type=float)
    params.add_argument("--first_depthwise", action='store_true', dest='first_depthwise')

    params.add_argument("--class_subset", default=None, type=int)
    params.add_argument("--dataset", default="mnist",
                        choices=['cifar10', 'fashion_mnist', 'mnist', 'olivetti', 'caltech'])
    params.add_argument("--eval_batch_size", default=32, type=int)

    params.add_argument("--prod_num_c0", default=32, type=int)
    params.add_argument("--prod_num_c1", default=32, type=int)
    params.add_argument("--prod_num_c2", default=32, type=int)
    params.add_argument("--prod_num_c3", default=64, type=int)
    params.add_argument("--prod_num_c4", default=64, type=int)

    params.add_argument("--sum_num_c0", default=64, type=int)
    params.add_argument("--sum_num_c1", default=64, type=int)
    params.add_argument("--sum_num_c2", default=64, type=int)
    params.add_argument("--sum_num_c3", default=64, type=int)
    params.add_argument("--sum_num_c4", default=64, type=int)

    params.add_argument("--initial_accum_value", type=float, default=1e-4)
    params.add_argument("--sample_path", action="store_true", dest="sample_path")
    params.add_argument("--sample_prob", type=float, default=None)
    params.add_argument("--use_unweighted", action="store_true", dest="use_unweighted")

    params.add_argument("--stop_epsilon", default=1e-5, type=float)

    params.add_argument("--share_scales", action="store_true", dest="share_scales")
    params.add_argument("--precision_init", type=float, default=10)

    params.add_argument("--dist",
                        choices=['normal', 'laplace', 'cauchy', 'truncate', 'beta'],
                        default='normal')
    params.add_argument("--share_locs", action="store_true", dest="share_locs")

    # Augmentation
    params.add_argument("--rotation_range", default=0.0, type=float)
    params.add_argument("--horizontal_flip", action="store_true", dest="horizontal_flip")
    params.add_argument("--shear_range", default=0.0, type=float)
    params.add_argument("--width_shift_range", default=0.0, type=float)
    params.add_argument("--height_shift_range", default=0.0, type=float)
    params.add_argument("--zoom_range", default=0.0, type=float)

    params.add_argument("--unsupervised", action="store_false", dest="supervised")

    params.add_argument("--estimate_scale", action="store_true", dest="estimate_scale")

    params.add_argument("--normalize_data", action="store_true", dest="normalize_data")

    params.add_argument("--completion_by_marginal", action="store_true",
                        dest="completion_by_marginal")

    params.add_argument("--lr_decay_rate", type=float, default=0.96)
    params.add_argument("--lr_decay_steps", type=int, default=100000)

    params.set_defaults(novelty_detection=False, discrete=False, predict_each_epoch=False,
                        uniform_priors=False, reparam_weights=False, sparse_range=True,
                        share_scales=False, unnormalized_leafs=False, student_t=False,
                        fixed_mean=False, kwamy=False, depthwise_top=False, trainable_df=False,
                        share_locs=False, share_dfs=False, share_ds=True, horizontal_flip=False,
                        supervised=True, estimate_scale=False, only_root_marginalize=False,
                        normalize_data=False, tensor_spn=False, fixed_variance=False,
                        log_weights=False, equidistant_means=False, first_depthwise=False,
                        sample_path=False, use_unweighted=False)
    args = params.parse_args()
    pprint.pprint(vars(args))
    train(args)
