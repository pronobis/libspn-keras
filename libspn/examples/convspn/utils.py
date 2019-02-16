import tensorflow.keras as tfk
import pandas as pd
import os.path as opth
import json
import os
from skimage.io import imsave
from collections import defaultdict
import numpy as np
import tensorflow as tf
import tabulate
import datetime as dt
import tqdm


class DataIterator:

    def __init__(self, data, batch_size, shuffle=True):
        self._data = [data] if not isinstance(data, (list, tuple)) else data
        self._num_samples = len(data[0])
        self._ind = 0
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._pbar = None
        self._reset()
        self._increment = batch_size
        self._step = 0

    def next_batch(self, batch_size=None):
        batch_size = batch_size or self._batch_size
        end_ind = min(self._ind + batch_size, self._num_samples)

        out = tuple(d[self._ind:end_ind] for d in self._data)

        self._step += 1
        if end_ind == self._num_samples:
            self._reset()
        else:
            self._ind += batch_size
        return out

    def iter_epoch(self, desc="Epoch", batch_size=None):
        self._pbar = tqdm.trange(self.num_batches(batch_size=batch_size), desc=desc)
        for _ in self._pbar:
            yield self.augment(self.next_batch(batch_size=batch_size))
        self._pbar.close()
        self._pbar = None
        self._reset()

    def _reset(self):
        self._ind = 0
        self._step = 0
        self._permute()

    def display_progress(self, **kwargs):
        if self._pbar is None:
            raise ValueError("No progress bar to use")
        self._pbar.set_postfix(**kwargs)

    def _permute(self):
        if not self._shuffle:
            return
        perm = np.random.permutation(self._num_samples)
        self._data = [d[perm] for d in self._data]

    @property
    def end_of_epoch(self):
        return self._ind == 0

    def num_batches(self, batch_size=None):
        if batch_size is not None:
            return int(np.ceil(self._num_samples / batch_size))
        return int(np.ceil(self._num_samples / self._batch_size))

    @property
    def step(self):
        return self._step - 1

    def augment(self, minibatch):
        return minibatch


class ImageIterator(DataIterator):

    def __init__(self, data, batch_size, shuffle=True, width_shift_range=0., height_shift_range=0., 
                 shear_range=0., zoom_range=0., fill_mode='nearest', cval=0., rescale=None, 
                 horizontal_flip=False, vertical_flip=False, rotation_range=0., image_dims=None):
        assert len(data) in [1, 2], "Length data must be either 1 or 2 (images or images + labels)"
        self._pairs = len(data) == 2
        super().__init__(data, batch_size, shuffle)
        self._keras_generator = tfk.preprocessing.image.ImageDataGenerator(
            width_shift_range=width_shift_range, height_shift_range=height_shift_range,
            shear_range=shear_range, zoom_range=zoom_range, fill_mode=fill_mode, cval=cval,
            rescale=rescale, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
            rotation_range=rotation_range)
        assert image_dims, "Currently image dimensions must be provided"
        self._image_dims = list(image_dims) if isinstance(image_dims, tuple) else image_dims

    def augment(self, minibatch):
        it = self._keras_generator
        if it.rotation_range == it.zoom_range == it.shear_range == it.height_shift_range == \
               it.width_shift_range == 0.0 and not it.horizontal_flip and not it.vertical_flip:
            return minibatch

        out = []
        if self._pairs:
            images, labels = minibatch
        else:
            images = minibatch
        for im in images:
            out.append(
                self._keras_generator.random_transform(im.reshape(self._image_dims + [-1])).reshape(
                    im.shape))
        if self._pairs:
            return np.asarray(out), labels
        return np.asarray(out)


class ExperimentLogger:

    LEADING_ZEROS = 5
    RUN = 'run'
    METARESULTS = "results.csv"

    def __init__(self, name, base_path='results', cmd_args=None, csv_writer=None, debug=False):
        self._base_path = base_path
        self._name = name

        self._outdir = opth.join(base_path, name)

        os.makedirs(self._outdir, exist_ok=True)
        self._outdir = opth.abspath(self._outdir)
        subdirs = listdirs(self._outdir)

        if debug:
            self._run = "debug"
        else:
            if len(subdirs) == 0:
                last_run_id = -1
            else:
                last_run_id = max([int(dirname[-self.LEADING_ZEROS:]) for dirname in subdirs
                                   if dirname != "debug"])
            this_run_id = last_run_id + 1

            self._run = self.RUN + str(this_run_id).zfill(self.LEADING_ZEROS)
        self._experiment_dir = opth.join(self._outdir, self._run)
        os.makedirs(self._experiment_dir, exist_ok=debug)

        if cmd_args:
            self.write_hyperparameter_dict(vars(cmd_args))

        self._debug = debug
        self._csv_writer = csv_writer

        print("LOGGING AT {}".format(self._experiment_dir))

    @property
    def logdir(self):
        return self._experiment_dir

    def tfwriter(self, *subdirs, graph=None, exist_ok=False):
        newdir = opth.join(self._experiment_dir, *subdirs)
        os.makedirs(newdir, exist_ok=self._debug or exist_ok)
        return tf.summary.FileWriter(newdir, graph=graph)

    def write_hyperparameter_dict(self, hyperparameter_dict):
        hyperparameter_dict["run_name"] = self._run
        hyperparameter_dict["experiment_folder"] = self._experiment_dir

        hyperparameter_dict = {k: v if is_jsonable(v) else str(v) for k, v in
                               hyperparameter_dict.items()}
        with open(opth.join(self._experiment_dir, "hyperparams.json"), 'w') as fp:
            json.dump(hyperparameter_dict, fp, indent=1, skipkeys=True)

    def concat_df(self, df, fnm):
        path = self._sub_path(fnm)
        if opth.exists(path):
            df_old = pd.read_csv(path)
            df = pd.concat([df, df_old])
        df.to_csv(path, index=False, header=True, mode='w')
        return df

    def write_dicts(self, dicts, fnm, ensure_same_cols=True):
        path = opth.join(self._experiment_dir, fnm)
        if isinstance(dicts, dict):
            dicts = [dicts]
        df = pd.DataFrame(dicts)
        if opth.exists(path):
            stored_df = pd.read_csv(path)
            if ensure_same_cols and sorted(stored_df.columns) != sorted(df.columns):
                raise KeyError("Stored df has different columns. If you want to force writing, "
                               "provide ensure_same_cols=False")
            df = pd.concat([stored_df, df])
        df.to_csv(path, index=False, header=True, mode='w')
        return df

    def write_results(self, ensure_same_cols=False, **kwargs):
        self.write_dicts(kwargs, opth.join(self._outdir, self.METARESULTS),
                         ensure_same_cols=ensure_same_cols)

    def write_line(self, fnm, **kwargs):
        return self.write_dicts(kwargs, self._sub_path(fnm), ensure_same_cols=True)

    def write_image(self, img, fnm):
        if img.dtype not in [np.uint8, np.int32, np.int64]:
            img = (img * 255).astype(np.uint8)
        imsave(self._sub_path(fnm), img)

    def write_numpy(self, arr, fnm):
        np.save(self._sub_path(fnm), arr)

    def _sub_path(self, fnm):
        if opth.dirname(fnm):
            os.makedirs(opth.join(self._experiment_dir, opth.dirname(fnm)), exist_ok=True)
        return opth.join(self._experiment_dir, fnm)


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def time_format(seconds):
    learning_time_seconds = 0
    learning_time_minutes = 0
    learning_time_hours = 0
    learning_time_days = 0

    if seconds < 60:
        learning_time_seconds = seconds
    else:
        learning_time_minutes = seconds // 60
        learning_time_seconds = seconds - (learning_time_minutes * 60)

    if learning_time_minutes < 60:
        pass
    else:
        learning_time_hours = learning_time_minutes // 60
        learning_time_minutes = learning_time_minutes - (learning_time_hours * 60)

    if learning_time_hours < 24:
        pass
    else:
        learning_time_days = learning_time_hours // 24
        learning_time_hours = learning_time_hours - (learning_time_days * 24)

    time_string = ""
    if learning_time_days > 0:
        time_string = time_string + ("%d days : " % learning_time_days)
    if learning_time_hours > 0 or learning_time_days > 0:
        time_string = time_string + ("%2d hours : " % learning_time_hours)
    if learning_time_minutes > 0 or learning_time_hours > 0:
        time_string = time_string + ("%2d minutes : " % learning_time_minutes)
    time_string = time_string + ("%2d seconds" % learning_time_seconds)

    return time_string


def listdirs(path):
    return [d for d in os.listdir(path) if opth.isdir(opth.join(path, d))]


def next_logdir(base, datetime=False):
    os.makedirs(base, exist_ok=True)
    path = opth.join(base, "run" + str(len(listdirs(base))).zfill(4))
    if datetime:
        path += dt.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    os.makedirs(path)
    return path



class GroupedMetrics:

    def __init__(self, reporter=None, batch_size=None, reduce_fun=np.mean, transpose=False):
        self.tasks = []
        self._reduce_fun = reduce_fun
        self._reporter = reporter
        self._batch_size = batch_size
        self._transpose = transpose

    def add_task(self, fnm, iterator, fun, metric_names, batch_size=None, return_val=False,
                 desc="Epoch"):
        self.tasks.append((fnm, iterator, batch_size, fun,
                           [metric_names] if isinstance(metric_names, str) else metric_names,
                           return_val, desc))
        return self

    def evaluate_one_epoch(self, epoch):
        result_summary = defaultdict(dict)
        rets = []
        for fnm, iterator, batch_size, fun, metric_names, return_val, desc in self.tasks:
            batch_size = batch_size or self._batch_size
            metrics = defaultdict(list)
            for batch in iterator.iter_epoch(batch_size=batch_size, desc=desc):
                results = fun(*batch, epoch=epoch, step=iterator.step)
                # print(type(results))
                results = [results] if not isinstance(results, (tuple, list)) else results
                # print(metric_names, results)
                for key, val in zip(metric_names, results):
                    # if 'loss_c' in key:
                    #     print(key, val)
                    if isinstance(val, (int, float)) or np.isscalar(val):
                        metrics[key].append(val)
                    elif isinstance(val, np.ndarray):
                        if val.size > 1:
                            metrics[key].extend(val.ravel())
                        elif val.size == 1:
                            metrics[key].append(val.ravel()[0])
                    elif isinstance(val, (tuple, list)):
                        metrics[key].extend(val)
                    else:
                        raise TypeError("Value type {} not supported.".format(type(val)))
            if self._reduce_fun is not None and callable(self._reduce_fun):
                results = {k: self._reduce_fun(v) for k, v in metrics.items()}
            else:
                results = metrics
            if return_val:
                rets.append(results)
            result_summary[fnm].update(results)

        if self._reporter:
            dfs = []
            for fnm, results in result_summary.items():
                if all(isinstance(r, list) for r in results.values()):
                    df = pd.DataFrame(data=results)
                    df['epoch'] = epoch
                    results_df = self._reporter.concat_df(df, fnm)
                else:
                    results_df = self._reporter.write_line(fnm, epoch=epoch, **results)
                results_df['fnm'] = fnm
                dfs.append(results_df)

            df = pd.concat(dfs)
            if self._reduce_fun is None:
                df = df.groupby(['epoch', 'fnm']).mean().reset_index()
            print(self._reporter._name + " intermediate results:")
            to_display = df.sort_values(by=['epoch', 'fnm']).tail(10).sort_values(by=['fnm', 'epoch'])
            print(tabulate.tabulate(to_display.transpose() if self._transpose else to_display,
                headers=df.columns, tablefmt='grid', showindex=False))

        return rets[0] if len(rets) == 1 else rets