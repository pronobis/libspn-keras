import abc
import os
import tensorflow as tf
import time
import numpy as np
from context import libspn as spn

from libspn.tests.profiler import profile_report
import itertools
import argparse
import pandas as pd
import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
import colorama as col

col.init()

red = col.Fore.RED
blue = col.Fore.BLUE
green = col.Fore.GREEN
yellow = col.Fore.YELLOW
magenta = col.Fore.MAGENTA


class AbstractPerformanceUnit(abc.ABC):

    def __init__(self, name, dtype):
        self.name = name
        self._dtype = dtype

    def build(self, inputs, conf, num_stack):
        """ Returns the Op to evaluate  """
        self._placeholders = self._build_placeholders(inputs)
        start_time = time.time()
        with tf.device("/gpu:0" if conf.gpu else "/cpu:0"):
            # TODO support stacking here
            op, init_ops = self._build_op(inputs, self._placeholders, conf)
            init_ops_list = [init_ops]
            for _ in range(num_stack - 1):
                op_list = op if isinstance(op, list) else [op]
                with tf.control_dependencies(op_list):
                    op, init = self._build_op(inputs, self._placeholders, conf)
                    init_ops_list.append(init)

        setup_time = time.time() - start_time
        init_single_op = tf.group(*init_ops_list) if not all(t is None for t in init_ops_list) \
            else None
        return op, init_single_op, self._placeholders, setup_time

    @abc.abstractmethod
    def _build_op(self, inputs, placeholders, conf):
        """ Actually builds the Op given inputs. Device is already set in callee """

    @abc.abstractmethod
    def true_out(self, inputs, conf):
        """ Returns the true output """

    def feed_dict(self, inputs):
        """ Creates the feed dict for this PerformanceUnit """
        return {ph: inp for ph, inp in zip(self._placeholders, inputs.values)}

    @abc.abstractmethod
    def _build_placeholders(self, inputs):
        """ Creates the placeholders """

    def description(self):
        return self.name

    def test_result(self, unit_name, run_times, init_time, setup_time, correct, conf, run_name,
                    **kwargs):
        return {
            "graph_size": len(tf.get_default_graph().get_operations()),
            "first_run_time": run_times[0],
            "rest_run_time": np.mean(run_times[1:]),
            "init_time": init_time,
            "setup_time": setup_time,
            "correct": correct,
            "gpu": conf.gpu,
            "log": conf.log,
            "inf_type": str(conf.inf_type),
            "unit_name": unit_name,
            "run_name": run_name,
            **kwargs
        }

    def _initialize_from(self, root):
        """ Initializes from root  """
        with tf.control_dependencies(None):
            # Explicit clearing of control dependencies allows for easy stacking of sub graphs
            return spn.initialize_weights(root)

    def _generate_weights(self, node, w=None):
        """ Generates the weights for a node using an optional default value """
        with tf.control_dependencies(None):
            # Explicit clearing of control dependencies allows for easy stacking of sub graphs
            if w is None:
                return node.generate_weights()
            return node.generate_weights(w)


def time_fn(fn):
    start_time = time.time()
    ret = fn()
    return ret, time.time() - start_time


class PerformanceTestConfig:

    def __init__(self, inf_type, log, gpu, **kwargs):
        self.inf_type = inf_type
        self.log = log
        self.gpu = gpu
        self.update(**kwargs)

        self.fields = {'inf_type': str(inf_type), 'log': log, 'gpu': gpu, **kwargs}

    def description(self):
        return ' '.join([
            "GPU" if self.gpu else "CPU",
            "LOG" if self.log else "Non-LOG",
            "Marginal" if self.inf_type == spn.InferenceType.MARGINAL else "MPE"]
        )

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class ConfigGenerator:

    def __init__(self, inference_types=None, log=None, gpu=None):
        self.gpu = bool_field(gpu)
        self.inf_type = inference_types or [spn.InferenceType.MARGINAL, spn.InferenceType.MPE]
        self.log = bool_field(log)
        self._fields = {
            'inf_type': self.inf_type,
            'log': self.log,
            'gpu': self.gpu
        }

    def iterate(self, order=('inf_type', 'log', 'gpu')):
        fields_not_in_order = tuple(sorted([f for f in self._fields.keys() if f not in order]))
        for values in itertools.product(
                *[self._fields[field] for field in order + fields_not_in_order]):
            yield PerformanceTestConfig(**{name: val for name, val in zip(order, values)})


class SumConfigGenerator(ConfigGenerator):

    def __init__(self, inference_types=None, log=None, latent_indicators=None, gpu=None):
        super().__init__(inference_types=inference_types, log=log, gpu=gpu)
        self.latent_indicators = bool_field(latent_indicators)

    def field_names(self):
        return {'latent_indicators': self.latent_indicators, **super().field_names()}

    def description(self, conf):
        return super().description(conf) + "IndicatorLeaf={}".format(conf['latent_indicators'])


def bool_field(a):
    ret = a or [False, True]
    return [ret] if not isinstance(ret, list) else ret


class PerformanceTestArgs(argparse.ArgumentParser):
    def __init__(self):
        super(PerformanceTestArgs, self).__init__()
        default_logdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
        self.add_argument('--log-devices', action='store_true',
                          help="Log on which device op is run. Affects run time!")
        self.add_argument('--without-cpu', action='store_true',
                          help="Do not run CPU tests")
        self.add_argument('--without-gpu', action='store_true',
                          help="Do not run GPU tests")
        self.add_argument('--num-runs', default=100, type=int,
                          help="Number of times each test is run")
        self.add_argument('--profile', default=False, action='store_true',
                          help="Run test one more time and profile")
        self.add_argument("--num-stack", default=50, type=int,
                          help="Number of units to stack")
        self.add_argument('--no-logs', action='store_true', dest='no_logs')
        self.add_argument('--logdir', default=default_logdir, type=str,
                          help="Path to log dir")
        self.add_argument('--write-mode', default='safe',
                          choices=['overwrite', 'append', 'safe', 'off'],
                          type=str, help="Whether to overwrite current CSV, append or to do "
                                         "neither")
        self.add_argument('--exit-on-fail', action='store_true', dest='exit_on_fail')
        self.add_argument('--rows', default=500, type=int)
        self.add_argument('--cols', default=100, type=int)
        self.add_argument("--run-name", default='run0001',
                          help='Unique identifier for runs. Can be used for easy cross-run'
                               'comparison of results later.')
        self.add_argument("--plot", action='store_true', dest='plot',
                          help="Plot the results at the end of execution")
        self.add_argument('--plot-only', action='store_true', dest='plot_only',
                          help="Skip testing and just plot with the given configuration")
        self.add_argument("--gpu-name", default='GTX1080', help="Name of GPU device")
        self.add_argument("--cpu-name", default='XeonE5v3', help="Name of CPU device")
        self.add_argument("--verbose", action="store_true", dest="verbose",
                          help="Whether to be verbose about testing output")
        self.add_argument("--tf-logging", action="store_true", dest="tf_logging")


class AbstractPerformanceTest(abc.ABC):

    def __init__(self, name, performance_units, test_args, config_generator):
        self._units = performance_units
        self._system_desc = "{}_{}".format(test_args.cpu_name, test_args.gpu_name)
        logdir = os.path.join(test_args.logdir, name, self._system_desc)
        if os.path.exists(logdir) and test_args.write_mode == 'safe':
            raise FileExistsError("Path already made and write mode set to safe, did you want to "
                                  "overwrite or append?")
        os.makedirs(logdir, exist_ok=True)
        self._logdir = logdir
        self._log_devs = test_args.log_devices
        self._num_runs = test_args.num_runs
        self._exit_on_fail = test_args.exit_on_fail

        self.name = name
        self._config = config_generator
        self._shape = (test_args.rows, test_args.cols)
        self._profile = test_args.profile
        self._write_mode = test_args.write_mode
        self._run_name = test_args.run_name
        self._plot = test_args.plot
        self._plot_only = test_args.plot_only
        self._num_stack = test_args.num_stack
        self._verbose = test_args.verbose

        if not test_args.tf_logging:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            tf.logging.set_verbosity(tf.logging.ERROR)

    def _run_single(self, unit, inputs, conf):
        # For each unit,
        run_times = []
        tf.reset_default_graph()
        op, init_ops, placeholders, setup_time = unit.build(inputs, conf, self._num_stack)
        true_out = unit.true_out(inputs, conf)
        feed_dict = unit.feed_dict(inputs)
        output_correct = True

        # Write graph to file
        op_description = unit.description()
        test_description = self.description()
        config_description = conf.description()
        writer = tf.summary.FileWriter(
            os.path.join(self._logdir, "graphs",
                         "{}_{}_{}".format(
                             test_description, op_description, config_description)),
            graph=tf.get_default_graph())
        writer.flush()
        writer.close()
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=self._log_devs)) as sess:
            init_time = time_fn(lambda: sess.run(init_ops))[1] if init_ops is not None else 0.0
            for n in range(self._num_runs):
                out, run_time = time_fn(lambda: sess.run(op, feed_dict=feed_dict))
                run_times.append(run_time)
                try:
                    if isinstance(out, list):
                        for o, to in zip(out, true_out):
                            self.assert_all_close(o, to)
                    else:
                        self.assert_all_close(out, true_out)
                except AssertionError:
                    output_correct = False
                    self.test_failed = True
                    if self._exit_on_fail:
                        raise RuntimeError("Incorrect output for test {}".format(self.name))

            if self._profile:
                # Create a suitable filename suffix
                # Create a profiling report
                suffix = "{}_{}_{}".format(test_description, op_description, config_description)
                profile_report(sess, op, feed_dict, os.path.join(self._logdir, "profiling"),
                               filename_prefix=self.name, filename_suffix=suffix)
            return unit.test_result(
                unit.name, run_times, init_time, setup_time, output_correct, conf,
                self._run_name)

    def random_numpy_tensor(self, shape):
        np.random.seed(self.seed())
        return np.random.rand(*shape)

    @abc.abstractmethod
    def description(self):
        """ Provides a textual description of the test """

    @abc.abstractmethod
    def generate_input(self):
        """ Generates the input """

    @staticmethod
    def seed():
        """A seed is convenient to ensure e.g. equal weights without having to communicate between
        classes
        """
        return 123

    def run(self, plot_metrics=('rest_run_time', 'graph_size')):
        if self._plot_only:
            self.plot_only()
            return

        results = []
        inputs = self.generate_input()
        for conf in self._config.iterate():
            print(blue + "Config " + conf.description() + col.Style.RESET_ALL)
            for unit in self._units:
                print(red + "Testing " + unit.description() + col.Style.RESET_ALL)
                results.append(self._run_single(unit, inputs, conf))
        self._report_and_write_results(results, plot_metrics)
        return results

    def _report_and_write_results(self, results, plot_metrics):
        df = pd.DataFrame(results)
        results_path = os.path.join(self._logdir, 'results.csv')
        # Print table to stdout
        print(tabulate.tabulate(df, headers=df.columns, tablefmt='grid', showindex=False))

        # Write results to CSV
        if self._write_mode == 'append' and os.path.exists(results_path):
            df.to_csv(results_path, index=False, mode='a', header=False)
            df = pd.read_csv(results_path)
        elif self._write_mode == 'overwrite' or not os.path.exists(results_path):
            df.to_csv(results_path, index=False, mode='w')
        if self._plot:
            # Plot results
            self._make_plots(df, plot_metrics)
        return df

    def plot_only(self, plot_metrics=('rest_run_time', 'graph_size')):
        results_path = os.path.join(self._logdir, 'results.csv')
        df = pd.read_csv(results_path)
        self._make_plots(df, plot_metrics=plot_metrics)

    def _make_plots(self, df, plot_metrics):
        def filter_df_with_dict(df, d):
            ret = df.copy()
            for k, v in d.items():
                ret = ret[ret[k] == v].copy()
            return ret

        # Create separate plots per config
        config_dfs = []
        for conf in self._config.iterate():
            df_filtered = filter_df_with_dict(df, conf.fields)
            for metric in plot_metrics:
                plot = sns.barplot(x=df_filtered["unit_name"], y=df_filtered[metric], capsize=0.2)
                plot.set_title(conf.description().replace('-', ' '))
                plot.get_figure().savefig(
                    os.path.join(self._logdir, conf.description() + "_" + metric + '.png'))
                plt.clf()
            df_filtered['config'] = conf.description()
            config_dfs.append(df_filtered)

        # Create aggregate plots for GPU vs CPU
        for gpu in self._config.gpu:
            for metric in plot_metrics:
                plot = sns.barplot(x=df['unit_name'], y=df[metric])
                plot.set_title("Fully aggregated " + ('GPU' if gpu else 'CPU'))
                plot.get_figure().savefig(
                    os.path.join(self._logdir,
                                 'aggregate' + ('GPU' if gpu else 'CPU') + "_" + metric + '.png'))
                plt.clf()

        # Create a single plot showing all configs at once
        all_configs = pd.concat(config_dfs)
        for metric in plot_metrics:
            plot = sns.factorplot(x='unit_name', y=metric, col='config', kind='bar',
                                  data=all_configs, col_wrap=4)
            plot.savefig(
                os.path.join(self._logdir, 'all_configs' + "_" + metric + '.png'))
            plt.clf()

        # Also plot statistics for different runs
        for metric in plot_metrics:
            plot = sns.factorplot(x='unit_name', y=metric, col='run_name', kind='bar',
                                  data=all_configs, col_wrap=4)
            plot.savefig(
                os.path.join(self._logdir, 'per_run' + "_" + metric + '.png'))
            plt.clf()

    def assert_all_close(self, out, true_out, rtol=1e-6, atol=1e-6, msg=None):
        if not np.allclose(out, true_out, rtol=rtol, atol=atol):
            # Prints more details than np.testing.assert_allclose.
            #
            # NOTE: numpy.allclose (and numpy.testing.assert_allclose)
            # checks whether two arrays are element-wise equal within a
            # tolerance. The relative difference (rtol * abs(b)) and the
            # absolute difference atol are added together to compare against
            # the absolute difference between a and b.  Here, we want to
            # print out which elements violate such conditions.
            cond = np.logical_or(
                np.abs(out - true_out) > atol + rtol * np.abs(true_out),
                np.isnan(out) != np.isnan(true_out))
            if out.ndim:
                x = out[np.where(cond)]
                y = true_out[np.where(cond)]
                if self._verbose:
                    print("not close where = ", np.where(cond))
            else:
                # np.where is broken for scalars
                x, y = out, true_out
            if self._verbose:
                print("not close out      = ", x)
                print("not close true_out = ", y)
                print("not close dif      = ", np.abs(x - y))
                print("not close tol      = ", atol + rtol * np.abs(y))
                print("dtype = %s, shape  = %s" % (out.dtype, out.shape))
            np.testing.assert_allclose(out, true_out, rtol=rtol, atol=atol, err_msg=msg)
