# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from libspn import log
from abc import ABC, abstractmethod
import argparse
import sys
import traceback
import colorama as col
import yaml
import collections


def update_dict(d, u):
    """Update dictionary recursively."""
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = update_dict(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


class App(ABC):
    """A helper class for building scripts/applications.

    The logic of the app should be placed in :func:`run`. To define command
    line arguments, override :func:`define_args` and :func:`process_args`.

    To print inside the app, use either a logger defied in the app, or color
    print with :func:`print1` and :func:`print2`. Finally, one can simply use
    the regular Python :func:`print`. To report a fatal error, use
    :func:`error`. All output (including exceptions) is saved to a file if
    ``--out`` is specified in the command line.

    Args:
        description (str): App description.
    """

    class Parser(argparse.ArgumentParser):
        """Custom parser with better error display."""

        def __init__(self, app, prog=None, usage=None, description=None,
                     epilog=None, parents=[],
                     prefix_chars='-', fromfile_prefix_chars=None,
                     argument_default=None, conflict_handler='error',
                     add_help=True, allow_abbrev=True):
            super().__init__(prog=prog, usage=usage, description=description,
                             epilog=epilog, parents=parents,
                             prefix_chars=prefix_chars,
                             fromfile_prefix_chars=fromfile_prefix_chars,
                             argument_default=argument_default,
                             conflict_handler=conflict_handler,
                             add_help=add_help, allow_abbrev=allow_abbrev,
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            self._app = app

        def error(self, message):
            self.print_help()
            self._app.error(message)

    class StreamFork():
        """Forks a stream to another stream and a file."""

        def __init__(self, stream, file):
            self.stream = stream
            self.file = file

        def write(self, message):
            self.stream.write(message)
            self.file.write(message)

        def flush(self):
            self.stream.flush()
            self.file.flush()

    def __init__(self, description):
        self.description = 'LibSPN: ' + description
        col.init()
        self._out_file = None
        self._orig_stdout = sys.stdout  # Used to not copy color codes to file
        self._orig_stderr = sys.stderr  # Used to not copy color codes to file

    def main(self):
        """Main function of the app."""
        # Argument parsing
        parser = App.Parser(app=self,
                            description=self.description)
        parser_optional = parser._action_groups.pop()
        commands = parser.add_subparsers(title='sub-commands')
        parser._action_groups.append(parser_optional)
        self.define_args(parser, commands)
        other_params = parser.add_argument_group(title="other")
        other_params.add_argument('-c', '--conf', action='append',
                                  type=str, default=[],
                                  help="read configuration from file")
        other_params.add_argument('-v', '--debug1', action='store_true',
                                  help="print log messages at level DEBUG1")
        other_params.add_argument('-vv', '--debug2', action='store_true',
                                  help="print log messages at level DEBUG2")
        other_params.add_argument('-o', '--out', type=str,
                                  metavar='FILE',
                                  help="save output to FILE")
        # Update defaults from config files
        conf_files = self._parse_config_args(parser, commands)
        self._read_config_files(parser, commands, conf_files)
        # Parse all arguments
        args = self._parse_args(parser, commands)
        # Redirect copy of output to a file
        if args.out:
            self._out_file = open(args.out, 'w')
            sys.stdout = App.StreamFork(sys.stdout, self._out_file)
            sys.stderr = App.StreamFork(sys.stderr, self._out_file)
        # Configure logger to output to the new stderr at specified level
        if args.debug2:
            log_level = log.DEBUG2
        elif args.debug1:
            log_level = log.DEBUG1
        else:
            log_level = log.INFO
        log.config_logger(log_level, stream=sys.stderr)
        # Process and print
        self._print_header(args)
        self.process_args(parser, args)
        # Run the app
        try:
            self.run(args)
        except Exception as e:
            # Print exception traceback to save it to file before
            # the file is closed in finally
            print(traceback.format_exc(), end='')
            sys.exit(1)
        finally:
            if self._out_file is not None:
                # Revert streams and close file
                sys.stderr = self._orig_stderr
                sys.stdout = self._orig_stdout
                log.config_logger(log_level, stream=sys.stderr)
                self._out_file.close()

    @abstractmethod
    def run(self, args):
        """Implement app functionality here."""

    @abstractmethod
    def define_args(self, parser, commands):
        """Define argparse arguments here.

        Args:
            parse (argparse.ArgumentParser): The root parser.
            commands (argparse._SubParsersAction): Use to define commands.
        """

    def process_args(self, parser, args):
        """Test and process values of arguments in ``args`` here. Report
        parsing error using ``parser.error()``."""

    def print1(self, msg):
        """Print with color 1."""
        if self._out_file is not None:
            print(msg, file=self._out_file)
        print(col.Fore.YELLOW + msg + col.Style.RESET_ALL,
              file=self._orig_stdout)

    def print2(self, msg):
        """Print with color 2."""
        if self._out_file is not None:
            print(msg, file=self._out_file)
        print(col.Fore.BLUE + msg + col.Style.RESET_ALL,
              file=self._orig_stdout)

    def error(self, msg=None):
        """Report an error and exit the app."""
        msg = "ERROR: " + str(msg)
        if msg is not None:
            if self._out_file is not None:
                print(msg, file=self._out_file)
            print(col.Fore.RED + msg + col.Style.RESET_ALL,
                  file=self._orig_stderr)
        sys.exit(1)

    @staticmethod
    def bool_arg(v):
        """Enables easy parsing of Boolean command line arguments."""
        t_vals = ('yes', 'true', 't', 'y', '1')
        f_vals = ('no', 'false', 'f', 'n', '0')
        if v.lower() in t_vals:
            return True
        elif v.lower() in f_vals:
            return False
        else:
            raise argparse.ArgumentTypeError('boolean value expected (%s or %s).'
                                             % (', '.join(t_vals), ', '.join(f_vals)))

    def _parse_config_args(self, parser, commands):
        """Parse arguments responsible for specifying config files."""
        # Divide argv by commands
        split_argv = [[]]
        for c in sys.argv[1:]:
            if c in commands.choices:
                split_argv.append([c])
            else:
                split_argv[-1].append(c)
        # Parse
        args, _ = parser.parse_known_args(split_argv[0])
        return args.conf

    def _parse_args(self, parser, commands):
        """Parse arguments considering commands."""
        # Divide argv by commands
        split_argv = [[]]
        for c in sys.argv[1:]:
            if c in commands.choices:
                split_argv.append([c])
            else:
                split_argv[-1].append(c)
        # Initialize final namespace
        args = argparse.Namespace()
        for c in commands.choices:
            setattr(args, c, None)
        # Parse pre-command
        parser.parse_args(split_argv[0], namespace=args)
        # Parse commands
        for argv in split_argv[1:]:  # Commands
            # Initialize namespace for the command
            n = argparse.Namespace()
            setattr(args, argv[0], n)
            # Parse
            parser.parse_args(argv, namespace=n)
        return args

    def _print_header(self, args):
        self.print1("======================================")
        self.print1(self.description)
        self.print1("======================================")
        self.print1("Args:")
        for name, val in sorted(vars(args).items()):
            if name not in {'out', 'debug1', 'debug2', 'conf'}:
                var = getattr(args, name)
                if isinstance(var, argparse.Namespace):  # command
                    self.print1("- %s:" % name)
                    for n, v in sorted(vars(var).items()):
                        if not hasattr(args, n):  # Ignore root arguments
                            self.print1("  - %s: %s" % (n, v))
                else:
                    self.print1("- %s: %s" % (name, val))

        self.print1("======================================")

    def _read_config_files(self, parser, commands, conf_files):
        """Read config files and update defaults of arguments."""
        def update_defaults(parser, key, value):
            action = [a for a in parser._actions if a.dest == key]
            if action:
                a = action[0]
                # Check value against choices
                if a.choices and value not in a.choices:
                    parser.error("invalid config value '%s' for key '%s' (choose from: %s)"
                                 % (value, key, ', '.join(a.choices)))
                # Update
                a.default = value
            else:
                parser.error("invalid config key '%s'" % key)

        # Load all config files
        config = {}
        for cfile in conf_files:
            try:
                with open(cfile, 'r') as stream:
                    update_dict(config, yaml.load(stream))
            except FileNotFoundError:
                parser.error("config file '%s' not found" % cfile)
        # Update defaults
        for k1, v1 in config.items():
            if isinstance(v1, dict):
                # Subcommands
                for subcmd_name, subcmd_parser in commands.choices.items():
                    if subcmd_name == k1:
                        # Subcommand arguments
                        for k2, v2 in v1.items():
                            update_defaults(subcmd_parser, k2, v2)
            else:
                # Arguments
                update_defaults(parser, k1, v1)

