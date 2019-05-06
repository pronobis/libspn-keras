"""Logging routines.

To use in module-level functions::

    from libspn.log import get_logger

    logger = get_logger()

    def fun():
        logger.debug1("Debug message.")

To use in class methods::

    from libspn.log import get_logger

    class Klass:

        __logger = get_logger()     # __ so that derived class can have
        __debug1 = __logger.debug   # a different logger

        def method(self):
            self.__debug1("Debug message.")

If expensive processing is required to calculate input for a debug message,
it can be made conditional on the debug level as follows::

    if Klass.logger.is_debug1():
        val = get_expensive_value()
        Klass.logger.debug1("Debug message with value %s." % val)

Note: Don't forget ``()`` behind ``is_debug1``, which will be evaluated to
``True``!

To quickly setup logging in the application using LibSPN::

    import libspn as spn
    spn.config_logger(spn.DEBUG2)
"""

import sys
import logging
import inspect
import pprint

# Log levels
WARNING = logging.WARNING
"""Log level for warnings."""

INFO = logging.INFO
"""Log level for infos."""

DEBUG1 = logging.DEBUG
"""Log level for debug messages."""

DEBUG2 = DEBUG1 - 1
"""Log level for additional (more verbose) debug messages."""

DEBUG3 = DEBUG2 - 1
"""Log level for additional (even more verbose) debug messages."""


# Add two levels of debugging messages
logging.addLevelName(DEBUG1, "DEBUG1")
logging.addLevelName(DEBUG2, "DEBUG2")
logging.addLevelName(DEBUG3, "DEBUG3")

# Define pretty printer
_pprinter = pprint.PrettyPrinter(indent=1)


class Logger(logging.getLoggerClass()):
    """Custom logger with two debug levels and pretty printing of arguments."""

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def warning(self, msg, *args, **kwargs):
        if self.isEnabledFor(WARNING):
            # Pretty print all arguments
            args = tuple(_pprinter.pformat(a) for a in args)
            self._log(WARNING, msg, args, **kwargs)

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(INFO):
            # Pretty print all arguments
            args = tuple(_pprinter.pformat(a) for a in args)
            self._log(INFO, msg, args, **kwargs)

    def debug1(self, msg, *args, **kwargs):
        if self.isEnabledFor(DEBUG1):
            # Pretty print all arguments
            args = tuple(_pprinter.pformat(a) for a in args)
            self._log(DEBUG1, msg, args, **kwargs)

    def debug2(self, msg, *args, **kwargs):
        if self.isEnabledFor(DEBUG2):
            # Pretty print all arguments
            args = tuple(_pprinter.pformat(a) for a in args)
            self._log(DEBUG2, msg, args, **kwargs)

    def debug3(self, msg, *args, **kwargs):
        if self.isEnabledFor(DEBUG3):
            # Pretty print all arguments
            args = tuple(_pprinter.pformat(a) for a in args)
            self._log(DEBUG3, msg, args, **kwargs)

    def is_warning(self):
        return self.isEnabledFor(WARNING)

    def is_info(self):
        return self.isEnabledFor(INFO)

    def is_debug1(self):
        return self.isEnabledFor(DEBUG1)

    def is_debug2(self):
        return self.isEnabledFor(DEBUG2)

    def is_debug3(self):
        return self.isEnabledFor(DEBUG3)


logging.setLoggerClass(Logger)


def get_logger():
    """Return a logger named ``<module>.<class>``."""
    frm = inspect.stack()[1]
    # Use class name to name the logger and add spn.
    # If no class, use the spn logger
    # We use frm[3] for compatibility with Python 3.4
    # In Python>=3.5 can use frm.function (it is a named tuple)
    if frm[3] == "<module>":
        logger_name = "spn"
    else:
        logger_name = "spn." + frm[3]
    # Module name can be obtained this way:
    # module = inspect.getmodule(frm.frame)
    # module.__name__
    logger = logging.getLogger(logger_name)
    return logger


def config_logger(level=WARNING, file_name=None, stream=sys.stderr):
    """Configure the formatting and level of the logger.

    To display all debug messages, use this in the application using LibSPN::

        import libspn as spn
        spn.config_logger(spn.DEBUG2)

    Args:
        level: Min logger level.
        file_name(str): If not None, log will be saved to the given file.
        stream: If not None, log will be output to the given stream.
    """
    # Create formatter
    formatter = logging.Formatter(
        "[%(levelname)s] [%(name)s:%(funcName)s] %(message)-s")
    # Create handlers
    handlers = []
    if stream is not None:
        handlers.append(logging.StreamHandler(stream))
    if file_name:
        handlers.append(logging.FileHandler(file_name))
    if not handlers:
        handlers.append(logging.NullHandler)
    # Replace all existing handlers of the root logger
    logging.root.handlers = []
    for h in handlers:
        h.setFormatter(formatter)
        logging.root.addHandler(h)
    # Set level
    logging.root.setLevel(level)
