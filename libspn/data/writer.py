# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
import numpy as np


class DataWriter(ABC):
    """An abstract class defining the interface of a data writer.

    Args:
        path (str): Full path to the file.
    """

    def __init__(self, path):
        self._path = path

    @abstractmethod
    def write(self, data):
        """Write arrays of data.

        Args:
            data: An array, a list of arrays or a dictionary of arrays with
                  the data to write.
        """


class CSVDataWriter(DataWriter):
    """Writer that writes data in the CSV format.

    Args:
        path (str): Full path to the file.
        delimiter (str): A one-character string used to separate fields.
        fmt_int (str): The format used for storing integers.
        fmt_float (str): The format used for storing floats.
    """

    def __init__(self, path, delimiter=',', fmt_int="%d", fmt_float="%.18e"):
        super().__init__(path)
        self._delimiter = delimiter
        self._fmt_int = fmt_int
        self._fmt_float = fmt_float
        self.__mode = 'wb'  # Initial mode

    def write(self, data):
        """Write arrays of data. The first call to write erases any existing
        file, while all subsequent calls will append to the same file.

        Args:
            data: An array or a list of arrays with the data to write.
        """
        # Ensure list
        if not isinstance(data, list) and not isinstance(data, tuple):
            data = [data]

        # Get columns
        cols = []
        fmt = ['%d', '%f', '%f']
        for d in data:
            if d.ndim == 1:
                cols.append(d)
            elif d.ndim == 2:
                cols.extend(d.T)
            else:
                raise ValueError("Arrays must be 1 or 2 dimensional")

        # Get structured array and formats
        arr = np.rec.fromarrays(cols)
        fmt = [self._fmt_int
               if np.issubdtype(c.dtype, np.integer)
               else self._fmt_float
               for c in cols]

        # Write
        with open(self._path, self.__mode) as csvfile:
            np.savetxt(csvfile, arr, delimiter=',', fmt=fmt)

        # Append further writes
        self.__mode = 'ab'
