import numpy as np
from scipy.signal import lfilter

from vendor.nfb.pynfb.signal_processing.filters import BaseFilter

from .. import TIME_AXIS, PYNFB_TIME_AXIS


def pynfb_ndarray_function_wrapper(pynfb_function):
    """Wraps a pynfb function to take account which axis it uses as the time axis"""
    def wrapped(ndarray: np.ndarray):
        if TIME_AXIS == PYNFB_TIME_AXIS:
            return pynfb_function(ndarray)
        else:
            return pynfb_function(ndarray.T).T
    return wrapped


class ExponentialMatrixSmoother(BaseFilter):
    def __init__(self, factor, column_count):
        self.a = [1, -factor]
        self.b = [1 - factor]
        self.column_count = column_count
        self.reset()

    def apply(self, chunk: np.ndarray):
        y, self.zi = lfilter(self.b, self.a, chunk, axis=0, zi=self.zi)
        return y

    def reset(self):
        self.zi = np.zeros((max(len(self.a), len(self.b)) - 1, self.column_count))
