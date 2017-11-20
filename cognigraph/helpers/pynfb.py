import numpy as np

from .. import TIME_AXIS, PYNFB_TIME_AXIS


def pynfb_ndarray_function_wrapper(pynfb_function):
    """Wraps a pynfb function to take account which axis it uses as the time axis"""
    def wrapped(ndarray: np.ndarray):
        if TIME_AXIS == PYNFB_TIME_AXIS:
            return pynfb_function(ndarray)
        else:
            return pynfb_function(ndarray.T).T