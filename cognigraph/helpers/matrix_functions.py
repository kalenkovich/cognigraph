import numpy as np

from .. import TIME_DIMENSION_ID, CHANNEL_DIMENSION_ID


def make_time_dimension_second(ndarray: np.ndarray):
    """ To different people chunks of data are either TIME x CHANNEL or CHANNEL x TIME. We don't want to impose our
    preference on users so this can be set with TIME_DIMENSION_ID constant. But at times we do need to have it in a
    particular way. Hence, this function. """
    if TIME_DIMENSION_ID == 1:
        return ndarray
    elif TIME_DIMENSION_ID == 0:
        return ndarray.T


def put_time_dimension_back_from_second(ndarray: np.ndarray):
    if TIME_DIMENSION_ID == 1:
        return ndarray
    elif TIME_DIMENSION_ID == 0:
        return ndarray.T


put_time_dimension_back_from_second.__doc__ = make_time_dimension_second.__doc__
