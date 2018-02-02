import uuid

import pylsl as lsl
import numpy as np
from pylsl.pylsl import fmt2string, string2fmt

from .. import TIME_AXIS

string2fmt['float64'] = string2fmt['double64']
LSL_TIME_DIMENSION_ID = 0


def convert_lsl_format_to_numpy(lsl_channel_format: int):
    return fmt2string[lsl_channel_format]


def convert_numpy_format_to_lsl(numpy_channel_format: np.dtype):
    return string2fmt[str(numpy_channel_format)]


def create_lsl_outlet(name, frequency, channel_format, channel_labels, channel_types, type=''):
    # Create StreamInfo
    channel_count = len(channel_labels)
    source_id = str(uuid.uuid4())
    info = lsl.StreamInfo(name=name, type=type, channel_count=channel_count, nominal_srate=frequency,
                          channel_format=channel_format, source_id=source_id)

    # Add channel labels
    desc = info.desc()
    channels_tag = desc.append_child(name='channels')
    for label, channel_type in zip(channel_labels, channel_types):
        single_channel_tag = channels_tag.append_child(name='channel')
        single_channel_tag.append_child_value(name='label', value=label)
        single_channel_tag.append_child_value(name='type', value=channel_type)

    # Create outlet
    return lsl.StreamOutlet(info)


def _transpose_if_need_be(ndarray: np.ndarray):
    """ Just a helper function for the next two conversion functions """
    if LSL_TIME_DIMENSION_ID != TIME_AXIS:
        return ndarray.T
    else:
        return ndarray


def convert_lsl_chunk_to_numpy_array(lsl_chunk, dtype=None):
    """
    An LSL chunk is a list of lists. Converting it to a numpy array and vice versa is obviously a trivial matter not
    worthy of a designated function. What *does* need taking care of, is whether we should transpose the numpy array or
    not. In LSL time is the first dimension. We might or might not adhere to this convention, which is reflected in the
    TIME_AXIS constant from the base package.
    """
    ndarray = np.array(lsl_chunk, dtype=dtype)
    return _transpose_if_need_be(ndarray)


def convert_numpy_array_to_lsl_chunk(ndarray):
    ndarray = _transpose_if_need_be(ndarray)
    return ndarray.tolist()


convert_numpy_array_to_lsl_chunk.__doc__ = convert_lsl_chunk_to_numpy_array.__doc__


def read_channel_labels_from_info(info: lsl.StreamInfo):
    channels_tag = info.desc().child('channels')
    if channels_tag.empty():
        return None
    else:
        # TODO: this is hard to read. Write a generator for children with a given name in helpers
        labels = list()
        types = list()
        single_channel_tag = channels_tag.child(name="channel")
        for channel_id in range(info.channel_count()):
            labels.append(single_channel_tag.child_value(name='label'))
            types.append(single_channel_tag.child_value(name='type'))
            single_channel_tag = single_channel_tag.next_sibling(name='channel')
        return labels, types