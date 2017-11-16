import pylsl as lsl
import numpy as np
from pylsl.pylsl import fmt2string, string2fmt
import uuid


def convert_lsl_format_to_numpy(lsl_channel_format: int):
    return fmt2string[lsl_channel_format]


def convert_numpy_format_to_lsl(numpy_channel_format: np.dtype):
    return string2fmt[str(np.dtype)]


def create_lsl_outlet(name, type, frequency, channel_format, channel_labels):
    # Create StreamInfo
    channel_cnt = len(channel_labels)
    source_id = str(uuid.uuid4())
    info = lsl.StreamInfo(name=name, type=type, channel_count=channel_cnt, nominal_srate=frequency,
                          channel_format=channel_format, source_id=source_id)

    # Add channel labels
    desc = info.desc()
    channels_tag = desc.append_child(name='channels')
    for label in channel_labels:
        single_channel_tag = channels_tag.append_child(name='channel')
        single_channel_tag.append_child_value(name='label', value=label)

    # Create outlet
    return lsl.StreamOutlet(info)
