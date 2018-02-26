import mne
from mne.io.meas_info import _kind_dict
import numpy as np

from .. import MISC_CHANNEL_TYPE


def calculate_interpolation_matrix(info: mne.Info):
    """
    Calculates a matrix that interpolates bad channels
    :param info: mne.Info dict with names of channels to be interpolated under the 'bads' key
    :return: matrix M such that M.dot(data) interpolates bad channels (assuming rows in data correspond to channels)
    """
    fill_eeg_channel_locations(info)
    channel_count = info['nchan']
    dummy_raw = mne.io.RawArray(data=np.identity(channel_count), info=info, verbose='ERROR')
    dummy_raw.interpolate_bads(mode='fast')

    return dummy_raw.get_data()


def fill_eeg_channel_locations(info: mne.Info):
    """
    Add standard locations for EEG channels if none were provided. Operates in place.
    """

    # To interpolate we need all EEG channels to have locations, including bad channels
    eeg_channel_ids = mne.pick_types(info, eeg=True, exclude=[])

    # EEG electrode locations are stored in the first three elements of the 'loc' list in info['chs'][idx].
    # If no location is set, then all of them will be zero.
    channels_wo_location = [info['chs'][eeg_channel_id] for eeg_channel_id in eeg_channel_ids
                            if np.all(info['chs'][eeg_channel_id]['loc'][:3] == [0, 0, 0])]

    if any(channels_wo_location):
        print('{channel_count} EEG channels have no locations assigned. Default values will be used.'.format(
            channel_count=len(channels_wo_location)
        ))
        eeg_montage = mne.channels.read_montage(kind='standard_1005')  # type: mne.channels.Montage
        for channel in channels_wo_location:
            try:
                channel_idx_in_montage = eeg_montage.ch_names.index(channel['ch_name'])
            except ValueError:
                ch_name = channel['ch_name']
                print('Could not find channel {name} in the standard montage. Will set it as {misc}'.format(
                    name=ch_name, misc=MISC_CHANNEL_TYPE))
                _set_channel_as_misc(channel)
            else:
                channel['loc'][:3] = eeg_montage.pos[channel_idx_in_montage]


def _set_channel_as_misc(channel):
    kind = _kind_dict[MISC_CHANNEL_TYPE][0]
    channel['kind'] = kind


def read_channel_types(info: mne.Info):
    return [mne.io.pick.channel_type(info, i) for i in range(info['nchan'])]


def channel_labels_saver(mne_info: mne.Info):
    return tuple(mne_info['ch_names'], )


def get_average_reference_projection(channel_count: int):
    """
    Calculates average-reference projection matrix assuming all channel_count channels are used
    :param channel_count: number of channels
    :return: np.ndarray of shape (channel_count, channel_count)
    """
    n = channel_count
    return np.eye(n) - np.ones((n, n)) / n
