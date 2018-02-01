import os

import numpy as np
import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator

from .. import MISC_CHANNEL_TYPE
from ..helpers.misc import all_upper

data_path = sample.data_path(verbose='ERROR')
sample_dir = os.path.join(data_path, 'MEG', 'sample')
neuromag_forward_file_path = os.path.join(sample_dir, 'sample_audvis-meg-oct-6-meg-inv.fif')
neuromag_inverse_file_path = os.path.join(sample_dir, 'sample_audvis-meg-oct-6-fwd.fif')
standard_1005_forward_file_path = os.path.join(sample_dir, 'sample_1005-eeg-oct-6-fwd.fif')
standard_1005_inverse_file_path = os.path.join(sample_dir, 'sample_1005-eeg-oct-6-eeg-inv.fif')


def _fake_standard_1005_info(channel_labels):
    montage_1005 = mne.channels.read_montage(kind='standard_1005')
    montage_labels_upper = all_upper(montage_1005.ch_names)
    ch_types = ['eeg' if label.upper() in montage_labels_upper else MISC_CHANNEL_TYPE
                for label in channel_labels]
    fake_info = mne.create_info(ch_names=channel_labels, sfreq=1000, ch_types=ch_types,)
    return fake_info


def _make_standard_1005_inverse_operator(channel_labels):

    forward = mne.read_forward_solution(standard_1005_forward_file_path, verbose='ERROR')
    fake_info = _fake_standard_1005_info(channel_labels)
    G = forward['sol']['data']
    q = np.trace(G.dot(G.T)) / G.shape[0]
    cov = mne.Covariance(data=(q * np.identity(fake_info['nchan'])),  # G.dot(G.T) and cov now have one scale
                         names=fake_info['ch_names'],
                         bads=fake_info['bads'],
                         projs=fake_info['projs'],
                         nfree=1)

    return mne.minimum_norm.make_inverse_operator(fake_info, forward, cov, verbose='ERROR')


def get_inverse_model_matrix_from_labels(channel_labels, snr, method):
    montage_1005 = mne.channels.read_montage(kind='standard_1005')
    montage_labels_upper = all_upper(montage_1005.ch_names)

    if max(label.startswith('MEG ') for label in channel_labels) is True:

        inverse_operator = read_inverse_operator(neuromag_inverse_file_path)
        mne_inverse_model_file_path = neuromag_inverse_file_path

    elif any([channel_label.upper() in montage_labels_upper for channel_label in channel_labels]):

        inverse_operator = _make_standard_1005_inverse_operator(channel_labels)
        mne_inverse_model_file_path = standard_1005_inverse_file_path

    full_inverse_model_matrix = _matrix_from_inverse_operator(inverse_operator, snr, method)
    inverse_model_matrix = _pick_channels_from_matrix(full_inverse_model_matrix, channel_labels,
                                                      inverse_operator['info']['ch_names'])

    return inverse_model_matrix, mne_inverse_model_file_path


def _pick_channels_from_matrix(matrix, channel_labels, matrix_channel_labels):
    """
    From matrix take only the columns that correspond to channel_labels - in the right order
    :param matrix: each column in matrix corresponds to one channel
    :param channel_labels: labels that we need
    :param matrix_channel_labels: labels that we have
    :return: np.ndarray with len(channel_labels) columns and the same number of rows as matrix has. Each row corresponds
    to one label from channel_labels in the right order. If that label was not represented in matrix then the row will
    be all zeros
    """
    picker_matrix = np.zeros((len(matrix_channel_labels), len(channel_labels)))
    for (label_index, label) in enumerate(channel_labels):
        try:
            label_index_in_operator = matrix_channel_labels.index(label)
            picker_matrix[label_index_in_operator, label_index] = 1
        except ValueError:
            pass
    return matrix.dot(picker_matrix)


def _matrix_from_inverse_operator(inverse_operator, snr, method) -> np.ndarray:
    # Create a dummy mne.Raw object
    channel_count = inverse_operator['info']['nchan']
    I = np.identity(channel_count)
    dummy_info = inverse_operator['info']
    dummy_info['sfreq'] = 1000
    dummy_info['projs'] = list()
    dummy_raw = mne.io.RawArray(data=I, info=inverse_operator['info'], verbose='ERROR')
    contains_eeg_channels = len(mne.pick_types(dummy_info, meg=False, eeg=True)) > 0
    if contains_eeg_channels:
        dummy_raw.set_eeg_reference(verbose='ERROR')

    # Applying inverse modelling to an identity matrix gives us the forward model matrix
    lambda2 = 1.0 / snr ** 2
    stc = mne.minimum_norm.apply_inverse_raw(dummy_raw, inverse_operator, lambda2, method, pick_ori='normal',
                                             verbose='ERROR')

    return stc.data


def get_inverse_model_matrix(mne_inverse_model_file_path, channel_labels, snr, method):
    inverse_operator = read_inverse_operator(mne_inverse_model_file_path)
    full_inverse_model_matrix = _matrix_from_inverse_operator(inverse_operator, snr, method)
    inverse_model_matrix = _pick_channels_from_matrix(full_inverse_model_matrix, channel_labels,
                                                      inverse_operator['info']['ch_names'])
    return inverse_model_matrix


def get_default_forward_file(mne_info: mne.Info):
    """
    Based on the labels of channels in mne_info return either neuromag or standard 1005 forward model file
    :param mne_info - mne.Info instance
    :return: str: path to the forward-model file
    """
    channel_labels_upper = all_upper(mne_info['ch_names'])

    if max(label.startswith('MEG ') for label in channel_labels_upper) is True:
        return neuromag_forward_file_path

    else:
        montage_1005 = mne.channels.read_montage(kind='standard_1005')
        montage_labels_upper = all_upper(montage_1005.ch_names)
        if any([label_upper in montage_labels_upper for label_upper in channel_labels_upper]):
            return standard_1005_forward_file_path


def assemble_gain_matrix(forward_model_path: str, mne_info: mne.Info):
    """
    Assemble the gain matrix from the forward model so that its rows correspond to channels in mne_info
    :param forward_model_path:
    :param mne_info:
    :return: np.ndarray with as many rows as there are dipoles in the forward model and as many rows as there are
    channels in mne_info. Throws an error if less than half of channels are present in the forward model.
    """
    channel_labels_upper = all_upper(mne_info['ch_names'])

    forward = mne.read_forward_solution(forward_model_path, verbose='ERROR')
    mne.convert_forward_solution(forward, force_fixed=True, copy=False, verbose='ERROR')
    G_forward = forward['sol']['data']
    channel_labels_forward = all_upper(forward['info']['ch_names'])

    return _pick_channels_from_matrix(G_forward.T, channel_labels_upper, channel_labels_forward).T
