import os

import numpy as np
import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator

data_path = sample.data_path(verbose='ERROR')
neuromag_inverse_file_path = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-oct-6-meg-inv.fif')
standard_1005_forward_file_path = os.path.join(data_path, 'MEG', 'sample', 'sample_1005-eeg-oct-6-fwd.fif')
standard_1005_inverse_file_path = os.path.join(data_path, 'MEG', 'sample', 'sample_1005-eeg-oct-6-eeg-inv.fif')


def _fake_standard_1005_info(channel_labels):
    montage_1005 = mne.channels.read_montage(kind='standard_1005')
    montage_labels_upper = [label.upper() for label in montage_1005.ch_names]
    ch_types = ['eeg' if label.upper() in montage_labels_upper else 'misc'
                for label in channel_labels]
    fake_info = mne.create_info(ch_names=channel_labels, sfreq=1000, ch_types=ch_types,)
    return fake_info


def _make_standard_1005_inverse_operator(channel_labels):

    forward = mne.read_forward_solution(standard_1005_forward_file_path)
    fake_info = _fake_standard_1005_info(channel_labels)
    G = forward['sol']['data']
    q = np.trace(G.dot(G.T)) / G.shape[0]
    cov = mne.Covariance(data=(q * np.identity(fake_info['nchan'])),  # G.dot(G.T) and cov now have one scale
                         names=fake_info['ch_names'],
                         bads=fake_info['bads'],
                         projs=fake_info['projs'],
                         nfree=1)

    return mne.minimum_norm.make_inverse_operator(fake_info, forward, cov)


def get_inverse_model_matrix_from_labels(channel_labels, snr, method):
    montage_1005 = mne.channels.read_montage(kind='standard_1005')
    montage_labels_upper = [montage_label.upper() for montage_label in montage_1005.ch_names]

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


def _pick_channels_from_matrix(full_inverse_model_matrix, channel_labels, inverse_operator_channel_labels):

    # Let's pick an appropriate row for each label in channel_labels. Channels that are not in the inverse
    # operator will get an all-zeros row.
    inverse_operator_channel_count = len(inverse_operator_channel_labels)
    picker_matrix = np.zeros((inverse_operator_channel_count, len(channel_labels)))
    for (label_index, label) in enumerate(channel_labels):
        try:
            label_index_in_operator = inverse_operator_channel_labels.index(label)
            picker_matrix[label_index_in_operator, label_index] = 1
        except ValueError:
            pass
    return full_inverse_model_matrix.dot(picker_matrix)


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
    stc = mne.minimum_norm.apply_inverse_raw(dummy_raw, inverse_operator, lambda2, method)

    return stc.data


def get_inverse_model_matrix(mne_inverse_model_file_path, channel_labels, snr, method):
    inverse_operator = read_inverse_operator(mne_inverse_model_file_path)
    full_inverse_model_matrix = _matrix_from_inverse_operator(inverse_operator, snr, method)
    inverse_model_matrix = _pick_channels_from_matrix(full_inverse_model_matrix, channel_labels,
                                                      inverse_operator['info']['ch_names'])
    return inverse_model_matrix
