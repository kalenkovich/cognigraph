import os

import numpy as np
import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator

from .. import MISC_CHANNEL_TYPE
from ..helpers.misc import all_upper

data_path = sample.data_path(verbose='ERROR')
sample_dir = os.path.join(data_path, 'MEG', 'sample')
neuromag_forward_file_path = os.path.join(sample_dir, 'sample_audvis-meg-oct-6-fwd.fif')
standard_1005_forward_file_path = os.path.join(sample_dir, 'sample_1005-eeg-oct-6-fwd.fif')


def _pick_columns_from_matrix(matrix: np.ndarray, output_column_labels: list, column_labels: list) -> np.ndarray:
    """
    From matrix take only the columns that correspond to column_labels - in the right order.
    :param matrix: each column in matrix has a label (eg. EEG channel name)
    :param output_column_labels: labels that we need
    :param column_labels: labels that we have
    :return: np.ndarray with len(output_column_labels) columns and the same number of rows as matrix has.
    """
    picker_matrix = np.zeros((len(column_labels), len(output_column_labels)))
    for (label_index, label) in enumerate(output_column_labels):
        try:
            label_index_in_operator = column_labels.index(label)
            picker_matrix[label_index_in_operator, label_index] = 1
        except ValueError:
            pass
    return matrix.dot(picker_matrix)


def matrix_from_inverse_operator(inverse_operator, mne_info, snr, method) -> np.ndarray:
    # Create a dummy mne.Raw object
    channel_count = mne_info['nchan']
    I = np.identity(channel_count)
    dummy_raw = mne.io.RawArray(data=I, info=mne_info, verbose='ERROR')
    contains_eeg_channels = len(mne.pick_types(mne_info, meg=False, eeg=True)) > 0
    if contains_eeg_channels:
        dummy_raw.set_eeg_reference(verbose='ERROR')

    # Applying inverse modelling to an identity matrix gives us the inverse model matrix
    lambda2 = 1.0 / snr ** 2
    stc = mne.minimum_norm.apply_inverse_raw(dummy_raw, inverse_operator, lambda2, method,
                                             verbose='ERROR')

    return stc.data


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

    return _pick_columns_from_matrix(G_forward.T, channel_labels_upper, channel_labels_forward).T


def make_inverse_operator(forward_model_file_path, mne_info, sigma2=1):
    # sigma2 is what will be used to scale the identity covariance matrix. This will not affect the MNE solution though.
    # The inverse operator will use channels common to forward_model_file_path and mne_info.
    forward = mne.read_forward_solution(forward_model_file_path, verbose='ERROR')
    cov = mne.Covariance(data=sigma2 * np.identity(mne_info['nchan']),
                         names=mne_info['ch_names'],
                         bads=mne_info['bads'],
                         projs=mne_info['projs'],
                         nfree=1
                         )

    return mne.minimum_norm.make_inverse_operator(mne_info, forward, cov, verbose='ERROR')
