import os

import numpy as np
import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator

from .node import ProcessorNode
from ..helpers.matrix_functions import make_time_dimension_second, put_time_dimension_back_from_second


class InverseModel(ProcessorNode):
    def __init__(self, mne_inverse_model_file_path=None, snr=1.0, method='MNE'):
        super().__init__()
        self._inverse_model_matrix = None
        self.mne_inverse_model_file_path = mne_inverse_model_file_path
        self.snr = snr
        self.method = method

        self.channel_labels = None
        self.channel_cnt = None

    def update(self):
        input_array = self.input_node.output
        self.output = self._apply_inverse_model_matrix(input_array)

    def _apply_inverse_model_matrix(self, input_array: np.ndarray):
        W = self._inverse_model_matrix  # VERTICES x CHANNELS
        output_array = W.dot(make_time_dimension_second(input_array))
        return put_time_dimension_back_from_second(output_array)

    def initialize(self):
        channel_labels = self.traverse_back_and_find('channel_labels')
        channel_cnt = len(channel_labels)
        self._inverse_model_matrix = self._assemble_inverse_model_matrix(channel_labels)
        self.channel_cnt = self._inverse_model_matrix.shape[0]
        self.channel_labels = ['vertex #{}'.format(i + 1) for i in range(self.channel_cnt)]

    def _assemble_inverse_model_matrix(self, channel_labels):
        # Either use the user-provided inverse model file or choose one of the default ones based on channel labels.
        if self.mne_inverse_model_file_path is None:
            self.mne_inverse_model_file_path = self._pick_inverse_model_based_on_channel_labels(channel_labels)
        inverse_operator = read_inverse_operator(self.mne_inverse_model_file_path, verbose='ERROR')

        # First, let's extract the inverse model matrix for all the channels in the inverse operator
        full_operator_matrix = self._matrix_from_inverse_operator(inverse_operator, snr=self.snr, method=self.method)

        # Now let's pick an appropriate row for each label in channel_labels. Channels that are not in the inverse
        # operator will get an all-zeros row.
        inverse_operator_channel_cnt = inverse_operator['info']['nchan']
        inverse_operator_channel_labels = inverse_operator['info']['ch_names']
        picker_matrix = np.zeros((inverse_operator_channel_cnt, len(channel_labels)))
        for (label_index, label) in enumerate(channel_labels):
            try:
                label_index_in_operator = inverse_operator_channel_labels.index(label)
                picker_matrix[label_index_in_operator, label_index] = 1
            except ValueError:
                pass
        return full_operator_matrix.dot(picker_matrix)

    def _matrix_from_inverse_operator(self, inverse_operator, snr, method) -> np.ndarray:
        # Create a dummy mne.Raw object
        channel_cnt = inverse_operator['info']['nchan']
        I = np.identity(channel_cnt)
        dummy_info = inverse_operator['info']
        dummy_info['sfreq'] = self.traverse_back_and_find('frequency')
        dummy_info['projs'] = list()
        dummy_raw = mne.io.RawArray(data=I, info=inverse_operator['info'], verbose='ERROR')
        # dummy_raw.set_eeg_reference(verbose='ERROR')

        # Applying inverse modelling to an identity matrix gives us the forward model matrix
        lambda2 = 1.0 / snr ** 2
        stc = mne.minimum_norm.apply_inverse_raw(dummy_raw, inverse_operator, lambda2, method)
        return stc.data

    def _pick_inverse_model_based_on_channel_labels(self, channel_labels):
        if max(label.startswith('MEG ') for label in channel_labels):
            # sample.data_path() will also download 1.5 Gb so call it only in this branch
            neuromag_inverse = os.path.join(sample.data_path(),
                                            'MEG', 'sample', 'sample_audvis-meg-oct-6-meg-inv.fif')
            return neuromag_inverse
