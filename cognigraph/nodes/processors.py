import numpy as np
from mne.datasets import sample

from .node import ProcessorNode
from ..helpers.matrix_functions import make_time_dimension_second, put_time_dimension_back_from_second


class InverseModel(ProcessorNode):
    def __init__(self, mne_inverse_model_file_path=None):
        super().__init__()
        self._forward_model_matrix = None
        self._mne_inverse_model_file_path = mne_inverse_model_file_path

    def update(self):
        input_array = self.input_node.output
        self.output = self._apply_inverse_model_matrix(input_array)

    def _apply_inverse_model_matrix(self, input_array: np.ndarray):
        W = self._forward_model_matrix  # VERTICES x CHANNELS
        output_array = W.dot(make_time_dimension_second(input_array))
        return put_time_dimension_back_from_second(output_array)

    def init(self):
        channel_labels = self.traverse_back_and_find('channel_labels')
        self._forward_model_matrix = self._assemble_inverse_model_matrix()

        self.channel_cnt = None
        self.channel_labels = None
        assert(self.channel_cnt is not None and self.channel_labels is not None)

    @staticmethod
    def make_inverse_operator():

        from mne.minimum_norm import read_inverse_operator
        data_path = sample.data_path()
        filename_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
        return read_inverse_operator(filename_inv, verbose='ERROR')

    @staticmethod
    def _assemble_inverse_model_matrix(inverse_operator):

        from mne.datasets import sample
        data_path = sample.data_path()
        fname_raw = data_path + '/MEG/sample/sample_audvis_raw.fif'
        raw = mne.io.read_raw_fif(fname_raw, verbose='ERROR')
        info = raw.info
        channel_cnt = info['nchan']

        I = np.identity(channel_cnt)
        dummy_raw = mne.io.RawArray(data=I, info=info, verbose='ERROR')
        dummy_raw.set_eeg_reference(verbose='ERROR')

        # Applying inverse modelling to an identity matrix gives us the forward model matrix
        snr = 1.0  # use smaller SNR for raw data
        lambda2 = 1.0 / snr ** 2
        method = "MNE"  # use sLORETA method (could also be MNE or dSPM)
        stc = mne.minimum_norm.apply_inverse_raw(dummy_raw, inverse_operator, lambda2, method)
        return stc.data

