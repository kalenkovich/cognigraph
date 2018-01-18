from typing import Tuple

import numpy as np
import mne

from .node import ProcessorNode
from ..helpers.matrix_functions import make_time_dimension_second, put_time_dimension_back_from_second
from ..helpers.inverse_model import get_inverse_model_matrix, get_inverse_model_matrix_from_labels
from ..helpers.pynfb import pynfb_ndarray_function_wrapper, ExponentialMatrixSmoother
from vendor.nfb.pynfb.signal_processing import filters


class Preprocessing(ProcessorNode):

    def __init__(self, duration):


    def _on_input_history_invalidation(self):
        self._reset_statistics()

    def _check_value(self, key, value):
        pass

    CHANGES_IN_THESE_REQUIRE_RESET = ('mne_info', )

    def _initialize(self):
        pass

    def _reset(self) -> bool:
        pass

    def _update(self):
        pass

    @property
    def UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION(self) -> Tuple[str]:
        pass


class InverseModel(ProcessorNode):
    def _on_input_history_invalidation(self):
        # The methods implemented in this node do not rely on past inputs
        pass

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    CHANGES_IN_THESE_REQUIRE_RESET = ('mne_inverse_model_file_path', 'snr', 'method')

    def _check_value(self, key, value):
        if key == 'method':
            if value not in self.SUPPORTED_METHODS:
                raise ValueError('Method {} is not supported. We support only {}'.format(value, self.SUPPORTED_METHODS))

        if key == 'snr':
            if value <= 0:
                raise ValueError('snr (signal-to-noise ratio) must be a positive number. See mne-python docs.')

    def _reset(self):
        self.mne_inverse_model_file_path = self._user_provided_inverse_model_file_path
        self._should_reinitialize = True
        self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    SUPPORTED_METHODS = ['MNE', 'dSPM', 'sLORETA']

    def __init__(self, mne_inverse_model_file_path=None, snr=1.0, method='MNE'):
        super().__init__()
        self._inverse_model_matrix = None
        self._user_provided_inverse_model_file_path = mne_inverse_model_file_path
        self._mne_inverse_model_file_path = mne_inverse_model_file_path
        self.snr = snr
        self.method = method

        self.mne_info = None

    @property
    def mne_inverse_model_file_path(self):
        return self._mne_inverse_model_file_path

    @mne_inverse_model_file_path.setter
    def mne_inverse_model_file_path(self, value):
        self._user_provided_inverse_model_file_path = self._mne_inverse_model_file_path = value

    def _update(self):
        input_array = self.input_node.output
        self.output = self._apply_inverse_model_matrix(input_array)

    def _apply_inverse_model_matrix(self, input_array: np.ndarray):
        W = self._inverse_model_matrix  # VERTICES x CHANNELS
        output_array = W.dot(make_time_dimension_second(input_array))
        return put_time_dimension_back_from_second(output_array)

    def _initialize(self):
        mne_info = self.traverse_back_and_find('mne_info')
        channel_labels = mne_info['ch_names']
        if self._user_provided_inverse_model_file_path is None:
            self._inverse_model_matrix, self._mne_inverse_model_file_path = \
                get_inverse_model_matrix_from_labels(channel_labels, snr=self.snr, method=self.method)
        else:
            self._inverse_model_matrix = get_inverse_model_matrix(self._user_provided_inverse_model_file_path,
                                                                  channel_labels, snr=self.snr, method=self.method)

        frequency = mne_info['sfreq']
        channel_count = self._inverse_model_matrix.shape[0]
        channel_labels = ['vertex #{}'.format(i + 1) for i in range(channel_count)]
        self.mne_info = mne.create_info(channel_labels, frequency)

class LinearFilter(ProcessorNode):

    def _on_input_history_invalidation(self):
        if self._linear_filter is not None:
            self._linear_filter.reset()

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    CHANGES_IN_THESE_REQUIRE_RESET = ('lower_cutoff', 'upper_cutoff')

    def _check_value(self, key, value):
        if value is None:
            pass

        elif key == 'lower_cutoff':
            if hasattr(self, 'upper_cutoff') and self.upper_cutoff is not None and value > self.upper_cutoff:
                raise ValueError('Lower cutoff cannot be set higher that the upper cutoff')
            if value < 0:
                raise ValueError('Lower cutoff must be a positive number')

        elif key == 'upper_cutoff':
            if hasattr(self, 'upper_cutoff') and self.lower_cutoff is not None and value < self.lower_cutoff:
                raise ValueError('Upper cutoff cannot be set lower that the lower cutoff')
            if value < 0:
                raise ValueError('Upper cutoff must be a positive number')

    def _reset(self):
        self._should_reinitialize = True
        self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    def __init__(self, lower_cutoff, upper_cutoff):
        super().__init__()
        self.lower_cutoff = lower_cutoff
        self.upper_cutoff = upper_cutoff
        self._linear_filter = None  # type: filters.ButterFilter

    def _initialize(self):
        mne_info = self.traverse_back_and_find('mne_info')
        frequency = mne_info['sfreq']
        channel_count = mne_info['nchan']
        if not (self.lower_cutoff is None and self.upper_cutoff is None):
            band = (self.lower_cutoff, self.upper_cutoff)
            self._linear_filter = filters.ButterFilter(band, fs=frequency, n_channels=channel_count)
            self._linear_filter.apply = pynfb_ndarray_function_wrapper(self._linear_filter.apply)
        else:
            self._linear_filter = None

    def _update(self):
        input = self.input_node.output
        if self._linear_filter is not None:
            self.output = self._linear_filter.apply(input)
        else:
            self.output = input


class EnvelopeExtractor(ProcessorNode):
    def _on_input_history_invalidation(self):
        pass

    def _check_value(self, key, value):
        if key == 'factor':
            if value <= 0 or value >= 1:
                raise ValueError('Factor must be a number between 0 and 1')

        if key == 'method':
            if value not in self.SUPPORTED_METHODS:
                raise ValueError('Method {} is not supported. We support only {}'.format(value, self.SUPPORTED_METHODS))

    def _reset(self):
        self._envelope_extractor.reset()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    CHANGES_IN_THESE_REQUIRE_RESET = ('method', 'factor')
    SUPPORTED_METHODS = ('Exponential smoothing', )

    def __init__(self, factor=0.9):
        super().__init__()
        self.method = 'Exponential smoothing'
        self.factor = factor
        self._envelope_extractor = None  # type: ExponentialMatrixSmoother

    def _initialize(self):
        channel_count = self.traverse_back_and_find('mne_info')['nchan']
        self._envelope_extractor = ExponentialMatrixSmoother(factor=self.factor, column_count=channel_count)
        self._envelope_extractor.apply = pynfb_ndarray_function_wrapper(self._envelope_extractor.apply)

    def _update(self):
        input = self.input_node.output
        self.output = self._envelope_extractor.apply(input)


# TODO: implement this function
def pynfb_filter_based_processor_class(pynfb_filter_class):
    """Returns a ProcessorNode subclass with the functionality of pynfb_filter_class

    pynfb_filter_class: a subclass of pynfb.signal_processing.filters.BaseFilter

    Sample usage 1:

    LinearFilter = pynfb_filter_based_processor_class(filters.ButterFilter)
    linear_filter = LinearFilter(band, fs, n_channels, order)

    Sample usage 2 (this would correspond to a different implementation of this function):

    LinearFilter = pynfb_filter_based_processor_class(filters.ButterFilter)
    linear_filter = LinearFilter(band, order)

    In this case LinearFilter should provide fs and n_channels parameters to filters.ButterFilter automatically
    """
    class PynfbFilterBasedProcessorClass(ProcessorNode):
        def _on_input_history_invalidation(self):
            pass

        def _check_value(self, key, value):
            pass

        @property
        def CHANGES_IN_THESE_REQUIRE_RESET(self) -> Tuple[str]:
            pass

        @property
        def UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION(self) -> Tuple[str]:
            pass

        def _reset(self):
            pass

        def __init__(self):
            pass

        def _initialize(self):
            pass

        def _update(self):
            pass
    return PynfbFilterBasedProcessorClass
