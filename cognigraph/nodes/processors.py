from typing import Tuple, List
import math

import numpy as np
import mne
from mne.preprocessing import find_outliers

from .node import ProcessorNode, Message
from ..helpers.matrix_functions import (make_time_dimension_second, put_time_dimension_back_from_second,
                                        apply_quad_form_to_columns, get_a_subset_of_channels)
from ..helpers.inverse_model import (get_default_forward_file, assemble_gain_matrix, make_inverse_operator,
                                     matrix_from_inverse_operator)
from ..helpers.pynfb import pynfb_ndarray_function_wrapper, ExponentialMatrixSmoother
from ..helpers.channels import calculate_interpolation_matrix, channel_labels_saver
from .. import TIME_AXIS
from vendor.nfb.pynfb.signal_processing import filters


class Preprocessing(ProcessorNode):

    def __init__(self, collect_for_x_seconds: int =60):
        super().__init__()
        self.collect_for_x_seconds = collect_for_x_seconds  # type: int

        self._samples_collected = None  # type: int
        self._samples_to_be_collected = None  # type: int
        self._enough_collected = None  # type: bool
        self._means = None  # type: np.ndarray
        self._mean_sums_of_squares = None  # type: np.ndarray
        self._bad_channel_indices = None  # type: List[int]
        self._interpolation_matrix = None  # type: np.ndarray

        self._reset_statistics()

    def _on_input_history_invalidation(self):
        self._reset_statistics()

    def _check_value(self, key, value):
        pass

    CHANGES_IN_THESE_REQUIRE_RESET = ('collect_for_x_seconds', )

    def _initialize(self):
        frequency = self.traverse_back_and_find('mne_info')['sfreq']
        self._samples_to_be_collected = int(math.ceil(self.collect_for_x_seconds * frequency))

    def _reset(self) -> bool:
        self._reset_statistics()
        self._input_history_is_no_longer_valid = True
        return self._input_history_is_no_longer_valid

    def _reset_statistics(self):
        self._samples_collected = 0
        self._enough_collected = False
        self._means = 0
        self._mean_sums_of_squares = 0
        self._bad_channel_indices = []

    def _update(self):
        # Have we collected enough samples without the new input?
        enough_collected = self._samples_collected >= self._samples_to_be_collected
        if not enough_collected:
            if self.input_node.output is not None and self.input_node.output.shape[TIME_AXIS] > 0:
                self._update_statistics()

        elif not self._enough_collected:  # We just got enough samples
            self._enough_collected = True
            standard_deviations = self._calculate_standard_deviations()
            self._bad_channel_indices = find_outliers(standard_deviations)
            if any(self._bad_channel_indices):
                self._interpolation_matrix = self._calculate_interpolation_matrix()
                message = Message(there_has_been_a_change=True,
                                  output_history_is_no_longer_valid=True)
                self._deliver_a_message_to_receivers(message)

        self.output = self._interpolate(self.input_node.output)

    def _update_statistics(self):
        input_array = self.input_node.output.astype(np.dtype('float64'))
        # Using float64 is necessary because otherwise rounding error in recursive formula accumulate
        n = self._samples_collected
        m = input_array.shape[TIME_AXIS]  # number of new samples
        self._samples_collected += m

        self._means = (self._means * n + np.sum(input_array, axis=TIME_AXIS)) / (n + m)
        self._mean_sums_of_squares = (self._mean_sums_of_squares * n
                                    + np.sum(input_array ** 2, axis=TIME_AXIS)) / (n + m)

    def _calculate_standard_deviations(self):
        n = self._samples_collected
        return np.sqrt(n / (n - 1) * (self._mean_sums_of_squares - self._means ** 2))

    def _calculate_interpolation_matrix(self):
        mne_info = self.traverse_back_and_find('mne_info').copy()  # type: mne.Info
        mne_info['bads'] = [mne_info['ch_names'][i] for i in self._bad_channel_indices]
        print('The following channels: {bads} were marked as bad and will be interpolated'.format(
            bads=mne_info['bads']
        ))
        return calculate_interpolation_matrix(mne_info)

    def _interpolate(self, input_array: np.ndarray):
        if input_array is None or self._interpolation_matrix is None:
            return input_array
        else:
            if TIME_AXIS == 1:
                return self._interpolation_matrix.dot(input_array)
            elif TIME_AXIS == 0:
                return self._interpolation_matrix.dot(input_array.T).T

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info': channel_labels_saver}


class InverseModel(ProcessorNode):
    def _on_input_history_invalidation(self):
        # The methods implemented in this node do not rely on past inputs
        pass

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    CHANGES_IN_THESE_REQUIRE_RESET = ('mne_inverse_model_file_path', 'snr', 'method')
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info': channel_labels_saver}

    def _check_value(self, key, value):
        if key == 'method':
            if value not in self.SUPPORTED_METHODS:
                raise ValueError('Method {} is not supported. We support only {}'.format(value, self.SUPPORTED_METHODS))

        if key == 'snr':
            if value <= 0:
                raise ValueError('snr (signal-to-noise ratio) must be a positive number. See mne-python docs.')

    def _reset(self):
        self._should_reinitialize = True
        self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    SUPPORTED_METHODS = ['MNE', 'dSPM', 'sLORETA']

    def __init__(self, forward_model_path=None, snr=1.0, method='MNE'):
        super().__init__()

        self.snr = snr
        self._user_provided_forward_model_file_path = forward_model_path
        self._default_forward_model_file_path = None
        self.mne_info = None

        self._inverse_model_matrix = None
        self.method = method

    @property
    def mne_forward_model_file_path(self):
        return self._user_provided_forward_model_file_path or self._default_forward_model_file_path

    @mne_forward_model_file_path.setter
    def mne_forward_model_file_path(self, value):
        # This setter is for public use, hence the "user_provided"
        self._user_provided_forward_model_file_path = value

    def _update(self):
        input_array = self.input_node.output
        self.output = self._apply_inverse_model_matrix(input_array)

    def _apply_inverse_model_matrix(self, input_array: np.ndarray):
        W = self._inverse_model_matrix  # VERTICES x CHANNELS
        output_array = W.dot(make_time_dimension_second(input_array))
        return put_time_dimension_back_from_second(output_array)

    def _initialize(self):
        mne_info = self.traverse_back_and_find('mne_info')

        if self._user_provided_forward_model_file_path is None:
            self._default_forward_model_file_path = get_default_forward_file(mne_info)

        G = assemble_gain_matrix(self.mne_forward_model_file_path, mne_info)
        self._gain_matrix = G

        inverse_operator = make_inverse_operator(self.mne_forward_model_file_path, mne_info)
        self._inverse_model_matrix = matrix_from_inverse_operator(inverse_operator=inverse_operator, mne_info=mne_info,
                                                                  snr=self.snr, method=self.method)

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
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info': lambda info: (info['nchan'], )}

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
        self._should_reinitialize = True
        self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    def _on_input_history_invalidation(self):
        self._envelope_extractor.reset()

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    CHANGES_IN_THESE_REQUIRE_RESET = ('method', 'factor')
    SUPPORTED_METHODS = ('Exponential smoothing', )
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info': lambda info: (info['nchan'],)}

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


class Beamformer(ProcessorNode):

    SUPPORTED_OUTPUT_TYPES = ('power', 'activation')

    def __init__(self, snr: float =3.0, output_type: str ='power', is_adaptive: bool =False,
                 fixed_orientation: bool = True,
                 forward_model_path: str =None, forgetting_factor_per_second: float =0.99):
        super().__init__()

        self.snr = snr  # type: float
        self._user_provided_forward_model_file_path = forward_model_path  # type: str
        self._default_forward_model_file_path = None  # type: str
        self.mne_info = None  # type: mne.Info

        self.output_type = output_type  # type: np.dtype
        self.is_adaptive = is_adaptive  # type: bool
        self._initialized_as_adaptive = None  # type: bool
        self.fixed_orientation = fixed_orientation  # type: bool
        self._initialized_as_fixed = None  # type: bool

        self._channel_indices = None  # type: list
        self._gain_matrix = None  # type: np.ndarray
        self._Rxx = None  # type: np.ndarray
        self._kernel = None # type: np.ndarray
        self.forgetting_factor_per_second = forgetting_factor_per_second  # type: float
        self._forgetting_factor_per_sample = None  # type: float



    @property
    def mne_forward_model_file_path(self):
        return self._user_provided_forward_model_file_path or self._default_forward_model_file_path

    @mne_forward_model_file_path.setter
    def mne_forward_model_file_path(self, value):
        # This setter is for public use, hence the "user_provided"
        self._user_provided_forward_model_file_path = value

    def _initialize(self):
        mne_info = self.traverse_back_and_find('mne_info')

        if self._user_provided_forward_model_file_path is None:
            self._default_forward_model_file_path = get_default_forward_file(mne_info)

        self._gain_matrix, self._channel_indices = assemble_gain_matrix(self.mne_forward_model_file_path, mne_info,
                                                                        drop_missing=True,
                                                                        force_fixed=self.fixed_orientation)

        G = self._gain_matrix
        if self.is_adaptive is False:
            Rxx = G.dot(G.T)
            self._kernel = self._calculate_kernel(Rxx)

        elif self.is_adaptive is True:
            self._Rxx = 0
            # Optimization
            if not self._gain_matrix.flags['F_CONTIGUOUS']:
                self._gain_matrix = np.asfortranarray(self._gain_matrix)

        frequency = mne_info['sfreq']
        self._forgetting_factor_per_sample = np.power(self.forgetting_factor_per_second, 1 / frequency)

        if self.fixed_orientation is True:
            vertex_count = self._gain_matrix.shape[1]
        else:
            vertex_count = int(self._gain_matrix.shape[1] / 3)
        channel_labels = ['vertex #{}'.format(i + 1) for i in range(vertex_count)]
        self.mne_info = mne.create_info(channel_labels, frequency)

        self._initialized_as_adaptive = self.is_adaptive
        self._initialized_as_fixed = self.fixed_orientation

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info',)
    CHANGES_IN_THESE_REQUIRE_RESET = ('snr', 'output_type', 'is_adaptive', 'fixed_orientation')
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info': channel_labels_saver}

    def _calculate_kernel(self, Rxx):
        Rxx_inv = self._regularized_inverse(Rxx)

        G = self._gain_matrix

        if self.fixed_orientation is True:
            denominators = 1 / apply_quad_form_to_columns(A=Rxx_inv, X=G)
            return G.T.dot(Rxx_inv) * np.expand_dims(denominators, TIME_AXIS)

        else:  # Free orientation
            vertex_count = int(G.shape[1] / 3)
            kernel = np.zeros(np.flipud(G.shape))
            for idx in range(vertex_count):
                vertex_slice = slice(idx * 3, idx * 3 + 3)
                Gi = G[:, vertex_slice]
                denominator = Gi.T.dot(Rxx_inv).dot(Gi)
                kernel[vertex_slice, :] = np.linalg.inv(denominator).dot(Gi.T.dot(Rxx_inv))
            return kernel

    def _regularized_inverse(self, Rxx):
        electrode_count = Rxx.shape[0]
        _lambda = 1 / self.snr ** 2 * Rxx.trace() / electrode_count
        return np.linalg.inv(Rxx + _lambda * np.eye(electrode_count))

    def _update(self):

        input_array = get_a_subset_of_channels(self.input_node.output, self._channel_indices)

        if self.is_adaptive is False:
            kernel = self._kernel
        else:
            self._update_covariance_matrix(input_array)
            kernel = self._calculate_kernel(self._Rxx)

        output = put_time_dimension_back_from_second(
            kernel.dot(make_time_dimension_second(input_array))
        )

        if self.fixed_orientation is True:
            if self.output_type == 'power':
                output = output ** 2
        else:
            vertex_count = int(self._gain_matrix.shape[1] / 3)
            output = np.sum(np.power(output, 2).reshape((vertex_count, 3, -1)), axis=1)
            if self.output_type == 'activation':
                output = np.sqrt(output)

        self.output = output

    def _reset(self) -> bool:

        # Only changing adaptiveness or fixed_orientation requires reinitialization
        if (self._initialized_as_adaptive is not self.is_adaptive
                or self._initialized_as_fixed is not self.fixed_orientation):
            self._should_reinitialize = True
            self.initialize()

        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    def _on_input_history_invalidation(self):
        # Only adaptive version relies on history
        if self._initialized_as_adaptive is True:
            self._should_reinitialize = True
            self.initialize()

    def _check_value(self, key, value):
        if key == 'output_type':
            if value not in self.SUPPORTED_OUTPUT_TYPES:
                raise ValueError('Method {} is not supported. We support only {}'
                                 .format(value, self.SUPPORTED_OUTPUT_TYPES))

        if key == 'snr':
            if value <= 0:
                raise ValueError('snr (signal-to-noise ratio) must be a positive number. See mne-python docs.')

        if key == 'is_adaptive':
            if not isinstance(value, bool):
                raise ValueError('Beamformer can either be adaptive or not. This should be a boolean')

    def _update_covariance_matrix(self, input_array):
        alpha = self._forgetting_factor_per_sample
        sample_count = input_array.shape[TIME_AXIS]
        # Exponential smoothing of XX'
        for sample in make_time_dimension_second(input_array).T:
            sample_2d = sample[:, np.newaxis]
            self._Rxx = alpha * self._Rxx + (1 - alpha) * sample_2d.dot(sample_2d.T)


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
