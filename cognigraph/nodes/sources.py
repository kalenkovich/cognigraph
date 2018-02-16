import os
import time

import pylsl as lsl
import numpy as np
import mne

from cognigraph.helpers.matrix_functions import get_a_time_slice
from .. import TIME_AXIS, DTYPE
from .node import SourceNode
from ..helpers.lsl import convert_lsl_chunk_to_numpy_array, convert_lsl_format_to_numpy, read_channel_labels_from_info
from ..helpers.brainvision import read_brain_vision_data


class LSLStreamSource(SourceNode):
    """ Class for reading data from an LSL stream defined by its name """

    CHANGES_IN_THESE_REQUIRE_RESET = ('source_name',)

    def _check_value(self, key, value):
        pass  # Whether we can find one stream with self.source_name will be checked on initialize  # TODO: move here

    SECONDS_TO_WAIT_FOR_THE_STREAM = 0.5

    def __init__(self, stream_name=None):
        super().__init__()
        self.source_name = stream_name
        self._inlet = None # type: lsl.StreamInlet

    @property
    def stream_name(self):
        return self.source_name

    @property
    def frequency(self):
        return self.mne_info['sfreq']

    @stream_name.setter
    def stream_name(self, stream_name):
        self.source_name = stream_name

    def _initialize(self):

        stream_infos = lsl.resolve_byprop('name', self.source_name, timeout=self.SECONDS_TO_WAIT_FOR_THE_STREAM)
        if len(stream_infos) == 0:
            raise ValueError('Could not find an LSL stream with name {}'.format(self.source_name))
        elif len(stream_infos) > 1:
            raise ValueError('There are multiple LSL streams with name {}, so I don''t know which to use'
                             .format(self.source_name))
        else:
            info = stream_infos[0]
            self._inlet = lsl.StreamInlet(info)
            self._inlet.open_stream()
            frequency = info.nominal_srate()
            self.dtype = DTYPE
            channel_labels, channel_types = read_channel_labels_from_info(self._inlet.info())
            self.mne_info = mne.create_info(channel_labels, frequency, ch_types=channel_types)

    def _update(self):
        lsl_chunk, timestamps = self._inlet.pull_chunk()
        self.output = convert_lsl_chunk_to_numpy_array(lsl_chunk, dtype=self.dtype)


class BrainvisionSource(SourceNode):
    SUPPORTED_EXTENSIONS = ('.vhdr', '.eeg', '.vmrk')
    CHANGES_IN_THESE_REQUIRE_RESET = ('source_name', )

    MAX_SAMPLES_IN_CHUNK = 1024

    def __init__(self, file_path=None):
        super().__init__()
        self.source_name = None
        self._file_path = None
        self.file_path = file_path  # This will also populate self.source_name
        self.data = None  # type: np.ndarray
        self.loop_the_file = False
        self.is_alive = True

        self._time_of_the_last_update = None
        self._samples_already_read = None

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, file_path):
        if file_path in (None, ''):
            self._file_path = None
            self.source_name = None
        else:
            basename = os.path.basename(file_path)
            file_name, extension = os.path.splitext(basename)
            if extension not in self.SUPPORTED_EXTENSIONS:
                raise ValueError('Cannot read {}. Extension must be one of the following: {}'.format(
                    basename, self.SUPPORTED_EXTENSIONS
                ))
            else:
                self._file_path = file_path
                self.source_name = file_name

    def _initialize(self):
        self._time_of_the_last_update = None
        self._samples_already_read = 0

        if self.file_path is not None:
            vhdr_file_path = os.path.splitext(self.file_path)[0] + '.vhdr'
            self.data, self.mne_info = \
                read_brain_vision_data(vhdr_file_path=vhdr_file_path, time_axis=TIME_AXIS)
            self.dtype = DTYPE
            self.data = self.data.astype(self.dtype)

    def _update(self):
        if self.data is None:
            return

        current_time = time.time()

        if self._time_of_the_last_update is not None:

            seconds_since_last_update = current_time - self._time_of_the_last_update
            self._time_of_the_last_update = current_time
            frequency = self.mne_info['sfreq']

            # How many sample we would like to read
            max_samples_in_chunk = np.int64(seconds_since_last_update * frequency)
            # Lower it to the amount we can process in a reasonable amount of time
            max_samples_in_chunk = min(max_samples_in_chunk, self.MAX_SAMPLES_IN_CHUNK)

            # We will have read max_samples_in_chunk samples unless we hit the end
            samples_in_data = self.data.shape[TIME_AXIS]
            stop_idx = self._samples_already_read + max_samples_in_chunk
            self.output = get_a_time_slice(self.data, start_idx=self._samples_already_read, stop_idx=stop_idx)
            actual_samples_in_chunk = self.output.shape[TIME_AXIS]
            self._samples_already_read = self._samples_already_read + actual_samples_in_chunk

            # If we do hit the end we need to either start again or stop completely depending on loop_the_file
            if self._samples_already_read == samples_in_data:
                if self.loop_the_file is True:
                    self._samples_already_read = 0
                else:
                    self.is_alive = False

        else:
            self._time_of_the_last_update = current_time

    def _check_value(self, key, value):
        pass
