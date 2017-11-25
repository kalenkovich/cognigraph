import pylsl as lsl

from .node import SourceNode
from ..helpers.lsl import convert_lsl_chunk_to_numpy_array, convert_lsl_format_to_numpy


class LSLStreamSource(SourceNode):
    """ Class for reading data from an LSL stream defined by its name """

    CHANGES_IN_THESE_REQUIRE_RESET = ('source_name',)

    def _check_value(self, key, value):
        pass  # Whether we can find one stream with self.source_name will be checked on initialize  # TODO: move here

    def reset(self):
        # There is nothing to reset really. So we wil just go ahead and initialize
        self.initialize()

    SECONDS_TO_WAIT_FOR_THE_STREAM = 0.5

    def __init__(self, stream_name=None):
        super().__init__()
        self.source_name = stream_name
        self._inlet = None # type: lsl.StreamInlet

    @property
    def stream_name(self):
        return self.source_name

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
            self.frequency = info.nominal_srate()
            self.dtype = convert_lsl_format_to_numpy(self._inlet.channel_format)
            self.channel_count = self._inlet.channel_count
            self.channel_labels = self._read_channel_labels_from_info(self._inlet.info())

    @staticmethod
    def _read_channel_labels_from_info(info: lsl.StreamInfo):
        channels_tag = info.desc().child('channels')
        if channels_tag.empty():
            return None
        else:
            # TODO: this is hard to read. Write a generator for children with a given name in helpers
            labels = list()
            single_channel_tag = channels_tag.child(name="channel")
            for channel_id in range(info.channel_count()):
                labels.append(single_channel_tag.child_value(name='label'))
                single_channel_tag = single_channel_tag.next_sibling(name='channel')
            return labels

    def _update(self):
        super()._update()
        lsl_chunk, timestamps = self._inlet.pull_chunk()
        self.output = convert_lsl_chunk_to_numpy_array(lsl_chunk)
