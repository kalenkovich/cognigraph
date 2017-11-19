from .node import OutputNode
from ..helpers.lsl import convert_numpy_format_to_lsl, convert_numpy_array_to_lsl_chunk, create_lsl_outlet


class LSLStreamOutput(OutputNode):
    def __init__(self, stream_name=None):
        super().__init__()
        self._stream_name = stream_name
        self._outlet = None

    @property
    def stream_name(self):
        return self._stream_name

    def init(self):
        super().init()

        # If no name was supplied we will use a modified version of the source name (a file or a stream name)
        source_name = self.traverse_back_and_find('source_name')
        self._stream_name = self._stream_name or (source_name + '_output')

        # Get other info from somewhere down the predecessor chain
        frequency = self.traverse_back_and_find('frequency')
        dtype = self.traverse_back_and_find('dtype')
        channel_format = convert_numpy_format_to_lsl(dtype)
        channel_labels = self.traverse_back_and_find('channel_labels')

        self._outlet = create_lsl_outlet(name=self._stream_name, frequency=frequency, channel_format=channel_format,
                                         channel_labels=channel_labels)

    def update(self):
        chunk = self.input_node.output
        lsl_chunk = convert_numpy_array_to_lsl_chunk(chunk)
        self._outlet.push_chunk(lsl_chunk)