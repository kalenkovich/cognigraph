import time

import pylsl as lsl
import numpy as np

from cognigraph.helpers.lsl import (create_lsl_outlet, convert_lsl_format_to_numpy, convert_numpy_format_to_lsl,
                                    convert_lsl_chunk_to_numpy_array, convert_numpy_array_to_lsl_chunk
                                    )


class Node(object):
    """ Any processing step (including getting and outputing data) is an instance of this class """

    def __init__(self):
        self._input_node = None  # type: Node
        self._output = None  # type: np.ndarray

    @property
    def input_node(self):
        return self._input_node

    @input_node.setter
    def input_node(self, node: 'Node'):
        self._input_node = node

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, array: np.ndarray):
        self._output = array

    def init(self):
        pass  # TODO: implement

    def update(self) -> None:
        pass  # TODO: implement

    def traverse_back_and_find(self, item: str):
        """ This function will walk up the node tree until it finds a node with an attribute <item> """
        try:
            return getattr(self._input_node, item)
        except AttributeError as e:
            msg = 'None of the predecessor nodes containts attribute {}'.format(item)
            raise AttributeError(msg).with_traceback(e.__traceback__)

class SourceNode(Node):
    """ Objects of this class read data from a source """

    def __init__(self, seconds_to_live=None):
        super().__init__()
        self._frequency = None
        self._dtype = None
        self._channel_count = None
        self._channel_labels = None
        self._source_name = None

        # TODO: remove this self-destruction nonsense
        self._should_self_destruct = seconds_to_live is not None
        if self._should_self_destruct:
            self._birthtime = None
            self._seconds_to_live = seconds_to_live
            self._is_alive = True

    @property
    def dtype(self):
        return self._dtype

    @property
    def frequency(self):
        return self._frequency

    @property
    def source_name(self):
        return self._source_name

    @property
    def channel_labels(self):
        return self._channel_labels

    @property
    def is_alive(self):
        # TODO: remove this self-destruction nonsense
        if self._should_self_destruct:
            current_time_in_s = time.time()
            if current_time_in_s > self._birthtime + self._seconds_to_live:
                self._is_alive = False
        return self._is_alive

    def init(self):
        super().init()
        if self._should_self_destruct:
            self._birthtime = time.time()

    def update(self):
        super().update()


class LSLStreamSource(SourceNode):
    """ Class for reading data from an LSL stream defined by its name """
    SECONDS_TO_WAIT_FOR_THE_STREAM = 0.5

    def __init__(self, stream_name=None):
        super().__init__()
        self._source_name = stream_name
        self._inlet = None

    def set_stream_name(self, stream_name):
        self._source_name = stream_name

    def init(self):
        stream_infos = lsl.resolve_byprop('name', self._source_name, timeout=self.SECONDS_TO_WAIT_FOR_THE_STREAM)
        if len(stream_infos) == 0:
            raise ValueError('Could not find an LSL stream with name {}'.format(self._source_name))
        elif len(stream_infos) > 1:
            raise ValueError('There are multiple LSL streams with name {}, so I don''t know which to use'
                             .format(self._source_name))
        else:
            info = stream_infos[0]
            self._inlet = lsl.StreamInlet(info)
            self._frequency = info.nominal_srate()
            self._dtype = convert_lsl_format_to_numpy(self._inlet.channel_format)
            self._channel_count = self._inlet.channel_count
            self._channel_labels = self._read_channel_labels_from_info(self._inlet.info())

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

    def update(self) -> object:
        super().update()
        lsl_chunk, timestamps = self._inlet.pull_chunk()
        self.output = convert_lsl_chunk_to_numpy_array(lsl_chunk)


class ProcessorNode(Node):
    pass  # TODO: implement


class OutputNode(Node):
    pass  # TODO: implement

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