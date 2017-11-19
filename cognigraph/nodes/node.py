import time
import numpy as np


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


class ProcessorNode(Node):
    pass  # TODO: implement


class OutputNode(Node):
    pass  # TODO: implement

