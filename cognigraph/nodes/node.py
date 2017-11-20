import time
import numpy as np


class Node(object):
    """ Any processing step (including getting and outputing data) is an instance of this class """

    def __init__(self):
        self.input_node = None  # type: Node
        self.output = None  # type: np.ndarray

    def initialize(self):
        pass  # TODO: implement

    def update(self) -> None:
        pass  # TODO: implement

    def traverse_back_and_find(self, item: str):
        """ This function will walk up the node tree until it finds a node with an attribute <item> """
        try:
            return getattr(self.input_node, item)
        except AttributeError as e:
            msg = 'None of the predecessor nodes contains attribute {}'.format(item)
            raise AttributeError(msg).with_traceback(e.__traceback__)


class SourceNode(Node):
    """ Objects of this class read data from a source """

    def __init__(self, seconds_to_live=None):
        super().__init__()
        self.frequency = None
        self.dtype = None
        self._channel_count = None
        self.channel_labels = None
        self.source_name = None

        # TODO: remove this self-destruction nonsense
        self._should_self_destruct = seconds_to_live is not None
        if self._should_self_destruct:
            self._birthtime = None
            self._seconds_to_live = seconds_to_live
            self._is_alive = True

    @property
    def is_alive(self):
        # TODO: remove this self-destruction nonsense
        if self._should_self_destruct:
            current_time_in_s = time.time()
            if current_time_in_s > self._birthtime + self._seconds_to_live:
                self._is_alive = False
        return self._is_alive

    def initialize(self):
        super().initialize()
        if self._should_self_destruct:
            self._birthtime = time.time()

    def update(self):
        super().update()


class ProcessorNode(Node):
    pass  # TODO: implement


class OutputNode(Node):
    pass  # TODO: implement
