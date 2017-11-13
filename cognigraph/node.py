import time

class Node(object):
    """ Any processing step (including getting and outputing data) is an instance of this class """

    def __init__(self):
        pass  # TODO: implement

    def init(self):
        pass  # TODO: implement

    def run(self):
        pass  # TODO: implement


class InputNode(Node):
    """ Objects of this class read data from a source """

    def __init__(self, seconds_to_live=None):
        super().__init__()
        self._birthtime = None  # TODO: remove this self-destruction nonsense
        self._seconds_to_live = seconds_to_live
        self._is_alive = True
        # TODO: implement reasonably or leave empty

    @property
    def is_alive(self):
        current_time_in_s = time.time()
        if current_time_in_s > self._birthtime + self._seconds_to_live:
            self._is_alive = False
        return self._is_alive

    def init(self):
        super().init()
        self._birthtime = time.time()

    def run(self):
        super.run()





class ProcessorNode(Node):
    pass  # TODO: implement


class OutputNode(Node):
    pass  # TODO: implement
