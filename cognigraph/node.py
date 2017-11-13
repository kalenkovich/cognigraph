class Node(object):
    """ Any processing step (including getting and outputing data) is an instance of this class """

    def __init__(self):
        pass


class InputNode(Node):
    """ Objects of this class read data from a source """

    def __init__(self):
        super().__init__()


class ProcessorNode(Node):
    pass


class OutputNode(Node):
    pass
