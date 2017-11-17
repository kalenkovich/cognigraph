from .node import Node, SourceNode, ProcessorNode, OutputNode
from .helpers.decorators import accepts


class Pipeline(object):
    """
    This class facilitates connecting data inputs to a sequence of signal processors and outputs.

    All elements in the pipeline are objects of class Node and inputs, processors and outputs should be objects of the
    corresponding subclasses of Node.

    Sample usage:

    p = Pipeline();
    p.input = lsl_input_stream
    p.addProcessor(linear_filter_1_40)
    p.addOutput(brainvision_output);
    p.run()
    """

    def __init__(self):
        self._input = None
        self._frequency = None
        self._processors = list()
        self._outputs = list()

    @property
    def input(self):
        return self._input


    @input.setter
    @accepts(object, SourceNode)
    def input(self, input_node):
        self._input = input_node

    @accepts(object, ProcessorNode)
    def add_processor(self, processor_node):
        if processor_node not in self._processors:
            self._processors.append(processor_node)

    @accepts(object, OutputNode)
    def add_output(self, output_node):
        if output_node not in self._outputs:
            self._outputs.append(output_node)

    def run(self):
        all_nodes = [self._input] + self._processors + self._outputs
        for node in all_nodes:
            node.init()

        while self.input.is_alive: # TODO: also stop if all outputs are dead
            for node in all_nodes:
                node.update()