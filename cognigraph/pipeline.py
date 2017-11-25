from typing import List

from cognigraph.nodes.node import Node, SourceNode, ProcessorNode, OutputNode
from .helpers.decorators import accepts
from .helpers.misc import class_name_of


class Pipeline(object):
    """
    This class facilitates connecting data inputs to a sequence of signal processors and outputs.

    All elements in the pipeline are objects of class Node and inputs, processors and outputs should be objects of the
    corresponding subclasses of Node.

    Sample usage:

    pipeline = Pipeline()
    pipeline.source = sources.LSLStreamSource(stream_name='cognigraph-mock-stream')
    linear_filter = processors.LinearFilter(lower_cutoff=0.1, upper_cutoff=40)
    pipeline.add_processor(linear_filter)
    pipeline.add_processor(processors.InverseModel(method='MNE'))
    pipeline.add_output(outputs.ThreeDeeBrain())
    pipeline.initialize_all_nodes()
    """

    def __init__(self):
        self._source = None  # type: SourceNode
        self._processors = list()  # type: List[ProcessorNode]
        self._outputs = list()  # type: List[OutputNode]
        self._inputs_of_outputs = list()  # type: List[(SourceNode, ProcessorNode)]

    @property
    def source(self):
        return self._source

    @source.setter
    @accepts(object, SourceNode)
    def source(self, input_node):
        self._source = input_node
        self._reconnect_outputs_to_last_node()  # In case some outputs were added before anything else

    @property
    def all_nodes(self) -> List[Node]:
        list_with_source = [self.source] if self.source is not None else list()  # type: List[Node]
        return list_with_source + self._processors + self._outputs

    @property
    def frequency(self) -> (int, float):
        try:
            return self.source.frequency
        except AttributeError:
            raise ValueError("No source has been set in the pipeline")


    @accepts(object, ProcessorNode)
    def add_processor(self, processor_node):
        if processor_node not in self._processors:
            last_node = self._last_node_before_outputs()
            processor_node.input_node = processor_node.input_node or last_node
            self._processors.append(processor_node)
            self._reconnect_outputs_to_last_node()
        else:
            msg = "Trying to add a {} that has already been added".format(class_name_of(processor_node))
            raise ValueError(msg)

    @accepts(object, OutputNode, (SourceNode, ProcessorNode))
    def add_output(self, output_node, input_node=None):
        """If input_node is None, output_node will be kept connected to whatever node that is currently last"""
        if output_node not in self._outputs:
            self._outputs.append(output_node)
            # If input_node is None we will need to reconnect output_node. So we keep track of those Nones.
            self._inputs_of_outputs.append(input_node)
            output_node.input_node = input_node or self._last_node_before_outputs()
        else:
            msg = "Trying to add a {} that has already been added".format(class_name_of(output_node))
            raise ValueError(msg)

    def _last_node_before_outputs(self):
        try:
            return self._processors[-1]
        except IndexError:
            return self.source

    def initialize_all_nodes(self):
        for node in self.all_nodes:
            node.initialize()

    def update_all_nodes(self):
        for node in self.all_nodes:
            node.update()

    def run(self):
        while self.source.is_alive:  # TODO: also stop if all outputs are dead
            for node in self.all_nodes:
                node.update()

    def _reconnect_outputs_to_last_node(self):
        """Reconnects all outputs that did not have an input node specified when added"""
        last_node = self._last_node_before_outputs()
        for output, input in zip(self._outputs, self._inputs_of_outputs):
            output.input_node = input or last_node

    def _reconnect_first_processor(self):
        try:
            self._processors[0].input_node = self.source
        except IndexError:  # No processors have been added yet
            pass
