# from . import context  # Adds the cognigraph folder to sys.path
from cognigraph.node import InputNode, ProcessorNode, OutputNode
from cognigraph.pipeline import Pipeline

import unittest


class PipelineTestingSuite(unittest.TestCase):

    def setUp(self):
        self.input_node = InputNode(seconds_to_live=120)
        self.processor_nodes = [ProcessorNode() for _ in range(2)]
        self.output_node = OutputNode()

    def test_if_pipeline_works_at_all(self):
        pipeline = Pipeline()
        pipeline.input = self.input_node
        for processor_node in self.processor_nodes:
            pipeline.add_processor(processor_node=processor_node)
        pipeline.add_output(self.output_node)


if __name__ == '__main__':
    unittest.main()