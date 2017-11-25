# from . import context  # Adds the cognigraph folder to sys.path
import os
import subprocess
import sys
import unittest

from cognigraph.nodes.node import SourceNode, ProcessorNode, OutputNode
from cognigraph.pipeline import Pipeline


class PipelineTestingSuite(unittest.TestCase):

    def setUp(self):
        self.input_node = SourceNode(seconds_to_live=0.0001)
        self.processor_nodes = [ProcessorNode() for _ in range(2)]
        self.output_node = OutputNode()

    def test_if_pipeline_works_at_all(self):
        pipeline = Pipeline()
        pipeline.source = self.input_node
        for processor_node in self.processor_nodes:
            pipeline.add_processor(processor_node=processor_node)
        pipeline.add_output(self.output_node)
        pipeline.run()

class LSLStreamInputTestingSuite(unittest.TestCase):

    def setUp(self):

        self.mock_stream_process = subprocess.Popen([sys.executable, os.path.join()])

    def test_init_with_mock_lsl_stream(self):
        pass


# TODO: write actual fucking unit test you lazy sob

import time

import pylsl as lsl
import numpy as np

from cognigraph.nodes.sources import LSLStreamSource
from cognigraph.nodes.processors import InverseModel
from cognigraph.nodes.outputs import LSLStreamOutput, ThreeDeeBrain
from cognigraph.helpers.lsl import convert_lsl_chunk_to_numpy_array

# LSL in and out

source = LSLStreamSource(stream_name='cognigraph-mock-stream')
output = LSLStreamOutput()
output.input_node = source
source.initialize()
output.initialize()
output_info = lsl.resolve_byprop('name', output.stream_name)[0]
inlet = lsl.StreamInlet(output_info)
inlet.open_stream()

source.update()  # The inlet only receives samples after the first request for data, so this is empty
time.sleep(0.1)
source.update()
output.update()

time.sleep(0.001)  # Time for the chunk to get pushed
lsl_chunk, timestamps = inlet.pull_chunk()
numpy_chunk = convert_lsl_chunk_to_numpy_array(lsl_chunk)

assert(np.array_equal(source.output, numpy_chunk))

# Add inverse modelling

inverse = InverseModel()
inverse.input_node = source
output.input_node = inverse

inverse.initialize()

source.update()
inverse.update()

# TODO: change to use TIME_AXIS
assert(source.output.shape[1] == inverse.output.shape[1])
assert(source.output.shape[0] != inverse.output.shape[0])
assert(inverse.output.shape[0] == inverse.channel_count)
assert(len(inverse.channel_labels) == inverse.channel_count)


# Visualize sources
brain = ThreeDeeBrain()
brain.input_node = inverse
brain.initialize()

if __name__ == '__main__':
    unittest.main()