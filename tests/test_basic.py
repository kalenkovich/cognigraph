# from . import context  # Adds the cognigraph folder to sys.path
from cognigraph.node import SourceNode, ProcessorNode, OutputNode, TIME_DIMENSION_ID, LSL_TIME_DIMENSION_ID
from cognigraph.pipeline import Pipeline

import unittest
import subprocess
import sys
import os


class PipelineTestingSuite(unittest.TestCase):

    def setUp(self):
        self.input_node = SourceNode(seconds_to_live=0.0001)
        self.processor_nodes = [ProcessorNode() for _ in range(2)]
        self.output_node = OutputNode()

    def test_if_pipeline_works_at_all(self):
        pipeline = Pipeline()
        pipeline.input = self.input_node
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

from cognigraph.node import LSLStreamSource, LSLStreamOutput
from cognigraph.helpers.lsl import convert_lsl_chunk_to_numpy_array

source = LSLStreamSource(stream_name='cognigraph-mock-stream')
output = LSLStreamOutput()
output.input_node = source
source.init()
output.init()
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

if __name__ == '__main__':
    unittest.main()