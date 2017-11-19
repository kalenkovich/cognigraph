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

# Add inverse modelling

inverse = InverseModel()
inverse.input_node = source
output.input_node = inverse

inverse.init()

source.update()
inverse.update()

# TODO: change to use TIME_AXIS
assert(source.output.shape[1] == inverse.output.shape[1])
assert(source.output.shape[0] != inverse.output.shape[0])
assert(inverse.output.shape[0] == inverse.channel_cnt)
assert(len(inverse.channel_labels) == inverse.channel_cnt)


# Visualize sources
brain = ThreeDeeBrain()
brain.input_node = inverse
brain.init()