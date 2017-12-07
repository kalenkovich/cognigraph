# TODO: write actual fucking unit test you lazy sob

import time

import pylsl as lsl
import numpy as np

from cognigraph.nodes.sources import LSLStreamSource
from cognigraph.nodes.processors import InverseModel, LinearFilter, EnvelopeExtractor
from cognigraph.nodes.outputs import LSLStreamOutput, ThreeDeeBrain
from cognigraph.helpers.lsl import convert_lsl_chunk_to_numpy_array
from cognigraph.helpers.matrix_functions import last_sample
from cognigraph import TIME_AXIS, CHANNEL_AXIS

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

brain._brain_painter.widget.show()
brain.update()


# Linear filter

linear_filter = LinearFilter(lower_cutoff=1, upper_cutoff=None)
linear_filter.input_node = source
linear_filter.initialize()
linear_filter.update()

# this linear filter should at least remove DC. Thus, new means should be somewhat close to zero
means = np.abs(np.mean(linear_filter.output, axis=TIME_AXIS))
mean_max = np.mean(np.max(linear_filter.output, axis=TIME_AXIS))
assert(np.all(means < 0.1 * mean_max))

linear_filter.lower_cutoff = None
linear_filter.initialize()
linear_filter.update()

assert(linear_filter.output is source.output)


# Envelope extractor

envelope_extractor = EnvelopeExtractor()
envelope_extractor.input_node = linear_filter
envelope_extractor.initialize()
envelope_extractor.update()

# TODO: come up with an actual way to test this stuff
assert(envelope_extractor.output is not None)


from cognigraph.nodes.sources import BrainvisionSource
source = BrainvisionSource(r"C:\Users\evgenii\Downloads\brainvision\Bulavenkova_A_2017-10-24_15-33-18_Rest.vmrk")
source.update()