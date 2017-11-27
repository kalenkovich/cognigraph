import timeit
import numpy as np


# RingBuffer
from cognigraph.helpers.ring_buffer import RingBuffer, RingBufferTest

row_cnt = 7000
maxlen = 12000
buffer = RingBuffer(row_cnt=row_cnt, maxlen=maxlen)
buffer_test = RingBufferTest(row_cnt=row_cnt, maxlen=maxlen)
samples_in_chunk = 40
chunk = np.random.random((samples_in_chunk, row_cnt))

def extend():
    buffer.extend(chunk.T)
    x = buffer.data
timeit.timeit(extend, number=10)

def extend_test():
    buffer_test.extend(chunk.T)
    x = buffer_test.data
timeit.timeit(extend_test, number=10)


def roll():
    np.roll(buffer._data, shift=-samples_in_chunk, axis=1)
timeit.timeit(roll, number=10)/10




# LocalDesync
from nfb.pynfb.brain.brain import LocalDesync

from cognigraph.helpers.ring_buffer import RingBufferTest
import numpy as np



test = RingBufferTest(row_cnt=1, maxlen=6)
new_data = np.arange(19).reshape((1, 19))

pieces = [new_data[:, 0:2], new_data[:, 2:5], new_data[:, 5:7], new_data[:, 7:11], new_data[:, 11:19]]

test.clear()
for piece in pieces:
    print("Adding {}".format(piece))
    test.extend(piece)
    print(test.data)

test.extend(pieces[0])