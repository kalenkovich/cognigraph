import time
import sys

import pylsl as lsl
import numpy as np
from multiprocessing import Process

from cognigraph.helpers.lsl import create_lsl_outlet


class MockLSLStream(Process):
    # TODO: implement so that we do not have to run this file as a script
    pass


frequency = 100
channel_cnt = 8
name = 'cognigraph-mock-stream'
stream_type='EEG'
channel_format = lsl.cf_float32
channel_labels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Ft9']
outlet = create_lsl_outlet(name=name, type=stream_type, frequency=frequency, channel_format=channel_format,
                           channel_labels=channel_labels)

while True:
    try:
        mysample = np.random.random((channel_cnt, 1))
        outlet.push_sample(mysample)
        time.sleep(1/frequency)
    except:
        del outlet
        sys.exit()
