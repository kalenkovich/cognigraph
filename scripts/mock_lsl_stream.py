import time
import sys
from multiprocessing import Process

import pylsl as lsl
import numpy as np
import mne


from cognigraph.helpers.lsl import create_lsl_outlet


class MockLSLStream(Process):
    # TODO: implement so that we do not have to run this file as a script
    pass

    def __init__(self, meg_cnt, eeg_cnt, other_cnt):
        self.meg_cnt = meg_cnt
        self.eeg_cnt = eeg_cnt
        self.other_cnt = other_cnt

frequency = 100
name = 'cognigraph-mock-stream'
stream_type = 'EEG'
channel_format = lsl.cf_float32

channel_labels_1005 = mne.channels.read_montage('standard_1005').ch_names
channel_labels_neuromag = mne.channels.read_layout(kind='Vectorview-all').names

meg_cnt = eeg_cnt = other_cnt = 8
channel_labels = (
    [name for name in channel_labels_1005 if any(char.isdigit() for char in name) or name.endswith('z')][:eeg_cnt]
    + channel_labels_neuromag[:meg_cnt]
    + ['Other {}'.format(i + 1) for i in range(other_cnt)]
)
channel_count = len(channel_labels)


outlet = create_lsl_outlet(name=name, type=stream_type, frequency=frequency, channel_format=channel_format,
                           channel_labels=channel_labels)

while True:
    try:
        mysample = np.random.random((channel_count, 1))
        outlet.push_sample(mysample)
        time.sleep(1/frequency)
    except:
        del outlet
        sys.exit()
