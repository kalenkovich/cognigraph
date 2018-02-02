import time
import sys
from multiprocessing import Process

import pylsl as lsl
import numpy as np
import mne
from mne.datasets import sample


from cognigraph.helpers.lsl import create_lsl_outlet
from cognigraph.helpers.channels import read_channel_types
from cognigraph import MISC_CHANNEL_TYPE


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

# Get neuromag channels from a random raw file
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
neuromag_info = mne.io.Raw(raw_fname).info
channel_labels_neuromag = neuromag_info['ch_names']
channel_types_neuromag = read_channel_types(neuromag_info)

meg_cnt = eeg_cnt = other_cnt = 8
channel_labels = (
    [name for name in channel_labels_1005 if any(char.isdigit() for char in name) or name.endswith('z')][:eeg_cnt]
    + channel_labels_neuromag[:meg_cnt]
    + ['Other {}'.format(i + 1) for i in range(other_cnt)]
)
channel_types = ['eeg'] * eeg_cnt + channel_types_neuromag[:meg_cnt] + [MISC_CHANNEL_TYPE] * other_cnt
channel_count = len(channel_labels)


outlet = create_lsl_outlet(name=name, type=stream_type, frequency=frequency, channel_format=channel_format,
                           channel_labels=channel_labels, channel_types=channel_types)

while True:
    try:
        mysample = np.random.random((channel_count, 1))
        outlet.push_sample(mysample)
        time.sleep(1/frequency)
    except:
        del outlet
        sys.exit()
