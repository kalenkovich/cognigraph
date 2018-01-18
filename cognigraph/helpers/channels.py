import os.path as op

import mne
from mne import io
from mne.datasets import sample


def calculate_interpolation_matrix(channel_labels, bad_channel_flags):
    fake_raw = None;
    fake_raw
    return



data_path = sample.data_path(verbose='ERROR')
sample_raw_path = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
raw = io.read_raw_fif(sample_raw_path)  # type: mne.io.Raw
raw.load_data();
raw2 = raw.interpolate_bads(reset_bads=False, mode='fast')
