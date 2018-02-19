from scripts.old_gui import *

import os

import mne

import cognigraph.nodes.node as node
from cognigraph.helpers.brainvision import read_brain_vision_data

import numpy as np
np.warnings.filterwarnings('ignore')

# Set bad channels and calculate interpolation matrix manually
bad_channel_labels = ['Fp2', 'F5', 'C5', 'F2', 'PPO10h', 'POO1', 'FCC2h']
preprocessing._bad_channel_indices = mne.pick_channels(source.mne_info['ch_names'], include=bad_channel_labels)
preprocessing._samples_to_be_collected = 0
preprocessing._enough_collected = True

preprocessing._interpolation_matrix = preprocessing._calculate_interpolation_matrix()
message = node.Message(there_has_been_a_change=True,
                       output_history_is_no_longer_valid=True)
preprocessing._deliver_a_message_to_receivers(message)


# Set the data in source to the (80, 100) s time interval
vhdr_file_path = os.path.splitext(source.file_path)[0] + '.vhdr'
start_s, stop_s = 80, 100
with source.not_triggering_reset():
    source.data, _ = read_brain_vision_data(vhdr_file_path, time_axis=TIME_AXIS, start_s=start_s, stop_s=stop_s)

beamformer.is_adaptive = True