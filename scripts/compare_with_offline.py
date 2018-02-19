import os

import numpy as np
import scipy
import mne

from cognigraph.helpers.matrix_functions import get_a_time_slice
from cognigraph.helpers.misc import class_name_of
from cognigraph.helpers.brainvision import read_brain_vision_data
from scripts.old_gui import *
from scripts.eyes_inverse_offline import *
from cognigraph.nodes.node import Node
import cognigraph.nodes.node as node

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


#
linear_filter.input_node = source
def compare_viz_with_offline():
    start_idx = source._samples_already_read
    source.update()
    stop_idx = start_idx + source.output.shape[TIME_AXIS]

    preprocessing.update()
    linear_filter.update()
    signal_viewer.update()

    envelope_extractor.output = get_a_time_slice(stc.data, start_idx=start_idx, stop_idx=stop_idx)
    three_dee_brain.update()

    brain.set_time(start_s + stop_idx / frequency)


linear_filter.input_node = preprocessing
# Compare inverse with offline
source._samples_already_read = 0
timer.timeout.disconnect()
def compare_with_offline():
    start_idx = source._samples_already_read
    pipeline.update_all_nodes()
    stop_idx = start_idx + source.output.shape[TIME_AXIS]
    brain.set_time(start_s + stop_idx / frequency)
timer.timeout.connect(compare_with_offline)