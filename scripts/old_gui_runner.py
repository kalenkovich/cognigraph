from cognigraph.helpers.channels import fill_eeg_channel_locations
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

beamformer.fixed_orientation = False
beamformer.is_adaptive = False
beamformer.update()

# Find the most posterior vertex in the left hemisphere
fwd = mne.read_forward_solution(beamformer.mne_forward_model_file_path)
lh = fwd['src'][0]
lh_dipole_coordinates = lh['rr'][lh['vertno']]
most_posterior_idx = np.argmin(lh_dipole_coordinates[:, 1])

# Draw to check location
vertex_count = beamformer.mne_info['nchan']
input_node = three_dee_brain.input_node
shape = (vertex_count, 1)
shape = (shape[1-TIME_AXIS], shape[TIME_AXIS])
input_node.output = np.zeros(shape=(shape, 1))
input_node.output[most_posterior_idx, :] = 1

three_dee_brain.limits_mode = 'Local'
three_dee_brain.update()

# Forward to the right sensors and put as the output of the linear filter

channel_count = source.mne_info['nchan']
shape = (channel_count, 1)
shape = (shape[1-TIME_AXIS], shape[TIME_AXIS])
linear_filter.output = np.zeros(shape=shape)
# Unit moments in all directions
G_vertex = beamformer._gain_matrix[:, most_posterior_idx * 3:][:, :3]
linear_filter.output[beamformer._channel_indices] = (
    G_vertex.sum(axis=1, keepdims=True)
)

# Plot topography
fill_eeg_channel_locations(source.mne_info)
data_channel_indices = mne.io.pick._pick_data_channels(source.mne_info)
mne.viz.plot_topomap(linear_filter.output[data_channel_indices, 0], source.mne_info)
# Looks weird.
#
# Use unit moment in normal direction
linear_filter.output = np.zeros(shape=shape)
normal = lh['nn'][most_posterior_idx]
linear_filter.output[beamformer._channel_indices] = (
    (G_vertex.dot(normal))[:, np.newaxis]
)
mne.viz.plot_topomap(linear_filter.output[data_channel_indices, 0], source.mne_info)


beamformer.update()
three_dee_brain.input_node = beamformer
three_dee_brain.update()

# Apply average-reference projection
channels_used_count = len(beamformer._channel_indices)
P = (np.eye(channels_used_count)
     - np.ones((channels_used_count, channels_used_count)) / channels_used_count)
linear_filter.output[beamformer._channel_indices, :] = \
    linear_filter[beamformer._channel_indices, :]

