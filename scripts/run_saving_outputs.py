import os

import numpy as np
import scipy
import mne

from cognigraph.helpers.misc import class_name_of
from cognigraph.helpers.brainvision import read_brain_vision_data
from scripts.old_gui import *
from cognigraph.nodes.node import Node


def force_reinitialize(node: Node):
    node._should_reinitialize = True
    node.initialize()


def force_reset(node: Node):
    node._should_reset = True
    node.reset()


def reinitialize(pipeline: Pipeline):
    for node in pipeline.all_nodes:
        node._should_reinitialize = True
        node.initialize()


def get_to_interpolation():
    force_reinitialize(source)
    force_reinitialize(preprocessing)

    def run_until_interpolation_is_ready():
        source.update()
        preprocessing.update()
        if preprocessing._enough_collected is True:
            timer.stop()

    timer.timeout.disconnect()
    timer.timeout.connect(run_until_interpolation_is_ready)
    timer.start()


def reset_all_but_interpolation():
    for node in pipeline.all_nodes:
        if node is not preprocessing:
            force_reset(node)
    preprocessing._no_pending_changes = True

    # Just a little botching
    for node in pipeline.all_nodes:
        if node is not source and 'mne_info' in node._saved_from_upstream:
            node._saved_from_upstream['mne_info'] = node.traverse_back_and_find('mne_info')

def run_saving_the_outputs(duration_in_s=20):

    duration_in_samples = source.mne_info['sfreq'] * duration_in_s
    processor_count = len(pipeline._processors)

    def run_for_duration():
        # Update source and processors
        source.update()
        for processor in pipeline._processors:
            processor.update()

        # Save outputs of the processors
        for processor, idx in zip(pipeline._processors, range(processor_count)):
            output = processor.output
            if output is not None:
                if saved_outputs[idx] is None:
                    preallocated_shape = list(output.shape)
                    preallocated_shape[TIME_AXIS] = int(duration_in_samples * 1.1)
                    preallocated_shape = tuple(preallocated_shape)
                    saved_outputs[idx] = np.append(output, np.zeros(preallocated_shape), axis=TIME_AXIS)
                    collected_samples[idx] = output.shape[TIME_AXIS]
                else:
                    new_samples_count = output.shape[TIME_AXIS]
                    time_slice = slice(collected_samples[idx], collected_samples[idx] + new_samples_count)
                    indices = list(np.s_[:, :])
                    indices[TIME_AXIS] = time_slice
                    indices = tuple(indices)
                    saved_outputs[idx][indices] = output
                    collected_samples[idx] += new_samples_count

        # Check if we've collected enough and stop the timer if we have
        if collected_samples[0] >= duration_in_samples:
            print('Run long enough')
            timer.stop()

    timer.timeout.disconnect()
    timer.timeout.connect(run_for_duration)
    timer.start()


def save_outputs_to_mat_file(file_path):
    mdict = {class_name_of(processor): saved_output
             for saved_output, processor in zip(saved_outputs, pipeline._processors)}
    scipy.io.savemat(file_path, mdict)


preprocessing._samples_to_be_collected = 2000
get_to_interpolation()

reset_all_but_interpolation()
vhdr_file_path = os.path.splitext(source.file_path)[0] + '.vhdr'
start_s, stop_s = 80, 100
with source.not_triggering_reset():
    source.data, _ = read_brain_vision_data(vhdr_file_path, time_axis=TIME_AXIS, start_s=start_s, stop_s=stop_s)



saved_outputs = [None] * len(pipeline._processors)
collected_samples = np.zeros(len(pipeline._processors), dtype=int)
run_saving_the_outputs(duration_in_s=20)


mat_file_path = r"D:\Cognigraph\eyes\cogni_output.mat"
save_outputs_to_mat_file(mat_file_path)


# Viz sources

from cognigraph.helpers.inverse_model import standard_1005_forward_file_path
forward = mne.read_forward_solution(standard_1005_forward_file_path)
left_hemi, right_hemi = forward['src']
vertices = [left_hemi['vertno'], right_hemi['vertno']]


subject = 'sample'
data_path = mne.datasets.sample.data_path()
import os
subjects_dir = os.path.join(data_path, 'subjects')

offset_s = 2
offset = int(offset_s * frequency)
stc = mne.SourceEstimate(data=saved_outputs[3][:, offset:], vertices=vertices,
                         tmin=start_s + offset_s, tstep=1/frequency)
brain = stc.plot(subject=subject, subjects_dir=subjects_dir,
         time_viewer=True, hemi='both', clim=dict(kind='percent', pos_lims=(99, 99.5, 100)))
