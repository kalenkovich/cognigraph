import sys

from pyqtgraph import QtCore, QtGui

from cognigraph.pipeline import Pipeline
from cognigraph.nodes import sources, processors, outputs
from cognigraph import TIME_AXIS
from cognigraph.gui.window import GUIWindow

app = QtGui.QApplication(sys.argv)

# BOBE
pipeline = Pipeline()

file_path = r"/home/evgenii/Downloads/brainvision/Bulavenkova_A_2017-10-24_15-33-18_Rest.vhdr"
source = sources.BrainvisionSource(file_path=file_path)
pipeline.source = source

inverse = processors.InverseModel()
pipeline.add_processor(inverse)

three_dee = outputs.ThreeDeeBrain()
pipeline.add_output(three_dee)

pipeline.initialize_all_nodes()

three_dee.widget.show()

source.output = source.data.take(indices=(0,), axis=TIME_AXIS)

inverse.update()
three_dee.update()

# Sample
pipeline = Pipeline()

source = sources.BrainvisionSource(file_path='')
pipeline.source = source

import mne
data_path = mne.datasets.sample.data_path()
file_path = data_path + '/MEG/sample/sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(file_path, verbose='ERROR')
start, stop = raw.time_as_index([0, 60])
source.data = raw.get_data(start=start, stop=stop)

source.frequency = raw.info['sfreq']
source.channel_labels = raw.info['ch_names']
source.channel_count = len(source.channel_labels)


source.MAX_SAMPLES_IN_CHUNK = 1

import numpy as np
source.update()
assert(np.array_equal(source.output, source.data[:, 0][:, None]))


# Our solution
inverse = processors.InverseModel()
pipeline.add_processor(inverse)
inverse.initialize()


inverse.update()


# mne-python solution
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
inverse_operator = mne.minimum_norm.read_inverse_operator(fname_inv)


lambda2 = 1/inverse.snr**2
method = inverse.method

start, stop = raw.time_as_index([0, 15])

raw.set_eeg_reference()

stc = mne.minimum_norm.apply_inverse_raw(
    raw, inverse_operator, lambda2, method,
    start=start, stop=stop, pick_ori='normal')

assert(np.allclose(inverse.output, stc.data[:, 0][:, None]))


##############################################33

three_dee = outputs.ThreeDeeBrain()
pipeline.add_output(three_dee)
three_dee.initialize()
three_dee.update()
three_dee.widget.show()



three_dee.widget.show()

source.output = source.data.take(indices=(0,), axis=TIME_AXIS)
inverse.update()
three_dee.update()

###############################################3


# epoched

fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
evoked = mne.read_evokeds(fname_evoked, condition=0, baseline=(None, 0))

stc = apply_inverse(evoked, inv, lambda2, 'dSPM', pick_ori='vector')



# Timing run after window.initialize() in old_gui.py

pipeline.update_all_nodes()
pipeline.update_all_nodes()

source = pipeline.source
lin_filter, inverse, envelope = pipeline._processors

source.MAX_SAMPLES_IN_CHUNK = 100
def run_source():
    source = pipeline.source
    _time_of_the_last_update = source._time_of_the_last_update
    source.update()
    source._time_of_the_last_update = _time_of_the_last_update
print(source.output.shape)

%timeit run_source()
print(source.output.shape)


%timeit lin_filter.update()

%timeit inverse.update()

%timeit envelope.update()


%timeit pipeline._outputs[0].update()
