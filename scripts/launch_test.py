import sys

from pyqtgraph import QtCore, QtGui
import mne
import os

from cognigraph.helpers.brainvision import read_brain_vision_data
from cognigraph.pipeline import Pipeline
from cognigraph.nodes import sources, processors, outputs, node
from cognigraph import TIME_AXIS
from cognigraph.gui.window import GUIWindow

app = QtGui.QApplication(sys.argv)

# Собираем узлы в пайплайн

pipeline = Pipeline()

source = sources.BrainvisionSource(file_path=launch_test_filepath)
source.loop_the_file = True
source.MAX_SAMPLES_IN_CHUNK = 30
pipeline.source = source


# Processors
preprocessing = processors.Preprocessing(collect_for_x_seconds=120)
pipeline.add_processor(preprocessing)

linear_filter = processors.LinearFilter(lower_cutoff=8.0, upper_cutoff=12.0)
pipeline.add_processor(linear_filter)

inverse_model = processors.InverseModel(method='MNE', snr=3.0)
pipeline.add_processor(inverse_model)

envelope_extractor = processors.EnvelopeExtractor()
pipeline.add_processor(envelope_extractor)

# Outputs
global_mode = outputs.ThreeDeeBrain.LIMITS_MODES.GLOBAL
three_dee_brain = outputs.ThreeDeeBrain(limits_mode=global_mode, buffer_length=6)
pipeline.add_output(three_dee_brain)
pipeline.add_output(outputs.LSLStreamOutput())

signal_viewer = outputs.SignalViewer()
pipeline.add_output(signal_viewer, input_node=linear_filter)


# Создаем окно

window = GUIWindow(pipeline=pipeline)
window.init_ui()
window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
window.show()


# Инициализируем все узлы
window.initialize()


# Симулируем работу препроцессинга по отлову шумных каналов

# Set bad channels and calculate interpolation matrix manually
bad_channel_labels = ['Fp2', 'F5', 'C5', 'F2', 'PPO10h', 'POO1', 'FCC2h']
preprocessing._bad_channel_indices = mne.pick_channels(source.mne_info['ch_names'], include=bad_channel_labels)
preprocessing._samples_to_be_collected = 0
preprocessing._enough_collected = True

preprocessing._interpolation_matrix = preprocessing._calculate_interpolation_matrix()
message = node.Message(there_has_been_a_change=True,
                       output_history_is_no_longer_valid=True)
preprocessing._deliver_a_message_to_receivers(message)


# Обрезаем данные в диапазоне с приличной записью
vhdr_file_path = os.path.splitext(source.file_path)[0] + '.vhdr'
start_s, stop_s = 80, 100
with source.not_triggering_reset():
    source.data, _ = read_brain_vision_data(vhdr_file_path, time_axis=TIME_AXIS, start_s=start_s, stop_s=stop_s)


# Создаем таймер, который все это будет гонять

def run():
    pipeline.update_all_nodes()


timer = QtCore.QTimer()
timer.timeout.connect(run)
frequency = pipeline.frequency
timer.setInterval(1000. / frequency * 10)


# Убираем предупреждения numpy, иначе в iPython некрасиво как-то Ж)
import numpy as np
np.warnings.filterwarnings('ignore')