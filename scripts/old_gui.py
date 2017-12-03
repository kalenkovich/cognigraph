import sys

from pyqtgraph import QtCore, QtGui

from cognigraph.pipeline import Pipeline
from cognigraph.nodes import sources, processors, outputs
from cognigraph import TIME_AXIS
from cognigraph.gui.window import GUIWindow

app = QtGui.QApplication(sys.argv)

pipeline = Pipeline()

#pipeline.source = sources.LSLStreamSource(stream_name='cognigraph-mock-stream')
file_path = r"/home/evgenii/Downloads/brainvision/Bulavenkova_A_2017-10-24_15-33-18_Rest.vhdr"
pipeline.source = sources.BrainvisionSource(file_path=file_path)

linear_filter = processors.LinearFilter(lower_cutoff=1, upper_cutoff=40)
pipeline.add_processor(linear_filter)
pipeline.add_processor(processors.InverseModel(method='MNE'))
pipeline.add_processor(processors.EnvelopeExtractor())
pipeline.add_output(outputs.ThreeDeeBrain())
pipeline.add_output(outputs.LSLStreamOutput())
# pipeline.initialize_all_nodes()

window = GUIWindow(pipeline=pipeline)
window.init_ui()
window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
window.show()

base_controls = window._controls._base_controls
source_controls = base_controls.source_controls
processors_controls = base_controls.processors_controls
outputs_controls = base_controls.outputs_controls

window.initialize()


def run():
    pipeline.update_all_nodes()
    # print(pipeline.source.output.shape[TIME_AXIS])


timer = QtCore.QTimer()
timer.timeout.connect(run)
frequency = pipeline.frequency
timer.setInterval(1000. / frequency * 10)

if __name__ == '__main__':
    import sys

    timer.start()

    # TODO: this runs when in iPython. It should not.
    # Start Qt event loop unless running in interactive mode or using pyside.
    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     sys.exit(QtGui.QApplication.instance().exec_())