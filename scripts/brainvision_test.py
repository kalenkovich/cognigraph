%gui qt

from pyqtgraph import QtCore

from cognigraph.pipeline import Pipeline
from cognigraph.nodes import sources, processors, outputs
from cognigraph.gui.window import GUIWindow

pipeline = Pipeline()

file_path = r"C:\Users\evgenii\Downloads\brainvision\Bulavenkova_A_2017-10-24_15-33-18_Rest.vmrk"
pipeline.source = sources.BrainvisionSource(file_path=file_path)

# linear_filter = processors.LinearFilter(lower_cutoff=0.1, upper_cutoff=40)
# pipeline.add_processor(linear_filter)
# pipeline.add_processor(processors.InverseModel(method='MNE'))
# pipeline.add_processor(processors.EnvelopeExtractor())
# pipeline.add_output(outputs.ThreeDeeBrain())
# pipeline.add_output(outputs.LSLStreamOutput())
# pipeline.initialize_all_nodes()

window = GUIWindow(pipeline=pipeline)
window.init_ui()
window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
window.show()