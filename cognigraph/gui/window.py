from typing import List

from pyqtgraph import QtGui

from ..pipeline import Pipeline
from .controls import ControlsWidget


class GUIWindow(QtGui.QMainWindow):
    def __init__(self, pipeline=Pipeline()):
        super().__init__()
        self._pipeline = pipeline  # type: Pipeline
        self._controls_widget = ControlsWidget(pipeline=self._pipeline)
        self._node_widgets = list()  # type: List[QtGui.QWidget]

    def init_ui(self):
        self._pipeline.initialize_all_nodes()
        self._node_widgets = self._get_node_widgets()
        self._controls_widget.initialize()

        central_widget = QtGui.QWidget()
        self.setCentralWidget(central_widget)

        widgets_layout = QtGui.QHBoxLayout()
        for node_widget in self._node_widgets:
            widgets_layout.addWidget(node_widget)

        main_layout = QtGui.QHBoxLayout()
        main_layout.addLayout(widgets_layout)
        main_layout.addWidget(self._controls_widget)

        self.centralWidget().setLayout(main_layout)

    def _get_node_widgets(self):
        node_widgets = list()
        for node in self._pipeline.all_nodes:
            try:
                node_widgets.append(node.widget)
            except AttributeError:
                pass
        return node_widgets
