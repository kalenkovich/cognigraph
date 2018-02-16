from typing import List

from pyqtgraph import QtGui

from ..pipeline import Pipeline
from .controls import Controls


class GUIWindow(QtGui.QMainWindow):
    def __init__(self, pipeline=Pipeline()):
        super().__init__()
        self._pipeline = pipeline  # type: Pipeline
        self._controls = Controls(pipeline=self._pipeline)
        self._controls_widget = self._controls.widget

        self.main_layout = None  # type: QtGui.QBoxLayout
        self.widgets_layout = None  # type: QtGui.QBoxLayout

    def init_ui(self):
        self._controls.initialize()

        central_widget = QtGui.QWidget()
        self.setCentralWidget(central_widget)

        widgets_layout = QtGui.QHBoxLayout()
        main_layout = QtGui.QHBoxLayout()
        main_layout.addLayout(widgets_layout)
        self._controls_widget.setMinimumWidth(400)
        main_layout.addWidget(self._controls_widget)

        self.centralWidget().setLayout(main_layout)
        self.main_layout = main_layout
        self.widgets_layout = widgets_layout

    def initialize(self):
        self._pipeline.initialize_all_nodes()
        for node_widget in self._node_widgets:
            node_widget.setMinimumWidth(400)
            self.widgets_layout.addWidget(node_widget)

    @property
    def _node_widgets(self) -> List[QtGui.QWidget]:
        node_widgets = list()
        for node in self._pipeline.all_nodes:
            try:
                node_widgets.append(node.widget)
            except AttributeError:
                pass
        return node_widgets
