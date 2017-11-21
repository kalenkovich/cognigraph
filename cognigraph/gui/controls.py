from pyqtgraph import QtGui


class ControlsWidget(QtGui.QWidget):
    def __init__(self, pipeline):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QtGui.QVBoxLayout()

        labels = list()
        labels.append(QtGui.QLabel('Source parameters'))
        labels.append(QtGui.QLabel('First processor parameters'))
        labels.append(QtGui.QLabel('Second processor parameters'))
        labels.append(QtGui.QLabel('Output parameters'))

        for label in labels:
            layout.addWidget(label)
        self.setLayout(layout)

    def initialize(self):
        pass

