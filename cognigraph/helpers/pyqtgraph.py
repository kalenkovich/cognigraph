from pyqtgraph import QtGui, QtCore
from pyqtgraph.parametertree import Parameter, parameterTypes, ParameterTree


class MyGroupParameter(parameterTypes.GroupParameter):
    def __init__(self, **opts):
        super().__init__(**opts)
        self.widget = None

    # Lets set in three cases:
    # 1) Value is not an instance of Parameter,
    # 2) Value is a child parameter and key is free
    # 3) The key is '_parent'. AFAIK it is the only Parameter-type attribute set by pyqtgraph.parametertree
    def __setattr__(self, key, value):

        if isinstance(value, Parameter) and key != '_parent':
            if key in self.__dict__:
                if ~isinstance(self.__dict__[key], Parameter):
                    msg = 'Can''t set {} to {} because it is already set to something other than a Parameter'.format(
                        key, value
                    )
                    raise ValueError(msg)
                else:
                    msg = 'Attribute {} is already set to {}. Before overwriting, use removeChild.'.format(
                        key, value
                    )
                    raise ValueError(msg)
            else:
                if value not in self.childs:
                    msg = 'Only children Parameter instances can be set as attributes. Call addChild* first.'
                    raise ValueError(msg)

        super().__setattr__(key, value)

    def removeChild(self, child):
        # If an attribute was set to refer to the child, remove it as well
        for key, value in self.__dict__.items():
            if value is child:
                delattr(self, key)
                break

        super().removeChild(child)

    def create_widget(self):
        self.widget = ParameterTree(showHeader=False)
        self.widget.setParameters(self, showTop=True)
        size_policy_preferred = QtGui.QSizePolicy.Preferred
        self.widget.setSizePolicy(size_policy_preferred, size_policy_preferred)
        return self.widget


class FileDialogParameterItem(parameterTypes.WidgetParameterItem):
    def __init__(self, param, depth):
        super().__init__(param, depth)
        self.hideWidget = False

    def makeWidget(self):

        opts = self.param.opts.copy()
        if 'limits' in opts:
            opts['minimum'], opts['maximum'] = opts['limits']
        else:
            raise ValueError("You have to provide 'limits' for this parameter")
        self.slider_widget = Slider(**opts)

        self.slider_widget.sigChanged = self.slider_widget.slider.sliderReleased
        self.slider_widget.value = self.slider_widget.value
        self.slider_widget.setValue = self.slider_widget.setValue
        return QtGui.QFileDialog()


class FileDialogParameter(Parameter):
    """Used for displaying a slider within the tree."""
    itemClass = FileDialogParameterItem


# All slider-related stuff was adapted from https://stackoverflow.com/a/42011414/3042770


class Slider(QtGui.QWidget):
    def __init__(self, minimum, maximum, value=None, suffix='', prec=1, parent=None, **kwargs):
        super().__init__(parent=parent)
        self.suffix = suffix
        self.prec = prec

        self.outerLayout = QtGui.QHBoxLayout(self)
        self.outerLayout.setContentsMargins(0, 0, 0, 0)
        self.outerLayout.setSpacing(0)

        self.label = QtGui.QLabel(self)
        self.outerLayout.addWidget(self.label)

        # Start of innerLayout - slider with spacer items on its sides
        self.innerLayout = QtGui.QHBoxLayout()

        spacerItem = QtGui.QSpacerItem(0, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.innerLayout.addItem(spacerItem)

        self.slider = QtGui.QSlider(self)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.innerLayout.addWidget(self.slider)

        spacerItem1 = QtGui.QSpacerItem(0, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.innerLayout.addItem(spacerItem1)
        # End of innerLayout

        self.outerLayout.addLayout(self.innerLayout)
        self.resize(self.sizeHint())

        self.minimum = minimum
        self.maximum = maximum
        self.slider.valueChanged.connect(self._setLabelValue_based_on_slider_value)
        if value:
            self.slider.setValue(self._value_to_slider_value(value))
            self.x = value
        else:
            self.x = self.minimum
        self._setLabelValue_based_on_slider_value(self.slider.value())

    def _setLabelValue_based_on_slider_value(self, slider_value):
        self.x = self._slider_value_to_value(slider_value)

        if self.prec == 0:
            label_text = "{0:d}{1}".format(int(self.x), self.suffix)
        else:
            label_text = "{0:.{2}g}{1}".format(self.x, self.suffix, self.prec)

        max_len = (
            max(len(str(int(limit))) for limit in (self.minimum, self.maximum))
            + (0 if self.prec ==0 else self.prec+1) + len(self.suffix)
        )

        self.label.setText("{:{width}}".format(label_text, width=max_len))

    def value(self):
        return self._slider_value_to_value(self.slider.value())

    def setValue(self, value):
        self.slider.setValue(self._value_to_slider_value(value))

    def _slider_value_to_value(self, slider_value):
        return self.minimum + (slider_value / (self.slider.maximum() - self.slider.minimum())) * (
            self.maximum - self.minimum)

    def _value_to_slider_value(self, value):
        return self.slider.minimum() + value / (self.maximum - self.minimum) * (
            self.slider.maximum() - self.slider.minimum())


class SliderParameterItem(parameterTypes.WidgetParameterItem):
    def __init__(self, param, depth):
        super().__init__(param, depth)
        self.hideWidget = False

    def makeWidget(self):

        opts = self.param.opts.copy()
        if 'limits' in opts:
            opts['minimum'], opts['maximum'] = opts['limits']
        else:
            raise ValueError("You have to provide 'limits' for this parameter")
        self.slider_widget = Slider(**opts)

        self.slider_widget.sigChanged = self.slider_widget.slider.sliderReleased
        self.slider_widget.value = self.slider_widget.value
        self.slider_widget.setValue = self.slider_widget.setValue
        return self.slider_widget


class SliderParameter(Parameter):
    """Used for displaying a slider within the tree."""
    itemClass = SliderParameterItem
