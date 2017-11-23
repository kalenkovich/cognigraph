from pyqtgraph import QtGui
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
                        key
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
