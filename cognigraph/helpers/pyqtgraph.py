from pyqtgraph import QtGui
from pyqtgraph.parametertree import parameterTypes, ParameterTree


class MyGroupParameter(parameterTypes.GroupParameter):
    def __init__(self, **opts):
        super().__init__(**opts)
        self.widget = None

    def addChild(self, child, *args, **kwargs):
        attr_name = self._construct_attr_name(child)

        # Can't use hasattr because Parameter.__getattr__ looks for attr in self.names which is a dict of children's
        # names
        if attr_name in self.__dict__:
            child_name = self._get_name_of_a_child(child)
            msg = "Can't add child '{}' to '{}' because it already has an attribute named '{}'".format(
                child_name, self.name(), attr_name)
            raise ValueError(msg)
        else:
            child = super().addChild(child, *args, **kwargs)
            setattr(self, attr_name, child)

    def removeChild(self, child):
        super().removeChild(child)
        attr_name = self._construct_attr_name(child)
        delattr(self, attr_name)

    @staticmethod
    def _get_name_of_a_child(child):
        # child can be a Parameter instance or it can be a dict
        try:
            child_name = child.name()
        except AttributeError:
            child_name = child['name']
        return child_name

    def _construct_attr_name(self, child):
        child_name = self._get_name_of_a_child(child)
        return '_'.join(child_name.lower().split(' '))

    def create_widget(self):
        self.widget = ParameterTree(showHeader=False)
        self.widget.setParameters(self, showTop=True)
        size_policy_preferred = QtGui.QSizePolicy.Preferred
        self.widget.setSizePolicy(size_policy_preferred, size_policy_preferred)
        return self.widget
