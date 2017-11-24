from collections import namedtuple

from pyqtgraph.parametertree import parameterTypes, ParameterTree, Parameter

from ..pipeline import Pipeline
from ..nodes import (
    sources as source_nodes,
    processors as processor_nodes,
    outputs as output_nodes
)
from .node_controls import (
    sources as source_controls,
    processors as processors_controls,
    outputs as outputs_controls
)
from ..helpers.pyqtgraph import MyGroupParameter
from ..helpers.misc import class_name_of


NodeControlClasses = namedtuple('NodeControlClasses', ['node_class', 'controls_class'])


class ProcessorsControls(MyGroupParameter):
    SUPPORTED_PROCESSORS = [
        NodeControlClasses(processor_nodes.LinearFilter, processors_controls.LinearFilterControls),
        NodeControlClasses(processor_nodes.InverseModel, processors_controls.InverseModelControls),
        NodeControlClasses(processor_nodes.EnvelopeExtractor, processors_controls.EnvelopeExtractorControls),
    ]

    def __init__(self, pipeline, **kwargs):
        self._pipeline = pipeline
        super().__init__(**kwargs)

        for processor_node in pipeline._processors:
            controls_class = self._find_controls_class_for_a_node(processor_node)
            self.addChild(controls_class(processor_node=processor_node))

    @classmethod
    def _find_controls_class_for_a_node(cls, processor_node):
        for node_control_classes in cls.SUPPORTED_PROCESSORS:
            if isinstance(processor_node, node_control_classes.node_class):
                return node_control_classes.controls_class

        # Raise an error if processor node is not supported
        msg = ("Node of class {0} is not supported by {1}.\n"
               "Add NodeControlClasses(node_class, controls_class) to {1}.SUPPORTED_PROCESSORS").format(
                class_name_of(processor_node), cls.__name__
            )
        raise ValueError(msg)


class OutputsControls(MyGroupParameter):
    pass


class BaseControls(MyGroupParameter):
    def __init__(self, pipeline):
        super().__init__(name='Base controls', type='BaseControls')
        self._pipeline = pipeline

        source_controls = SourceControls(pipeline=pipeline, name='Source')
        processors_controls = ProcessorsControls(pipeline=pipeline, name='Processors')
        output_controls = OutputsControls(pipeline=pipeline, name='Outputs')

        self.addChildren([source_controls, processors_controls, output_controls])


class SourceControls(MyGroupParameter):
    SourceClasses = namedtuple('SourceClasses', ['node_class', 'controls_class'])
    SOURCE_OPTIONS = {
        'LSL stream': SourceClasses(source_nodes.LSLStreamSource,
                                    source_node_controls.LSLStreamSourceControls),
    }
    SOURCE_TYPE_COMBO_NAME = 'kind'  # Can't use 'type' bc there is already an attribute with this name
    SOURCE_TYPE_PLACEHOLDER = ''
    SOURCE_CONTROLS_NAME = 'source controls'

    def __init__(self, pipeline, **kwargs):
        self._pipeline = pipeline
        super().__init__(**kwargs)

        labels = [self.SOURCE_TYPE_PLACEHOLDER] + [label for label in self.SOURCE_OPTIONS]
        source_type_combo = parameterTypes.ListParameter(name=self.SOURCE_TYPE_COMBO_NAME, title='Source type: ',
                                                         values=labels, value=labels[0])
        source_type_combo.sigValueChanged.connect(self._on_source_type_changed)
        self.addChild(source_type_combo)

        self._invisible_placeholder_parameter = Parameter(name=self.SOURCE_CONTROLS_NAME, visible=False)
        self.addChild(self._invisible_placeholder_parameter)

    def _on_source_type_changed(self, param, value):
        self.removeChild(self.source_controls)
        if value != self.SOURCE_TYPE_PLACEHOLDER:
            source_classes = self.SOURCE_OPTIONS[value]
            source_controls = source_classes.controls_class(pipeline=self._pipeline, name=self.SOURCE_CONTROLS_NAME)
            self.addChild(source_controls)
        else:
            self.addChild(self._invisible_placeholder_parameter)


class Controls(object):
    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self._pipeline = pipeline  # type: Pipeline
        self._base_controls = BaseControls(pipeline=self._pipeline)
        self.widget = self._base_controls.create_widget()

    def initialize(self):
        pass


