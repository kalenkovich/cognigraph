from collections import namedtuple, OrderedDict

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


class MultipleNodeControls(MyGroupParameter):
    """Base class for grouping of node settings (processors or outputs). Source is supported by a separate class."""

    @property
    def SUPPORTED_NODES(self):
        raise NotImplementedError

    def __init__(self, nodes, **kwargs):
        self._nodes = nodes
        super().__init__(**kwargs)

        for node in nodes:
            controls_class = self._find_controls_class_for_a_node(node)
            self.addChild(controls_class(node))

    @classmethod
    def _find_controls_class_for_a_node(cls, processor_node):
        for node_control_classes in cls.SUPPORTED_NODES:
            if isinstance(processor_node, node_control_classes.node_class):
                return node_control_classes.controls_class

        # Raise an error if processor node is not supported
        msg = ("Node of class {0} is not supported by {1}.\n"
               "Add NodeControlClasses(node_class, controls_class) to {1}.SUPPORTED_NODES").format(
                class_name_of(processor_node), cls.__name__
            )
        raise ValueError(msg)


class ProcessorsControls(MultipleNodeControls):
    SUPPORTED_NODES = [
        NodeControlClasses(processor_nodes.LinearFilter, processors_controls.LinearFilterControls),
        NodeControlClasses(processor_nodes.InverseModel, processors_controls.InverseModelControls),
        NodeControlClasses(processor_nodes.EnvelopeExtractor, processors_controls.EnvelopeExtractorControls),
    ]


class OutputsControls(MultipleNodeControls):
    SUPPORTED_NODES = [
        NodeControlClasses(output_nodes.LSLStreamOutput, outputs_controls.LSLStreamOutputControls),
        NodeControlClasses(output_nodes.ThreeDeeBrain, outputs_controls.ThreeDeeBrainControls),
    ]


class BaseControls(MyGroupParameter):
    def __init__(self, pipeline):
        super().__init__(name='Base controls', type='BaseControls')
        self._pipeline = pipeline

        # TODO: Change names to delineate source_controls as defined here and source_controls - gui.node_controls.source
        source_controls = SourceControls(pipeline=pipeline, name='Source')
        processors_controls = ProcessorsControls(nodes=pipeline._processors, name='Processors')
        outputs_controls = OutputsControls(nodes=pipeline._outputs, name='Outputs')

        self.source_controls = self.addChild(source_controls)
        self.processors_controls = self.addChild(processors_controls)
        self.outputs_controls = self.addChild(outputs_controls)


class SourceControls(MyGroupParameter):
    """Represents a drop-down list with the names of supported source types. Selecting a type creates controls for that
    type below the drop-down.
    """

    # Order is important. Entries with node subclasses must precede entries with the parent class
    SOURCE_OPTIONS = OrderedDict((
        ('LSL stream', NodeControlClasses(source_nodes.LSLStreamSource,
                                          source_controls.LSLStreamSourceControls)),
        ('Brainvision data', NodeControlClasses(source_nodes.BrainvisionSource,
                                                source_controls.BrainvisionSourceControls)),
    ))

    SOURCE_TYPE_COMBO_NAME = 'Source type: '
    SOURCE_TYPE_PLACEHOLDER = ''
    SOURCE_CONTROLS_NAME = 'source controls'

    def __init__(self, pipeline: Pipeline, **kwargs):
        self._pipeline = pipeline
        super().__init__(**kwargs)

        labels = [self.SOURCE_TYPE_PLACEHOLDER] + [label for label in self.SOURCE_OPTIONS]
        source_type_combo = parameterTypes.ListParameter(name=self.SOURCE_TYPE_COMBO_NAME,
                                                         values=labels, value=labels[0])
        source_type_combo.sigValueChanged.connect(self._on_source_type_changed)
        self.source_type_combo = self.addChild(source_type_combo)

        if pipeline.source is not None:
            for source_option, classes in self.SOURCE_OPTIONS.items():
                if isinstance(pipeline.source, classes.node_class):
                    self.source_type_combo.setValue(source_option)

    def _on_source_type_changed(self, param, value):
        try:
            source_controls = self.source_controls
            self.removeChild(source_controls)
        except AttributeError:  # No source type has been chosen
            pass
        if value != self.SOURCE_TYPE_PLACEHOLDER:
            # Update source controls
            source_classes = self.SOURCE_OPTIONS[value]
            controls = source_classes.controls_class(pipeline=self._pipeline, name=self.SOURCE_CONTROLS_NAME)
            self.source_controls = self.addChild(controls)

            # Update source
            if not isinstance(self._pipeline.source, source_classes.node_class):
                self._pipeline.source = self.source_controls.create_node()


class Controls(object):
    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self._pipeline = pipeline  # type: Pipeline
        self._base_controls = BaseControls(pipeline=self._pipeline)
        self.widget = self._base_controls.create_widget()

    def initialize(self):
        pass


