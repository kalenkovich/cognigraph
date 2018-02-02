from ...nodes.node import ProcessorNode
from ...nodes import processors
from ...helpers.pyqtgraph import MyGroupParameter, parameterTypes
from ...helpers.misc import class_name_of


class ProcessorNodeControls(MyGroupParameter):

    DISABLED_NAME = 'Disable: '

    @property
    def PROCESSOR_CLASS(self):
        raise NotImplementedError

    @property
    def CONTROLS_LABEL(self):
        raise NotImplementedError

    def __init__(self, processor_node: PROCESSOR_CLASS = None, **kwargs):
        super().__init__(name=self.CONTROLS_LABEL, **kwargs)

        if processor_node is None:
            raise ValueError("Right now we can create controls only for an already existing node")

        self._processor_node = processor_node  # type: self.PROCESSOR_CLASS
        self._create_parameters()
        self._add_disable_parameter()

    def _create_parameters(self):
        raise NotImplementedError

    def _add_disable_parameter(self):
        disabled_value = False  # TODO: change once disabling is implemented
        disabled = parameterTypes.SimpleParameter(type='bool', name=self.DISABLED_NAME, value=disabled_value,
                                                  readonly=False)
        disabled.sigValueChanged.connect(self._on_disabled_changed)
        self.disabled = self.addChild(disabled)

    def _on_disabled_changed(self, param, value):
        self._processor_node.disabled = value


class PreprocessingControls(ProcessorNodeControls):
    PROCESSOR_CLASS = processors.Preprocessing
    CONTROLS_LABEL = 'Preprocessing'

    DURATION_NAME = 'Baseline duration: '

    def _create_parameters(self):

        duration_value = self._processor_node.collect_for_x_seconds
        duration = parameterTypes.SimpleParameter(type='int', name=self.DURATION_NAME, suffix='s',
                                                  limits=(30, 180), value=duration_value)
        self.duration = self.addChild(duration)
        self.duration.sigValueChanged.connect(self._on_duration_changed)

    def _on_duration_changed(self, param, value):
        self._processor_node.collect_for_x_seconds = value

class LinearFilterControls(ProcessorNodeControls):
    PROCESSOR_CLASS = processors.LinearFilter
    CONTROLS_LABEL = 'Linear filter'

    LOWER_CUTOFF_NAME = 'Lower cutoff: '
    UPPER_CUTOFF_NAME = 'Upper cutoff: '

    def _create_parameters(self):

        lower_cutoff_value = self._processor_node.lower_cutoff
        upper_cutoff_value = self._processor_node.upper_cutoff

        lower_cutoff = parameterTypes.SimpleParameter(type='float', name=self.LOWER_CUTOFF_NAME,
                                                      decimals=1, suffix='Hz',
                                                      limits=(0, upper_cutoff_value), value=lower_cutoff_value)
        upper_cutoff = parameterTypes.SimpleParameter(type='int', name=self.UPPER_CUTOFF_NAME, suffix='Hz',
                                                      limits=(lower_cutoff_value, 100), value=upper_cutoff_value)

        self.lower_cutoff = self.addChild(lower_cutoff)
        self.upper_cutoff = self.addChild(upper_cutoff)

        lower_cutoff.sigValueChanged.connect(self._on_lower_cutoff_changed)
        upper_cutoff.sigValueChanged.connect(self._on_upper_cutoff_changed)

    def _on_lower_cutoff_changed(self, param, value):
        # Update the node
        if value == 0.0:
            self._processor_node.lower_cutoff = None
        else:
            self._processor_node.lower_cutoff = value  # TODO: implement on the filter side
        # Update the upper cutoff so that it is not lower that the lower one
        if self.upper_cutoff.value() != 0.0:
            self.upper_cutoff.setLimits((value, 100))

    def _on_upper_cutoff_changed(self, param, value):
        # Update the node
        if value == 0.0:
            self._processor_node.upper_cutoff = None
            value = 100
        else:
            self._processor_node.upper_cutoff = value  # TODO: implement on the filter side

        if self.lower_cutoff.value() != 0:
            # Update the lower cutoff so that it is not higher that the upper one
            self.lower_cutoff.setLimits((0, value))


class InverseModelControls(ProcessorNodeControls):
    CONTROLS_LABEL = 'Inverse modelling'
    PROCESSOR_CLASS = processors.InverseModel

    METHODS_COMBO_NAME = 'Method: '

    def _create_parameters(self):

        method_values = self.PROCESSOR_CLASS.SUPPORTED_METHODS
        method_value = self._processor_node.method
        methods_combo = parameterTypes.ListParameter(name=self.METHODS_COMBO_NAME,
                                                     values=method_values, value=method_value)
        methods_combo.sigValueChanged.connect(self._on_method_changed)
        self.methods_combo = self.addChild(methods_combo)

    def _on_method_changed(self, param, value):
        self._processor_node.method = value


class EnvelopeExtractorControls(ProcessorNodeControls):
    PROCESSOR_CLASS = processors.EnvelopeExtractor
    CONTROLS_LABEL = 'Extract envelope: '

    FACTOR_NAME = 'Factor: '
    METHODS_COMBO_NAME = 'Method: '

    def _create_parameters(self):

        method_values = ['Exponential smoothing']  # TODO: change once we support more methods
        method_value = self._processor_node.method
        methods_combo = parameterTypes.ListParameter(name=self.METHODS_COMBO_NAME,
                                                     values=method_values, value=method_value)
        methods_combo.sigValueChanged.connect(self._on_method_changed)
        self.methods_combo = self.addChild(methods_combo)

        factor_value = self._processor_node.factor
        factor_spin_box = parameterTypes.SimpleParameter(type='float', name=self.FACTOR_NAME,
                                                         decimals=2, limits=(0.5, 0.99), value=factor_value)
        factor_spin_box.sigValueChanged.connect(self._on_factor_changed)
        self.factor_spin_box = self.addChild(factor_spin_box)

    def _on_method_changed(self):
        pass  # TODO: implement

    def _on_factor_changed(self):
        pass  # TODO: implement


class BeamformerControls(ProcessorNodeControls):
    PROCESSOR_CLASS = processors.Beamformer
    CONTROLS_LABEL = 'Beamformer'

    ADAPTIVENESS_NAME = 'Use adaptive version: '
    SNR_NAME = 'SNR: '
    OUTPUT_TYPE_COMBO_NAME = 'Output type: '
    FORGETTING_FACTOR_NAME = 'Forgetting factor (per second): '

    def _create_parameters(self):
        # snr: float = 3.0, output_type: str = 'power', is_adaptive: bool = False,
        # forgetting_factor_per_second = 0.99
        is_adaptive = self._processor_node.is_adaptive
        adaptiveness_check = parameterTypes.SimpleParameter(type='bool', name=self.ADAPTIVENESS_NAME, value=is_adaptive,
                                                            readonly=False)
        adaptiveness_check.sigValueChanged.connect(self._on_adaptiveness_changed)
        self.adaptiveness_check = self.addChild(adaptiveness_check)

        snr_value = self._processor_node.snr
        snr_spin_box = parameterTypes.SimpleParameter(type='float', name=self.SNR_NAME,
                                                      decimals=1, limits=(1.0, 10.0), value=snr_value)
        snr_spin_box.sigValueChanged.connect(self._on_snr_changed)
        self.snr_spin_box = self.addChild(snr_spin_box)

        output_type_value = self._processor_node.output_type
        output_type_values = self.PROCESSOR_CLASS.SUPPORTED_OUTPUT_TYPES
        output_type_combo = parameterTypes.ListParameter(name=self.OUTPUT_TYPE_COMBO_NAME,
                                                         values=output_type_values, value=output_type_value)
        output_type_combo.sigValueChanged.connect(self._on_output_type_changed)
        self.output_type_combo = self.addChild(output_type_combo)

        forgetting_factor_value = self._processor_node.forgetting_factor_per_second
        forgetting_factor_spin_box = parameterTypes.SimpleParameter(type='float', name=self.FORGETTING_FACTOR_NAME,
                                                                    decimals=2, limits=(0.90, 0.99),
                                                                    value=forgetting_factor_value)
        forgetting_factor_spin_box.sigValueChanged.connect(self._on_forgetting_factor_changed)
        self.forgetting_factor_spin_box.addChild(forgetting_factor_spin_box)

    def _on_adaptiveness_changed(self, param, value):
        self.forgetting_factor_spin_box.show(value)
        self._processor_node.is_adaptive = value

    def _on_snr_changed(self, param, value):
        self._processor_node.snr = value

    def _on_output_type_changed(self, param, value):
        self._processor_node.output_type = value

    def _on_forgetting_factor_changed(self, param, value):
        self._processor_node.forgetting_factor_per_second = value
