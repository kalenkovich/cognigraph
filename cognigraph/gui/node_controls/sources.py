from pyqtgraph.parametertree import parameterTypes
import pylsl

from ...helpers.pyqtgraph import MyGroupParameter


class LSLStreamSourceControls(MyGroupParameter):
    STREAM_NAME_PLACEHOLDER = 'Click here to choose a stream'
    STREAM_NAMES_COMBO_NAME = 'stream names combo'

    def __init__(self, pipeline, **kwargs):

        self._pipelne = pipeline

        kwargs['title'] = 'LSL stream settings'
        super().__init__(**kwargs, )

        stream_names = [info.name() for info in pylsl.resolve_streams()]
        values = [self.STREAM_NAME_PLACEHOLDER] + stream_names
        stream_names_combo = parameterTypes.ListParameter(name=self.STREAM_NAMES_COMBO_NAME, title='Choose a stream: ',
                                                          values=values, value=self.STREAM_NAME_PLACEHOLDER)
        stream_names_combo.sigValueChanged.connect(self._on_stream_name_picked)
        stream_names_combo.sigDefaultChanged.connect(self._remove_placeholder_option)

        self.addChild(stream_names_combo)

    def _on_stream_name_picked(self, param, value):
        pass

    def _remove_placeholder_option(self, default):
        stream_names_combo = self.param(self.STREAM_NAMES_COMBO_NAME)
        values = stream_names_combo.opts['values']  # type: list
        values.remove(self.STREAM_NAME_PLACEHOLDER)
        self.setLimits(values)
