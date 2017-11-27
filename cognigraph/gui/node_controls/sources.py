from pyqtgraph.parametertree import parameterTypes
import pylsl

from ...helpers.pyqtgraph import MyGroupParameter


class LSLStreamSourceControls(MyGroupParameter):
    STREAM_NAME_PLACEHOLDER = 'Click here to choose a stream'
    STREAM_NAMES_COMBO_NAME = 'Choose a stream: '

    def __init__(self, pipeline, **kwargs):

        self._pipeline = pipeline

        kwargs['title'] = 'LSL stream'
        super().__init__(**kwargs, )

        stream_names = [info.name() for info in pylsl.resolve_streams()]
        values = [self.STREAM_NAME_PLACEHOLDER] + stream_names
        stream_names_combo = parameterTypes.ListParameter(name=self.STREAM_NAMES_COMBO_NAME,
                                                          values=values, value=self.STREAM_NAME_PLACEHOLDER)
        stream_names_combo.sigValueChanged.connect(self._on_stream_name_picked)
        self.stream_names_combo = self.addChild(stream_names_combo)

    def _on_stream_name_picked(self, param, value):
        pass

    def _remove_placeholder_option(self, default):
        stream_names_combo = self.param(self.STREAM_NAMES_COMBO_NAME)
        values = stream_names_combo.opts['values']  # type: list
        try:
            values.remove(self.STREAM_NAME_PLACEHOLDER)
            self.setLimits(values)
        except ValueError:  # The placeholder option has already been removed
            pass


class BrainvisionSourceControls(MyGroupParameter):
    FILE_PATH_STR_NAME = 'Path to Brainvision file: '

    def __init__(self, pipeline, **kwargs):
        self._pipeline = pipeline

        kwargs['title'] = 'Brainvision file'
        super().__init__(**kwargs, )

        try:
            file_path = pipeline.source.file_path
        except:
            file_path = ''
        file_path_str = parameterTypes.SimpleParameter(type='str', name=self.FILE_PATH_STR_NAME, value=file_path)
        file_path_str.sigValueChanged.connect(self._on_file_path_changed)
        self.file_path_str = self.addChild(file_path_str)

    def _on_file_path_changed(self, param, value):
        self._pipeline.source.file_path = value
