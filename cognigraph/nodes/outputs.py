import os
from types import SimpleNamespace

import mne
import nibabel as nib
import numpy as np
import pyqtgraph.opengl as gl
from matplotlib import cm
from matplotlib.colors import Colormap as matplotlib_Colormap
from mne.datasets import sample
from scipy import sparse

from ..helpers.pysurfer.smoothing_matrix import smoothing_matrix as calculate_smoothing_matrix, mesh_edges
from .node import OutputNode
from .. import CHANNEL_AXIS
from ..helpers.lsl import convert_numpy_format_to_lsl, convert_numpy_array_to_lsl_chunk, create_lsl_outlet
from ..helpers.matrix_functions import last_sample, make_time_dimension_second
from ..helpers.ring_buffer import RingBuffer


class LSLStreamOutput(OutputNode):

    def _check_value(self, key, value):
        pass  # TODO: check that value as a string usable as a stream name

    CHANGES_IN_THESE_REQUIRE_RESET = ('stream_name', )

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = (
        'source_name', 'frequency', 'dtype', 'channel_labels'
    )

    def reset(self):
        # It is impossible to change then name of an already started stream so we have to initialize again
        self.initialize()

    def __init__(self, stream_name=None):
        super().__init__()
        self._provided_stream_name = stream_name
        self.stream_name = None
        self._outlet = None

    def _initialize(self):

        # If no name was supplied we will use a modified version of the source name (a file or a stream name)
        source_name = self.traverse_back_and_find('source_name')
        self.stream_name = self._provided_stream_name or (source_name + '_output')

        # Get other info from somewhere down the predecessor chain
        frequency = self.traverse_back_and_find('frequency')
        dtype = self.traverse_back_and_find('dtype')
        channel_format = convert_numpy_format_to_lsl(dtype)
        channel_labels = self.traverse_back_and_find('channel_labels')

        self._outlet = create_lsl_outlet(name=self.stream_name, frequency=frequency, channel_format=channel_format,
                                         channel_labels=channel_labels)

    def _update(self):
        chunk = self.input_node.output
        lsl_chunk = convert_numpy_array_to_lsl_chunk(chunk)
        self._outlet.push_chunk(lsl_chunk)


class ThreeDeeBrain(OutputNode):
    def _check_value(self, key, value):
        pass

    CHANGES_IN_THESE_REQUIRE_RESET = ('buffer_length', 'take_abs', )

    def reset(self):
        self.limits_buffer.clear()

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = (
        'mne_inverse_model_file_path', 'channel_labels'
    )

    LIMITS_MODES = SimpleNamespace(GLOBAL='Global', LOCAL='Local', MANUAL='Manual')

    def __init__(self, take_abs=True, limits_mode=LIMITS_MODES.LOCAL, buffer_length=1, **brain_painter_kwargs):
        super().__init__()

        self.limits_mode = limits_mode
        self.lock_limits = False
        self.buffer_length = buffer_length
        self.limits_buffer = None  # type: RingBuffer
        self.take_abs = take_abs
        self.colormap_limits = SimpleNamespace(lower=None, upper=None)

        self.brain_painter = BrainPainter(mne_inverse_model_file_path=None, **brain_painter_kwargs)

    def _initialize(self):
        mne_inverse_model_file_path = self.traverse_back_and_find('mne_inverse_model_file_path')
        self.brain_painter.initialize(mne_inverse_model_file_path)

        frequency = self.traverse_back_and_find('frequency')
        buffer_sample_count = np.int(self.buffer_length * frequency)
        self.limits_buffer = RingBuffer(row_cnt=2, maxlen=buffer_sample_count)

    def _update(self):
        sources = self.input_node.output
        if self.take_abs:
            sources = np.abs(sources)
        self._update_colormap_limits(sources)
        normalized_sources = self._normalize_sources(last_sample(sources))
        self.brain_painter.draw(normalized_sources)

    def _update_colormap_limits(self, sources):
        self.limits_buffer.extend(np.array([
            make_time_dimension_second(np.min(sources, axis=CHANNEL_AXIS)),
            make_time_dimension_second(np.max(sources, axis=CHANNEL_AXIS)),
        ]))

        if self.limits_mode == self.LIMITS_MODES.GLOBAL:
            mins, maxs = self.limits_buffer.data
            self.colormap_limits.lower = np.percentile(mins, q = 5)
            self.colormap_limits.upper = np.percentile(maxs, q = 95)
        elif self.limits_mode == self.LIMITS_MODES.LOCAL:
            sources = last_sample(sources)
            self.colormap_limits.lower = np.min(sources)
            self.colormap_limits.upper = np.max(sources)
        elif self.limits_mode == self.LIMITS_MODES.MANUAL:
            pass

    def _normalize_sources(self, last_sources):
        minimum = self.colormap_limits.lower
        maximum = self.colormap_limits.upper
        return (last_sources - minimum) / (maximum - minimum)

    @property
    def widget(self):
        if self.brain_painter.widget is not None:
            return self.brain_painter.widget
        else:
            raise AttributeError('{} does not have widget yet. Probably has not been initialized')


class BrainPainter(object):
    def __init__(self, mne_inverse_model_file_path, threshold_pct=50,
                 brain_colormap: matplotlib_Colormap = cm.Greys,
                 data_colormap: matplotlib_Colormap = cm.Reds,
                 show_curvature=True, surfaces_dir=None, buffer_length=1):
        """
        This is the last step. Object of this class draws whatever data are given to it on the cortex mesh. No changes,
        except for thresholding, are made.

        :param threshold_pct: Only values exceeding this percentage threshold will be shown
        :param show_curvature: If True, concave areas will be shown in darker grey, convex - in lighter
        :param surfaces_dir: Path to the Fressurfer surf directory. If None, mne's sample's surfaces will be used.
        """

        self.threshold_pct = threshold_pct
        self.show_curvature = show_curvature

        self.brain_colormap = brain_colormap
        self.data_colormap = data_colormap

        self.surfaces_dir = surfaces_dir  # type: str
        self.mesh_data = None  # type: gl.MeshData
        self.smoothing_matrix = None  # type: np.ndarray
        self.widget = None  # type: gl.GLViewWidget

        self.background_colors = None  # type: np.ndarray  # N x 4
        self.mesh_item = None  # type: gl.GLMeshItem

    def initialize(self, mne_inverse_model_file_path):
        self.surfaces_dir = self.surfaces_dir or self._guess_surfaces_dir_based_on(mne_inverse_model_file_path)
        self.mesh_data = self._get_mesh_data_from_surfaces_dir()
        self.smoothing_matrix = self._get_smoothing_matrix(mne_inverse_model_file_path)

        if self.widget is None:
            self.widget = self._create_widget()
        else:  # Do not recreate the widget, just clear it
            for item in self.widget.items:
                self.widget.removeItem(item)

        self.background_colors = self._calculate_background_colors(self.show_curvature)
        self.mesh_data.setVertexColors(self.background_colors)
        self.mesh_item = gl.GLMeshItem(meshdata=self.mesh_data, shader='shaded')
        self.widget.addItem(self.mesh_item)

    def draw(self, normalized_values):

        sources_smoothed = self.smoothing_matrix.dot(normalized_values)
        colors = self.data_colormap(sources_smoothed)

        threshold = self.threshold_pct / 100
        invisible_mask = sources_smoothed <= threshold
        colors[invisible_mask] = self.background_colors[invisible_mask]
        colors[~invisible_mask] *= self.background_colors[~invisible_mask, 0, np.newaxis]

        self.mesh_data.setVertexColors(colors)
        self.mesh_item.meshDataChanged()

    def _get_mesh_data_from_surfaces_dir(self, cortex_type='inflated') -> gl.MeshData:
        surf_paths = [os.path.join(self.surfaces_dir, '{}.{}'.format(h, cortex_type))
                      for h in ('lh', 'rh')]
        lh_mesh, rh_mesh = [nib.freesurfer.read_geometry(surf_path) for surf_path in surf_paths]
        lh_vertexes, lh_faces = lh_mesh
        rh_vertexes, rh_faces = rh_mesh

        # Move all the vertexes so that the lh has x (L-R) <= 0 and rh - >= 0
        lh_vertexes[:, 0] -= np.max(lh_vertexes[:, 0])
        rh_vertexes[:, 0] -= np.min(rh_vertexes[:, 0])

        # Combine two meshes
        vertexes = np.r_[lh_vertexes, rh_vertexes]
        lh_vertex_cnt = lh_vertexes.shape[0]
        faces = np.r_[lh_faces, lh_vertex_cnt + rh_faces]

        # Move the mesh so that the center of the brain is at (0, 0, 0) (kinda)
        vertexes[:, 1:2] -= np.mean(vertexes[:, 1:2])

        # Invert vertex normals for more reasonable lighting (I am not sure if the pyqtgraph's shader has a bug or
        # gl.MeshData's calculation of normals does
        mesh_data = gl.MeshData(vertexes=vertexes, faces=faces)
        mesh_data._vertexNormals = mesh_data.vertexNormals() * (-1)

        return mesh_data

    def _get_mesh_data_from_inverse_operator(self, inverse_operator_file_path) -> (list, gl.MeshData):
        # mne's inverse operator is a dict with the geometry information under the key 'src'.
        # inverse_operator['src'] is a list two items each of which corresponds to one hemisphere.
        inverse_operator = mne.minimum_norm.read_inverse_operator(inverse_operator_file_path, verbose='ERROR')
        left_hemi, right_hemi = inverse_operator['src']

        # Each hemisphere is represented by a dict containing the list of all vertices from the original mesh (with
        # default options in FreeSurfer that is ~150K vertices). These are stored under the key 'rr'.

        # Only a small subset of these vertices was likely used during the construction of the inverse operator. The
        # mesh containing only the used vertices is represented by an array of faces stored under the 'use_tris' key.
        # This submesh still contains some extra vertices so that it is still a manifold.

        # Each face is a row with the indices of the vertices of that face. The indexing is into the 'rr' array
        # containing all the vertices.

        # Let's now combine two meshes into one. Also save the indexes of the sources
        vertexes = np.r_[left_hemi['rr'], right_hemi['rr']]
        lh_vertex_cnt = left_hemi['rr'].shape[0]
        faces = np.r_[left_hemi['use_tris'], lh_vertex_cnt + right_hemi['use_tris']]
        sources_idx = np.r_[left_hemi['vertno'], lh_vertex_cnt + right_hemi['vertno']]

        return sources_idx, gl.MeshData(vertexes=vertexes, faces=faces)

    def _create_widget(self):
        widget = gl.GLViewWidget()
        # Set the camera at a distance proportional to the size of the mesh along the widest dimension
        max_ptp = max(np.ptp(self.mesh_data.vertexes(), axis=0))
        widget.setCameraPosition(distance=(1.5 * max_ptp))
        return widget

    def _calculate_background_colors(self, show_curvature):
        if show_curvature:
            curvature_file_paths = [os.path.join(self.surfaces_dir,
                                                 "{}.curv".format(h)) for h in ('lh', 'rh')]
            curvatures = [nib.freesurfer.read_morph_data(path) for path in curvature_file_paths]
            curvature = np.concatenate(curvatures)
            return self.brain_colormap((curvature > 0) / 3 + 1 / 3)  # 1/3 for concave, 2/3 for convex
        else:
            background_color = self.brain_colormap(0.5)
            total_vertex_cnt = self.mesh_data.vertexes().shape[0]
            return np.tile(background_color, total_vertex_cnt)

    @staticmethod
    def _guess_surfaces_dir_based_on(mne_inverse_model_file_path):
        # If tha inverse model that was used is from the mne's sample dataset, then we can use curvatures from there
        path_to_sample = os.path.realpath(sample.data_path())
        if os.path.realpath(mne_inverse_model_file_path).startswith(path_to_sample):
            return os.path.join(path_to_sample, "subjects", "sample", "surf")

    @staticmethod
    def read_smoothing_matrix():
        lh_npz = np.load('playground/vs_pysurfer/smooth_mat_lh.npz')
        rh_npz = np.load('playground/vs_pysurfer/smooth_mat_rh.npz')

        smooth_mat_lh = sparse.coo_matrix((
            lh_npz['data'], (lh_npz['row'], lh_npz['col'])),
            shape=lh_npz['shape'] + rh_npz['shape'])

        lh_row_cnt, lh_col_cnt = lh_npz['shape']
        smooth_mat_rh = sparse.coo_matrix((
            rh_npz['data'], (rh_npz['row'] + lh_row_cnt, rh_npz['col'] + lh_col_cnt)),
            shape=rh_npz['shape'] + lh_npz['shape'])

        return smooth_mat_lh.tocsc() + smooth_mat_rh.tocsc()

    def _get_smoothing_matrix(self, mne_inverse_model_file_path):
        """Creates or loads a smoothing matrix that lets us interpolate source values onto all mesh vertices"""
        # Not all the vertices in the inverse model mesh are sources. sources_idx actually indexes into the union of
        # high-definition meshes for left and right hemispheres. The smoothing matrix then lets us assign a color to
        # each vertex. If in future we decide to use low-definition mesh from the inverse model for drawing, we should
        # index into that.
        # Shorter: the coordinates of the jth source are in self.mesh_data.vertexes()[sources_idx[j], :]
        smoothing_matrix_file_path = os.path.splitext(mne_inverse_model_file_path)[0] + '-smoothing-matrix.npz'
        try:
            return sparse.load_npz(smoothing_matrix_file_path)
        except FileNotFoundError:
            print('Calculating smoothing matrix. This might take a while the first time.')
            sources_idx, _ = self._get_mesh_data_from_inverse_operator(mne_inverse_model_file_path)
            adj_mat = mesh_edges(self.mesh_data.faces())
            smoothing_matrix = calculate_smoothing_matrix(sources_idx, adj_mat)
            sparse.save_npz(smoothing_matrix_file_path, smoothing_matrix)
            return smoothing_matrix
