import os

import numpy as np
from mayavi import mlab
import mne
from mne.preprocessing.bads import find_outliers

from cognigraph.helpers.channels import fill_eeg_channel_locations

subject = 'sample'
data_path = mne.datasets.sample.data_path()
subjects_dir = os.path.join(data_path, 'subjects')

fwd_file_path = os.path.join(data_path, 'MEG', 'sample', 'sample_1005-eeg-oct-6-fwd.fif')
fwd = mne.read_forward_solution(fwd_file_path)


fwd_info = fwd['info']

# Load file, find bad channels, interpolate, check that alpha peaks around 10 Hz, filter in
file_path = r"D:\Cognigraph\eyes\Koleno.vhdr"
raw = mne.io.brainvision.read_raw_brainvision(file_path, preload=True)  # type: mne.io.Raw
bad_indices = find_outliers(np.std(raw.get_data(), axis=1), max_iter=3)
bads = [raw.info['chs'][idx]['ch_name'] for idx in bad_indices]
raw.info['bads'] = bads
fill_eeg_channel_locations(raw.info)
raw.interpolate_bads(mode='fast')
# raw.plot_psd()
raw.filter(l_freq=8, h_freq=12)
raw.set_eeg_reference(projection=True)


# Data covariance - GG'
data_cov = mne.compute_raw_covariance(raw)
channel_indices = mne.pick_channels(fwd['info']['ch_names'], data_cov['names'])
G = fwd['sol']['data'][channel_indices]
data_cov['data'] = G.dot(G.T)
reg=0.05

start, stop = raw.time_as_index((80, 100))

info = raw.info
weight_norm = None
filters = mne.beamformer.make_lcmv(info=info, forward=fwd, data_cov=data_cov,
                                   reg=reg, noise_cov=None,
                                   weight_norm=weight_norm)

stc = mne.beamformer.lcmv_raw(raw, fwd, None, data_cov=data_cov, start=start,
                              stop=stop, weight_norm='unit-noise-gain')

brain = stc.plot(subject=subject, subjects_dir=subjects_dir,
                 time_viewer=True, initial_time=87.654,
                 hemi='both', clim=dict(kind='percent', lims=(95, 97.5, 100)))


#
lh = fwd['src'][0]
lh_dipole_coordinates = lh['rr'][lh['vertno']]
most_posterior_idx = np.argmin(lh_dipole_coordinates[:, 1])

G_vertex = G[:, most_posterior_idx * 3:][:, :3]
normal = lh['nn'][most_posterior_idx]
raw.data[channel_indices, -1] = (
    (G_vertex.dot(normal))[:, np.newaxis]
)