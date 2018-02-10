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

# Check alignment
trans_file_path = os.path.join(data_path, 'MEG', 'sample', 'sample_1005-eeg-oct-6-fwd-trans.fif')
mri_head_t = fwd['mri_head_t']
trans = {'from': mri_head_t['to'], 'to': mri_head_t['from'], 'trans': np.linalg.inv(mri_head_t['trans'])}
mne.transforms.write_trans(trans_file_path, trans)

source_space = fwd['src']

mne.viz.plot_alignment(info=fwd_info, trans=trans_file_path, subject=subject,
                       # surfaces=['outer_skin', 'pial'],  # can't see the source space
                       eeg='projected',
                       meg=False, src=source_space, subjects_dir=subjects_dir)

# Load file, find bad channels, interpolate, check that alpha peaks around 10 Hz, filter in
file_path = r"D:\Cognigraph\eyes\Koleno.vhdr"
raw = mne.io.brainvision.read_raw_brainvision(file_path, preload=True)  # type: mne.io.Raw
bad_indices = find_outliers(np.std(raw.get_data(), axis=1), max_iter=3)
bads = [raw.info['chs'][idx]['ch_name'] for idx in bad_indices]
raw.info['bads'] = bads
fill_eeg_channel_locations(raw.info)
raw.interpolate_bads(mode='fast')
raw.plot_psd()
raw.filter(l_freq=8, h_freq=12)
raw.set_eeg_reference(projection=True)

# Construct inverse, apply
info = raw.info
G = fwd['sol']['data']
q = np.trace(G.dot(G.T)) / G.shape[0]

picks = mne.pick_types(info, eeg=True, meg=False)
q *= np.mean(np.var(raw.get_data(picks=picks)))

cov = mne.Covariance(data=(q * np.identity(info['nchan'])),  # "whitened" G.dot(G.T) and cov now have one scale
                     names=info['ch_names'],
                     bads=info['bads'],
                     projs=info['projs'],
                     nfree=1)

inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, depth=0.8, verbose='ERROR')
start, stop = raw.time_as_index((80, 100))
stc = mne.minimum_norm.apply_inverse_raw(raw, start=start, stop=stop,
                                         inverse_operator=inv, lambda2=0.1, method='MNE')
brain = stc.plot(subject=subject, subjects_dir=subjects_dir,
                 time_viewer=True, initial_time=82.0,
                 hemi='both', clim=dict(kind='percent', lims=(95, 97.5, 100)))
