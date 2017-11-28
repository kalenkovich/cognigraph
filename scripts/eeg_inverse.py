# This script requires mayavi and thus python 2.7

from mayavi import mlab
%gui qt

import os

import numpy as np
import mne
from mne.datasets import sample

subject = 'sample'
data_path = sample.data_path()
subjects_dir = os.path.join(data_path, 'subjects')

source_space = mne.setup_source_space(subject, subjects_dir=subjects_dir,
                                      spacing='oct6')

montage_1005 = mne.channels.read_montage(kind='standard_1005')
fake_info = mne.create_info(ch_names=montage_1005.ch_names, sfreq=1000, ch_types='eeg', montage=montage_1005)


# fake_raw = mne.io.RawArray(np.zeros((len(fake_info['ch_names']), 1)), fake_info)
# fake_raw_file_path = r'C:\Users\evgenii\Downloads\fake_raw.fif'
# fake_raw.save(fake_raw_file_path, overwrite=True)

# In the coregistration GUI save the tranformation to the file path below
# mne.gui.coregistration(inst=fake_raw_file_path, subject=subject, subjects_dir=subjects_dir)

trans_file_path = r"C:\Users\evgenii\Downloads\sample-trans.fif"
mne.viz.plot_alignment(info=fake_info, trans=trans_file_path, subject=subject, dig=True,
                       meg=False, src=source_space,subjects_dir=subjects_dir)

# bem
conductivity = (0.3, 0.006, 0.3)
model = mne.make_bem_model(subject='sample', ico=4, conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)


fwd = mne.make_forward_solution(info=fake_info, trans=trans_file_path, src=source_space, bem=bem, eeg=True,
                                mindist=5.0, n_jobs=2)  # the last 2 copied from one of the mne-python's examples

fwd_file_path = os.path.join(data_path, 'MEG', 'sample', 'sample_1005-eeg-oct-6-fwd.fif')
mne.write_forward_solution(fwd_file_path, fwd)
fwd = mne.read_forward_solution(fwd_file_path)

cov = mne.Covariance(data=np.identity(fake_info['nchan']),
                     names=fake_info['ch_names'],
                     bads=fake_info['bads'],
                     projs=fake_info['projs'],
                     nfree=1)

inv = mne.minimum_norm.make_inverse_operator(fake_info, fwd, cov)
inv_file_path = os.path.join(data_path, 'MEG', 'sample', 'sample_1005-eeg-oct-6-eeg-inv.fif')
mne.minimum_norm.write_inverse_operator(inv_file_path, inv=inv)

