import os


import mne
data_path = mne.datasets.sample.data_path()
file_path = data_path + '/MEG/sample/sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(file_path, verbose='ERROR')
start, stop = raw.time_as_index([0, 60])



fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
inverse_operator = mne.minimum_norm.read_inverse_operator(fname_inv)


lambda2 = 1/1.0**2
method = 'MNE'

start, stop = raw.time_as_index([0, 15])

raw.set_eeg_reference()

stc = mne.minimum_norm.apply_inverse_raw(
    raw, inverse_operator, lambda2, method,
    start=start, stop=stop, pick_ori='normal')


subject = 'sample'
data_path = mne.datasets.sample.data_path()
subjects_dir = os.path.join(data_path, 'subjects')

stc.plot(subjects_dir=subjects_dir)



fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
evoked = mne.read_evokeds(fname_evoked, condition=0, baseline=(None, 0))

stc_evoked = mne.minimum_norm.apply_inverse(evoked=evoked, inverse_operator=inverse_operator, lambda2=lambda2, method=method,
                                     pick_ori='normal')  # type: mne.SourceEstimate
stc_evoked.plot(subjects_dir=subjects_dir, hemi='both', time_viewer=True)