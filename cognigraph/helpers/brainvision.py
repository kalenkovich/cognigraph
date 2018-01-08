import mne
import mne.io.brainvision.brainvision as brainvision

# Start of accommodating spelling mistakes
brainvision._check_mrk_version_original = brainvision._check_mrk_version
brainvision._check_hdr_version_original = brainvision._check_hdr_version

def _check_mrk_version(header):
    bobe_mrk_header = 'BrainVision Data Exchange Marker File, Version 1.0'
    if header == bobe_mrk_header:
        return True
    else:
        return brainvision._check_mrk_version_original(header)


def _check_hdr_version(header):
    bobe_hdr_header = 'BrainVision Data Exchange Header File Version 1.0'
    if header == bobe_hdr_header:
        return True
    else:
        return brainvision._check_hdr_version_original(header)

brainvision._check_hdr_version = _check_hdr_version
brainvision._check_mrk_version = _check_mrk_version
# End of accommodating spelling mistakes


def read_brain_vision_data(vhdr_file_path, time_axis):
    raw = mne.io.read_raw_brainvision(vhdr_fname=vhdr_file_path, verbose='ERROR')
    data = raw.get_data()
    mne_info = raw.info.copy()
    return data, mne_info
