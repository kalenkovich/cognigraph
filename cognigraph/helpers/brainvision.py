import mne
import mne.io.brainvision.brainvision as brainvision

# Start of accommodating spelling mistakes
brainvision._check_mrk_version_original = brainvision._check_mrk_version
brainvision._check_hdr_version_original = brainvision._check_hdr_version

BRAINVISION_TIME_AXIS = 1

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


def read_brain_vision_data(vhdr_file_path, time_axis, start_s: int=0, stop_s: int=None):
    raw = mne.io.read_raw_brainvision(vhdr_fname=vhdr_file_path, verbose='ERROR')  # type: mne.io.Raw

    # Get the required time slice. mne.io.Raw.get_data takes array indices, not time
    start = 0 if start_s is None else raw.time_as_index(start_s)[0]
    stop = min(raw.n_times if stop_s is None else raw.time_as_index(stop_s), raw.n_times)
    data = raw.get_data(start=start, stop=stop)

    mne_info = raw.info.copy()

    if time_axis == BRAINVISION_TIME_AXIS:
        data = data.T

    return data, mne_info