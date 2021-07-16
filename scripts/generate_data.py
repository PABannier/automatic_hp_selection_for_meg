import os
import joblib
import mne
from mne.datasets import sample


CONDITIONS = ["Left Auditory", "Right Auditory", "Left visual",
                "Right visual"]
OUT_DIR = "../data/sure"


def load_data():
    data_path = sample.data_path()
    fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
    ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
    cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'

    noise_cov = mne.read_cov(cov_fname)
    evoked = mne.read_evokeds(ave_fname, condition=condition,
                              baseline=(None, 0))
    evoked.crop(tmin=0.04, tmax=0.18)

    evoked = evoked.pick_types(eeg=False, meg=True)
    forward = mne.read_forward_solution(fwd_fname)
    return evoked, forward, noise_cov


if __name__ == "__main__":

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    for condition in CONDITIONS:
        raise NotImplementedError
        # stc = solve_sure(condition)
        # condition_fname = condition.lower().replace(" ", "_") + "_sure.pkl"
        # out_path = os.path.join(OUT_DIR, condition_fname)
        # with open(out_path, "wb") as out_file:
        #     joblib.dump(stc, out_file)
