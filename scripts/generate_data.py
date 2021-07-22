import os
import joblib
import mne
from mne.datasets import sample

from hp_selection.sure import solve_using_sure
from hp_selection.spatial_cv import solve_using_spatial_cv
from hp_selection.temporal_cv import solve_using_temporal_cv
from hp_selection.utils import apply_solver


CONDITIONS = ["Left Auditory", "Right Auditory", "Left visual",
              "Right visual"]


def load_data(condition):
    data_path = sample.data_path()
    fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
    ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
    cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'

    noise_cov = mne.read_cov(cov_fname)
    evoked = mne.read_evokeds(ave_fname, condition=condition,
                              baseline=(None, 0))
    evoked.crop(tmin=0.05, tmax=0.15)

    evoked = evoked.pick_types(eeg=False, meg=True)
    forward = mne.read_forward_solution(fwd_fname)
    return evoked, forward, noise_cov


def save_stc(stc, condition, solver):
    out_dir = "../data/%s/" % solver

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    fname = condition.lower().replace(" ", "_") + ".pkl"
    out_path = os.path.join(out_dir, fname)

    with open(out_path, "wb") as out_file:
        joblib.dump(stc, out_file)


if __name__ == "__main__":
    for condition in CONDITIONS:
        evoked, forward, noise_cov = load_data(condition)

        # SURE
        stc = solve_using_sure(evoked, forward, noise_cov)
        save_stc(stc, condition, "sure")

        # Spatial CV
        # stc = apply_solver(solve_using_spatial_cv, evoked, forward, noise_cov)
        # save_stc(stc, condition, "spatial_cv")

        # Temporal CV
        # stc = apply_solver(solve_using_temporal_cv, evoked, forward, noise_cov)
        # save_stc(stc, condition, "temporal_cv")
