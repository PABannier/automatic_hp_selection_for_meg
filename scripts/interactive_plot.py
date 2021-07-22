import argparse
import joblib
import os
import mne
from mne.datasets import sample
from mne.viz import plot_sparse_source_estimates


parser = argparse.ArgumentParser()
parser.add_argument(
    "--criterion",
    help="choice of criterion to reconstruct the channels. available: "
    + "spatial_cv, temporal_cv, lambda_map, sure",
)
parser.add_argument(
    "--condition",
    help="choice of condition. available: "
    + "Left Auditory, Right Auditory, Left visual, Right visual",
)

args = parser.parse_args()
CRITERION, CONDITION = args.criterion, args.condition

DATA_DIR = "../data/"


def load_data():
    data_path = sample.data_path()
    fwd_fname = data_path + "/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif"
    ave_fname = data_path + "/MEG/sample/sample_audvis-ave.fif"
    cov_fname = data_path + "/MEG/sample/sample_audvis-shrunk-cov.fif"

    noise_cov = mne.read_cov(cov_fname)
    evoked = mne.read_evokeds(ave_fname, condition=CONDITION,
                              baseline=(None, 0))
    evoked.crop(tmin=0.04, tmax=0.18)
    evoked = evoked.pick_types(eeg=False, meg=True)

    forward = mne.read_forward_solution(fwd_fname)

    return evoked, forward, noise_cov


if __name__ == "__main__":
    looose, depth = 0.9, 0.9
    evoked, forward, noise_cov = load_data()

    fname = CONDITION.lower().replace(" ", "_") + ".pkl"
    fpath = os.path.join(DATA_DIR, CRITERION, fname)

    stc = joblib.load(fpath)
    plot_sparse_source_estimates(forward["src"], stc, bgcolor=(1, 1, 1),
                                 opacity=0.1)
