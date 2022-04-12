import argparse, os, joblib
import mne
from mne.viz import plot_sparse_source_estimates
from mne.datasets import somato, sample
from calibromatic.utils import load_somato_data, load_data


parser = argparse.ArgumentParser()
parser.add_argument(
    "--criterion",
    help="choice of criterion to reconstruct the channels. available: "
    + "spatial_cv, temporal_cv, lambda_map, sure",
)
parser.add_argument(
    "--condition",
    help="choice of condition. available: "
    + "auditory/left, auditory/right, visual/left, visual/right, somato",
)
parser.add_argument(
    "--simu",
    help="use simulated data",
    action='store_true',
)
parser.add_argument(
    "--mf",
    help="use maxfiltered data",
    action='store_true',
)

args = parser.parse_args()
CRITERION, CONDITION = args.criterion, args.condition
simulated, maxfilter = args.simu, args.mf

DATA_DIR = "../data/"


if __name__ == "__main__":
    looose, depth = 0.9, 0.9
    if CONDITION == "somato":
        data_path = somato.data_path()
        subject = "01"
        task = "somato"
        fwd_fname = os.path.join(data_path, "derivatives",
                                "sub-{}".format(subject),
                                "sub-{}_task-{}-fwd.fif".format(subject, task))
        forward = mne.read_forward_solution(fwd_fname)
    else:
        data_path = sample.data_path()
        fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
        forward = mne.read_forward_solution(fwd_fname)

    fname = CONDITION.lower().replace("/", "_")
    if simulated:
        fname += "_simu"
    if maxfilter:
        fname += "_mf"
    fpath = os.path.join(DATA_DIR, CRITERION, fname + ".pkl")

    stc = joblib.load(fpath)
    plot_sparse_source_estimates(forward["src"], stc, bgcolor=(1, 1, 1),
                                 opacity=0.1)
