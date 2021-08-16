import argparse, os, joblib
from mne.viz import plot_sparse_source_estimates
from hp_selection.utils import load_somato_data, load_data


parser = argparse.ArgumentParser()
parser.add_argument(
    "--criterion",
    help="choice of criterion to reconstruct the channels. available: "
    + "spatial_cv, temporal_cv, lambda_map, sure",
)
parser.add_argument(
    "--condition",
    help="choice of condition. available: "
    + "Left Auditory, Right Auditory, Left visual, Right visual, somato",
)

args = parser.parse_args()
CRITERION, CONDITION = args.criterion, args.condition

DATA_DIR = "../data/"


if __name__ == "__main__":
    looose, depth = 0.9, 0.9
    if CONDITION == "somato":
        evoked, forward, noise_cov = load_somato_data()
    else:
        evoked, forward, noise_cov = load_data(CONDITION, maxfilter=False)

    fname = CONDITION.lower().replace(" ", "_") + ".pkl"
    fpath = os.path.join(DATA_DIR, CRITERION, fname)

    stc = joblib.load(fpath)
    plot_sparse_source_estimates(forward["src"], stc, bgcolor=(1, 1, 1),
                                 opacity=0.1)
