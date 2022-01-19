import argparse
import joblib
import os.path as op

from numba import njit
import matplotlib.pyplot as plt

import mne

parser = argparse.ArgumentParser()
parser.add_argument(
    "--criterion",
    help="choice of criterion to reconstruct the channels. available: "
    + "sure, spatial_cv, temporal_cv, lambda_map",
)
parser.add_argument(
    "--condition",
    help="choice of condition. available: "
    + "auditory/left, auditory/right, visual/left, visual/right",
)

args = parser.parse_args()

CRITERION = args.criterion
CONDITION = args.condition
HEMISPHERES = ["lh", "rh"]


def add_foci_to_brain_surface(brain, stc, ax):
    for i_hemi, hemi in enumerate(["lh", "rh"]):
        surface_coords = brain.geo[hemi].coords
        hemi_data = stc.lh_data if hemi == "lh" else stc.rh_data
        for k in range(len(stc.vertices[i_hemi])):
            activation_idx = stc.vertices[i_hemi][k]
            foci_coords = surface_coords[activation_idx]

            # In milliseconds
            (line,) = ax.plot(stc.times * 1e3, 1e9 * hemi_data[k])
            brain.add_foci(foci_coords, hemi=hemi, color=line.get_color())

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (nAm)")


@njit
def add_margin(nonwhite_col, margin=5):
    margin_nonwhite_col = nonwhite_col.copy()
    for i in range(len(nonwhite_col)):
        if nonwhite_col[i] and not nonwhite_col[i - 1]:
            margin_nonwhite_col[i - (1 + margin): i - 1] = True
        elif not nonwhite_col[i] and nonwhite_col[i - 1]:
            margin_nonwhite_col[i: i + margin] = True
    return margin_nonwhite_col


if __name__ == "__main__":
    mne.viz.set_3d_backend('pyvistaqt')
    data_path = mne.datasets.sample.data_path()
    subjects_dir = op.join(data_path, "subjects")
    fname_evoked = op.join(data_path, "MEG", "sample", "sample_audvis-ave.fif")
    fname_fwd = data_path + "/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif"

    evoked = mne.read_evokeds(fname_evoked, CONDITION)
    evoked.pick_types(meg="grad").apply_baseline((None, 0.0))
    max_t = evoked.get_peak()[1]

    forward = mne.read_forward_solution(fname_fwd)

    fname = CONDITION.lower().replace(" ", "_") + ".pkl"
    fpath = op.join("../data", CRITERION, fname)
    stc = joblib.load(fpath)

    lower_bound = round(stc.data.min() * 1e9)
    upper_bound = round(stc.data.max() * 1e9)

    colormap = "inferno"
    clim = dict(kind="value", lims=(lower_bound, 0, upper_bound))

    # Plot the STC, get the brain image, crop it:
    brain = stc.plot(
        views=["med"],
        hemi="split" if len(HEMISPHERES) > 1 else HEMISPHERES[0],
        size=(500, 250),
        subjects_dir=subjects_dir,
        background="w",
        clim="auto",
        colorbar=False,
        colormap=colormap,
        time_viewer=False,
        show_traces=False,
        cortex="classic",
        volume_options=dict(resolution=1),
    )

    t = 0.05
    brain.set_time(t)

    fig = plt.figure(figsize=(4.5, 4.5))
    axes = [
        plt.subplot2grid((7, 1), (0, 0), rowspan=4),
        plt.subplot2grid((7, 1), (4, 0), rowspan=3),
    ]

    add_foci_to_brain_surface(brain, stc, axes[1])

    out_fname = CONDITION.lower().replace(" ", "_") + "_" + CRITERION + "_brain.pdf"
    fig_dir = op.join("../figures", out_fname)

    brain.save_image(fig_dir)
