# from hp_selection.metric_utils import delta_precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import recall_score, precision_score
# from scipy.stats import wasserstein_distance

import mne
# from mne.viz import plot_sparse_source_estimates

from hp_selection.sure import solve_using_sure
from hp_selection.spatial_cv import solve_using_spatial_cv
from hp_selection.lambda_map import solve_using_lambda_map
from hp_selection.temporal_cv import solve_using_temporal_cv
from hp_selection.utils import apply_solver

from hp_selection.utils import load_data

CONDITION = "auditory/left"
# AMPLITUDE_RANGE = [(30 + 25 * i, 30 + 25 * j) for i, j in
#                    zip(range(0, 10), range(0, 10))]

AMPLITUDE_RANGE = [(120, 120)]
# SOLVERS = ["temporal_cv", "spatial_cv", "lambda_map", "sure"]
SOLVERS = ["spatial_cv"]

def delta_f1_score(stc, true_stc, forward, subject, labels, extent,
                   subjects_dir):
    vertices = []
    vertices.append(forward["src"][0]["vertno"])
    vertices.append(forward["src"][1]["vertno"])

    # Reverse the effect of make_sparse_stc
    # stc now contains a sparse matrix with a lot of zero rows and few non-zero
    # rows containing the activations

    stc.expand(vertices)
    true_stc.expand(vertices)

    estimated_activations = np.abs(stc.data).sum(axis=-1)
    true_activations = np.abs(true_stc.data).sum(axis=-1)

    estimated_as = estimated_activations != 0
    true_as = true_activations != 0

    map_hemis = {"lh": 0, "rh": 1}

    # XXX: We assume one source per hemisphere
    ext_vertices_lh = mne.label.grow_labels(
        subject, labels[0].vertices, extent, map_hemis[labels[0].hemi],
        subjects_dir=subjects_dir)[0].vertices
    ext_vertices_rh = mne.label.grow_labels(
        subject, labels[1].vertices, extent, map_hemis[labels[1].hemi],
        subjects_dir=subjects_dir)[0].vertices

    ext_vertices_lh = np.intersect1d(ext_vertices_lh,
                                     forward["src"][0]["vertno"])
    ext_vertices_rh = np.intersect1d(ext_vertices_rh,
                                     forward["src"][1]["vertno"])

    ext_indices_lh = np.searchsorted(forward["src"][0]["vertno"],
                                     ext_vertices_lh)
    ext_indices_rh = np.searchsorted(forward["src"][1]["vertno"],
                                     ext_vertices_rh)

    ext_pred_lh = np.zeros(len(forward["src"][0]["vertno"]), dtype=int)
    ext_pred_lh[ext_indices_lh] = 1
    ext_pred_rh = np.zeros(len(forward["src"][1]["vertno"]), dtype=int)
    ext_pred_rh[ext_indices_rh] = 1

    extended_true_as = np.concatenate((ext_pred_lh, ext_pred_rh))

    delta_precision = precision_score(extended_true_as, estimated_as,
                                      zero_division=0)
    recall = recall_score(true_as, estimated_as, zero_division=0)

    if delta_precision == 0 and recall == 0:
        return 0, delta_precision, recall
    f1 = 2 * (delta_precision * recall) / (delta_precision + recall)

    return f1, delta_precision, recall


if __name__ == "__main__":
    simulated = True
    maxfilter = True

    precis = 1e-3
    n_points = int(1 / precis) + 1
    recall = np.linspace(1.0, 0.0, n_points)
    dict_precision = {}
    dict_delta_precision = {}

    extent = 20

    subject = "sample"
    subjects_dir = mne.datasets.sample.data_path() + '/subjects'

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)

    for solver in SOLVERS:
        f1_scores = list()
        delta_precision_scores = list()
        recall_scores = list()

        for amplitude in AMPLITUDE_RANGE:
            evoked, forward, noise_cov, true_stc, labels = load_data(
                CONDITION, maxfilter=maxfilter, simulated=simulated,
                amplitude=amplitude, return_stc=True, return_labels=True, 
                resolution=6)

            if solver == "sure":
                stc = solve_using_sure(evoked, forward, noise_cov, depth=0.99)
            elif solver == "lambda_map":
                stc = apply_solver(solve_using_lambda_map, evoked, forward,
                                   noise_cov, depth=0.99)
            elif solver == "spatial_cv":
                stc = apply_solver(solve_using_spatial_cv, evoked, forward,
                                   noise_cov, depth=0.99)
            elif solver == "temporal_cv":
                stc = apply_solver(solve_using_temporal_cv, evoked, forward,
                                   noise_cov, depth=0.99)
            else:
                raise ValueError("Unknown solver!")

            f1, delta_precision, recall = delta_f1_score(stc, true_stc, forward,
                subject, labels, extent, subjects_dir)

            delta_precision_scores.append(delta_precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        amplitudes = [x[0] for x in AMPLITUDE_RANGE]
        ax1.plot(amplitudes, delta_precision_scores, label=solver)
        ax2.plot(amplitudes, recall_scores, label=solver)
        ax3.plot(amplitudes, f1_scores, label=solver)

    plt.suptitle(r"Simulated - Source amplitude vs $\delta$-F1 score")
    ax3.set_xlabel("Source amplitude (nAm)")
    ax1.set_ylabel(r"$\delta$-precision")
    ax2.set_ylabel("recall")
    ax3.set_ylabel(r"$\delta$-F1")

    plt.legend()
    plt.show()
