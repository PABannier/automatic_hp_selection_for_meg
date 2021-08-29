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
amplitude_range = [(30 + 25 * i, 30 + 25 * j) for i, j in
                   zip(range(0, 10), range(0, 10))]

if __name__ == "__main__":
    simulated = True
    maxfilter = True

    precis = 1e-3
    n_points = int(1 / precis) + 1
    recall = np.linspace(1.0, 0.0, n_points)
    dict_precision = {}
    dict_delta_precision = {}

    extent = 10

    subject = "sample"
    subjects_dir = mne.datasets.sample.data_path() + '/subjects'

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)

    for solver in ["temporal_cv", "spatial_cv", "lambda_map", "sure"]:
        f1_scores = list()
        delta_precision_scores = list()
        recall_scores = list()

        for amplitude in amplitude_range:
            evoked, forward, noise_cov, true_stc, labels = load_data(
                CONDITION, maxfilter=maxfilter, simulated=simulated,
                amplitude=amplitude, return_stc=True, return_labels=True)

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

            vertices = []
            vertices.append(forward["src"][0]["vertno"])
            vertices.append(forward["src"][1]["vertno"])

            stc.expand(vertices)
            true_stc.expand(vertices)

            estimated_active_set = np.abs(stc.data).sum(axis=-1)
            true_active_set = np.abs(true_stc.data).sum(axis=-1)

            # estimated_active_set /= estimated_active_set.sum()
            # true_active_set /= true_active_set.sum()

            # Compute delta-precision
            support_bin = true_active_set != 0
            normalized_score = estimated_active_set

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

            support_ext_bin = np.concatenate((ext_pred_lh, ext_pred_rh))
            # support_ext_bin = support_ext_bin.astype(int)

            delta_precision = precision_score(support_ext_bin,
                                              normalized_score != 0,
                                              zero_division=1)
            recall = recall_score(support_bin, normalized_score != 0)

            f1 = 2 * (delta_precision * recall) / (delta_precision + recall)
            delta_precision_scores.append(delta_precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        amplitudes = [x[0] for x in amplitude_range]
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
