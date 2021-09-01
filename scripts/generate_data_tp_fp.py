import joblib

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import multilabel_confusion_matrix

import mne

from hp_selection.sure import solve_using_sure
from hp_selection.spatial_cv import solve_using_spatial_cv
from hp_selection.lambda_map import solve_using_lambda_map
from hp_selection.temporal_cv import solve_using_temporal_cv
from hp_selection.utils import apply_solver

from hp_selection.utils import load_data

CONDITION = "auditory/left"
# AMPLITUDES = [(30, 30), (120, 120), (180, 180)]
AMPLITUDES = [(180, 180)]
# SOLVERS = ["temporal_cv", "spatial_cv", "lambda_map", "sure"]
SOLVERS = ["temporal_cv"]

DEPTH = 0.99

if __name__ == "__main__":
    simulated = True
    maxfilter = True

    extent = 10

    RESULTS = []  # List of dictionaries

    subject = "sample"
    subjects_dir = mne.datasets.sample.data_path() + "/subjects"

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    for i, amplitude in enumerate(AMPLITUDES):
        result_tmp = {}
        for j, solver in enumerate(SOLVERS):
            evoked, forward, noise_cov, true_stc, labels = load_data(
                CONDITION, maxfilter=maxfilter, simulated=simulated,
                amplitude=amplitude, return_stc=True, return_labels=True
            )

            if solver == "sure":
                stc = solve_using_sure(evoked, forward, noise_cov, depth=DEPTH)
            elif solver == "lambda_map":
                stc = apply_solver(solve_using_lambda_map, evoked, forward,
                                   noise_cov, depth=DEPTH)
            elif solver == "spatial_cv":
                stc = apply_solver(solve_using_spatial_cv, evoked, forward,
                                   noise_cov, depth=DEPTH)
            elif solver == "temporal_cv":
                stc = apply_solver(solve_using_temporal_cv, evoked, forward,
                                   noise_cov, depth=DEPTH)
            else:
                raise ValueError("Unknown solver!")

            vertices = []
            vertices.append(forward["src"][0]["vertno"])
            vertices.append(forward["src"][1]["vertno"])

            stc.expand(vertices)
            true_stc.expand(vertices)

            estimated_active_set = np.abs(stc.data).sum(axis=-1)
            true_active_set = np.abs(true_stc.data).sum(axis=-1)

            support_bin = true_active_set != 0
            estimated_support_bin = estimated_active_set != 0

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

            # MCM = multilabel_confusion_matrix(support_ext_bin,
            #                                   estimated_support_bin)

            MCM = multilabel_confusion_matrix(support_bin, estimated_support_bin)
            tp_sum = MCM[:, 1, 1].sum()
            fp_sum = MCM[:, 0, 1].sum()

            result_tmp[solver] = (fp_sum, tp_sum)

        RESULTS.append(result_tmp.copy())

    with open("../data/tp_fp_results.pkl", "wb") as outfile:
        joblib.dump(RESULTS, outfile)
