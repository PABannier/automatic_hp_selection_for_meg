from inspect import Attribute
from itertools import product
import time

from numpy.core.function_base import linspace
from joblib import parallel_backend
from joblib import Parallel, delayed
import joblib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import recall_score, precision_score
from scipy.stats import wasserstein_distance

import mne

from calibromatic.sure import solve_using_sure
from calibromatic.spatial_cv import solve_using_spatial_cv
from calibromatic.lambda_map import solve_using_lambda_map
from calibromatic.temporal_cv import solve_using_temporal_cv
from calibromatic.utils import apply_solver, load_data

N_JOBS = 4
INNER_MAX_NUM_THREADS = 1

CONDITION = "auditory/left"
RESOLUTION = 6

EXTENT = 7
AMPLITUDE_RANGE = [(i, i) for i in np.linspace(100, 700, num=5)]
# AMPLITUDE_RANGE = [(i*10, i*10) for i in range(1, 11)]

MAXFILTER = False
SIMULATED = True
SOLVERS = ["spatial_cv", "sure", "lambda_map"]
# SOLVERS = ["sure", "spatial_cv", "temporal_cv", "lambda_map"]

def delta_f1_score(stc, true_stc, forward, subject, labels, extent,
                   subjects_dir):
    vertices = []
    vertices.append(forward["src"][0]["vertno"])
    vertices.append(forward["src"][1]["vertno"])

    # Reverse the effect of make_sparse_stc
    # stc now contains a sparse matrix with a lot of zero rows and few non-zero
    # rows containing the activations

    if type(stc) == tuple:
        stc = stc[0]

    stc.expand(vertices)
    true_stc.expand(vertices)

    est_activations = np.abs(stc.data).sum(axis=-1)
    true_activations = np.abs(true_stc.data).sum(axis=-1)

    estimated_as = est_activations != 0
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

    emd = wasserstein_distance(est_activations, true_activations)

    if delta_precision == 0 and recall == 0:
        return 0, delta_precision, recall, emd
    f1 = 2 * (delta_precision * recall) / (delta_precision + recall)
    # except AttributeError:
    #     print("=" * 20)
    #     print("ERROR")
    #     print("=" * 20)
    #     f1 = "err"
    #     delta_precision = "err"
    #     recall = "err"
    #     emd = "err"

    return f1, delta_precision, recall, emd


def solve_condition_for_amplitude(solver, amplitude, subject, subjects_dir):
    out = load_data(CONDITION, maxfilter=MAXFILTER, simulated=SIMULATED,
                    amplitude=amplitude, return_stc=True,
                    return_labels=True, resolution=RESOLUTION)
    if SIMULATED:
        evoked, forward, noise_cov, true_stc, labels = out
    else:
        evoked, forward, noise_cov = out

    delta_f1, delta_precision, recall, emd = np.nan, np.nan, np.nan, np.nan

    start_time = time.time()
    if solver == "sure":
        stc = solve_using_sure(evoked, forward, noise_cov, depth=0.9)
    elif solver == "lambda_map":
        stc = apply_solver(solve_using_lambda_map, evoked, forward,
                            noise_cov, depth=0.9)
    elif solver == "spatial_cv":
        stc = apply_solver(solve_using_spatial_cv, evoked, forward,
                            noise_cov, depth=0.9)
    elif solver == "temporal_cv":
        stc = apply_solver(solve_using_temporal_cv, evoked, forward,
                            noise_cov, depth=0.9)
    else:
        raise ValueError("Unknown solver!")

    duration = time.time() - start_time

    if SIMULATED:
        delta_f1, delta_precision, recall, emd = delta_f1_score(stc, true_stc,
            forward, subject, labels, EXTENT, subjects_dir)

    return {
        "solver": solver,
        "amplitude": amplitude[0],
        "duration": duration,
        "delta_f1_score": delta_f1,
        "delta_precision": delta_precision,
        "recall": recall,
        "emd": emd
    }


if __name__ == "__main__":
    subject = "sample"
    subjects_dir = mne.datasets.sample.data_path() + '/subjects'

    with parallel_backend("loky", inner_max_num_threads=INNER_MAX_NUM_THREADS):
        experiment_results = Parallel(N_JOBS)(
            delayed(solve_condition_for_amplitude)(solver, amplitude, subject,
            subjects_dir) for solver, amplitude in product(SOLVERS,
            AMPLITUDE_RANGE)
        )

    with open("experiment_results_lower_extent.pkl", "wb") as outfile:
        joblib.dump(experiment_results, outfile)
