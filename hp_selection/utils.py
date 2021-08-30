import functools
import os
import joblib
from pathlib import Path

import numpy as np

import mne
from mne.datasets import sample, somato
from mne.inverse_sparse.mxne_inverse import (_prepare_gain, is_fixed_orient,
                                             _reapply_source_weighting,
                                             _make_sparse_stc)
from mne.inverse_sparse.mxne_optim import iterative_mixed_norm_solver
from mne.preprocessing import find_bad_channels_maxwell


def groups_norm2(A, n_orient=1):
    """Compute squared L2 norms of groups inplace."""
    n_positions = A.shape[0] // n_orient
    return np.sum(np.power(A, 2, A).reshape(n_positions, -1), axis=1)


def sum_squared(X):
    X_flat = X.ravel(order="F" if np.isfortran(X) else "C")
    return np.dot(X_flat, X_flat)


def norm_l2_05(X, n_orient, copy=True):
    """Compute l_2,p norm"""
    if X.size == 0:
        return 0.0
    if copy:
        X = X.copy()
    return np.sum(np.sqrt(np.sqrt(groups_norm2(X, n_orient))))


def norm_l2_inf(X, n_orient, copy=True):
    """Compute l_2,inf norm"""
    if X.size == 0:
        return 0.0
    if copy:
        X = X.copy()
    return np.sqrt(np.max(groups_norm2(X, n_orient)))


def compute_alpha_max(G, M, n_orient):
    """Compute alpha max"""
    return norm_l2_inf(np.dot(G.T, M), n_orient, copy=True)


def apply_solver(solver, evoked, forward, noise_cov, depth=0.9, loose=0.9,
                 n_mxne_iter=5, random_state=0):
    """
    Preprocess M/EEG data and apply solver to the preprocess data.
    """
    all_ch_names = evoked.ch_names

    # Handle depth weighting and whitening (here is no weights)
    forward, gain, gain_info, whitener, source_weighting, _ = _prepare_gain(
        forward, evoked.info, noise_cov, pca=True, depth=depth,
        loose=loose, weights=None, weights_min=None, rank=None)  # info

    # Select channels of interest
    sel = [all_ch_names.index(name) for name in gain_info['ch_names']]
    M = evoked.data[sel]

    # Spatial whitening
    M = np.dot(whitener, M)
    n_orient = 1 if is_fixed_orient(forward) else 3

    X, active_set = solver(gain, M, n_orient, n_mxne_iter=n_mxne_iter,
                           random_state=random_state)
    X = _reapply_source_weighting(X, source_weighting, active_set)

    stc = _make_sparse_stc(X, active_set, forward, tmin=evoked.times[0],
                           tstep=1. / evoked.info['sfreq'])
    return stc


def build_full_coefficient_matrix(active_set, n_times, coef):
    """Building full coefficient matrix and filling active set with
    non-zero coefficients"""
    final_coef_ = np.zeros((len(active_set), n_times))
    if coef is not None:
        final_coef_[active_set] = coef
    return final_coef_


def solve_irmxne_problem(G, M, alpha, n_orient, n_mxne_iter=5, tol=1e-8):
    X, active_set, _ = iterative_mixed_norm_solver(M, G, alpha, n_mxne_iter,
                                                   n_orient=n_orient,
                                                   debias=False, tol=tol)
    return X, active_set


@functools.lru_cache(None)
def get_dgemm():
    from scipy import linalg

    return linalg.get_blas_funcs("gemm", (np.empty(0, np.float64),))


def norm_l2_1(X, n_orient, copy=True):
    if X.size == 0:
        return 0.0
    if copy:
        X = X.copy()
    return np.sum(np.sqrt(groups_norm2(X, n_orient)))


def primal_mtl(X, Y, coef, active_set, alpha, n_orient=1):
    """Primal objective function for multi-task
    LASSO
    """
    Y_hat = np.dot(X[:, active_set], coef)
    R = Y - Y_hat
    penalty = norm_l2_1(coef, n_orient, copy=True)
    nR2 = sum_squared(R)
    p_obj = 0.5 * nR2 + alpha * penalty
    return p_obj


def get_duality_gap_mtl(X, Y, coef, active_set, alpha, n_orient=1):
    Y_hat = np.dot(X[:, active_set], coef)
    R = Y - Y_hat
    penalty = norm_l2_1(coef, n_orient, copy=True)
    nR2 = sum_squared(R)
    p_obj = 0.5 * nR2 + alpha * penalty

    dual_norm = norm_l2_inf(np.dot(X.T, R), n_orient, copy=False)
    scaling = alpha / dual_norm
    scaling = min(scaling, 1.0)
    d_obj = (scaling - 0.5 * (scaling ** 2)) * nR2 + scaling * np.sum(
        R * Y_hat
    )
    gap = p_obj - d_obj
    return gap, p_obj, d_obj


def load_data(condition, maxfilter=True, simulated=False, amplitude=(200, 500),
              return_stc=False, return_labels=False, resolution=3):
    data_path = sample.data_path()

    if resolution == 6:
        fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
        forward = mne.read_forward_solution(fwd_fname)
    else:
        forward = compute_forward(data_path, None, resolution=resolution)

    labels = []

    # if not maxfilter and simulated:
    #     ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
    #     cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'

    #     noise_cov = mne.read_cov(cov_fname)
    #     evoked = mne.read_evokeds(ave_fname, condition=condition,
    #                               baseline=(None, 0))
    # else:
    raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

    # Standard sample event IDs. These values will correspond to the third column
    # in the events matrix.
    event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
                    'visual/right': 4, 'smiley': 5, 'button': 32}

    stc = None

    if simulated:
        info = mne.io.read_info(raw_fname)
        tstep = 1 / info['sfreq']

        src = forward['src']
        fname_event = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
        fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'
        subject = "sample"
        subjects_dir = data_path + '/subjects'
        events = mne.read_events(fname_event)
        noise_cov = mne.read_cov(fname_cov)

        events = events[:40] # 40

        activations = {
            'auditory/left':
                [('G_temp_sup-G_T_transv-lh', amplitude[0]),          # label, activation (nAm)
                ('G_temp_sup-G_T_transv-rh', amplitude[1])],
            'auditory/right':
                [('G_temp_sup-G_T_transv-lh', 50),
                ('G_temp_sup-G_T_transv-rh', 20)],
            'visual/left':
                [('S_calcarine-lh', 20),
                ('S_calcarine-rh', 50)],
            'visual/right':
                [('S_calcarine-lh', 30),
                ('S_calcarine-rh', 10)],
        }

        annot = 'aparc.a2009s'

        # Load the 4 necessary label names.
        region_names = list(activations.keys())

        def data_fun(times, latency, duration):
            """Function to generate source time courses for evoked responses,
            parametrized by latency and duration."""
            f = 15  # oscillating frequency, beta band [Hz]
            sigma = 0.5 * duration
            sinusoid = np.sin(2 * np.pi * f * (times - latency))
            gf = np.exp(- (times - latency - (sigma / 4.) * rng.rand(1)) ** 2 /
                        (2 * (sigma ** 2)))
            return 1e-9 * sinusoid * gf

        times = np.arange(150, dtype=np.float64) / info['sfreq']
        duration = 0.03
        rng = np.random.RandomState(7)
        source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep)

        for region_id, region_name in enumerate(region_names, 1):
            if region_name != condition:
                continue
            events_tmp = events[np.where(events[:, 2] == region_id)[0], :]
            for i in range(2):
                label_name = activations[region_name][i][0]
                label_tmp = mne.read_labels_from_annot(subject, annot,
                                                        subjects_dir=subjects_dir,
                                                        regexp=label_name,
                                                        verbose=False)[0]
                label_tmp.vertices = np.intersect1d(label_tmp.vertices,
                                                    src[i]["vertno"])
                label_tmp.values = np.ones(len(label_tmp.vertices))  # Hacky but works
                label_tmp = mne.label.select_sources(subject, label_tmp,
                                                     subjects_dir=subjects_dir)
                labels.append(label_tmp)
                amplitude_tmp = activations[region_name][i][1]
                if region_name.split('/')[1][0] == label_tmp.hemi[0]:
                    latency_tmp = 0.115
                else:
                    latency_tmp = 0.1
                wf_tmp = data_fun(times, latency_tmp, duration)
                source_simulator.add_data(label_tmp,
                                          amplitude_tmp * wf_tmp,
                                          events_tmp)

        raw = mne.simulation.simulate_raw(info, source_simulator, forward=forward)
        raw.set_eeg_reference(projection=True)
        mne.simulation.add_noise(raw, cov=noise_cov, random_state=0)

        stc = source_simulator.get_stc()
    else:
        raw = mne.io.read_raw_fif(raw_fname, verbose=False)

    if maxfilter:
        fine_cal_file = os.path.join(data_path, 'SSS', 'sss_cal_mgh.dat')
        crosstalk_file = os.path.join(data_path, 'SSS', 'ct_sparse_mgh.fif')

        # raw.info['bads'] = []
        # raw_check = raw.copy()
        # auto_noisy_chs, auto_flat_chs, _ = find_bad_channels_maxwell(
        #     raw_check, cross_talk=crosstalk_file, calibration=fine_cal_file,
        #     return_scores=True, verbose=False
        # )

        # bads = raw.info['bads'] + auto_noisy_chs + auto_flat_chs + ['MEG 2313']
        # raw.info['bads'] = bads

        raw = mne.preprocessing.maxwell_filter(raw, cross_talk=crosstalk_file,
                                               calibration=fine_cal_file,
                                               verbose=False)

    events = mne.find_events(raw)
    reject = dict(grad=4000e-13, eog=350e-6)
    picks = mne.pick_types(raw.info, meg=True, eog=True)

    event_id, tmin, tmax = event_id[condition], -1.0, 3.0
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        reject=reject, preload=True, baseline=(None, 0))
    # evoked = epochs.filter(1, None).average()
    evoked = epochs.average()

    noise_cov = mne.compute_covariance(epochs, rank="info", tmax=0.0)

    evoked = evoked.pick_types(meg=True, eeg=False)
    evoked.crop(tmin=0.05, tmax=0.15)  # Choose a timeframe not too large

    if return_stc and simulated:
        if return_labels:
            return evoked, forward, noise_cov, stc, labels
        return evoked, forward, noise_cov, stc
    return evoked, forward, noise_cov


def load_somato_data():
    data_path = somato.data_path()
    subject = "01"
    task = "somato"

    raw_fname = os.path.join(data_path, "sub-{}".format(subject), "meg",
                             "sub-{}_task-{}_meg.fif".format(subject, task))
    fwd_fname = os.path.join(data_path, "derivatives",
                             "sub-{}".format(subject),
                             "sub-{}_task-{}-fwd.fif".format(subject, task))

    # Read evoked
    raw = mne.io.read_raw_fif(raw_fname)
    events = mne.find_events(raw, stim_channel="STI 014")
    reject = dict(grad=4000e-13, eog=350e-6)
    picks = mne.pick_types(raw.info, meg=True, eog=True)

    event_id, tmin, tmax = 1, -1.0, 3.0
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        reject=reject, preload=True)
    evoked = epochs.filter(1, None).average()
    evoked = evoked.pick_types(meg=True)
    evoked.crop(tmin=0.03, tmax=0.05)  # Choose a timeframe not too large

    # Handling forward solution
    forward = mne.read_forward_solution(fwd_fname)
    noise_cov = mne.compute_covariance(epochs, rank="info", tmax=0.0)

    return evoked, forward, noise_cov


def load_data_from_camcan(folder_name, data_path, orient):
    data_path = Path(data_path)

    fwd_fname = data_path / "meg" / f"{folder_name}_task-passive-fwd.fif"
    ave_fname = data_path / "meg" / f"{folder_name}_task-passive-ave.fif"
    cleaned_epo_fname = (
        data_path / "meg" / f"{folder_name}_task-passive_cleaned-epo.fif"
    )

    # Building noise covariance
    cleaned_epochs = mne.read_epochs(cleaned_epo_fname)
    noise_cov = mne.compute_covariance(cleaned_epochs, tmax=0, rank="info")

    evokeds = mne.read_evokeds(ave_fname, condition=None, baseline=(None, 0))
    evoked = evokeds[-2]

    CAMCAN_DATA_PATH = Path(f"../data/camcan/evokeds/{folder_name}")
    CAMCAN_DATA_PATH.mkdir(parents=True, exist_ok=True)
    evoked.save(CAMCAN_DATA_PATH / f"{orient}_full-ave.fif")

    forward = mne.read_forward_solution(fwd_fname)

    evoked.crop(tmin=0.08, tmax=0.15)  # 0.08 - 0.15
    evoked = evoked.pick_types(eeg=False, meg=True)

    return evoked, forward, noise_cov


# ================================================
# ================ LOW RESOLUTION ================
# ================================================

def compute_forward(data_path, info, resolution=3):
    path_fwd = data_path + \
        '/MEG/sample/sample_audvis-meg-eeg-oct-%i-fwd.fif' % resolution
    if not os.path.isfile(path_fwd):
        fwd = compute_forward_(data_path, info, resolution)
        mne.write_forward_solution(path_fwd, fwd, overwrite=True)
    else:
        fwd = mne.read_forward_solution(path_fwd)
    return fwd


def compute_forward_(data_path, info, resolution=3):
    if resolution == 6:
        path_fwd = data_path + \
            '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
        fwd = mne.read_forward_solution(path_fwd)
        return fwd
    # if not os.path.isfile(path_fwd):
    #     fwd = compute_forward(data_path, raw.info, resolution)
    #     mne.write_forward_solution(path_fwd, fwd, overwrite=True)
    # else:
    spacing = "ico%d" % resolution
    src_fs = mne.setup_source_space(
        subject='sample',
        spacing=spacing,
        subjects_dir=data_path+"/subjects",
        add_dist=False)
    bem_fname = data_path + \
        "/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif"
    bem = mne.read_bem_solution(bem_fname)

    fwd = mne.make_forward_solution(
        info, trans=data_path + "/MEG/sample/sample_audvis_raw-trans.fif",
        src=src_fs, bem=bem, meg=True, eeg=True, mindist=2.,
        n_jobs=2)
    path_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-%i-fwd.fif' \
        % resolution
    mne.write_forward_solution(path_fwd, fwd, overwrite=True)
    return fwd
