import os
import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.time_frequency import fit_iir_model_raw
from mne.viz import plot_sparse_source_estimates
from mne.simulation import simulate_sparse_stc, simulate_evoked
from mne.cov import _smart_eigh


# from data.utils import compute_forward, get_data_from_X_S_and_B_star
# from sgcl.utils import clp_sqrt, get_S_Sinv
# from data.artificial import get_data_from_X_S_and_B_star
# import data.artificial
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
        subjects_dir=data_path + "/subjects",
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


def get_fwd_and_cov(
        resolution=3, meg=True, eeg=True, bads=['MEG 2443', 'EEG 053'],
        data_path=sample.data_path()):
    ##########################################################################
    # Load real data as templates
    print("Loading data..............................................................")
    raw = mne.io.read_raw_fif(
        data_path + '/MEG/sample/sample_audvis_raw.fif', preload=True)
    raw.info['bads'] = bads  # mark bad channels
    cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'
    cov = mne.read_cov(cov_fname)
    raw.drop_channels(ch_names=raw.info['bads'])
    ################################################################################
    # import to resize foreward
    path_fwd = data_path + \
        '/MEG/sample/sample_audvis-meg-eeg-oct-%i-fwd.fif' % resolution
    if not os.path.isfile(path_fwd):
        fwd = compute_forward(data_path, raw.info, resolution)
        mne.write_forward_solution(path_fwd, fwd, overwrite=True)
    else:
        fwd = mne.read_forward_solution(path_fwd)

    fwd = mne.convert_forward_solution(fwd, force_fixed=True)
    fwd = mne.pick_types_forward(
        fwd, meg=meg, eeg=eeg, exclude=raw.info['bads'])
    raw.pick_types(meg=meg, eeg=eeg)

    fwd = mne.pick_channels_forward(fwd, raw.ch_names)
    cov = mne.pick_channels_cov(cov, raw.ch_names)
    cov['bads'] = bads
    cov = mne.pick_channels_cov(cov)
    return fwd, cov, raw.info


def get_B_star(stc, fwd, dns=False):
    n_sources = fwd['sol']['data'].shape[1]
    list_node_lh = stc.vertices[0]
    list_node_rh = stc.vertices[1]
    mask_B_star = np.zeros(n_sources, dtype=bool)
    for node in list_node_lh:
        active_feature = np.where(fwd["src"][0]["vertno"] == node)[0][0]
        mask_B_star[active_feature] = True
    for node in list_node_rh:
        # TODO can somebody check this to see if it is correct ?
        active_feature = np.where(
            fwd["src"][1]["vertno"] == node)[0][0] + \
            len(fwd["src"][0]["vertno"])
        mask_B_star[active_feature] = True
    dense_B_star = stc.data
    if dns:
        return mask_B_star, dense_B_star
    else:
        B_star = get_B_from_supp_and_dense(mask_B_star, dense_B_star)
        return B_star


def get_semi_real_data(
        n_times=100, n_dipoles=2, amplitude=1e-8, resolution=3,
        seed=0, meg=True, eeg=True, random_state=42,
        data_path=sample.data_path(), bads=['MEG 2443', 'EEG 053'],
        sanity_check=True):

    # load forward and covariance matrix
    fwd, cov, info = get_fwd_and_cov(
        resolution=resolution, meg=meg, eeg=eeg, bads=bads,
        data_path=data_path)

    cov_data = cov.data
    if cov_data.ndim == 1:
        cov_data = np.diag(cov_data)

    X = fwd['sol']['data']

    # set labels
    label_names = ['Aud-lh', 'Aud-rh', 'Vis-lh', 'Vis-rh']
    labels = [
        mne.read_label(data_path + '/MEG/sample/labels/%s.label' % ln)
        for ln in label_names]

    ###########################################################################
    # Generate source time courses from 2 dipoles and the corresponding data
    times = np.arange(n_times, dtype=np.float) / info['sfreq']
    rng = check_random_state(seed)

    def data_fun_int(times):
        """Function to generate random source time courses"""
        return amplitude * np.sin((1 + 3 * rng.randn(1) / 2.) * np.pi *
                                  times / times.max())

    stc = simulate_sparse_stc(
        fwd['src'], n_dipoles=n_dipoles, times=times,
        random_state=random_state, labels=labels, data_fun=data_fun_int)
    # recover B_star from stc and fwd
    B_star = get_B_star(stc, fwd)
    assert (norm(B_star, axis=1) != 0).sum() == n_dipoles

    Y = X @ B_star

    if sanity_check:
        plt.plot(Y.T)
        plt.show()

        idx_nnz = np.where(norm(B_star != 0, axis=1))[0]
        plt.plot(B_star[idx_nnz, :].T)
        plt.show()

    # TODO XXX create evoked object from data X
    return X, fwd, cov




def get_B_from_supp_and_dense(mask, dense):
    n_sources = mask.shape[0]
    n_times = dense.shape[1]
    B = np.zeros((n_sources, n_times))
    B[mask, :] = dense
    return B


def get_my_whitener(cov_data, info=None, ch_names=None, rank=None,
                  pca=False, scalings=None, prepared=False):
    n_chan = cov_data.shape[0]
    eig, eigvecs, _ = _smart_eigh(
        cov_data, info, None, None, info["projs"], ch_names)

    nzero = (eig > 0)
    eig[~nzero] = 0.  # get rid of numerical noise (negative) ones
    n_nzero = np.sum(nzero)

    whitener = np.zeros((n_chan, 1), dtype=np.float)
    whitener[nzero, 0] = 1.0 / np.sqrt(eig[nzero])
    #   Rows of eigvec are the eigenvectors
    whitener = whitener * eigvecs  # C ** -0.5
    colorer = np.sqrt(eig) * eigvecs.T  # C ** 0.5
    return whitener, colorer, cov_data, n_nzero


def rescale_forward(X):
    X_init = X.copy()

    X = X.copy()

    log_abs_X = np.log(np.abs(X))

    # rescaling of each line
    colorer0 = norm(X, axis=1)
    X /= colorer0[:, np.newaxis]

    log_norm_X_axis1 = compute_log_norm_axis1(X_init, axis=1)
    log_abs_X -= log_norm_X_axis1[:, np.newaxis]

    source_weighting = norm(X, axis=0, ord=2)
    X /= source_weighting

    log_norm_X_axis_0 = compute_log_norm_axis0(log_abs_X, axis=0)
    log_abs_X -= log_norm_X_axis_0[np.newaxis, :]

    X = np.exp(log_abs_X) * np.sign(X_init)
    return X


def rescale_cov_data(X, cov_data):
    X_init = X.copy()
    cov_data_init = cov_data.copy()
    X = X.copy()
    cov_data = cov_data.copy()

    log_abs_X = np.log(np.abs(X))
    log_abs_cov_data = np.log(np.abs(cov_data))

    colorer0 = norm(X, axis=1)
    cov_data /= np.outer(colorer0, colorer0)

    log_norm_X_axis1 = compute_log_norm_axis1(X_init, axis=1)
    log_abs_cov_data -= log_norm_X_axis1[:, np.newaxis]
    log_abs_cov_data -= log_norm_X_axis1[np.newaxis, :]

    # assert np.allclose( np.exp(log_abs_cov_data * np.sign(cov_data_init)), cov_data)

    cov_data = np.exp(log_abs_cov_data) * np.sign(cov_data_init)
    return cov_data


def source_weight(X, B_star):
    X_init = X.copy()
    B_star_init = B_star.copy()

    X = X.copy()
    B_star = B_star.copy()

    log_abs_X = np.log(np.abs(X))
    log_abs_B_star = np.log(np.abs(B_star))

    # source weighting
    source_weighting = norm(X, axis=0, ord=2)
    X /= source_weighting
    B_star *= source_weighting[:, np.newaxis]
    colorer_source_weight = 1 / source_weighting

    log_norm_X_axis_0 = compute_log_norm_axis0(log_abs_X, axis=0)
    log_abs_X -= log_norm_X_axis_0[np.newaxis, :]
    log_abs_B_star += log_norm_X_axis_0[:, np.newaxis]

    assert np.allclose( np.exp(log_abs_X) * np.sign(X_init), X)
    assert np.allclose( np.exp(log_abs_B_star) * np.sign(B_star_init), B_star)

    X =  np.exp(log_abs_X) * np.sign(X_init)
    B_star = np.exp(log_abs_B_star) * np.sign(B_star_init)
    return X, B_star, colorer_source_weight


def rescale_X_all_epochs_and_B_star(
    X, all_epochs, B_star, cov_data, rescale_lines=True, source_weight=True):

    X_init = X.copy()
    all_epochs_init = all_epochs.copy()
    B_star_init = B_star.copy()
    cov_data_init = cov_data.copy()

    X = X.copy()
    all_epochs = all_epochs.copy()
    cov_data = cov_data.copy()
    B_star = B_star.copy()

    log_abs_X = np.log(np.abs(X))
    log_abs_all_epochs = np.log(np.abs(all_epochs))
    log_abs_cov_data = np.log(np.abs(cov_data))
    log_abs_B_star = np.log(np.abs(B_star))

    if rescale_lines:
        # rescaling of each line
        colorer0 = norm(X, axis=1)
        X /= colorer0[:, np.newaxis]
        all_epochs /= colorer0[np.newaxis, :, np.newaxis]
        cov_data /= np.outer(colorer0, colorer0)

        log_norm_X_axis1 = compute_log_norm_axis1(X_init, axis=1)
        log_abs_X -= log_norm_X_axis1[:, np.newaxis]
        log_abs_all_epochs -= log_norm_X_axis1[np.newaxis, :, np.newaxis]
        log_abs_cov_data -= log_norm_X_axis1[:, np.newaxis]
        log_abs_cov_data -= log_norm_X_axis1[np.newaxis, :]

        assert np.allclose(np.exp(log_abs_X) * np.sign(X_init), X)
        assert np.allclose(np.exp(log_abs_all_epochs) * np.sign(all_epochs_init), all_epochs)
        # assert np.allclose( np.exp(log_abs_cov_data * np.sign(cov_data_init)), cov_data )

    if source_weight:
        # source weighting
        source_weighting = norm(X, axis=0, ord=2)
        X /= source_weighting
        B_star *= source_weighting[:, np.newaxis]
        colorer_source_weight = 1 / source_weighting

        log_norm_X_axis_0 = compute_log_norm_axis0(log_abs_X, axis=0)
        log_abs_X -= log_norm_X_axis_0[np.newaxis, :]
        log_abs_B_star += log_norm_X_axis_0[:, np.newaxis]

        assert np.allclose( np.exp(log_abs_X) * np.sign(X_init), X)
        assert np.allclose( np.exp(log_abs_B_star) * np.sign(B_star_init), B_star)
        assert np.allclose( np.exp(log_abs_cov_data) * np.sign(cov_data_init), cov_data)


    Y = all_epochs.mean(axis=0)
    # normalize Y = all_epochs.mean() to 1
    scaling_factor = norm(Y, ord='fro')
    # scaling_factor = np.abs(all_epochs.mean(axis=0)).max()
    all_epochs /= scaling_factor
    colorer0 *= scaling_factor
    B_star /= scaling_factor
    colorer_source_weight *= scaling_factor
    cov_data /= scaling_factor ** 2

    log_norm_Y = compute_log_norm( Y )
    # log_norm_Y = compute_log_norm( ((np.exp(log_abs_all_epochs) * np.sign(all_epochs_init)).mean(axis=0)) )
    log_abs_all_epochs -= log_norm_Y
    log_abs_B_star -= log_norm_Y
    log_abs_cov_data -= 2 * log_norm_Y

    assert np.allclose(colorer0[:, np.newaxis] * X / colorer_source_weight, X_init)
    assert np.allclose(colorer0[np.newaxis, :, np.newaxis] * all_epochs, all_epochs_init)
    # assert np.allclose(cov_data * np.outer(colorer0, colorer0), cov_data_init)
    assert np.allclose(B_star * colorer_source_weight[:, np.newaxis], B_star_init)

    assert np.allclose( np.exp(log_abs_X) * np.sign(X_init), X)
    # assert np.allclose( np.exp(log_abs_B_star) * np.sign(B_star_init), B_star)
    # assert np.allclose( np.exp(log_abs_all_epochs) * np.sign(all_epochs_init), all_epochs)
    assert np.allclose( np.exp(log_abs_cov_data) * np.sign(cov_data_init), cov_data)
    if True:
        all_epochs = np.exp(log_abs_all_epochs) * np.sign(all_epochs_init)
        cov_data = np.exp(log_abs_cov_data) * np.sign(cov_data_init)
        X =  np.exp(log_abs_X) * np.sign(X_init)
    return X, all_epochs, B_star, cov_data, (colorer0, colorer_source_weight)


def compute_log_norm_axis1(X, axis=1):
    abs_X = np.abs(X)
    max_abs_X = np.max(X, axis=axis)
    if axis == 1:
        abs_X = abs_X / max_abs_X[:, np.newaxis]
    else:
        abs_X = abs_X / max_abs_X[np.newaxis, :]
    log_norm = np.log(max_abs_X) + np.log(norm(abs_X, axis=axis))
    return log_norm

def compute_log_norm_axis0(log_abs_X, axis=0):
    max_log_abs_X = np.max(log_abs_X, axis=axis)

    log_abs_X = log_abs_X - max_log_abs_X[np.newaxis, :]
    abs_X = np.exp(log_abs_X)

    log_norm = max_log_abs_X + np.log(norm(abs_X, axis=axis))
    return log_norm

def compute_log_norm(X):
    log_abs_X = np.log(np.abs(X))
    max_log_abs_X = np.max(log_abs_X)

    log_abs_X = log_abs_X - max_log_abs_X
    abs_X = np.exp(log_abs_X)

    log_norm = max_log_abs_X + np.log(norm(abs_X, ord='fro'))
    return log_norm

def simulate_real_B_star(fwd, meeg_ch_names, n_dipoles=3, n_times=100, n_epochs=50, seed=0):
    X = fwd['sol']['data']
    _, n_sources = X.shape
    n_meg_sensors = get_n_meg_numbers(meeg_ch_names)
    rng = check_random_state(seed)

    B_star = np.zeros([n_sources, n_times])
    supp = rng.choice(n_sources, n_dipoles, replace=False)
    for index in supp:
        is_meg_ch = index <= n_meg_sensors
        B_star[index, :] = data_fun(n_times, rng, is_meg_ch=is_meg_ch)
    return B_star

def data_fun(n_times, rng, is_meg_ch=True):
    """Function to generate random source time courses"""
    arr_times = np.arange(n_times)
    line = ( np.cos(30. * (arr_times - arr_times[n_times //2 ]) ) *
            np.exp(- (arr_times - arr_times[n_times //2 ]) ** 2 * 0.1))
    if is_meg_ch:
        return line / norm(line) * 500e-9
    else:
        return line / norm(line) * 20e-6

def get_n_meg_numbers(ch_names):
    n_meg_sensors = 0
    for ch_name in ch_names:
        if ch_name.startswith('MEG'):
            n_meg_sensors += 1
    return n_meg_sensors

def drop_bad_ch_cov(cov_data, ch_names, bads):
    idx = [ii for ii, ch in enumerate(ch_names) if ch not in bads]
    cov_data = cov_data[idx][:, idx]
    return cov_data
