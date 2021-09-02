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
    print("Loading data.....................................................")
    raw = mne.io.read_raw_fif(
        data_path + '/MEG/sample/sample_audvis_raw.fif', preload=True)
    raw.info['bads'] = bads  # mark bad channels
    cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'
    cov = mne.read_cov(cov_fname)
    raw.drop_channels(ch_names=raw.info['bads'])
    ###########################################################################
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
        return amplitude * np.sin(
            (1 + 3 * rng.randn(1) / 2.) * np.pi * times / times.max())

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
