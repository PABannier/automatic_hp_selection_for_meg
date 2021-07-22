# import joblib
# import os.path as op

# from numba import njit
# import matplotlib.pyplot as plt

# import mne


# def merge_brain_plot(stcs):
#     subjects_dir = "subjects_dir"
#     src_fsaverage_fname = subjects_dir +
# "/fsaverage/bem/fsaverage-ico-5-src.fif"
#     src_fsaverage = mne.read_source_spaces(src_fsaverage_fname)

#     morphed_stcs = []

#     for stc in stcs:
#         morph = mne.compute_source_morph(
# stc, subject_from=subject_id,
#
# subject_to="fsaverage", spacing=None,
#
# sparse=True, subjects_dir=subjects_dir)
#         stc_fsaverage = morph.apply(stc)
#         morphed_stcs.append(stc_fsaverage)

