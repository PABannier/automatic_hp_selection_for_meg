import joblib, os
from tqdm import tqdm
from pathlib import Path

import matplotlib.pyplot as plt

import mne
from mne.viz import plot_sparse_source_estimates


def morph_stc(stcs, subject_ids, subject_dir):
    """Morph stc inplace onto generic brain"""
    for i, (stc, subject_id) in tqdm(enumerate(zip(stcs, subject_ids)),
                                     total=len(stcs)):
        morph = mne.compute_source_morph(stc, subject_from=subject_id,
                                         subject_to="fsaverage", spacing=None,
                                         sparse=True, subjects_dir=subject_dir)
        stc_fsaverage = morph.apply(stc)
        stcs[i] = stc_fsaverage

    return stcs


if __name__ == "__main__":
    STC_PATH = Path("../data/camcan/stcs")

    subjects_dir = "../data/camcan/subjects_dir/"
    src_fsaverage_fname = subjects_dir + "fsaverage/bem/fsaverage-ico-5-src.fif"
    src_fsaverage = mne.read_source_spaces(src_fsaverage_fname)

    stc_paths = os.listdir(STC_PATH)
    stc_paths = [x for x in stc_paths if x != ".DS_Store"][:3]
    subject_ids = stc_paths.copy()
    subject_ids = [x[4:] for x in subject_ids]
    stc_paths = [STC_PATH / Path(x) / "free.pkl" for x in stc_paths]

    stcs = [joblib.load(x) for x in stc_paths]
    morphed_stcs = morph_stc(stcs, subject_ids, subjects_dir)

    plot_sparse_source_estimates(src_fsaverage, morphed_stcs, bgcolor=(1, 1, 1),
                                 fig_name="Merged brains", opacity=0.1)





