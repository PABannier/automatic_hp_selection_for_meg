import joblib
import os
from tqdm import tqdm
from pathlib import Path

import mne
from mne.viz import plot_sparse_source_estimates

CRITERION = "sure"

def morph_stc(stcs, subject_ids, subject_dir):
    """Morph stc inplace onto generic brain"""
    morphed_stcs = []
    for i, (stc, subject_id) in tqdm(enumerate(zip(stcs, subject_ids)),
                                     total=len(stcs)):
        try:
            morph = mne.compute_source_morph(stc, subject_from=subject_id,
                                             subject_to="fsaverage",
                                             spacing=None,
                                             sparse=True,
                                             subjects_dir=subject_dir)
            stc_fsaverage = morph.apply(stc)
        except FileNotFoundError:
            print(f"file not found for {subject_id}")
        else:
            morphed_stcs.append(stc_fsaverage)

    return morphed_stcs


if __name__ == "__main__":
    out_stc_dirs = {
        "sure": "stcs",
        "spatial_cv": "stcs_scv",
        "temporal_cv": "stcs_tcv",
        "lambda_map": "stcs_lmap"
    }

    stc_path = Path(f"../data/camcan/{out_stc_dirs[CRITERION]}")

    # subjects_dir = "../data/camcan/subjects_dir/"
    subjects_dir = "/storage/store/data/camcan-mne/freesurfer/"
    src_fsaverage_fname = \
        subjects_dir + "fsaverage/bem/fsaverage-ico-5-src.fif"
    src_fsaverage = mne.read_source_spaces(src_fsaverage_fname)

    stc_paths = os.listdir(stc_path)
    stc_paths = [x for x in stc_paths if x != ".DS_Store"][:30]
    subject_ids = stc_paths.copy()
    subject_ids = [x[4:] for x in subject_ids]
    stc_paths = [stc_path / Path(x) / "free.pkl" for x in stc_paths]

    stcs = [joblib.load(x)[0] for x in stc_paths]
    morphed_stcs = morph_stc(stcs, subject_ids, subjects_dir)

    joblib.dump(morphed_stcs, f'../data/camcan/morphed_stc_{CRITERION}.pkl')

    plot_sparse_source_estimates(
        src_fsaverage, morphed_stcs, bgcolor=(1, 1, 1),
        fig_name="Merged brains", opacity=0.1
    )
