import joblib
import mne
from mne.viz import plot_sparse_source_estimates
# %%
from pathlib import Path

mne.viz.set_3d_backend('pyvistaqt')
sample_dir = Path(mne.datasets.sample.data_path())
subjects_dir = sample_dir / 'subjects'

src_fsaverage_fname = \
    subjects_dir / "fsaverage/bem/fsaverage-ico-5-src.fif"
src_fsaverage = mne.read_source_spaces(src_fsaverage_fname)

morphed_stcs = joblib.load('../data/camcan/morphed_stc.pkl')
plot_sparse_source_estimates(
    src_fsaverage, morphed_stcs, bgcolor=(1, 1, 1),
    fig_name="Merged brains", opacity=0.1
)
