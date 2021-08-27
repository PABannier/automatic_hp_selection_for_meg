# %%
from collections import Counter
import joblib
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_sparse_source_estimates
# %%
from pathlib import Path

CRITERION = "sure"
HEMISPHERES = ["lh", "rh"]

mne.viz.set_3d_backend('pyvistaqt')
sample_dir = Path(mne.datasets.sample.data_path())
subjects_dir = sample_dir / 'subjects'

src_fsaverage_fname = \
    subjects_dir / "fsaverage/bem/fsaverage-ico-5-src.fif"
src_fsaverage = mne.read_source_spaces(src_fsaverage_fname)

morphed_stcs = joblib.load(f'../data/camcan/morphed_stc_{CRITERION}.pkl')
plot_sparse_source_estimates(
    src_fsaverage, morphed_stcs, bgcolor=(1, 1, 1),
    fig_name="Merged brains", opacity=0.1
)

# See some stats on number of sources
n_vertices = [(stc.vertices[0].size, stc.vertices[1].size) for stc in morphed_stcs]
print(Counter(n_vertices))

# %% look at mean activity
vertices = [
    [src_fsaverage[0]['vertno']] + [stc.vertices[0] for stc in morphed_stcs],
    [src_fsaverage[1]['vertno']] + [stc.vertices[1] for stc in morphed_stcs],
]
vertices = map(np.concatenate, vertices)
vertices = list(map(np.unique, vertices))
dense_morphed_stcs = [stc.copy().expand(vertices) for stc in morphed_stcs]
mean_stc = 1e9 * sum(dense_morphed_stcs[1:], dense_morphed_stcs[0]) / len(morphed_stcs)

mean_stc.plot(subject='fsaverage', subjects_dir=subjects_dir, hemi="both")

# %% 

brain = mne.viz.Brain(
    subject_id='fsaverage',
    # views=["lat", "med"],
    views=["lat"],
    hemi="split" if len(HEMISPHERES) > 1 else HEMISPHERES[0],
    size=(500, 250),
    # size=(500, 500),
    subjects_dir=subjects_dir,
    background="w",
    surf='inflated',
    cortex="classic",
)

def add_foci_to_brain_surface(brain, stc, ax, color):
    for i_hemi, hemi in enumerate(HEMISPHERES):
        surface_coords = brain.geo[hemi].coords
        hemi_data = stc.lh_data if hemi == "lh" else stc.rh_data
        # i_hemi += 1  Dirty trick for rh
        for k in range(len(stc.vertices[i_hemi])):
            activation_idx = stc.vertices[i_hemi][k]
            foci_coords = surface_coords[activation_idx]

            # In milliseconds
            (line,) = ax.plot(stc.times * 1e3, 1e9 * hemi_data[k], color=color)
            brain.add_foci(foci_coords, hemi=hemi, color=line.get_color())

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (nAm)")


colors = plt.cm.tab20(np.linspace(0, 1, len(morphed_stcs)))
fig, ax = plt.subplots(1, 1)
for stc, col in zip(morphed_stcs, colors):
    add_foci_to_brain_surface(brain, stc, ax, color=col)

# %% add auditory labels from HCP atlas

labels = mne.read_labels_from_annot(
    'fsaverage', 'HCPMMP1_combined', 'both', subjects_dir=subjects_dir)

aud_labels = [label for label in labels if "Early Auditory" in label.name]
for label in aud_labels:
    try:
        brain.add_label(label, borders=False)
    except KeyError:
        print(f"Label {label} not found.")
        pass

if len(HEMISPHERES) > 1:
    outfile_name = f"agregate_brain_{CRITERION}_split_99_grad_only.png"
else:
    outfile_name = f"agregate_brain_{CRITERION}_{HEMISPHERES[0]}.png"


brain.save_image(f'../figures/{outfile_name}')
# %%
