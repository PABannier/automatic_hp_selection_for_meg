import os, joblib

from hp_selection.sure import solve_using_sure
from hp_selection.spatial_cv import solve_using_spatial_cv
from hp_selection.temporal_cv import solve_using_temporal_cv
from hp_selection.utils import apply_solver

from hp_selection.utils import load_data, load_somato_data

CONDITIONS = ["Left Auditory", "Right Auditory", "Left visual",
              "Right visual", "somato"]


def save_stc(stc, condition, solver):
    out_dir = "../data/%s/" % solver

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    fname = condition.lower().replace(" ", "_") + ".pkl"
    out_path = os.path.join(out_dir, fname)

    with open(out_path, "wb") as out_file:
        joblib.dump(stc, out_file)


if __name__ == "__main__":
    CONDITIONS = ["Left Auditory"]

    for condition in CONDITIONS:
        if condition == "somato":
            evoked, forward, noise_cov = load_somato_data()
        else:
            evoked, forward, noise_cov = load_data(condition, maxfilter=False)

        # SURE
        # stc = solve_using_sure(evoked, forward, noise_cov, loose=0)
        # save_stc(stc, condition, "sure")

        # Spatial CV
        # stc = apply_solver(solve_using_spatial_cv, evoked, forward, noise_cov)
        # save_stc(stc, condition, "spatial_cv")

        # Temporal CV
        stc = apply_solver(solve_using_temporal_cv, evoked, forward, noise_cov)
        save_stc(stc, condition, "temporal_cv")

