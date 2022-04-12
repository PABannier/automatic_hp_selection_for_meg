import os, joblib

from calibromatic.hp_selection.sure import solve_using_sure
from calibromatic.hp_selection.spatial_cv import solve_using_spatial_cv
from calibromatic.hp_selection.lambda_map import solve_using_lambda_map
from calibromatic.utils import apply_solver

from calibromatic.utils import load_data, load_somato_data

CONDITIONS = ["auditory/left", "auditory/right", "visual/left",
              "visual/right", "somato"]


def save_stc(stc, condition, solver):
    out_dir = "../data/%s/" % solver

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    fname = condition.lower().replace("/", "_") + ".pkl"
    out_path = os.path.join(out_dir, fname)

    with open(out_path, "wb") as out_file:
        joblib.dump(stc, out_file)


if __name__ == "__main__":
    simulated = False
    maxfilter = False

    for condition in CONDITIONS:
        if condition == "somato":
            evoked, forward, noise_cov = load_somato_data()
        else:
            evoked, forward, noise_cov = load_data(
                condition, maxfilter=maxfilter, simulated=simulated)

        stc_name = "lambda_map"
        if condition != "somato" and simulated:
            condition += "_simu"
        if condition != "somato" and maxfilter:
            condition += "_mf"

        stc = solve_using_sure(evoked, forward, noise_cov, loose=0.9, depth=0.9)
        save_stc(stc, condition, stc_name)

        stc = apply_solver(solve_using_spatial_cv, evoked, forward, noise_cov,
                           depth=0.9, loose=0.9)
        save_stc(stc, condition, "spatial_cv")

        # Lambda map
        stc = apply_solver(solve_using_lambda_map, evoked, forward, noise_cov,
                           depth=0.9, loose=0.9)
        save_stc(stc, condition, stc_name)
