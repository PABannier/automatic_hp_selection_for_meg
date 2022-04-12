from calibromatic.hp_selection.lambda_map import solve_using_lambda_map
from pathlib import Path
import os
from joblib import parallel_backend
from joblib import Parallel, delayed
import joblib

import pandas as pd

from calibromatic.utils import load_data_from_camcan, apply_solver
from calibromatic.hp_selection.spatial_cv import solve_using_spatial_cv
from calibromatic.hp_selection.temporal_cv import solve_using_temporal_cv
from calibromatic.hp_selection.sure import solve_using_sure

N_JOBS = 30  # -1
INNER_MAX_NUM_THREADS = 1
CRITERION = "sure"  # Can be: sure, spatial_cv, temporal_cv, lambda_map

DERIVATIVES_PATH = Path("/storage/store2/work/rhochenb/Data/Cam-CAN/BIDS")
DATA_PATH = DERIVATIVES_PATH / "derivatives/mne-study-template"

PARTICIPANTS_INFO = Path(
    "/storage/store/data/camcan/BIDSsep/passive/participants.tsv"
)

out_stc_dirs = {
    "sure": "stcs",
    "spatial_cv": "stcs_scv",
    "temporal_cv": "stcs_tcv",
    "lambda_map": "stcs_lmap"
}

OUTPUT_DIR = Path(f"../data/camcan/{out_stc_dirs[CRITERION]}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CRASHING_PATIENTS = [
    "sub-CC210250",
    "sub-CC310086",
    "sub-CC320160",
    "sub-CC320616",
    "sub-CC320870",
    "sub-CC321557",
    "sub-CC610653",
    "sub-CC610671",
]

EXISTING_PATIENTS = [
    "sub-CC620406",
    "sub-CC610052",
    "sub-CC520480",
    "sub-CC720188",
    "sub-CC420137",
    "sub-CC220697",
    "sub-CC720407",
    "sub-CC320461",
    "sub-CC620314",
    "sub-CC410248",
    "sub-CC310086",
    "sub-CC120120",
    "sub-CC320870",
    "sub-CC520584",
    "sub-CC420004",
    "sub-CC520980",
    "sub-CC120347",
    "sub-CC520200",
    "sub-CC721291",
    "sub-CC610653",
    "sub-CC210250",
    "sub-CC520211",
    "sub-CC610372",
    "sub-CC410226",
    "sub-CC121685",
    "sub-CC420217",
    "sub-CC410220",
    "sub-CC521040",
    "sub-CC720670",
    "sub-CC321557",
    "sub-CC610508",
    "sub-CC320448",
    "sub-CC320687",
    "sub-CC320680",
    "sub-CC320616",
    "sub-CC410119",
    "sub-CC110033",
    "sub-CC520868",
    "sub-CC610671",
    "sub-CC320160",
    "sub-CC420143",
]

EXISTING_PATIENTS = [
    x for x in EXISTING_PATIENTS if x not in CRASHING_PATIENTS
]


def solve_camcan_inverse_problem(folder_name, data_path, criterion):
    evoked, forward, noise_cov = load_data_from_camcan(folder_name, data_path,
                                                       "free")

    if criterion == "sure":
        stc = solve_using_sure(evoked, forward, noise_cov, loose=0)
    elif criterion == "spatial_cv":
        stc = apply_solver(solve_using_spatial_cv, evoked, forward, noise_cov)
    elif criterion == "temporal_cv":
        stc = apply_solver(solve_using_temporal_cv, evoked, forward, noise_cov)
    elif criterion == "lambda_map":
        stc = apply_solver(solve_using_lambda_map, evoked, forward, noise_cov)
    else:
        raise Exception("Wrong criterion!")

    return stc, evoked, forward, noise_cov


def solve_for_patient(folder_path, criterion):
    folder_name = folder_path.split("/")[-1]
    print(f"Solving #{folder_name}")

    patient_path = DATA_PATH / folder_name
    stc = solve_camcan_inverse_problem(folder_name, patient_path, criterion)

    out_dir = OUTPUT_DIR / f"{folder_name}"
    out_path_stc = out_dir / "free.pkl"
    out_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(stc, out_path_stc)


if __name__ == "__main__":
    # Load 30 youngest patients
    participant_info_df = pd.read_csv(PARTICIPANTS_INFO, sep="\t")
    participant_info_df = participant_info_df[participant_info_df["age"] < 30]

    patient_folders = list(participant_info_df["participant_id"])
    # patient_folders = EXISTING_PATIENTS

    patient_folders = [DATA_PATH / x for x in patient_folders]
    patient_folders = [str(x) for x in patient_folders if os.path.isdir(x)]

    # Select subjects
    with parallel_backend("loky", inner_max_num_threads=INNER_MAX_NUM_THREADS):
        Parallel(N_JOBS)(
            delayed(solve_for_patient)(folder_name, CRITERION)
            for folder_name in patient_folders
        )
