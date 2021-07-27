from pathlib import Path
import os
from joblib import parallel_backend
from joblib import Parallel, delayed
import joblib

import pandas as pd

from hp_selection.utils import solve_camcan_inverse_problem

N_JOBS = 20  # -1
INNER_MAX_NUM_THREADS = 1

DATA_PATH = Path(
    "../../../../../rhochenb/Data/Cam-CAN/BIDS/derivatives/mne-study-template"
)

PARTICIPANTS_INFO = Path(
    "../../../../../../../store/data/camcan/BIDSsep/passive/participants.tsv"
)

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

def solve_for_patient(folder_path, criterion, loose=0.9):
    folder_name = folder_path.split("/")[-1]
    print(f"Solving #{folder_name}")

    patient_path = DATA_PATH / folder_name
    stc = solve_camcan_inverse_problem(folder_name, patient_path, criterion)

    out_dir = f"stcs/{folder_name}"
    out_path_stc = out_dir + f"/free.pkl"

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    joblib.dump(stc, out_path_stc)


if __name__ == "__main__":
    # Load 30 youngest patients
    participant_info_df = pd.read_csv(PARTICIPANTS_INFO, sep="\t")
    participant_info_df = participant_info_df[participant_info_df["age"] < 30]

    patient_folders = list(participant_info_df["participant_id"])
    patient_folders = [DATA_PATH / x for x in patient_folders]
    patient_folders = [str(x) for x in patient_folders if os.path.isdir(x)]

    # Select subjects
    with parallel_backend("loky", inner_max_num_threads=INNER_MAX_NUM_THREADS):
        Parallel(N_JOBS)(
            delayed(solve_for_patient)(folder_name)
            for folder_name in patient_folders
        )
