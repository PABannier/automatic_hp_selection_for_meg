from hp_selection.utils import load_data
import joblib

import numpy as np
import matplotlib.pyplot as plt

AMPLITUDES = [(30, 30), (120, 120), (180, 180)]
SOLVERS = ["temporal_cv", "spatial_cv", "lambda_map", "sure"]
COLORS = ["cyan", "skyblue", "orange", "green"]

if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey="row")

    results = joblib.load("../data/tp_fp_results.pkl")

    for i, amplitude in enumerate(AMPLITUDES):
        for j, solver in enumerate(SOLVERS):
            (fp_sum, tp_sum) = results[i][solver]
            fp_sum, tp_sum = fp_sum, tp_sum
            print(fp_sum)
            print(tp_sum)
            axes[i].scatter(fp_sum, tp_sum, c=COLORS[j], label=SOLVERS[j])

        axes[i].set_xlabel(r"$\delta$-FP")
        axes[i].set_ylabel(r"$\delta$-TP")
        axes[i].set_title("Amplitude: %s nAm" % amplitude[0])

    plt.suptitle("FP - TP in low/medium/high SNR regime")
    plt.tight_layout()
    plt.legend()
    plt.show()
