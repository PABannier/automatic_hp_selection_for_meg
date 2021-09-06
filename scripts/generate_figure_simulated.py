import joblib

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from seaborn import color_palette
import pandas as pd

from celer.plot_utils import configure_plt
from mtl.utils_plot import _plot_legend_apart

configure_plt()

label_mapping = {
    "lambda_map": r"$\lambda$-MAP",
    "spatial_cv": "Spatial CV",
    "sure": "SURE"
}

fontsize = 18
fontsize2 = 15
lw = 1
savefig = True

mpl.rcParams["xtick.labelsize"] = fontsize
mpl.rcParams["ytick.labelsize"] = fontsize
dict_colors = color_palette("colorblind")

data = joblib.load("../data/experiment_results_lower_extent.pkl")
full_df = pd.DataFrame(data)

full_df = full_df[full_df["solver"].isin(["sure", "spatial_cv", "lambda_map"])]

print("============ SPEED ===========")

print(full_df.groupby(by="solver")["duration"].mean())

print("=" * 30)

fig, axarr = plt.subplots(1, 3, sharey="row", figsize=(14, 2))

for idx_solver, solver in enumerate(["lambda_map", "spatial_cv", "sure"]):
    df_solver = full_df[full_df["solver"] == solver]
    solver_label = label_mapping[solver]

    axarr[0].plot(df_solver["amplitude"], df_solver["recall"],
                  color=dict_colors[idx_solver], label=solver_label, lw=lw)
    axarr[1].plot(df_solver["amplitude"], df_solver["delta_precision"],
                  color=dict_colors[idx_solver], label=solver_label, lw=lw)
    axarr[2].plot(df_solver["amplitude"], df_solver["delta_f1_score"],
                  color=dict_colors[idx_solver], label=solver_label, lw=lw)

axarr[0].set_xlabel("Source amplitude (nAm)", fontsize=fontsize2)
axarr[1].set_xlabel("Source amplitude (nAm)", fontsize=fontsize2)
axarr[2].set_xlabel("Source amplitude (nAm)", fontsize=fontsize2)

axarr[0].set_title("Recall", fontsize=fontsize)
axarr[1].set_title(r"$\delta$-precision", fontsize=fontsize)
axarr[2].set_title(r"$\delta$-F1", fontsize=fontsize)

plt.tight_layout()
fig.show()

OUT_PATH = f"../figures/simulated_comparison"

if savefig:
    fig.savefig(OUT_PATH + ".pdf")
    fig.savefig(OUT_PATH + ".svg")
    _plot_legend_apart(axarr[0], OUT_PATH + "_legend.pdf")
    print("Figure saved.")
