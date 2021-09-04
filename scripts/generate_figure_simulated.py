# %%
import joblib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% 
data = joblib.load("../data/experiment_results.pkl")
data_v2 = joblib.load("../data/experiment_results_v2.pkl")
data_v3 = joblib.load("../data/experiment_results_v3.pkl")
# %%
df = pd.DataFrame(data)
df = df[df["solver"].isin(["temporal_cv", "spatial_cv"])]

df2 = pd.DataFrame(data_v2)
df3 = pd.DataFrame(data_v3)

full_df = pd.concat([df, df2, df3]).reset_index(drop=True)

# %%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 10), sharex="col")
fig.suptitle("SNR vs Summary statistics")

for solver in ["lambda_map", "spatial_cv", "temporal_cv", "sure"]:
    df_solver = full_df[full_df["solver"] == solver]

    ax1.plot(df_solver["amplitude"], df_solver["recall"], label=solver)
    ax2.plot(df_solver["amplitude"], df_solver["delta_precision"], label=solver)
    ax3.plot(df_solver["amplitude"], df_solver["delta_f1_score"], label=solver)
    ax4.plot(df_solver["amplitude"], df_solver["emd"], label=solver)

ax1.set_title("Recall")
ax2.set_title(r"$\delta$-precision")
ax3.set_title(r"$\delta$-F1")
ax4.set_title("EMD")

ax4.set_xlabel("Source amplitude (nAm)")

plt.legend()

# %%
# The case where amplitude = 10 is very long. We remove it as it is unrealistic to have
# such noisy signals and prefer to compute the mean of the remaining trials.
full_df[full_df["amplitude"] > 10].groupby(by=["solver"])["duration"].mean().plot(kind="bar", title="Average duration per solver (in s)")
