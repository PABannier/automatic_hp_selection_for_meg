# %%
import pandas as pd
import joblib


CRITERION = "lambda_map"

SELECTED_SUBJECTS = ['CC120120', 'CC110319', 'CC110069', 'CC110037', 'CC110033', 
                     'CC110126', 'CC110182', 'CC120061', 'CC110101', 'CC110087', 
                     'CC110045', 'CC120182', 'CC112141', 'CC110411', 'CC120319', 
                     'CC120376', 'CC110187', 'CC120218', 'CC120640', 'CC120347', 
                     'CC120313', 'CC120264', 'CC120462', 'CC120727', 'CC121144', 
                     'CC121317', 'CC121158']
SELECTED_SUBJECTS = ["sub-" + x for x in SELECTED_SUBJECTS]

report_data = joblib.load(f"../data/camcan/report_{CRITERION}.pkl")

# %% 
df = pd.DataFrame(report_data, columns=["patient_id", "reg", "explained_var", "num_left_act", "num_right_act"])
df = df[df["patient_id"].isin(SELECTED_SUBJECTS)]

# %%
df["num_left_act"] = df["num_left_act"].apply(lambda x: len(x))
df["num_right_act"] = df["num_right_act"].apply(lambda x: len(x))

df["total_act"] = df["num_left_act"] + df["num_right_act"]
df["explained_var"] = 1 - df["explained_var"] 

if CRITERION == "sure":
    df["reg"] *= 0.01

# %%
print(f"RECAPITULATIVE TABLE - {CRITERION.upper()}")
print("#" * 30)
print("Average reg: %.2f" % df["reg"].mean())
print("Average explained var: %.2f" % df["explained_var"].mean())
print("Average number of activations: %.2f" % df["total_act"].mean())
print("\n")
print("#" * 30)
print("% of zero activation: ", round(len(df[df["total_act"] == 0]) / len(df) * 100, 2))
print("% of one activation: ", round(len(df[df["total_act"] == 1]) / len(df) * 100, 2))
print("% of two activations: ", round(len(df[df["total_act"] == 2]) / len(df) * 100, 2))
print("% of > 2 activations: ", round(len(df[df["total_act"] > 2]) / len(df) * 100, 2))

TEX_TEMPLATE = """
    %!TEX root = ../neurips_2021.tex

    \begin{table}
    \caption{Agregated results - CamCan}
    \label{tab:camcan_res}
    \begin{tabular}{ p{3cm}|p{3cm} p{3cm} p{3cm}  }
    Average metrics & Spatial CV & SURE & $\lambda$-MAP\
    \hline
    $\lambda / \lambda_{\text{max}}$   & 0.30 & 0.61 & 0.9 \
    Explained variance &   0.67  & 0.32 & 0.06 \
    # of sources & 9.30 & 1.44 & 1.3 \
    \% of zero sources    & 0 & 0 & 29.63\
    \% of one source &   0  & 55.56 & 40.74\
    \% of two sources &   3.7  & 44.44 & 18.52\
    \% of $> 2$ sources & 96.3  & 0.0 & 11.11\
    %  \hline
    \label{tab:camcan_res}
    \end{tabular}
"""
