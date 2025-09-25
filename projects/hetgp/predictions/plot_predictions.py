# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv-3.12
#     language: python
#     name: python3
# ---

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

base_path = Path("projects/hetgp/predictions")

# %%
budget_list = np.array([3500, 4000, 4500, 5000, 5500])

# %% [markdown]
# Hadron - HetGP

# %%
hetGPMSE = pd.read_csv(base_path / "Hadron_HETGP_Prediction_by_bin.csv")
hetGPMSE

# %% [markdown]
# Hadron - HFGP

# %%
HFGPMSE = pd.read_csv(base_path / "Hadron_HFGP_Prediction_by_bin.csv")
HFGPMSE

# %%
sum_HFGPMSE = HFGPMSE.sum()
sum_hetGPMSE = hetGPMSE.sum()
compare_df = pd.DataFrame({"HF": sum_HFGPMSE, "HetGP": sum_hetGPMSE})

# Plot using line plot
compare_df.plot(kind="line", marker="o", figsize=(8, 5))
plt.title("MSE (sum over bins) by budget")
plt.xlabel("Budget")
plt.ylabel("Sum of MSE")
plt.grid(True)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %%
# Melt both DataFrames and add a 'Source' label
HFGPMSE_melted = HFGPMSE.melt(var_name="Variable", value_name="Value")
HFGPMSE_melted["Source"] = "HF"

hetGPMSE_melted = hetGPMSE.melt(var_name="Variable", value_name="Value")
hetGPMSE_melted["Source"] = "HetGP"

# Combine both for plotting
df_combined = pd.concat([HFGPMSE_melted, hetGPMSE_melted])

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x="Variable", y="Value", hue="Source", data=df_combined)
plt.title("MSE by pt bins")
plt.xlabel("Budget")
plt.ylabel("MSE")
plt.legend(title="DataFrame")
plt.show()

# %% [markdown]
# Jet - HetGP

# %%
hetGPMSE = pd.read_csv(base_path / "Jet_HETGP_Prediction_by_bin.csv")
hetGPMSE

# %% [markdown]
# Jet - HFGP

# %%
HFGPMSE = pd.read_csv(base_path / "Jet_HFGP_Prediction_by_bin.csv")
HFGPMSE

# %%
sum_HFGPMSE = HFGPMSE.sum()
sum_hetGPMSE = hetGPMSE.sum()
compare_df = pd.DataFrame({"HF": sum_HFGPMSE, "HetGP": sum_hetGPMSE})

# Plot using line plot
compare_df.plot(kind="line", marker="o", figsize=(8, 5))
plt.title("MSE (sum over bins) by budget")
plt.xlabel("Budget")
plt.ylabel("Sum of MSE")
plt.grid(True)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %%
# Melt both DataFrames and add a 'Source' label
HFGPMSE_melted = HFGPMSE.melt(var_name="Variable", value_name="Value")
HFGPMSE_melted["Source"] = "HF"

hetGPMSE_melted = hetGPMSE.melt(var_name="Variable", value_name="Value")
hetGPMSE_melted["Source"] = "HetGP"

# Combine both for plotting
df_combined = pd.concat([HFGPMSE_melted, hetGPMSE_melted])

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x="Variable", y="Value", hue="Source", data=df_combined)
plt.title("MSE by pt bins")
plt.xlabel("Budget")
plt.ylabel("MSE")
plt.legend(title="DataFrame")
plt.show()

# %%
hetGPMSE_melted

# %%
