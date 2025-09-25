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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from __future__ import annotations

import pickle  # noqa: F401
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # noqa: F401

# from SALib.analyze import sobol
# from SALib.sample import saltelli

base_path = Path("projects/hetgp/sensitivity")

# %%
filenameindex = ["4.8_28.8", "28.8_73.6", "73.6_165.0", "165.0_400.0"]


# %%
def plot(HF, HetGP, HFerr, HetGPerr, plotname) -> None:
    # Some example data to plot
    n_points = len(HF)

    HFerr = HFerr / HF.sum()
    HetGPerr = HetGPerr / HetGP.sum()

    HF = HF / HF.sum()
    HetGP = HetGP / HetGP.sum()

    # Define the overall figure layout
    fig, (ax) = plt.subplots(
        1,
        1,
        figsize=(10, 6),
        sharex=True,
    )

    # Based on https://stackoverflow.com/a/59421062 and https://matplotlib.org/3.0.0/gallery/statistics/barchart_demo.html
    # Need to define an index, which we will then shift as needed to plot each bar
    index = np.arange(n_points)
    width = 0.15

    # Plot in the main panel
    # Group A
    ax.bar(index, HF, width, yerr=HFerr, label="HF Model")
    # Group B: shift the next bar over so they sit side-by-side
    ax.bar(index + width, HetGP, width, yerr=HetGPerr, label="HetGP Model")
    ax.set_ylim([0, 1])

    ax.legend(loc="upper right", frameon=False)

    # It shouldn't hurt to align the labels if there's only one.
    fig.align_ylabels()

    # Adjust the layout.
    # NOTE: The subplots adjust needs to be after tight_layout to remove the inter-axis spacing
    fig.tight_layout()
    adjust_args = {
        # Remove spacing between subplots
        "hspace": 0,
        "wspace": 0,
        # Can manually adjust/reduce the spacing around the edges if desired - just uncomment below.
        # "left": 0.10,
        # "bottom": 0.08,
        # "right": 0.98,
        # "top": 0.98,
    }
    fig.subplots_adjust(**adjust_args)

    fig.savefig(plotname)
    plt.close(fig)


# %%
# global total index
for index in range(len(filenameindex)):
    filename_HF = base_path / "HFGP/Hadron_aggregate/Global_TotalIndex_" + filenameindex[index] + ".csv"
    HFdatabyaggbin = pd.read_csv(filename_HF)
    HF = np.array(HFdatabyaggbin["STagg"])
    HFerr = np.array(HFdatabyaggbin["ST_confagg"])
    filename_HetGP = base_path / "HetGP/Hadron_aggregate/Global_TotalIndex_" + filenameindex[index] + ".csv"
    HetGPdatabyaggbin = pd.read_csv(filename_HetGP)
    HetGP = np.array(HetGPdatabyaggbin["STagg"])
    HetGPerr = np.array(HetGPdatabyaggbin["ST_confagg"])
    plotname = base_path / "figures/Hadron_Global_TotalIndex_aggbin_" + filenameindex[index] + ".pdf"
    plot(HF, HetGP, HFerr, HetGPerr, plotname)

# %%
# local 5-95 total index
for index in range(len(filenameindex)):
    filename_HF = base_path / "HFGP/Hadron_aggregate/Local_5_95_TotalIndex_" + filenameindex[index] + ".csv"
    HFdatabyaggbin = pd.read_csv(filename_HF)
    HF = np.array(HFdatabyaggbin["STagg"])
    HFerr = np.array(HFdatabyaggbin["ST_confagg"])
    filename_HetGP = base_path / "HetGP/Hadron_aggregate/Local_5_95_TotalIndex_" + filenameindex[index] + ".csv"
    HetGPdatabyaggbin = pd.read_csv(filename_HetGP)
    HetGP = np.array(HetGPdatabyaggbin["STagg"])
    HetGPerr = np.array(HetGPdatabyaggbin["ST_confagg"])
    plotname = base_path / "figures/Hadron_Local_5_95_TotalIndex_aggbin_" + filenameindex[index] + ".pdf"
    plot(HF, HetGP, HFerr, HetGPerr, plotname)

# %%
# local 1-99 total index
for index in range(len(filenameindex)):
    filename_HF = base_path / "HFGP/Hadron_aggregate/Local_1_99_TotalIndex_" + filenameindex[index] + ".csv"
    HFdatabyaggbin = pd.read_csv(filename_HF)
    HF = np.array(HFdatabyaggbin["STagg"])
    HFerr = np.array(HFdatabyaggbin["ST_confagg"])
    filename_HetGP = base_path / "HetGP/Hadron_aggregate/Local_1_99_TotalIndex_" + filenameindex[index] + ".csv"
    HetGPdatabyaggbin = pd.read_csv(filename_HetGP)
    HetGP = np.array(HetGPdatabyaggbin["STagg"])
    HetGPerr = np.array(HetGPdatabyaggbin["ST_confagg"])
    plotname = base_path / "figures/Hadron_Local_1_99_TotalIndex_aggbin_" + filenameindex[index] + ".pdf"
    plot(HF, HetGP, HFerr, HetGPerr, plotname)
