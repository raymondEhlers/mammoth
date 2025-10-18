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

import pickle  # noqa: F401
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot as pb
import pandas as pd
import seaborn as sns  # noqa: F401

# from SALib.analyze import sobol
# from SALib.sample import saltelli

base_path = Path("projects/hetgp/sensitivity")

pb.configure()

# %%
filenameindex = ["4.8_28.8", "28.8_73.6", "73.6_165.0", "165.0_400.0"]

text_font_size = 22


# %%
def reorder_data(df: pd.DataFrame) -> pd.DataFrame:
    input_data_index_to_label = {
        0: "alpha_s",
        1: "Q0",
        2: "C1",
        3: "C2",
        4: "tau0",
        5: "C3",
    }
    new_order = ["alpha_s", "Q0", "tau0", "C1", "C2", "C3"]

    df.index = df.index.map(input_data_index_to_label)
    return df.loc[new_order]


# %%
# Original function that RJE provided to Irene
def plot_original(HF, HetGP, HFerr, HetGPerr, plotname) -> None:
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
    ax.bar(index, HF, width, yerr=HFerr, label="High fidelity GP")
    # Group B: shift the next bar over so they sit side-by-side
    ax.bar(index + width, HetGP, width, yerr=HetGPerr, label="VarP-GP")
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


label_to_display_label = {
    "alpha_s": r"$\alpha_{\text{s}}$",
    "Q0": r"$Q_{0}$",
    "tau0": r"$\tau_{0}$",
    "C1": "C1",
    "C2": "C2",
    "C3": "C3",
}
colors = {"hetgp": "#FF8301", "HF": "#845cba"}


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
    ax.bar(index, HF, width, yerr=HFerr, label="High fidelity GP", color=colors["HF"])
    # Group B: shift the next bar over so they sit side-by-side
    ax.bar(index + width, HetGP, width, yerr=HetGPerr, label="VarP-GP", color=colors["hetgp"])
    ax.set_ylim([0, 1])
    ax.xaxis.set_ticks(range(len(label_to_display_label)))
    ax.xaxis.set_ticklabels(label_to_display_label.values())
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Sobol' index")

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


def plot_new(
    HF,
    HetGP,
    HFerr,
    HetGPerr,
    plot_config: pb.PlotConfig,
):
    # Setup for data
    n_points = len(HF)

    HFerr = HFerr / HF.sum()
    HetGPerr = HetGPerr / HetGP.sum()

    HF = HF / HF.sum()
    HetGP = HetGP / HetGP.sum()

    # Based on https://stackoverflow.com/a/59421062 and https://matplotlib.org/3.0.0/gallery/statistics/barchart_demo.html
    # Need to define an index, which we will then shift as needed to plot each bar
    index = np.arange(n_points)
    width = 0.20

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, 6.25),
        sharex=True,
    )

    # Plot in the main panel
    # Group A
    ax.bar(index - width / 2, HF, width, yerr=HFerr, label="High fidelity GP", color=colors["HF"])
    # Group B: shift the next bar over so they sit side-by-side
    ax.bar(index + width / 2, HetGP, width, yerr=HetGPerr, label="VarP-GP", color=colors["hetgp"])
    ax.set_ylim([0, 1])
    ax.xaxis.set_ticks(range(len(label_to_display_label)))
    ax.xaxis.set_ticklabels(label_to_display_label.values())

    ax.axhline(y=1.0 / 6.0, color="black", linestyle="dashed", zorder=0)

    plot_config.apply(fig, ax=ax)

    _output_path = base_path / "figures"
    _output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(_output_path / f"{plot_config.name}.pdf")
    plt.close(fig)


# %%
# global total index
for index in range(len(filenameindex)):
    filename_HF = base_path / ("HFGP/hadron_aggregate/Global_TotalIndex_" + filenameindex[index] + ".csv")
    HFdatabyaggbin = reorder_data(pd.read_csv(filename_HF))
    HF = np.array(HFdatabyaggbin["STagg"])
    HFerr = np.array(HFdatabyaggbin["ST_confagg"])
    filename_HetGP = base_path / ("HetGP/hadron_aggregate/Global_TotalIndex_" + filenameindex[index] + ".csv")
    HetGPdatabyaggbin = reorder_data(pd.read_csv(filename_HetGP))
    HetGP = np.array(HetGPdatabyaggbin["STagg"])
    HetGPerr = np.array(HetGPdatabyaggbin["ST_confagg"])
    plotname = base_path / ("figures/hadron_Global_TotalIndex_aggbin_" + filenameindex[index] + ".pdf")

    # I considered including everything here (e.g. sqrt_s), but it doesn't matter overly much
    # for the purposes of this exercise. To just highlight the important information, I'm going to cut down to the minimal.
    hadron_low, hadron_high = map(float, filenameindex[index].split("_"))
    text = "Global Sobol' sensitivity"
    text += r", trained on JETSCAPE (MATTER + LBT)"
    text += "\n" + r"corresponding to CMS, $\textit{JHEP 04 (2017) 039}$"
    text += "\n" + r"0-5\%, Hadron $R_{\text{AA}}$, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$"
    text += "\n" + rf"${hadron_low:g} < p_{{\text{{T}}}} < {hadron_high:g}\:\text{{GeV}}/c$"
    plot_config = pb.PlotConfig(
        name=f"sensitivity_hadron_global_{filenameindex[index]}",
        panels=[
            # Main panel
            pb.Panel(
                axes=[
                    pb.AxisConfig(
                        "x",
                        label="Parameter",
                        font_size=text_font_size,
                        use_major_axis_multiple_locator_with_base=1,
                    ),
                    pb.AxisConfig(
                        "y",
                        label="Relative Sobol' index",
                        font_size=text_font_size,
                        range=(0, 1),
                    ),
                ],
                text=[
                    pb.TextConfig(x=0.95, y=0.81, text=text, font_size=18),
                    pb.TextConfig(x=0.99, y=0.17, text="Equal sensitivity", font_size=18),
                ],
                legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
            ),
        ],
        figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.11}),
    )

    # plot(HF, HetGP, HFerr, HetGPerr, plotname)
    plot_new(HF, HetGP, HFerr, HetGPerr, plot_config=plot_config)

# %%
HetGPdatabyaggbin

# %%
# local 5-95 total index
for index in range(len(filenameindex)):
    filename_HF = base_path / ("HFGP/hadron_aggregate/Local_5_95_TotalIndex_" + filenameindex[index] + ".csv")
    HFdatabyaggbin = pd.read_csv(filename_HF)
    HF = np.array(HFdatabyaggbin["STagg"])
    HFerr = np.array(HFdatabyaggbin["ST_confagg"])
    filename_HetGP = base_path / ("HetGP/hadron_aggregate/Local_5_95_TotalIndex_" + filenameindex[index] + ".csv")
    HetGPdatabyaggbin = pd.read_csv(filename_HetGP)
    HetGP = np.array(HetGPdatabyaggbin["STagg"])
    HetGPerr = np.array(HetGPdatabyaggbin["ST_confagg"])
    plotname = base_path / ("figures/Hadron_Local_5_95_TotalIndex_aggbin_" + filenameindex[index] + ".pdf")

    # I considered including everything here (e.g. sqrt_s), but it doesn't matter overly much
    # for the purposes of this exercise. To just highlight the important information, I'm going to cut down to the minimal.
    hadron_low, hadron_high = map(float, filenameindex[index].split("_"))
    text = r"5-95\% MAP Sobol' sensitivity"
    text += r", trained on JETSCAPE (MATTER + LBT)"
    text += "\n" + r"corresponding to CMS, $\textit{JHEP 04 (2017) 039}$"
    text += "\n" + r"0-5\%, Hadron $R_{\text{AA}}$, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$"
    text += "\n" + rf"${hadron_low:g} < p_{{\text{{T}}}} < {hadron_high:g}\:\text{{GeV}}/c$"
    plot_config = pb.PlotConfig(
        name=f"sensitivity_hadron_5_95_{filenameindex[index]}",
        panels=[
            # Main panel
            pb.Panel(
                axes=[
                    pb.AxisConfig(
                        "x",
                        label="Parameter",
                        font_size=text_font_size,
                        use_major_axis_multiple_locator_with_base=1,
                    ),
                    pb.AxisConfig(
                        "y",
                        label="Relative Sobol' index",
                        font_size=text_font_size,
                        range=(0, 1),
                    ),
                ],
                text=[
                    pb.TextConfig(x=0.95, y=0.81, text=text, font_size=18),
                    pb.TextConfig(x=0.99, y=0.17, text="Equal sensitivity", font_size=18),
                ],
                legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
            ),
        ],
        figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.11}),
    )

    # plot(HF, HetGP, HFerr, HetGPerr, plotname)
    plot_new(HF, HetGP, HFerr, HetGPerr, plot_config=plot_config)

# %%
# local 1-99 total index
for index in range(len(filenameindex)):
    filename_HF = base_path / ("HFGP/hadron_aggregate/Local_1_99_TotalIndex_" + filenameindex[index] + ".csv")
    HFdatabyaggbin = pd.read_csv(filename_HF)
    HF = np.array(HFdatabyaggbin["STagg"])
    HFerr = np.array(HFdatabyaggbin["ST_confagg"])
    filename_HetGP = base_path / ("HetGP/hadron_aggregate/Local_1_99_TotalIndex_" + filenameindex[index] + ".csv")
    HetGPdatabyaggbin = pd.read_csv(filename_HetGP)
    HetGP = np.array(HetGPdatabyaggbin["STagg"])
    HetGPerr = np.array(HetGPdatabyaggbin["ST_confagg"])
    plotname = base_path / ("figures/Hadron_Local_1_99_TotalIndex_aggbin_" + filenameindex[index] + ".pdf")

    # I considered including everything here (e.g. sqrt_s), but it doesn't matter overly much
    # for the purposes of this exercise. To just highlight the important information, I'm going to cut down to the minimal.
    hadron_low, hadron_high = map(float, filenameindex[index].split("_"))
    text = r"1-99\% MAP sobol sensitivity"
    text += r", trained on JETSCAPE (MATTER + LBT)"
    text += "\n" + r"corresponding to CMS, $\textit{JHEP 04 (2017) 039}$"
    text += "\n" + r"0-5\%, Hadron $R_{\text{AA}}$, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$"
    text += "\n" + rf"${hadron_low:g} < p_{{\text{{T}}}} < {hadron_high:g}\:\text{{GeV}}/c$"
    plot_config = pb.PlotConfig(
        name=f"sensitivity_hadron_1_99_{filenameindex[index]}",
        panels=[
            # Main panel
            pb.Panel(
                axes=[
                    pb.AxisConfig(
                        "x",
                        label="Parameter",
                        font_size=text_font_size,
                        use_major_axis_multiple_locator_with_base=1,
                    ),
                    pb.AxisConfig(
                        "y",
                        label="Sobol sensitivity",
                        font_size=text_font_size,
                        range=(0, 1),
                    ),
                ],
                text=[
                    pb.TextConfig(x=0.95, y=0.81, text=text, font_size=18),
                    pb.TextConfig(x=0.99, y=0.21, text="Equal sensitivity", font_size=18),
                ],
                legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
            ),
        ],
        figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.11}),
    )

    # plot(HF, HetGP, HFerr, HetGPerr, plotname)
    plot_new(HF, HetGP, HFerr, HetGPerr, plot_config=plot_config)

    # plot(HF, HetGP, HFerr, HetGPerr, plotname)

# %%
