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

import copy
from pathlib import Path

import attrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pachyderm.plot as pb
import pandas as pd
import seaborn as sns  # noqa: F401

# from SALib.analyze import sobol
# from SALib.sample import saltelli

base_path = Path("projects/hetgp/sensitivity")

pb.configure()

# %% [markdown]
# # Setup

# %%
hadron_pt_bin_labels = ["4.8_28.8", "28.8_73.6", "73.6_165.0", "165.0_400.0"]
jet_pt_bin_labels = ["100.0_177.0", "177.0_281.0", "281.0_999.0"]

text_font_size = 22


# %%
@attrs.define
class Data:
    hetgp: npt.NDArray[np.float64]
    hetgp_err: npt.NDArray[np.float64]
    hf: npt.NDArray[np.float64]
    hf_err: npt.NDArray[np.float64]

    def normalize(self) -> None:
        hetgp_sum = self.hetgp.sum()
        self.hetgp /= hetgp_sum
        self.hetgp_err /= hetgp_sum

        hf_sum = self.hf.sum()
        self.hf /= hf_sum
        self.hf_err /= hf_sum


_possible_range_labels = ["Global", "Local_5_95", "Local_1_99"]


def load_data(pt_bin_labels: list[float], range_label: str, load_hadron_data: bool) -> dict[str, Data]:
    # Validation
    if range_label not in _possible_range_labels:
        msg = f"Range label {range_label} not in available labels: {_possible_range_labels}. Please check your inputs."
        raise ValueError(msg)

    output = {}
    for pt_label in pt_bin_labels:
        observable_label = "hadron" if load_hadron_data else "jet"
        filename_HF = base_path / f"HFGP/{observable_label}_aggregate/{range_label}_TotalIndex_{pt_label}.csv"
        HFdatabyaggbin = pd.read_csv(filename_HF)
        filename_HetGP = base_path / f"HetGP/{observable_label}_aggregate/{range_label}_TotalIndex_{pt_label}.csv"
        HetGPdatabyaggbin = pd.read_csv(filename_HetGP)

        output[pt_label] = Data(
            hetgp=np.array(HetGPdatabyaggbin["STagg"]),
            hetgp_err=np.array(HetGPdatabyaggbin["ST_confagg"]),
            hf=np.array(HFdatabyaggbin["STagg"]),
            hf_err=np.array(HFdatabyaggbin["ST_confagg"]),
        )

    return output


# %%
hadron_data = {
    "global": load_data(pt_bin_labels=hadron_pt_bin_labels, range_label="Global", load_hadron_data=True),
    "5_95": load_data(pt_bin_labels=hadron_pt_bin_labels, range_label="Local_5_95", load_hadron_data=True),
    "1_99": load_data(pt_bin_labels=hadron_pt_bin_labels, range_label="Local_1_99", load_hadron_data=True),
}
jet_data = {
    "global": load_data(pt_bin_labels=jet_pt_bin_labels, range_label="Global", load_hadron_data=False),
    "5_95": load_data(pt_bin_labels=jet_pt_bin_labels, range_label="Local_5_95", load_hadron_data=False),
    "1_99": load_data(pt_bin_labels=jet_pt_bin_labels, range_label="Local_1_99", load_hadron_data=False),
}


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


# Plot parameters
label_to_display_label = {
    "alpha_s": r"$\alpha_{\text{s}}$",
    "Q0": r"$Q_{0}$",
    "tau0": r"$\tau_{0}$",
    "C1": "C1",
    "C2": "C2",
    "C3": "C3",
}
# NOTE: These are matching with those colors used in plot_predictions
colors = {"hetgp": "#FF8301", "HF": "#845cba"}


# %% [markdown]
# # Plots


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


def plot_parameters_on_ax(
    hetgp: npt.NDArray[np.float64],
    hetgp_err: npt.NDArray[np.float64],
    hf: npt.NDArray[np.float64],
    hf_err: npt.NDArray[np.float64],
    ax: mpl.axes.Axes,
    bar_width: float = 0.2,
) -> None:
    """Plot all parameters on an axis.

    This assumes the inputs are already properly normalized.
    """
    # Setup data
    n_points = len(hetgp)
    # Based on https://stackoverflow.com/a/59421062 and https://matplotlib.org/3.0.0/gallery/statistics/barchart_demo.html
    # Need to define an index, which we will then shift as needed to plot each bar
    index = np.arange(n_points)

    # The first is shifted left
    ax.bar(index - bar_width / 2, hf, bar_width, yerr=hf_err, label="High fidelity GP", color=colors["HF"])
    # The second is shifted right
    ax.bar(index + bar_width / 2, hetgp, bar_width, yerr=hetgp_err, label="VarP-GP", color=colors["hetgp"])

    # And then ensure we show the parameter names on the x-axis
    ax.xaxis.set_ticks(range(len(label_to_display_label)))
    ax.xaxis.set_ticklabels(label_to_display_label.values())

    # And then a line to mark equal sensitivity
    # NOTE: (it relies on the plot_config for the label)
    ax.axhline(y=1.0 / 6.0, color="black", linestyle="dashed", zorder=0)


def plot_compare_hetgp_hf(
    hetgp: npt.NDArray[np.float64],
    hetgp_err: npt.NDArray[np.float64],
    hf: npt.NDArray[np.float64],
    hf_err: npt.NDArray[np.float64],
    plot_config: pb.PlotConfig,
) -> None:
    """Individual stand-alone figure for one set of data."""
    # Normalize errors and data
    hf_err = hf_err / hf.sum()
    hetgp_err = hetgp_err / hetgp.sum()
    hf = hf / hf.sum()
    hetgp = hetgp / hetgp.sum()

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, 6.25),
        sharex=True,
    )

    # Plot in the main panel
    plot_parameters_on_ax(hetgp=hetgp, hetgp_err=hetgp_err, hf=hf, hf_err=hf_err, ax=ax)

    plot_config.apply(fig, ax=ax)

    _output_path = base_path / "figures"
    _output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(_output_path / f"{plot_config.name}.pdf")
    plt.close(fig)


def plot_compare_hetgp_hf_hadron(
    hadron_data_by_pt: dict[str, Data],
    plot_config: pb.PlotConfig,
) -> None:
    """Specific comparison function for hadron pt comparison combined into a single figure.

    NOTE:
        There's a special function for it since it will have a different axis configuration
        than jets due to different numbers of pt bins
    """
    # Normalize errors and data
    # NOTE: We need to copy the data first so our normalization doesn't impact the existing data
    hadron_data_by_pt = {k: copy.deepcopy(v) for k, v in hadron_data_by_pt.items()}
    for v in hadron_data_by_pt.values():
        v.normalize()

    # Setup axes with a header where we'll put shared information
    # Setup
    # We start with a standard grid, and then we'll modify it to define a header.
    # This is quite nice because we can utilize gridspec when necessary, but skip over
    # the complications of it when we don't need it.
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(10, 8),
        gridspec_kw={"height_ratios": [0.5, 6, 6]},
        sharex="col",
        sharey="row",
    )
    # According to gpt, all the axes will return the same gridspec.
    gs = axes[0, 0].get_gridspec()
    # Remove the underlying axes
    for _ax in axes[0, :]:
        _ax.remove()
    ax_header = fig.add_subplot(gs[0, :])

    # Plot in the main panel
    for data, ax in zip(hadron_data_by_pt.values(), axes[1:, :].flatten(), strict=True):
        plot_parameters_on_ax(hetgp=data.hetgp, hetgp_err=data.hetgp_err, hf=data.hf, hf_err=data.hf_err, ax=ax)

    plot_config.apply(fig, axes=[ax_header, *axes[1:, :].flatten()])
    # And remove the axis header
    ax_header.set_axis_off()

    _output_path = base_path / "figures"
    _output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(_output_path / f"{plot_config.name}.pdf")
    plt.close(fig)


# %% [markdown]
# ## Hadron, combined sensitivity

# %%
# Hadron, combined figure
text_font_size = 22
in_figure_font_size = 20

for selected_space in ["global", "posterior"]:
    # Labeling
    # The miniaml text contains only the most important text
    # Everything else (e.g. what is not critical to distringuishing the plots) is kept in the header.
    # I skipped that it was trained on JETSCAPE (MATTER+LBT) - it's bulky, and never varies.
    text = "Norm. total-effect Sobol' index"
    if selected_space == "global":
        text += "\n" + "Full design space"
    else:
        text += "\n" + r"1-99\% posterior"
    header_text = "\n" + r"Emulated: Hadron $R_{\text{AA}}$ in 0-5\% Pb-Pb, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$,"
    header_text += " " + r"CMS, $\textit{JHEP 04 (2017) 039}$"

    pt_labels = []
    for k in hadron_data["global"]:
        hadron_low, hadron_high = map(float, k.split("_"))
        pt_labels.append(rf"${hadron_low:g} < p_{{\text{{T}}}} < {hadron_high:g}\:\text{{GeV}}/c$")

    plot_config = pb.PlotConfig(
        name=f"sensitivity_hadron_{selected_space}_combined",
        panels=[
            # Header
            pb.Panel(
                axes=[],
                text=pb.TextConfig(x=0.01, y=2.0, text=header_text, font_size=18),
            ),
            # Main panels
            pb.Panel(
                axes=[
                    pb.AxisConfig(
                        "y",
                        label="Norm. TE Sobol' index",
                        font_size=text_font_size,
                        range=(0, 1),
                    ),
                ],
                text=[
                    pb.TextConfig(x=0.95, y=0.80, text=text, font_size=text_font_size),
                    pb.TextConfig(x=0.95, y=0.95, text=pt_labels[0], font_size=in_figure_font_size),
                ],
                # legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
            ),
            pb.Panel(
                axes=[
                    pb.AxisConfig(
                        "y",
                        range=(0, 1),
                    ),
                ],
                text=[
                    pb.TextConfig(x=0.95, y=0.95, text=pt_labels[1], font_size=in_figure_font_size),
                    pb.TextConfig(x=0.98, y=0.19, text="Equal sensitivity", font_size=18, text_kwargs={"zorder": 0}),
                ],
                legend=pb.LegendConfig(
                    location="upper right", anchor=(0.95, 0.80), font_size=text_font_size, marker_label_spacing=0.3
                ),
            ),
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
                        label="Norm. TE Sobol' index",
                        font_size=text_font_size,
                        range=(0, 0.995),
                    ),
                ],
                text=pb.TextConfig(x=0.95, y=0.95, text=pt_labels[2], font_size=in_figure_font_size),
                # legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
            ),
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
                        range=(0, 0.995),
                    ),
                ],
                text=[
                    pb.TextConfig(x=0.95, y=0.95, text=pt_labels[3], font_size=in_figure_font_size),
                    pb.TextConfig(x=0.98, y=0.19, text="Equal sensitivity", font_size=18, text_kwargs={"zorder": 0}),
                ],
                # legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
            ),
        ],
        figure=pb.Figure(edge_padding={"left": 0.09, "bottom": 0.09}),
    )

    # plot(HF, HetGP, HFerr, HetGPerr, plotname)
    plot_compare_hetgp_hf_hadron(
        hadron_data_by_pt=hadron_data["global" if selected_space == "global" else "1_99"], plot_config=plot_config
    )


# %% [markdown]
# ## Jet, combined sensitivity


# %%
def plot_compare_hetgp_hf_jet(
    jet_data_by_pt: dict[str, Data],
    plot_config: pb.PlotConfig,
) -> None:
    """Specific comparison function for jet pt comparison combined into a single figure.

    NOTE:
        There's a special function for it since it will have a different axis configuration
        than jets due to different numbers of pt bins
    """
    # Normalize errors and data
    # NOTE: We need to copy the data first so our normalization doesn't impact the existing data
    jet_data_by_pt = {k: copy.deepcopy(v) for k, v in jet_data_by_pt.items()}
    for v in jet_data_by_pt.values():
        v.normalize()

    # Setup axes with a header where we'll put shared information
    # Setup
    # We start with a standard grid, and then we'll modify it to define a header.
    # This is quite nice because we can utilize gridspec when necessary, but skip over
    # the complications of it when we don't need it.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8, 8),
        gridspec_kw={"height_ratios": [6, 6]},
        # sharex="col",
        sharex="none",
        sharey="row",
    )
    # NOTE: Will use the last ax as the header

    # Plot in the main panel
    for data, ax in zip(jet_data_by_pt.values(), axes.flatten()[:-1], strict=True):
        plot_parameters_on_ax(hetgp=data.hetgp, hetgp_err=data.hetgp_err, hf=data.hf, hf_err=data.hf_err, ax=ax)

    plot_config.apply(fig, axes=axes.flatten())
    # And remove the axis header
    axes[1, 1].set_axis_off()

    _output_path = base_path / "figures"
    _output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(_output_path / f"{plot_config.name}.pdf")
    plt.close(fig)


# %%
# Jet, combined figure
text_font_size = 22
in_figure_font_size = 20
for selected_space in ["global", "posterior"]:
    # I considered including everything here (e.g. sqrt_s), but it doesn't matter overly much
    # for the purposes of this exercise. To just highlight the important information, I'm going to cut down to the minimal.
    text = "Norm. total-effect Sobol' index"
    if selected_space == "global":
        text += "\n" + "Full design space"
    else:
        text += "\n" + r"1-99\% posterior"

    text_details = "\n" + r"Emulated: Jet $R_{\text{AA}}$, $R = 0.4$"
    text_details += "\n" + r"0-10\%, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$"
    text_details += "\n" + r"ATLAS, $\textit{PLB 790 (2019) 108-128}$"

    pt_labels = []
    for k in jet_data["global"]:
        low, high = map(float, k.split("_"))
        # It's actually 1 TeV, so better to just rewrite it properly...
        if high == 999:
            high = 1000
        pt_labels.append(rf"${low:g} < p_{{\text{{T}}}} < {high:g}\:\text{{GeV}}/c$")

    plot_config = pb.PlotConfig(
        name=f"sensitivity_jet_{selected_space}_combined",
        panels=[
            # Main panel
            pb.Panel(
                axes=[
                    pb.AxisConfig(
                        "y",
                        label="Norm. TE Sobol' index",
                        font_size=text_font_size,
                        range=(0, 1),
                    ),
                ],
                text=pb.TextConfig(x=0.95, y=0.95, text=pt_labels[0], font_size=in_figure_font_size),
                # legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
            ),
            pb.Panel(
                axes=[
                    pb.AxisConfig(
                        "y",
                        range=(0, 1),
                    ),
                ],
                text=[
                    pb.TextConfig(x=0.95, y=0.95, text=pt_labels[1], font_size=in_figure_font_size),
                    pb.TextConfig(x=0.98, y=0.19, text="Equal sensitivity", font_size=18, text_kwargs={"zorder": 0}),
                ],
                legend=pb.LegendConfig(
                    location="upper right", anchor=(0.95, 0.80), font_size=text_font_size, marker_label_spacing=0.3
                ),
            ),
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
                        label="Norm. TE Sobol' index",
                        font_size=text_font_size,
                        range=(0, 0.995),
                    ),
                ],
                text=[
                    pb.TextConfig(x=0.95, y=0.95, text=pt_labels[2], font_size=in_figure_font_size),
                    pb.TextConfig(x=0.98, y=0.19, text="Equal sensitivity", font_size=18, text_kwargs={"zorder": 0}),
                ],
                # legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
            ),
            # Header
            pb.Panel(
                axes=[],
                text=[
                    pb.TextConfig(x=0.05, y=0.71, text=text, font_size=18),
                    pb.TextConfig(x=0.05, y=0.62, text=text_details, font_size=14),
                ],
            ),
        ],
        figure=pb.Figure(edge_padding={"left": 0.105, "bottom": 0.09}),
    )

    plot_compare_hetgp_hf_jet(
        jet_data_by_pt=jet_data["global" if selected_space == "global" else "1_99"], plot_config=plot_config
    )


# %% [markdown]
# ## Pt dependence of parameter (alpha_s)


# %%
def plot_parameter_pt_dependence(
    data_by_pt: dict[str, Data], selected_parameter: str, plot_config: pb.PlotConfig
) -> None:
    """Plot pt dependence of a particular parameter."""

    # Normalize errors and data
    # NOTE: We need to copy the data first so our normalization doesn't impact the existing data
    data_by_pt = {k: copy.deepcopy(v) for k, v in data_by_pt.items()}
    for v in data_by_pt.values():
        v.normalize()

    # Need to determine where the values are in the array
    parameter_index = list(label_to_display_label.keys()).index(selected_parameter)
    pt_labels = list(data_by_pt.keys())
    hetgp = [data_by_pt[v].hetgp[parameter_index] for v in pt_labels]
    hetgp_err = [data_by_pt[v].hetgp_err[parameter_index] for v in pt_labels]
    hf = [data_by_pt[v].hf[parameter_index] for v in pt_labels]
    hf_err = [data_by_pt[v].hf_err[parameter_index] for v in pt_labels]
    index = np.arange(len(pt_labels))

    # Setup
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, 6.25),
        sharex=True,
    )

    ax.plot()

    bar_width = 0.2
    # The first is shifted left
    ax.bar(index - bar_width / 2, hf, bar_width, yerr=hf_err, label="High fidelity GP", color=colors["HF"])
    # The second is shifted right
    ax.bar(index + bar_width / 2, hetgp, bar_width, yerr=hetgp_err, label="VarP-GP", color=colors["hetgp"])

    # And then ensure we show the parameter names on the x-axis
    ax.xaxis.set_ticks(index)
    axis_labels = []
    for pt_label in pt_labels:
        low, high = map(float, pt_label.split("_"))
        axis_labels.append(f"[{low}, {high}]")
    ax.xaxis.set_ticklabels(axis_labels)

    # Apply styling
    plot_config.apply(fig, ax=ax)

    _output_path = base_path / "figures"
    _output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(_output_path / f"{plot_config.name}.pdf")
    plt.close(fig)


# %% [markdown]
# ### Hadron

# %%
parameter = "alpha_s"
text = f"{label_to_display_label[parameter]} sensitivity (full design space)"
text += "\n" + r"Hadron $R_{\text{AA}}$"
# text += ", " + rf"${hadron_low:g} < p_{{\text{{T}}}} < {hadron_high:g}\:\text{{GeV}}/c$"
header_text = r"Emulated: Hadron $R_{\text{AA}}$ in 0-5\% Pb-Pb, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$,"
header_text += " " + r"CMS, $\textit{JHEP 04 (2017) 039}$"
plot_config = pb.PlotConfig(
    name=f"sensitivity_hadron_global_{parameter}",
    panels=[
        # Main panel
        pb.Panel(
            axes=[
                pb.AxisConfig(
                    "x",
                    label=r"$p_{\text{T}}$ bin (GeV/$c$)",
                    font_size=text_font_size,
                    use_major_axis_multiple_locator_with_base=1,
                ),
                pb.AxisConfig(
                    "y",
                    label="Norm. total-effect Sobol' index",
                    font_size=text_font_size,
                    range=(0, 1),
                ),
            ],
            text=[
                pb.TextConfig(x=0.5, y=1.03, text=header_text, font_size=18, alignment="center"),
                pb.TextConfig(x=0.95, y=0.8, text=text, font_size=text_font_size),
            ],
            legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
        ),
    ],
    figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.115, "top": 0.94}),
)
plot_parameter_pt_dependence(hadron_data["global"], selected_parameter=parameter, plot_config=plot_config)

# %% [markdown]
# ### Jet

# %%
parameter = "alpha_s"
text = f"{label_to_display_label[parameter]} sensitivity (full design space)"
text += "\n" + r"Jet $R_{\text{AA}}$, $R = 0.4$"
header_text = r"Emulated: Jet $R_{\text{AA}}$, $R = 0.4$ in 0-10\% Pb-Pb, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$,"
header_text += " " + r"ATLAS, $\textit{PLB 790 (2019) 108-128}$"

plot_config = pb.PlotConfig(
    name=f"sensitivity_jet_global_{parameter}",
    panels=[
        # Main panel
        pb.Panel(
            axes=[
                pb.AxisConfig(
                    "x",
                    label=r"$p_{\text{T}}$ bin (GeV/$c$)",
                    font_size=text_font_size,
                    use_major_axis_multiple_locator_with_base=1,
                ),
                pb.AxisConfig(
                    "y",
                    label="Norm. total-effect Sobol' index",
                    font_size=text_font_size,
                    range=(0, 1),
                ),
            ],
            text=[
                pb.TextConfig(x=0.505, y=1.03, text=header_text, font_size=16, alignment="center"),
                pb.TextConfig(x=0.95, y=0.8, text=text, font_size=text_font_size),
            ],
            legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
        ),
    ],
    figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.115, "top": 0.94}),
)
plot_parameter_pt_dependence(jet_data["global"], selected_parameter=parameter, plot_config=plot_config)


# %% [markdown]
# ## Hadron vs jet at similar pt


# %%
def plot_hadron_vs_jet_sensitivity(hadron_data: Data, jet_data: Data, plot_config: pb.PlotConfig):
    # Normalize errors and data
    # NOTE: We need to copy the data first so our normalization doesn't impact the existing data
    hadron_data = copy.deepcopy(hadron_data)
    hadron_data.normalize()
    jet_data = copy.deepcopy(jet_data)
    jet_data.normalize()

    index = np.arange(len(hadron_data.hetgp))

    # Setup
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, 6.25),
        sharex=True,
    )

    # Darker options. RJE likes these less...
    # colors = {
    #     "hetgp": "#FF8301",      # Original orange
    #     "hetgp_alt": "#CC6900",  # Darker, richer orange
    #     "HF": "#845cba",         # Original purple
    #     "HF_alt": "#654494"      # Darker, richer purple
    # }
    # Lighter options
    colors = {
        "hetgp": "#FF8301",  # Original orange
        "hetgp_alt": "#FFB347",  # Lighter, softer orange
        "HF": "#845cba",  # Original purple
        "HF_alt": "#a47dd4",  # Lighter, softer purple
    }

    bar_width = 0.2
    ax.bar(
        index - bar_width * 3 / 2,
        hadron_data.hf,
        bar_width,
        yerr=hadron_data.hf_err,
        label="High fidelity GP (Hadron)",
        color=colors["HF"],
    )
    ax.bar(
        index - bar_width / 2,
        hadron_data.hetgp,
        bar_width,
        yerr=hadron_data.hetgp_err,
        label="VarP-GP (Hadron)",
        color=colors["hetgp"],
    )
    # The second is shifted right
    ax.bar(
        index + bar_width / 2,
        jet_data.hf,
        bar_width,
        yerr=jet_data.hf_err,
        label="High fidelity GP (Jet)",
        color=colors["HF_alt"],
    )
    ax.bar(
        index + bar_width * 3 / 2,
        jet_data.hetgp,
        bar_width,
        yerr=jet_data.hetgp_err,
        label="VarP-GP (Jet)",
        color=colors["hetgp_alt"],
    )

    # And then a line to mark equal sensitivity
    # NOTE: (it relies on the plot_config for the label)
    ax.axhline(y=1.0 / 6.0, color="black", linestyle="dashed", zorder=0)

    # And then ensure we show the parameter names on the x-axis
    ax.xaxis.set_ticks(range(len(label_to_display_label)))
    ax.xaxis.set_ticklabels(label_to_display_label.values())

    # Modify the legend ordering to group by emulator type.
    leg_handles, leg_labels = ax.get_legend_handles_labels()
    new_handles = [
        # HFGP
        leg_handles[0],
        leg_handles[2],
        # VarP-GP
        leg_handles[1],
        leg_handles[3],
    ]
    new_labels = [
        # HFGP
        leg_labels[0],
        leg_labels[2],
        # VarP-GP
        leg_labels[1],
        leg_labels[3],
    ]

    # Apply styling
    plot_config.apply(fig, ax=ax, legend_handles=new_handles, legend_labels=new_labels)

    _output_path = base_path / "figures"
    _output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(_output_path / f"{plot_config.name}.pdf")
    plt.close(fig)


header_text = r"Emulated: Central Pb-Pb at $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$"
header_text += "\n" + r"CMS hadron $R_{\text{AA}}$, $\textit{JHEP 04 (2017) 039}$"
header_text += ", " + r"ATLAS jet $R_{\text{AA}}$, $\textit{PLB 790 (2019) 108-128}$"
# text = "Full design space"
text = r"Hadron $R_{\text{AA}}$: $73.6 < p_{\text{T}}^{\text{jet}} < 165$ (GeV/$c$)"
text += "\n" + r"Jet $R_{\text{AA}}$: $100 < p_{\text{T}}^{\text{jet}} < 177$ (GeV/$c$)"

plot_config = pb.PlotConfig(
    name="sensitivity_hadron_vs_jet_global",
    panels=[
        # Main panel
        pb.Panel(
            axes=[
                pb.AxisConfig(
                    "x",
                    label="Parameters",
                    font_size=text_font_size,
                    use_major_axis_multiple_locator_with_base=1,
                ),
                pb.AxisConfig(
                    "y",
                    label="Norm. total-effect Sobol' index",
                    font_size=text_font_size,
                    range=(0, 1),
                ),
            ],
            text=[
                pb.TextConfig(x=0.01, y=1.01, text=header_text, font_size=16, alignment="lower left"),
                pb.TextConfig(x=0.03, y=0.96, text="Full design space", font_size=text_font_size),
                pb.TextConfig(x=0.95, y=0.58, text=text, font_size=text_font_size),
                pb.TextConfig(x=0.98, y=0.20, text="Equal sensitivity", font_size=16, text_kwargs={"zorder": 0}),
            ],
            legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
        ),
    ],
    figure=pb.Figure(edge_padding={"left": 0.10, "bottom": 0.11, "top": 0.90}),
)
plot_hadron_vs_jet_sensitivity(
    hadron_data["global"]["73.6_165.0"], jet_data["global"]["100.0_177.0"], plot_config=plot_config
)


# %% [markdown]
# ## Global vs local sensitivity in same figure


# %%
def plot_global_vs_local_sensitivity(data: dict[str, dict[str, Data]], pt_label: str, plot_config: pb.PlotConfig):
    # Normalize errors and data
    # NOTE: We need to copy the data first so our normalization doesn't impact the existing data
    data = copy.deepcopy(data)
    for region in ["global", "1_99"]:
        for v in data[region].values():
            v.normalize()

    # The length is the same for everything - it's just the parameters
    index = np.arange(len(label_to_display_label))

    # Setup
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, 6.25),
        sharex=True,
    )

    # Darker options. RJE likes these less...
    # colors = {
    #     "hetgp": "#FF8301",      # Original orange
    #     "hetgp_alt": "#CC6900",  # Darker, richer orange
    #     "HF": "#845cba",         # Original purple
    #     "HF_alt": "#654494"      # Darker, richer purple
    # }
    # Lighter options
    colors = {
        "hetgp": "#FF8301",  # Original orange
        "hetgp_alt": "#FFB347",  # Lighter, softer orange
        "HF": "#845cba",  # Original purple
        "HF_alt": "#a47dd4",  # Lighter, softer purple
    }

    bar_width = 0.2
    ax.bar(
        index - bar_width * 3 / 2,
        data["global"][pt_label].hf,
        bar_width,
        yerr=data["global"][pt_label].hf_err,
        label="High fidelity GP (Full design)",
        color=colors["HF"],
    )
    ax.bar(
        index - bar_width / 2,
        data["1_99"][pt_label].hf,
        bar_width,
        yerr=data["1_99"][pt_label].hf_err,
        label=r"High fidelity GP (1-99\%)",
        color=colors["HF_alt"],
    )
    # The second is shifted right
    ax.bar(
        index + bar_width / 2,
        data["global"][pt_label].hetgp,
        bar_width,
        yerr=data["global"][pt_label].hetgp_err,
        label="VarP-GP (Full design)",
        color=colors["hetgp"],
    )
    ax.bar(
        index + bar_width * 3 / 2,
        data["1_99"][pt_label].hetgp,
        bar_width,
        yerr=data["1_99"][pt_label].hetgp_err,
        label=r"VarP-GP (1-99\%)",
        color=colors["hetgp_alt"],
    )

    # And then ensure we show the parameter names on the x-axis
    ax.xaxis.set_ticks(range(len(label_to_display_label)))
    ax.xaxis.set_ticklabels(label_to_display_label.values())

    # And then a line to mark equal sensitivity
    # NOTE: (it relies on the plot_config for the label)
    ax.axhline(y=1.0 / 6.0, color="black", linestyle="dashed", zorder=0)

    # Apply styling
    plot_config.apply(fig, ax=ax)

    _output_path = base_path / "figures"
    _output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(_output_path / f"{plot_config.name}.pdf")
    plt.close(fig)


for observable in ["hadron", "jet"]:
    data = hadron_data if observable == "hadron" else jet_data
    if observable == "hadron":
        header_text = r"Emulated: Hadron $R_{\text{AA}}$ in 0-5\% Pb-Pb, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$,"
        header_text += " " + r"CMS, $\textit{JHEP 04 (2017) 039}$"
    else:
        header_text = (
            r"Emulated: Jet $R_{\text{AA}}$, $R = 0.4$ in 0-10\% Pb-Pb, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$,"
        )
        header_text += " " + r"ATLAS, $\textit{PLB 790 (2019) 108-128}$"

    for pt_label in data["global"]:
        # Pt bin
        low, high = map(float, pt_label.split("_"))
        # It's actually 1 TeV, so better to just rewrite it properly...
        if high == 999:
            high = 1000
        if observable == "hadron":  # noqa: SIM108
            minimal_text = r"Hadron $R_{\text{AA}}$"
        else:
            minimal_text = r"Jet $R_{\text{AA}}$, $R = 0.4$"
        minimal_text += "\n" + rf"${low:g} < p_{{\text{{T}}}} < {high:g}\:\text{{GeV}}/c$"

        plot_config = pb.PlotConfig(
            name=f"sensitivity_global_vs_1_99_{observable}_{pt_label}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "x",
                            label="Parameters",
                            font_size=text_font_size,
                            use_major_axis_multiple_locator_with_base=1,
                        ),
                        pb.AxisConfig(
                            "y",
                            label="Norm. total-effect Sobol' index",
                            font_size=text_font_size,
                            range=(0, 1),
                        ),
                    ],
                    text=[
                        pb.TextConfig(x=0.01, y=1.01, text=header_text, font_size=16, alignment="lower left"),
                        pb.TextConfig(x=0.95, y=0.60, text=minimal_text, font_size=text_font_size),
                        pb.TextConfig(
                            x=0.98, y=0.20, text="Equal sensitivity", font_size=16, text_kwargs={"zorder": 0}
                        ),
                    ],
                    legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.09, "bottom": 0.11, "top": 0.94}),
        )
        print(f"{observable=}, {pt_label=}")
        plot_global_vs_local_sensitivity(data=data, pt_label=pt_label, plot_config=plot_config)

# %% [markdown]
# # Older, standalone figures

# %%
# global total index
for index in range(len(hadron_pt_bin_labels)):
    filename_HF = base_path / ("HFGP/hadron_aggregate/Global_TotalIndex_" + hadron_pt_bin_labels[index] + ".csv")
    HFdatabyaggbin = reorder_data(pd.read_csv(filename_HF))
    HF = np.array(HFdatabyaggbin["STagg"])
    HFerr = np.array(HFdatabyaggbin["ST_confagg"])
    filename_HetGP = base_path / ("HetGP/hadron_aggregate/Global_TotalIndex_" + hadron_pt_bin_labels[index] + ".csv")
    HetGPdatabyaggbin = reorder_data(pd.read_csv(filename_HetGP))
    HetGP = np.array(HetGPdatabyaggbin["STagg"])
    HetGPerr = np.array(HetGPdatabyaggbin["ST_confagg"])
    plotname = base_path / ("figures/hadron_Global_TotalIndex_aggbin_" + hadron_pt_bin_labels[index] + ".pdf")

    # I considered including everything here (e.g. sqrt_s), but it doesn't matter overly much
    # for the purposes of this exercise. To just highlight the important information, I'm going to cut down to the minimal.
    hadron_low, hadron_high = map(float, hadron_pt_bin_labels[index].split("_"))
    text = "Norm. total-effect Sobol' index"
    text += r", trained on JETSCAPE (MATTER + LBT)"
    text += "\n" + r"corresponding to CMS, $\textit{JHEP 04 (2017) 039}$"
    text += "\n" + r"0-5\%, Hadron $R_{\text{AA}}$, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$"
    text += "\n" + rf"${hadron_low:g} < p_{{\text{{T}}}} < {hadron_high:g}\:\text{{GeV}}/c$"
    plot_config = pb.PlotConfig(
        name=f"sensitivity_hadron_global_{hadron_pt_bin_labels[index]}",
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
                        label="Norm. total-effect Sobol' index",
                        font_size=text_font_size,
                        range=(0, 1),
                    ),
                ],
                text=[
                    pb.TextConfig(x=0.95, y=0.81, text=text, font_size=18),
                    pb.TextConfig(x=0.99, y=0.17, text="Equal sensitivity", font_size=18, text_kwargs={"zorder": 1}),
                ],
                legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
            ),
        ],
        figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.11}),
    )

    # plot(HF, HetGP, HFerr, HetGPerr, plotname)
    plot_compare_hetgp_hf(HF, HetGP, HFerr, HetGPerr, plot_config=plot_config)

# %%
# local 5-95 total index
for index in range(len(hadron_pt_bin_labels)):
    filename_HF = base_path / ("HFGP/hadron_aggregate/Local_5_95_TotalIndex_" + hadron_pt_bin_labels[index] + ".csv")
    HFdatabyaggbin = pd.read_csv(filename_HF)
    HF = np.array(HFdatabyaggbin["STagg"])
    HFerr = np.array(HFdatabyaggbin["ST_confagg"])
    filename_HetGP = base_path / (
        "HetGP/hadron_aggregate/Local_5_95_TotalIndex_" + hadron_pt_bin_labels[index] + ".csv"
    )
    HetGPdatabyaggbin = pd.read_csv(filename_HetGP)
    HetGP = np.array(HetGPdatabyaggbin["STagg"])
    HetGPerr = np.array(HetGPdatabyaggbin["ST_confagg"])
    plotname = base_path / ("figures/Hadron_Local_5_95_TotalIndex_aggbin_" + hadron_pt_bin_labels[index] + ".pdf")

    # I considered including everything here (e.g. sqrt_s), but it doesn't matter overly much
    # for the purposes of this exercise. To just highlight the important information, I'm going to cut down to the minimal.
    hadron_low, hadron_high = map(float, hadron_pt_bin_labels[index].split("_"))
    text = r"5-95\% MAP norm. total-effect Sobol' index"
    text += r", trained on JETSCAPE (MATTER + LBT)"
    text += "\n" + r"corresponding to CMS, $\textit{JHEP 04 (2017) 039}$"
    text += "\n" + r"0-5\%, Hadron $R_{\text{AA}}$, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$"
    text += "\n" + rf"${hadron_low:g} < p_{{\text{{T}}}} < {hadron_high:g}\:\text{{GeV}}/c$"
    plot_config = pb.PlotConfig(
        name=f"sensitivity_hadron_5_95_{hadron_pt_bin_labels[index]}",
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
                        label="Norm. total-effect Sobol' index",
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
    plot_compare_hetgp_hf(HF, HetGP, HFerr, HetGPerr, plot_config=plot_config)

# %%
# local 1-99 total index
for index in range(len(hadron_pt_bin_labels)):
    filename_HF = base_path / ("HFGP/hadron_aggregate/Local_1_99_TotalIndex_" + hadron_pt_bin_labels[index] + ".csv")
    HFdatabyaggbin = pd.read_csv(filename_HF)
    HF = np.array(HFdatabyaggbin["STagg"])
    HFerr = np.array(HFdatabyaggbin["ST_confagg"])
    filename_HetGP = base_path / (
        "HetGP/hadron_aggregate/Local_1_99_TotalIndex_" + hadron_pt_bin_labels[index] + ".csv"
    )
    HetGPdatabyaggbin = pd.read_csv(filename_HetGP)
    HetGP = np.array(HetGPdatabyaggbin["STagg"])
    HetGPerr = np.array(HetGPdatabyaggbin["ST_confagg"])
    plotname = base_path / ("figures/Hadron_Local_1_99_TotalIndex_aggbin_" + hadron_pt_bin_labels[index] + ".pdf")

    # I considered including everything here (e.g. sqrt_s), but it doesn't matter overly much
    # for the purposes of this exercise. To just highlight the important information, I'm going to cut down to the minimal.
    hadron_low, hadron_high = map(float, hadron_pt_bin_labels[index].split("_"))
    text = r"1-99\% MAP sobol sensitivity"
    text += r", trained on JETSCAPE (MATTER + LBT)"
    text += "\n" + r"corresponding to CMS, $\textit{JHEP 04 (2017) 039}$"
    text += "\n" + r"0-5\%, Hadron $R_{\text{AA}}$, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$"
    text += "\n" + rf"${hadron_low:g} < p_{{\text{{T}}}} < {hadron_high:g}\:\text{{GeV}}/c$"
    plot_config = pb.PlotConfig(
        name=f"sensitivity_hadron_1_99_{hadron_pt_bin_labels[index]}",
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
    plot_compare_hetgp_hf(HF, HetGP, HFerr, HetGPerr, plot_config=plot_config)

    # plot(HF, HetGP, HFerr, HetGPerr, plotname)
