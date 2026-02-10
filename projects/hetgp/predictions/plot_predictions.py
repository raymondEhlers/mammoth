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

# %% [markdown]
# # Setup

# %%
from __future__ import annotations

from pathlib import Path
from typing import Any

import attrs
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot as pb
import pandas as pd
import seaborn as sns

base_path = Path("projects/hetgp/predictions")

pb.configure()

# %%
# NOTE(RJE): This is not actually used anywhere - we grab these values from the csv
budget_list = np.array([2100, 3500, 4000, 4500, 5000, 5500])


# %%
@attrs.define
class MethodStyle:
    color: str
    marker: str
    fillstyle: str
    label: str
    label_short: str
    zorder: int

    def kwargs_for_plot_errorbar(self) -> dict[str, Any]:
        """Most common kwargs for plotting with ax.errorbar.

        This can always be customized for individual plots, but this provides a good starting point.
        """
        d: dict[str, str | int | float] = {
            "color": self.color,
            "marker": self.marker,
            "markersize": 11,
            # Configure rest of marker presentation
            "markeredgecolor": self.color,
            "linestyle": "",
            "linewidth": 3,
        }
        # This marker is a bit small, so try to boost it up
        if self.marker in ["P", "d", "D"]:
            d["markersize"] += 2  # type: ignore[operator]
        if self.fillstyle == "none":
            # Update the fillstyle to be solid white. Transparent would be better, but it doesn't work because
            # it shows the errorbar lines going through the point :-(
            # One possibility: calculate the marker size, find the edge in display space, and then basically stop the lines
            # at the marker edge. I can imagine this might look terrible though and/or be buggy and/or be a lot of work.
            # So wait on this step for now (Aug 2023). See the ChatGPT logs and the debug section n the paper-plots notebook.
            d.update(
                {
                    "fillstyle": "full",
                    "markerfacecolor": "white",
                    "markeredgewidth": 3,
                }
            )
        else:
            d.update(
                {
                    "markerfacecolor": self.color,
                }
            )
        return d

    def kwargs_for_plot_error_boxes(self) -> dict[str, Any]:
        return {
            "color": self.color,
            "linewidth": 0,
            "alpha": 0.3,
            # This is common, but does need to be overridden sometimes!
            "zorder": 2,
        }


method_styles = {
    "hetgp": MethodStyle(
        # Middle blue
        # color="#2980b9",
        # Light blue
        # color="#4bafd0",
        # Orange
        color="#FF8301",
        marker="o",
        fillstyle="full",
        label="VarP-GP",
        label_short="VarP-GP",
        zorder=10,
    ),
    "high_fidelity": MethodStyle(
        # Purple
        color="#845cba",
        marker="s",
        fillstyle="full",
        label="HF-GP",
        label_short="HF-GP",
        zorder=4,
    ),
}


# %% [markdown]
# ## Figure definitions

# %% [markdown]
# ### Residual

# %%
# Raymond's revised figures


# Plot showing the fully integrated residual error
def plot_residual(HFGPMSE: pd.DataFrame, hetGPMSE: pd.DataFrame, plot_config: pb.PlotConfig) -> None:
    # Setup data
    sum_HFGPMSE = HFGPMSE.sum()
    sum_hetGPMSE = hetGPMSE.sum()
    compare_df = pd.DataFrame({"high_fidelity": sum_HFGPMSE, "hetgp": sum_hetGPMSE})

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, 6.25),
        sharex=True,
    )

    # Plot using line plot
    # NOTE: reversed so the high_fidelity will be plotted first, which looks nicer in the legend since
    #       it matches the order of the displayed data
    for column, styling in reversed(method_styles.items()):
        print(f"Plotting column '{column}'")
        styling_kwargs = styling.kwargs_for_plot_errorbar()
        styling_kwargs["linestyle"] = "-"
        ax.plot(compare_df[column].index, compare_df[column], label=styling.label, **styling_kwargs)

    # Customize the axis presentation
    # I want numbers to appear as 10^6, not 10^7
    ax.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))

    plot_config.apply(fig, ax=ax)

    _output_path = base_path / "figures"
    _output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(_output_path / f"{plot_config.name}.pdf")
    plt.close(fig)

    return compare_df


# %% [markdown]
# ### Boxplot


# %%
# Plot showing the box plot of a distribution
# For the paper, we look at the pt distribution integrated over the design points
def boxplot(HFGPMSE: pd.DataFrame, hetGPMSE: pd.DataFrame, plot_config: pb.PlotConfig) -> None:
    """Plot the boxplot.

    Args:
        HFGPMSE: MSE values of the HF-GP.
        hetGPMSE: MSE values of the VarP-GP.
        plot_config: Plot configuration
    """
    # Melt both DataFrames and add a 'Source' label
    HFGPMSE_melted = HFGPMSE.melt(var_name="Variable", value_name="Value")
    HFGPMSE_melted["Source"] = "HF-GP"

    hetGPMSE_melted = hetGPMSE.melt(var_name="Variable", value_name="Value")
    hetGPMSE_melted["Source"] = "VarP-GP"

    # Combine both for plotting
    df_combined = pd.concat([hetGPMSE_melted, HFGPMSE_melted])

    # sns.boxplot treats the x-axis as categorical. This causes all kinds of problems.
    # So we need to handle this by hand, rather than using the built-in functions.
    # Scale down the x-axis by 10^6, which we have to handle by hand.
    df_combined["Variable"] = df_combined["Variable"] / 1_000_000

    # Check for outliers that are outside of our figure and need to be labeled.
    # Calculated per source, it should be the count of the number of outliers outside of the y-axis, indexed by their x-axis value.
    # This will be used to specify labels for outliers off the figure. Default: None.
    # NOTE: The threshold is the y-axis max
    threshold = plot_config.panels[0].axes[1].range[1]
    outliers = (
        df_combined[df_combined["Value"] > threshold]
        .pivot_table(index="Source", columns="Variable", aggfunc="size", fill_value=0)
        .to_dict("index")
    )

    fig, ax = plt.subplots(
        1,
        1,
        # figsize=(10, 6.25),
        figsize=(10, 8),
        sharex=True,
    )

    # We match the order to the construction of the DataFrame, where we have hetGP first.
    colors = [method_styles["hetgp"].color, method_styles["high_fidelity"].color]

    # Get unique variables and sources
    variables = sorted(df_combined["Variable"].unique())
    sources = df_combined["Source"].unique()

    # Define box width and offset for grouping
    box_width = 0.8
    # Use the same grey that's used in seaborn for the lines
    grey_for_lines = (75.0 / 255, 75.0 / 255, 75.0 / 255)

    # Plot for each source
    for i, source in enumerate(sources):
        # Calculate positions (offset each source)
        positions = np.array(variables) + (i - len(sources) / 2 + 0.5) * box_width
        print(f"{source=}, {positions=}")

        # Prepare data for each variable
        data = [
            df_combined[(df_combined["Variable"] == var) & (df_combined["Source"] == source)]["Value"].values
            for var in variables
        ]

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            label=source,
            medianprops={
                "linewidth": 4,
                # "color": grey_for_lines,
                "color": "black",
            },
            whiskerprops={"linewidth": 2, "color": grey_for_lines},
            capprops={"linewidth": 2, "color": grey_for_lines},
            boxprops={
                "facecolor": colors[i],
                "edgecolor": grey_for_lines,
                "linewidth": 2,
            },
            flierprops={
                "marker": "o",
                "markersize": 5,
                "markeredgecolor": "black",
                "markeredgewidth": 0.3,
                "markerfacecolor": colors[i],
            },
        )

        # And then add a marker at the median
        # ax.plot(
        #     positions,
        #     [np.median(v) for v in data],
        #     marker="s",
        #     color="black",
        #     #color="#2E86C1",
        #     linestyle="",
        #     zorder=5,
        #     markersize=10,
        # )

        # Check for outliers to put outside of the figure

        source_outliers = outliers.get(source, {})
        for x_loc, count in source_outliers.items():
            print(f"Plotting source outlier: {x_loc=}, {count=}")
            # Need to shift the x_loc to correspond to the variable
            x_shifted = x_loc + (i - len(sources) / 2 + 0.5) * box_width
            ax.annotate(
                # count,
                # NOTE(RJE): I decided that not to include the count, since I'm not sure it brings another.
                #            It could if we included more, but since we're only doing this for one particular case,
                #            I think it's okay to include no label to simplify the figure.
                "",
                xy=(x_shifted, threshold),
                xytext=(x_shifted, threshold * 0.95),
                # NOTE(RJE): This is a cute trick splitting how the coordinates are interpreted, but
                #            not worth the extra complexity here.
                # xytext=(x_shifted, -0.05),
                # textcoords=("data", "axes fraction"),
                ha="center",
                fontsize=12,
                arrowprops={"arrowstyle": "->", "color": colors[i], "lw": 2},
            )

    # Manually set ticks at regular intervals
    # We have to do this by hand because the labels are otherwise set at where the boxes are actually positioned
    # NOTE: It's important to set both the ticks and labels - just one isn't enough since the boxplot
    #       function sets a bunch of things that we need to undo.
    tick_positions = np.arange(10.0, 30.0, 2.5)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions)

    # # Plot
    # # Original version, but it's much more restrictive compared to the mpl version
    # p = sns.boxplot(
    #     x="Variable",
    #     y="Value",
    #     hue="Source",
    #     palette=colors,
    #     saturation=1.0,
    #     ax=ax,
    #     width=0.6,
    #     medianprops={"linewidth": 3},
    #     whiskerprops={"linewidth": 2},
    #     capprops={"linewidth": 2},
    #     boxprops={
    #         "linewidth": 2,
    #     },
    #     flierprops={
    #         "marker": "o",
    #         "markersize": 5,
    #         "markeredgecolor": "black",
    #         "markeredgewidth": 0.3,
    #     },
    #     data=df_combined,
    # )

    ## Apply colors to fliers based on their position
    ## We can identify the fliers based on them not using lines
    ## NOTE: They classes are plotted one at a time - e.g. HetGP first, so we set
    ##       the colors based on the first half (HetGP) and the second half (HFGP)
    # lines = [line for line in ax.lines if line.get_linestyle() == "None"]
    # for i, line in enumerate(lines):
    #    # Determine which hue group this outlier belongs to
    #    # NOTE: This depends on the data structure. If there were more sources,
    #    #       we would need to adjust
    #    color_idx = 0 if i < len(lines) / 2 else 1
    #    # Set just the facecolor - we'll keep the black edges
    #    line.set_markerfacecolor(colors[color_idx])

    # Add the x 10^6 label at the bottom right. As noted above,
    # we cannot use the usual functionality due to how sns.boxplot creates axes.
    # n.b. 2026-01-14: I switched to the mpl boxplot function, so may have been able to use a regular
    #                  axis, but it wasn't worth changing this.
    plot_config.panels[0].text.append(pb.TextConfig(r"$\times 10^6$", 1.0, -0.11, font_size=20))

    plot_config.apply(fig, ax=ax)

    _output_path = base_path / "figures"
    _output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(_output_path / f"{plot_config.name}.pdf")
    plt.close(fig)

    return df_combined


# %%
def read_csv(p: Path) -> pd.DataFrame:
    """Read CSV with consistent types."""
    df = pd.read_csv(p)  # noqa: PD901
    # Treat columns as integers, so we can multiply them as expect
    df.columns = df.columns.astype(np.int64)
    # And then multiply the axis 5000 events to get the total number of events used for training
    df.columns = df.columns * 5000
    return df


# %% [markdown]
# # Hadrons
#
# Hadron - HetGP

# %%
# HetGP
hetGPMSE = read_csv(base_path / "Hadron_HETGP_Prediction_by_bin.csv")
# High fidelity
HFGPMSE = read_csv(base_path / "Hadron_HFGP_Prediction_by_bin.csv")
hetGPMSE, HFGPMSE

# %% [markdown]
# ## Residual
#
# ### Irene

# %%
# Irene's original...
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

# %% [markdown]
# ### Raymond

# %%
text_font_size = 22

# Labeling
# The miniaml text contains only the most important text
# Everything else (e.g. what is not critical to distringuishing the plots) is kept in the header.
# I skipped that it was trained on JETSCAPE (MATTER+LBT) - it's bulky, and never varies.
header_text = r"Emulated: Hadron $R_{\text{AA}}$ in 0-5\% Pb-Pb, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$,"
header_text += " " + r"CMS, $\textit{JHEP 04 (2017) 039}$"
minimal_text_obs = r"Hadron $R_{\text{AA}}$"
minimal_text_pt = r"$4.8 < p_{\text{T}} < 400\:\text{GeV}/c$"
plot_config = pb.PlotConfig(
    name="budget_residual_error_hadron",
    panels=[
        # Main panel
        pb.Panel(
            axes=[
                pb.AxisConfig(
                    "x",
                    label=r"Training data $N_{\text{event}}$",
                    font_size=text_font_size,
                ),
                pb.AxisConfig(
                    "y",
                    # label=r"$\sum_{p_{\text{T,}}\:\text{design}}$ MSE",
                    label=r"Aggregated MSE",
                    font_size=text_font_size,
                    range=(0.035, 0.24),
                    # range=(0.076, 0.235),
                    # range=(0.044, 0.155),
                ),
            ],
            text=[
                pb.TextConfig(x=0.5, y=1.03, text=header_text, font_size=18, alignment="center"),
                pb.TextConfig(x=0.95, y=0.70, text=minimal_text_obs, font_size=30),
                pb.TextConfig(x=0.95, y=0.62, text=minimal_text_pt, font_size=12),
            ],
            legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=text_font_size),
        ),
    ],
    figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.11, "top": 0.94}),
)

compare_df = plot_residual(HFGPMSE=HFGPMSE, hetGPMSE=hetGPMSE, plot_config=plot_config)

# %%
HFGPMSE.columns.astype(int)

# %%
compare_df.index

# %% [markdown]
# ## Boxplot
#
# ### Irene

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
# ### Raymond

# %%
text_font_size = 20
for log in [False, True]:
    # Labeling
    # The miniaml text contains only the most important text
    # Everything else (e.g. what is not critical to distringuishing the plots) is kept in the header.
    # I skipped that it was trained on JETSCAPE (MATTER+LBT) - it's bulky, and never varies.
    header_text = r"Emulated: Hadron $R_{\text{AA}}$ in 0-5\% Pb-Pb, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$,"
    header_text += " " + r"CMS, $\textit{JHEP 04 (2017) 039}$"
    minimal_text_obs = r"Hadron $R_{\text{AA}}$"
    minimal_text_pt = r"$4.8 < p_{\text{T}} < 400\:\text{GeV}/c$"
    y_max = 0.12 if log else 0.031
    plot_config = pb.PlotConfig(
        name=f"budget_residual_error_design_point_dist_hadron{'_log' if log else ''}",
        panels=[
            # Main panel
            pb.Panel(
                axes=[
                    pb.AxisConfig(
                        "x",
                        label=r"Training data $N_{\text{event}}$",
                        font_size=text_font_size,
                        range=(9.25, 28.75),
                    ),
                    pb.AxisConfig(
                        "y",
                        label=r"$\sum_{\text{design}}$ MSE",
                        font_size=text_font_size,
                        # range=(-0.0005, 0.038),
                        log=log,
                        # range=(0.0005, 0.08) if log else (-0.0005, 0.0185),
                        range=(0.0005, y_max) if log else (-0.0005, y_max),
                    ),
                ],
                text=[
                    pb.TextConfig(x=0.5, y=1.03, text=header_text, font_size=18, alignment="center"),
                    # pb.TextConfig(x=0.5, y=0.95, text=minimal_text_obs, font_size=30),
                    # pb.TextConfig(x=0.48, y=0.88, text=minimal_text_pt, font_size=12, alignment="upper right"),
                    pb.TextConfig(x=0.05, y=0.90, text=minimal_text_obs, font_size=30),
                    pb.TextConfig(x=0.05, y=0.83, text=minimal_text_pt, font_size=12),
                ],
                legend=pb.LegendConfig(location="upper right", anchor=(0.975, 0.95), font_size=22),
            ),
        ],
        figure=pb.Figure(edge_padding={"left": 0.12, "bottom": 0.11, "top": 0.94}),
    )

    r = boxplot(HFGPMSE=HFGPMSE, hetGPMSE=hetGPMSE, plot_config=plot_config)

# %%
r

# %% [markdown]
# # Jets
#
# Jet - HetGP

# %%
# HetGP
hetGPMSE = read_csv(base_path / "Jet_HETGP_Prediction_by_bin.csv")
# High fidelity
HFGPMSE = read_csv(base_path / "Jet_HFGP_Prediction_by_bin.csv")
hetGPMSE, HFGPMSE

# %% [markdown]
# ## Residual
#
# ### Irene

# %%
# Irene's original figure
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

# %% [markdown]
# ### Raymond

# %%
text_font_size = 22

# Labeling
# The miniaml text contains only the most important text
# Everything else (e.g. what is not critical to distringuishing the plots) is kept in the header.
# I skipped that it was trained on JETSCAPE (MATTER+LBT) - it's bulky, and never varies.
header_text = r"Emulated: $R = 0.4$ jet $R_{\text{AA}}$ in 0-10\% Pb-Pb, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$,"
header_text += " " + r"ATLAS, $\textit{PLB 790 (2019) 108-128}$"
# minimal_text_obs = r"Jet $R_{\text{AA}}$, $R = 0.4$"
minimal_text_obs = r"Jet $R_{\text{AA}}$"
minimal_text_pt = r"$100 < p_{\text{T}} < 1000\:\text{GeV}/c$"
plot_config = pb.PlotConfig(
    name="budget_residual_error_jet",
    panels=[
        # Main panel
        pb.Panel(
            axes=[
                pb.AxisConfig(
                    "x",
                    label=r"Training data $N_{\text{event}}$",
                    font_size=text_font_size,
                ),
                pb.AxisConfig(
                    "y",
                    # label=r"$\sum_{p_{\text{T,}}\:\text{design}}$ MSE",
                    label=r"Aggregated MSE",
                    font_size=text_font_size,
                    # range=(0.044, 0.2025),
                    range=(0.035, 0.24),
                ),
            ],
            text=[
                pb.TextConfig(x=0.5, y=1.03, text=header_text, font_size=16, alignment="center"),
                pb.TextConfig(x=0.95, y=0.70, text=minimal_text_obs, font_size=30),
                pb.TextConfig(x=0.95, y=0.62, text=minimal_text_pt, font_size=12),
            ],
            legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=text_font_size),
        ),
    ],
    figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.11, "top": 0.94}),
)

plot_residual(HFGPMSE=HFGPMSE, hetGPMSE=hetGPMSE, plot_config=plot_config)

# %% [markdown]
# ## Boxplot
#
# ### Irene

# %%
# Irene's
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

# %% [markdown]
# ### Raymond

# %%
text_font_size = 20
for log in [False, True]:
    # Labeling
    # The miniaml text contains only the most important text
    # Everything else (e.g. what is not critical to distringuishing the plots) is kept in the header.
    # I skipped that it was trained on JETSCAPE (MATTER+LBT) - it's bulky, and never varies.
    header_text = r"Emulated: $R = 0.4$ jet $R_{\text{AA}}$ in 0-10\% Pb-Pb, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$,"
    header_text += " " + r"ATLAS, $\textit{PLB 790 (2019) 108-128}$"
    # minimal_text_obs = r"Jet $R_{\text{AA}}$, $R = 0.4$"
    minimal_text_obs = r"Jet $R_{\text{AA}}$"
    minimal_text_pt = r"$100 < p_{\text{T}} < 1000\:\text{GeV}/c$"
    # y_max = 0.12 if log else 0.033
    y_max = 0.12 if log else 0.031
    plot_config = pb.PlotConfig(
        name=f"budget_residual_error_design_point_dist_jet{'_log' if log else ''}",
        panels=[
            # Main panel
            pb.Panel(
                axes=[
                    pb.AxisConfig(
                        "x",
                        label=r"Training data $N_{\text{event}}$",
                        font_size=text_font_size,
                        range=(9.25, 28.75),
                    ),
                    pb.AxisConfig(
                        "y",
                        label=r"$\sum_{\text{design}}$ MSE",
                        font_size=text_font_size,
                        # range=(-0.0005, 0.038),
                        log=log,
                        # range=(0.0005, 0.12) if log else (-0.0005, 0.0385),
                        # range=(0.0005, 0.12) if log else (-0.0005, 0.033),
                        range=(0.0005, y_max) if log else (-0.0005, y_max),
                    ),
                ],
                text=[
                    pb.TextConfig(x=0.5, y=1.03, text=header_text, font_size=18, alignment="center"),
                    pb.TextConfig(x=0.05, y=0.90, text=minimal_text_obs, font_size=30),
                    pb.TextConfig(x=0.05, y=0.83, text=minimal_text_pt, font_size=12),
                ],
                legend=pb.LegendConfig(location="upper right", anchor=(0.975, 0.95), font_size=22),
            ),
        ],
        figure=pb.Figure(edge_padding={"left": 0.12, "bottom": 0.11, "top": 0.94}),
    )

    r = boxplot(HFGPMSE=HFGPMSE, hetGPMSE=hetGPMSE, plot_config=plot_config)

# %%
r[r["Value"] > 0.02].groupby("Variable").size().to_dict()

# %%
