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
        label="High fidelity GP",
        label_short="HF-GP",
        zorder=4,
    ),
}

# %%
budget_list = np.array([3500, 4000, 4500, 5000, 5500])

# %% [markdown]
# # Hadrons
#
# Hadron - HetGP

# %%
# HetGP
hetGPMSE = pd.read_csv(base_path / "Hadron_HETGP_Prediction_by_bin.csv")
# High fidelity
HFGPMSE = pd.read_csv(base_path / "Hadron_HFGP_Prediction_by_bin.csv")
hetGPMSE, HFGPMSE

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

# %%
# Raymond's revised figure
sum_HFGPMSE = HFGPMSE.sum()
sum_hetGPMSE = hetGPMSE.sum()
compare_df = pd.DataFrame({"high_fidelity": sum_HFGPMSE, "hetgp": sum_hetGPMSE})

text_font_size = 22

# TODO(RJE): Note the pt range, collaboration, etc
# I considered including everything here (e.g. sqrt_s), but it doesn't matter overly much
# for the purposes of this exercise. To just highlight the important information, I'm going to cut down to the minimal.
text = r"Trained on JETSCAPE (MATTER + LBT)"
text += "\n" + r"corresponding to CMS, $\textit{JHEP 04 (2017) 039}$"
text += "\n" + r"0-5\%, Hadron $R_{\text{AA}}$, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$"
text += "\n" + r"$4.8 < p_{\text{T}} < 400\:\text{GeV}/c$"
plot_config = pb.PlotConfig(
    name="budget_residual_error_hadron",
    panels=[
        # Main panel
        pb.Panel(
            axes=[
                pb.AxisConfig(
                    "x",
                    label="Training data computational budget",
                    font_size=text_font_size,
                    use_major_axis_multiple_locator_with_base=1,
                ),
                pb.AxisConfig(
                    "y",
                    label=r"$\sum_{i \in p_{\text{T}}\:\text{bins},\:j \in \text{design pts.}}$ MSE$_{i,j}$",
                    font_size=text_font_size,
                    range=(0.076, 0.155),
                    # range=(0.044, 0.155),
                ),
            ],
            text=pb.TextConfig(x=0.95, y=0.81, text=text, font_size=18),
            legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
        ),
    ],
    figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.11}),
)

fig, ax = plt.subplots(
    1,
    1,
    figsize=(10, 6.25),
    sharex=True,
)

# Plot using line plot
for column, styling in method_styles.items():
    print(f"Plotting column '{column}'")
    styling_kwargs = styling.kwargs_for_plot_errorbar()
    styling_kwargs["linestyle"] = "-"
    ax.plot(compare_df[column].index, compare_df[column], label=styling.label, **styling_kwargs)
    # ax.plot(compare_df[column].index, compare_df[column], label=styling.label)
# compare_df.plot(kind="line", style={k: v.kwargs_for_plot_errorbar() for k, v in method_styles.items()}, ax=ax)
plot_config.apply(fig, ax=ax)
# plt.title("MSE (sum over bins) by budget")
# plt.xlabel("Budget")
# plt.ylabel("Sum of MSE")
# plt.grid(True)

_output_path = base_path / "figures"
_output_path.mkdir(parents=True, exist_ok=True)
fig.savefig(_output_path / f"{plot_config.name}.pdf")
plt.close(fig)

# TODO(RJE): Need to recall exactly what this budget means, and figure out how to translate it for the audience...

# %%
compare_df["hetgp"]

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
# # Jets
#
# Jet - HetGP

# %%
# HetGP
hetGPMSE = pd.read_csv(base_path / "Jet_HETGP_Prediction_by_bin.csv")
# High fidelity
HFGPMSE = pd.read_csv(base_path / "Jet_HFGP_Prediction_by_bin.csv")
hetGPMSE, HFGPMSE

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

# %%
# Raymond's revised jet figure
sum_HFGPMSE = HFGPMSE.sum()
sum_hetGPMSE = hetGPMSE.sum()
compare_df = pd.DataFrame({"high_fidelity": sum_HFGPMSE, "hetgp": sum_hetGPMSE})

text_font_size = 22

# TODO(RJE): Note the pt range, collaboration, etc
# I considered including everything here (e.g. sqrt_s), but it doesn't matter overly much
# for the purposes of this exercise. To just highlight the important information, I'm going to cut down to the minimal.
text = r"Trained on JETSCAPE (MATTER + LBT)"
text += "\n" + r"corresponding to ATLAS, $\textit{PLB 790 (2019) 108-128}$"
text += "\n" + r"0-10\%, $R = 0.4$ inclusive jet $R_{\text{AA}}$, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$"
text += "\n" + r"$100 < p_{\text{T}} < 1000\:\text{GeV}/c$"
plot_config = pb.PlotConfig(
    name="budget_residual_error_jet",
    panels=[
        # Main panel
        pb.Panel(
            axes=[
                pb.AxisConfig(
                    "x",
                    label="Training data computational budget",
                    font_size=text_font_size,
                    use_major_axis_multiple_locator_with_base=1,
                ),
                pb.AxisConfig(
                    "y",
                    label=r"$\sum_{i \in p_{\text{T}}\:\text{bins},\:j \in \text{design pts.}}$ MSE$_{i,j}$",
                    font_size=text_font_size,
                    range=(0.044, 0.155),
                ),
            ],
            text=pb.TextConfig(x=0.95, y=0.81, text=text, font_size=18),
            legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
        ),
    ],
    figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.11}),
)

fig, ax = plt.subplots(
    1,
    1,
    figsize=(10, 6.25),
    sharex=True,
)

# Plot using line plot
for column, styling in method_styles.items():
    print(f"Plotting column '{column}'")
    styling_kwargs = styling.kwargs_for_plot_errorbar()
    styling_kwargs["linestyle"] = "-"
    ax.plot(compare_df[column].index, compare_df[column], label=styling.label, **styling_kwargs)
    # ax.plot(compare_df[column].index, compare_df[column], label=styling.label)
# compare_df.plot(kind="line", style={k: v.kwargs_for_plot_errorbar() for k, v in method_styles.items()}, ax=ax)
plot_config.apply(fig, ax=ax)
# plt.title("MSE (sum over bins) by budget")
# plt.xlabel("Budget")
# plt.ylabel("Sum of MSE")
# plt.grid(True)

_output_path = base_path / "figures"
_output_path.mkdir(parents=True, exist_ok=True)
fig.savefig(_output_path / f"{plot_config.name}.pdf")
plt.close(fig)

# TODO(RJE): Need to recall exactly what this budget means, and figure out how to translate it for the audience...

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
# Melt both DataFrames and add a 'Source' label
HFGPMSE_melted = HFGPMSE.melt(var_name="Variable", value_name="Value")
HFGPMSE_melted["Source"] = "High Fidelity GP"

hetGPMSE_melted = hetGPMSE.melt(var_name="Variable", value_name="Value")
hetGPMSE_melted["Source"] = "VarP-GP"

# Combine both for plotting
df_combined = pd.concat([hetGPMSE_melted, HFGPMSE_melted])

text_font_size = 20

for log in [False, True]:
    # I considered including everything here (e.g. sqrt_s), but it doesn't matter overly much
    # for the purposes of this exercise. To just highlight the important information, I'm going to cut down to the minimal.
    text = r"Trained on JETSCAPE (MATTER + LBT)"
    text += "\n" + r"corresponding to ATLAS, $\textit{PLB 790 (2019) 108-128}$"
    text += "\n" + r"0-10\%, $R = 0.4$ inclusive jet $R_{\text{AA}}$, $\sqrt{s_{\text{NN}}} = 5.02\:\text{TeV}$"
    text += "\n" + r"$100 < p_{\text{T}} < 1000\:\text{GeV}/c$"
    plot_config = pb.PlotConfig(
        name=f"budget_residual_error_design_point_dist_jet{'_log' if log else ''}",
        panels=[
            # Main panel
            pb.Panel(
                axes=[
                    pb.AxisConfig(
                        "x",
                        label="Training data computational budget",
                        font_size=text_font_size,
                        use_major_axis_multiple_locator_with_base=1,
                    ),
                    pb.AxisConfig(
                        "y",
                        label=r"$\sum_{p_{\text{T}}\:\text{bins}}$ MSE",
                        font_size=text_font_size,
                        # range=(-0.0005, 0.038),
                        log=log,
                        range=(0.0005, 0.08) if log else (-0.0005, 0.038),
                    ),
                ],
                text=pb.TextConfig(x=0.63, y=0.97, text=text, font_size=18),
                legend=pb.LegendConfig(location="upper right", anchor=(0.975, 0.95), font_size=22),
            ),
        ],
        figure=pb.Figure(edge_padding={"left": 0.12, "bottom": 0.11}),
    )

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, 6.25),
        sharex=True,
    )

    # First is VarP-GP and second is High fidelity GP
    colors = ["#FF8301", "#845cba"]

    # Plot
    sns.boxplot(
        x="Variable",
        y="Value",
        hue="Source",
        palette=colors,
        saturation=1.0,
        ax=ax,
        width=0.6,
        medianprops={"linewidth": 3},
        whiskerprops={"linewidth": 2},
        capprops={"linewidth": 2},
        flierprops={
            "marker": "o",
            "markersize": 5,
            "markeredgecolor": "black",
            "markeredgewidth": 0.3,
        },
        data=df_combined,
    )
    # plt.show()

    # Apply colors to fliers based on their position
    # We can identify the fliers based on them not using lines
    lines = [line for line in ax.lines if line.get_linestyle() == "None"]
    for i, line in enumerate(lines):
        # Determine which hue group this outlier belongs to
        # NOTE: This depends on the data structure. If there were more sources,
        #       we would need to adjust
        color_idx = i % len(colors)
        # Set just the facecolor - we'll keep the black edges
        line.set_markerfacecolor(colors[color_idx])

    plot_config.apply(fig, ax=ax)

    _output_path = base_path / "figures"
    _output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(_output_path / f"{plot_config.name}.pdf")
    plt.close(fig)

# %%
