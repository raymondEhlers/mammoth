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
import pachyderm.plot as pb
import polars as pl

base_path = Path("projects/hetgp/design_toy")

pb.configure()

# %%
text_font_size = 22

# %%
data = pl.read_csv(
    base_path / "design_visualization.csv",
    # Column names are only changed at the very end, so we use the original column names for schema overrides.
    schema_overrides={"M_original": pl.Float64, "M_optimal": pl.Float64},
    new_columns=["x", "simple", "optimal"],
)

# %%
data


# %%
def plot(df: pl.DataFrame, plot_config: pb.PlotConfig, select_data: list[str] | None = None) -> None:
    if select_data is None:
        select_data = ["simple", "optimal"]
    single_selection = len(select_data) == 1

    fig, ax = plt.subplots(
        1,
        1,
        # The single selection is smaller than I usually like, but it makes it easier to get the
        # text/labels to be a reasonable size with just one plot_config. Plus, it's a pdf so I can
        # always just enlarge it in e.g. a presentation
        figsize=(10, 6.25) if not single_selection else (6, 4.8),
        sharex=True,
    )

    jitter = 0.0075 if len(select_data) > 1 else 0
    if "simple" in select_data:
        ax.plot(
            df["x"] - jitter,
            df["simple"],
            marker="s",
            markersize=15,
            linestyle="",
            color="#4bafd0",
            alpha=1.0 if single_selection else 0.9,
            label="Poor",
        )
    if "optimal" in select_data:
        ax.plot(
            df["x"] + jitter,
            df["optimal"],
            marker="o",
            markersize=15,
            linestyle="",
            color="#FF8301",
            alpha=1.0 if single_selection else 0.9,
            label="Optimized",
        )

    plot_config.apply(fig, ax=ax)
    # Tweak presentation
    import matplotlib as mpl

    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=0.2))

    _output_path = base_path / "figures"
    _output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(_output_path / f"{plot_config.name}.pdf")
    plt.close(fig)


# %%
# I considered including everything here (e.g. sqrt_s), but it doesn't matter overly much
# for the purposes of this exercise. To just highlight the important information, I'm going to cut down to the minimal.
text = ""
for select_data in [["simple"], ["optimal"]]:
    single_selection = len(select_data) == 1
    name = "combined"
    if single_selection:
        name = select_data[0]
    plot_config = pb.PlotConfig(
        name=f"design_toy_{name}",
        panels=[
            # Main panel
            pb.Panel(
                axes=[
                    pb.AxisConfig(
                        "x",
                        label="Parameter $x$",
                        # NOTE: Added extra size since the x looks quite small for a single selection
                        # font_size=text_font_size * 1.2 if single_selection else text_font_size,
                        # Now that the label is longer, I'm keeping them both the same size...
                        font_size=text_font_size,
                        use_major_axis_multiple_locator_with_base=1,
                    ),
                    pb.AxisConfig(
                        "y",
                        # label="Target event count (arb.)",
                        # Treated as if it's a percentage of a target event count.
                        label=r"Target event count (\%)",
                        font_size=text_font_size,
                        range=(15, 110),
                    ),
                ],
                text=pb.TextConfig(x=0.95, y=0.81, text=text, font_size=18),
                legend=pb.LegendConfig(
                    location="upper left", anchor=(0.025, 0.96), font_size=22, marker_label_spacing=0.01
                ),
                title=pb.TitleConfig(
                    text="{x} design".format(x="Poor" if name == "simple" else "Optimized"), size=text_font_size
                ),
            ),
        ],
        figure=(
            pb.Figure(edge_padding={"left": 0.15, "bottom": 0.15, "top": 0.92})
            if single_selection
            else pb.Figure(edge_padding={"left": 0.09, "bottom": 0.105, "top": 0.92})
        ),
    )

    plot(df=data, select_data=select_data, plot_config=plot_config)

# %%
