"""Example of how to create a two panel figure with bar plots

It will have a regular figure in the upper plot, and then a ratio in the lower
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).parent


def plot() -> None:
    # Some example data to plot
    n_points = 6
    group_A = np.array([1.5, 2.5, 1.5, 2, 2.3, 0.35], dtype=np.float32)
    group_B = np.array([1.25, 2.75, 1.5, 2.1, 1.8, 0.39], dtype=np.float32)

    # Define the overall figure layout
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Based on https://stackoverflow.com/a/59421062 and https://matplotlib.org/3.0.0/gallery/statistics/barchart_demo.html
    # Need to define an index, which we will then shift as needed to plot each bar
    index = np.arange(n_points)
    width = 0.15

    # Plot in the main panel
    # Group A
    ax.bar(index, group_A, width, label="Group A")
    # Group B: shift the next bar over so they sit side-by-side
    ax.bar(index + width, group_B, width, label="Group B")

    # Define the ratio (or whatever is desired)
    ratio = group_A / group_B

    # And then plot the ratio in the lower panel
    # The x position is shifted so that the center of the point splits the bars.
    ax_ratio.errorbar(
        index + width / 2,
        ratio,
        xerr=width * 2,
        yerr=np.zeros_like(ratio),
        linestyle="",
        marker="o",
    )

    # Shift x-axis labels so they split between the bars
    ax.set_xticks(index + width / 2)
    ax_ratio.set_xticks(index + width / 2)
    ax_ratio.set_xticklabels(["alpha_s", "Q0", "C1", "C2", "tau", "C3"])

    # Aesthetics
    ax.set_ylabel("Sensitivity")
    ax_ratio.set_ylim([0.61, 1.39])
    ax_ratio.set_xlabel("Parameter")
    ax_ratio.set_ylabel("Ratio")
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

    fig.savefig(here / "two_panel_figure.pdf")
    plt.close(fig)


if __name__ == "__main__":
    plot()
