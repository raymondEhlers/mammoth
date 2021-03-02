""" Plot EIC qt.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch> ORNL
"""

from typing import Mapping, Sequence, Tuple

import boost_histogram as bh
import mplhep
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pachyderm.plot

from pachyderm import binned_data, yaml


pachyderm.plot.configure()
# Enable ticks on all sides
# Unfortunately, some of this is overriding the pachyderm plotting style.
# That will have to be updated eventually...
matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["xtick.minor.top"] = True
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["ytick.minor.right"] = True


def plot(hist: binned_data.BinnedData) -> bool:
    bh_hist = hist.to_boost_histogram()

    hists = {}
    pt_ranges = [(10, 15), (20, 25), (30, 40)]

    for pt_range in pt_ranges:
        hists[pt_range] = binned_data.BinnedData.from_existing_data(
            bh_hist[bh.loc(pt_range[0]):bh.loc(pt_range[1]):bh.sum, :bh.loc(6)]
        )

        # Normalize
        print(f"pt_range: {pt_range}, normalization: {np.sum(hists[pt_range].values)}")
        hists[pt_range] /= np.sum(hists[pt_range].values)

    fig, ax = plt.subplots(figsize=(12, 9))

    for pt_range, h in hists.items():
        # data = ax.errorbar(
        #     h.axes[0].bin_centers,
        #     h.values,
        #     xerr=h.axes[0].bin_widths,
        #     yerr=h.errors,
        #     marker=".",
        #     #markersize=11,
        #     linestyle="",
        #     zorder=10,
        #     label=fr"${pt_range[0]} < p_{{\text{{T}}}} < {pt_range[1]}\:\text{{GeV}}/c$",
        # )
        mplhep.histplot(
            H=h.values,
            bins=h.axes[0].bin_edges,
            yerr=h.errors,
            label=fr"${pt_range[0]} < p_{{\text{{T}}}} < {pt_range[1]}\:\text{{GeV}}/c$",
            ax=ax,
        )

    jet_R = 1.0
    text = "PYTHIA 6, e+p 10+250 GeV"
    text += "\n" + r"$p_{\text{T}}^{\text{hard}} > 5\:\text{GeV}/c$"
    text += "\n" + fr"R = {jet_R:g} anti-$k_{{\text{{T}}}}$ jets"
    text += "\n" + fr"${1 + jet_R:g} < \eta_{{\text{{jet}}}} < {4 - jet_R:g}$"
    ax.set_xlabel(r"$q_{\text{T}}\:(\text{GeV}/c)$")
    ax.set_ylabel(r"$1/N_{\text{jets}}\:\text{d}N/\text{d}q_{\text{T}}\:(\text{GeV}/c)^{-1}$")
    ax.legend(
        loc="upper right",
        frameon=False,
    )
    ax.text(
        0.68, 0.97,
        text,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
        transform=ax.transAxes,
        fontsize=18,
    )

    fig.tight_layout()
    fig.savefig("qt.pdf")

    return True


if __name__ == "__main__":
    y = yaml.yaml(modules_to_register=[binned_data])
    with open("qt.yaml", "r") as f:
        h, = y.load(f)
    plot(h)
