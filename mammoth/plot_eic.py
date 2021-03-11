""" Plot EIC qt.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch> ORNL
"""

from pathlib import Path
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


def _base_plot_label(jet_R: float, eta_limits: Tuple[float, float], min_q2: float, x_limits: Tuple[float, float]) -> str:
    eta_min, eta_max = eta_limits
    text = r"PYTHIA 6, e+p 10+250 $\text{GeV}^2$"
    # I add the jet pt cut because it's used in trigger. It's not exactly required here (ie. we could drop it),
    # but I'm not sure how to label that trigger. We don't need the \vec{p} trigger label because we're above that
    # range in (most of) our main plots.
    text += "\n" + r"$p_{\text{T}}^{\text{hard}} > 5\:\text{GeV}/c$, $p_{\text{T}}^{\text{jet}} > 5\:\text{GeV}/c$"
    text += "\n" + fr"$Q^{{2}} > {min_q2} \text{{GeV}}^{{2}}$, ${x_limits[0]} < x < {x_limits[1]}$"
    text += "\n" + fr"R = {jet_R:g} anti-$k_{{\text{{T}}}}$ jets"
    text += "\n" + fr"${eta_min + jet_R:g} < \eta_{{\text{{jet}}}} < {eta_max - jet_R:g}$"
    return text


def plot_qt(hist: binned_data.BinnedData,
            jet_R: float,
            eta_limits: Tuple[float, float],
            base_plot_label: str,
            output_dir: Path,
            ) -> bool:
    # Setup
    # Set max for plotting purposes.
    max_qt = 6

    bh_hist = hist.to_boost_histogram()

    hists = {}
    pt_ranges = [(10, 15), (20, 25), (30, 40)]

    for pt_range in pt_ranges:
        hists[pt_range] = binned_data.BinnedData.from_existing_data(
            bh_hist[bh.loc(pt_range[0]):bh.loc(pt_range[1]):bh.sum, :bh.loc(max_qt)]
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

    text = base_plot_label
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
    fig.savefig(output_dir / "qt_pythia6_ep.pdf")
    plt.close(fig)

    return True


def plot_qt_pt(hist: binned_data.BinnedData,
               jet_R: float,
               eta_limits: Tuple[float, float],
               label: str,
               base_plot_label: str,
               output_dir: Path,
            ) -> bool:
    bh_hist = hist.to_boost_histogram()

    hists = {}
    p_ranges = [(100, 150), (150, 200), (200, 250)]

    for p_range in p_ranges:
        hists[p_range] = binned_data.BinnedData.from_existing_data(
            bh_hist[bh.loc(p_range[0]):bh.loc(p_range[1]):bh.sum, :]
        )

        # Normalize
        print(f"p_range: {p_range}, normalization: {np.sum(hists[p_range].values)}")
        hists[p_range] /= np.sum(hists[p_range].values)
        #hists[p_range] /= hists[p_range].axes[0].bin_widths

    fig, ax = plt.subplots(figsize=(12, 9))

    for p_range, h in hists.items():
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
            label=fr"${p_range[0]} < |\vec{{p}}| < {p_range[1]}\:\text{{GeV}}/c$",
            ax=ax,
        )

    text = base_plot_label
    ax.set_xlabel(r"$q_{\text{T}} / p_{\text{T}}^{\text{" + label + r"}}\:(\text{GeV}/c)$")
    ax.set_ylabel(r"$1/N_{\text{" + label + r"}}\:\text{d}N/\text{d}(q_{\text{T}}/p_{\text{T}}^{\text{" + label + r"}})\:(\text{GeV}/c)^{-1}$")
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
    fig.savefig(output_dir / f"qt_pt_{label}_pythia6_ep.pdf")
    plt.close(fig)

    return True


def plot_qt_pt_comparison(
    hist_pt_jet: binned_data.BinnedData,
    hist_pt_e: binned_data.BinnedData,
    jet_R: float,
    eta_limits: Tuple[float, float],
    base_plot_label: str,
    output_dir: Path,
) -> bool:
    bh_hist_pt_jet = hist_pt_jet.to_boost_histogram()
    bh_hist_pt_e = hist_pt_e.to_boost_histogram()

    hists = {}
    p_ranges = [(100, 150), (150, 200), (200, 250)]

    for p_range in p_ranges:
        # Project and normalize
        proj_pt_jet = binned_data.BinnedData.from_existing_data(
            bh_hist_pt_jet[bh.loc(p_range[0]):bh.loc(p_range[1]):bh.sum, :]
        )
        proj_pt_e = binned_data.BinnedData.from_existing_data(
            bh_hist_pt_e[bh.loc(p_range[0]):bh.loc(p_range[1]):bh.sum, :]
        )
        print(f"p_range: {p_range}, jet normalization: {np.sum(proj_pt_jet.values)}, e normalization: {np.sum(proj_pt_e.values)}")
        proj_pt_jet /= np.sum(proj_pt_jet.values)
        proj_pt_e /= np.sum(proj_pt_e.values)

        fig, ax = plt.subplots(figsize=(12, 9))

        mplhep.histplot(
            H=proj_pt_jet.values,
            bins=proj_pt_jet.axes[0].bin_edges,
            yerr=proj_pt_jet.errors,
            label=r"$q_{\text{T}} / p_{\text{T}}^{\text{jet}}\:(\text{GeV}/c)$",
            ax=ax,
        )
        mplhep.histplot(
            H=proj_pt_e.values,
            bins=proj_pt_e.axes[0].bin_edges,
            yerr=proj_pt_e.errors,
            label=r"$q_{\text{T}} / p_{\text{T}}^{\text{e}}\:(\text{GeV}/c)$",
            ax=ax,
        )

        text = base_plot_label
        text += "\n" + fr"${p_range[0]} < |\vec{{p}}| < {p_range[1]}\:\text{{GeV}}/c$"
        ax.set_xlabel(r"$q_{\text{T}} / p_{\text{T}}^{\text{X}}\:(\text{GeV}/c)$")
        ax.set_ylabel(r"$1/N_{\text{X}}\:\text{d}N/\text{d}(q_{\text{T}}/p_{\text{T}}^{\text{X}})\:(\text{GeV}/c)^{-1}$")
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
        fig.savefig(output_dir / f"qt_pt_p_range_{p_range[0]}_{p_range[1]}_pythia6_ep.pdf")
        plt.close(fig)

    return True



if __name__ == "__main__":
    jet_R = 0.7
    eta_limits = (1.1, 3.5)
    min_q2 = 100
    x_limits = (0.05, 0.6)
    base_plot_label = _base_plot_label(jet_R=jet_R, eta_limits=eta_limits, min_q2=min_q2, x_limits=x_limits)
    output_dir = Path("output") / "eic_qt"
    output_dir.mkdir(parents=True, exist_ok=True)

    y = yaml.yaml(modules_to_register=[binned_data])
    with open(output_dir / "qt.yaml", "r") as f:
        hists = y.load(f)
    print("Plotting qt")
    plot_qt(
        hist=hists["qt"],
        jet_R=jet_R,
        eta_limits=eta_limits,
        base_plot_label=base_plot_label,
        output_dir=output_dir,
    )
    print("Plotting qt/pt jet")
    plot_qt_pt(
        hist=hists["qt_pt"],
        jet_R=jet_R,
        eta_limits=eta_limits,
        label="jet",
        base_plot_label=base_plot_label,
        output_dir=output_dir,
    )
    print("Plotting qt/pt e")
    plot_qt_pt(
        hist=hists["qt_pte"],
        jet_R=jet_R,
        eta_limits=eta_limits,
        label="e",
        base_plot_label=base_plot_label,
        output_dir=output_dir,
    )
    print("Plotting qt/pt jet vs e comparison")
    plot_qt_pt_comparison(
        hist_pt_jet=hists["qt_pt"],
        hist_pt_e=hists["qt_pte"],
        jet_R=jet_R,
        eta_limits=eta_limits,
        base_plot_label=base_plot_label,
        output_dir=output_dir,
    )

