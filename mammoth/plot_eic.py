""" Plot EIC qt.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch> ORNL
"""

from pathlib import Path
from typing import Mapping, Sequence, Tuple, Union

import boost_histogram as bh
import mplhep
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pachyderm.plot
import seaborn as sns
from pachyderm import binned_data, yaml
from scipy import optimize


pachyderm.plot.configure()
# Enable ticks on all sides
# Unfortunately, some of this is overriding the pachyderm plotting style.
# That will have to be updated eventually...
matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["xtick.minor.top"] = True
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["ytick.minor.right"] = True


def _gaussian(x: Union[np.ndarray, float], mean: float, sigma: float, amplitude: float) -> Union[np.ndarray, float]:
    r"""Extended gaussian.

    .. math::

        f = A / \sqrt{2 * \pi * \sigma^{2}} * \exp{-\frac{(x - \mu)^{2}}{(2 * \sigma^{2}}}

    Args:
        x: Value(s) where the gaussian should be evaluated.
        mean: Mean of the gaussian distribution.
        sigma: Width of the gaussian distribution.
        amplitude: Amplitude of the gaussian
    Returns:
        Calculated gaussian value(s).
    """
    return amplitude / np.sqrt(2 * np.pi * np.square(sigma)) * np.exp(-1.0 / 2.0 * np.square((x - mean) / sigma))  # type: ignore


def _base_plot_label(jet_R: float, eta_limits: Tuple[float, float],
                     #min_q2: float, x_limits: Tuple[float, float]
                     ) -> str:
    eta_min, eta_max = eta_limits
    text = r"PYTHIA 6, e+p 10+275 $\text{GeV}^2$"
    #text += "\n" + fr"$Q^{{2}} > {min_q2} \text{{GeV}}^{{2}}$, ${x_limits[0]} < x < {x_limits[1]}$"
    text += "\n" + fr"R = {jet_R:g} anti-$k_{{\text{{T}}}}$ jets"
    text += "\n" + fr"${eta_min + jet_R:g} < \eta_{{\text{{jet}}}} < {eta_max - jet_R:g}$"
    return text


def _mean_values_label(mean_x: float, mean_Q2: float) -> str:
    return fr"$\langle Q^{{2}} \rangle = {round(mean_Q2)}\:\text{{GeV}}^{{2}}$, $\langle x \rangle = {mean_x:.03g}$"


def plot_qt(hist: binned_data.BinnedData,
            base_plot_label: str,
            means: Mapping[Tuple[float, float], Mapping[str, float]],
            output_dir: Path,
            ) -> bool:
    # Setup
    # Set max for plotting purposes.
    p_range_for_mean_label = (0, 300)
    max_qt = 6

    bh_hist = hist.to_boost_histogram()

    hists = {}
    pt_ranges = [(10, 15), (20, 25), (30, 40)]
    for pt_range in pt_ranges:
        hists[pt_range] = binned_data.BinnedData.from_existing_data(
            bh_hist[bh.loc(pt_range[0]):bh.loc(pt_range[1]):bh.sum, :: bh.rebin(2)]
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
            linewidth=2,
            ax=ax,
        )

    text = base_plot_label
    text += "\n" + _mean_values_label(mean_x = means[p_range_for_mean_label]["x"], mean_Q2 = means[p_range_for_mean_label]["Q2"])
    ax.set_xlabel(r"$q_{\text{T}}\:(\text{GeV}/c)$", fontsize=20)
    ax.set_xlim([-0.25, max_qt * 1.05])
    ax.set_ylabel(r"$1/N_{\text{jets}}\:\text{d}N/\text{d}q_{\text{T}}\:(\text{GeV}/c)^{-1}$", fontsize=20)
    ax.legend(
        loc="upper right",
        frameon=False,
        bbox_to_anchor=(0.97, 0.765),
        # If we specify an anchor, we want to reduce an additional padding
        # to ensure that we have accurate placement.
        borderaxespad=0,
        borderpad=0,
        fontsize=20,
    )
    ax.text(
        0.97, 0.97,
        text,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
        transform=ax.transAxes,
        fontsize=20,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "qt_pythia6_ep.pdf")
    plt.close(fig)

    return True


def plot_qt_pt_as_function_of_p(hist: binned_data.BinnedData,
                                label: str,
                                base_plot_label: str,
                                means: Mapping[Tuple[float, float], Mapping[str, float]],
                                output_dir: Path,
                                debug_fit: bool = False,
                                ) -> bool:
    # Setup
    bh_hist = hist.to_boost_histogram()

    hists = {}
    p_ranges = [(100, 150), (150, 200), (200, 250)]
    # Assuming linearly increasing.
    full_p_range = (p_ranges[0][0], p_ranges[-1][-1])

    for p_range in p_ranges:
        hists[p_range] = binned_data.BinnedData.from_existing_data(
            bh_hist[bh.loc(p_range[0]):bh.loc(p_range[1]):bh.sum, :]
        )

        # Normalize
        print(f"p_range: {p_range}, normalization: {np.sum(hists[p_range].values)}")
        hists[p_range] /= np.sum(hists[p_range].values)
        #hists[p_range] /= hists[p_range].axes[0].bin_widths

    for do_fit in [False, True]:
        fig, ax = plt.subplots(figsize=(12, 9))
        for i, (p_range, h) in enumerate(hists.items()):
            # Fit and plot
            width = -100
            if do_fit:
                print(f"p_range: {p_range}")
                x_linspace_min_for_plotting = -0.2 if debug_fit else 0.0
                if False:
                    fixed_gaussian_mean = 0.0
                    popt, _ = optimize.curve_fit(
                        lambda x, w, a: _gaussian(x, fixed_gaussian_mean, w, a), h.axes[0].bin_centers, h.values,
                        p0 = [0.1, 1],
                        maxfev = 2000,
                    )
                    print(f"Mean: {fixed_gaussian_mean}, Width: {popt[0]:.03g}, amplitude: {popt[1]:.03g}")
                    p = ax.plot(
                        #h.axes[0].bin_centers,
                        #_gaussian(h.axes[0].bin_centers, 0.025, *popt),
                        np.linspace(x_linspace_min_for_plotting, 0.5, 100),
                        _gaussian(np.linspace(x_linspace_min_for_plotting, 0.5, 100), fixed_gaussian_mean, *popt),
                        linestyle="--",
                        linewidth=2,
                        # color=p[0].step.get_color(),
                        # We want to be on top, even if plotted first.
                        zorder=10,
                    )
                    # Store for plotting
                    width = popt[0]
                else:
                    initial_mean = [0.05, 0.025, 0.01]
                    popt, _ = optimize.curve_fit(
                        _gaussian, h.axes[0].bin_centers, h.values,
                        p0 = [initial_mean[i], 0.05, 0.1],
                        #p0 = [0.0, 0.1],
                        maxfev = 50000,
                    )
                    #print(f"Mean: {popt[0]}, Width: {popt[1]}, amplitude: {popt[2]}")
                    print(f"Mean: {popt[0]:.03g}, Width: {popt[1]:.03g}")
                    p = ax.plot(
                        #h.axes[0].bin_centers,
                        #_gaussian(h.axes[0].bin_centers, *popt),
                        np.linspace(x_linspace_min_for_plotting, 0.5, 100),
                        _gaussian(np.linspace(x_linspace_min_for_plotting, 0.5, 100), *popt),
                        linestyle="--",
                        linewidth=2,
                        # color=p[0].step.get_color(),
                        # We want to be on top, even if plotted first.
                        zorder=10,
                    )
                    # Store for plotting
                    width = popt[1]

                # RMS from ROOT
                try:
                    import ROOT
                    h_ROOT = h.to_ROOT()
                    #fu = ROOT.TF1("fu", "[2] * TMath::Gaus(x,[0],[1])")
                    #fu.SetParameters(0, 0.1, 0.1)
                    #res = h_ROOT.Fit("fu")
                    #import IPython; IPython.embed()
                    print(f"RMS from ROOT: {h_ROOT.GetRMS():.03g}, Std Dev: {h_ROOT.GetStdDev():.03g}")
                except ImportError:
                    pass

            kwargs = {}
            plot_label = fr"${p_range[0]} < |\vec{{p}}| < {p_range[1]}\:\text{{GeV}}/c$"
            if do_fit:
                plot_label += fr", $\sigma = {width:.02g}$"
                kwargs = {
                    "color": p[0].get_color()
                }
            mplhep.histplot(
                H=h.values,
                bins=h.axes[0].bin_edges,
                yerr=h.errors,
                label=plot_label,
                linewidth=2,
                ax=ax,
                **kwargs,
            )

        text = base_plot_label
        text += "\n" + _mean_values_label(mean_x = means[full_p_range]["x"], mean_Q2 = means[full_p_range]["Q2"])
        if do_fit:
            text += "\n" + r"Gaussian fit"
        ax.set_xlabel(r"$q_{\text{T}} / p_{\text{T}}^{\text{" + label + r"}}$", fontsize=20)
        ax.set_ylabel(r"$1/N_{\text{" + label + r"}}\:\text{d}N/\text{d}(q_{\text{T}}/p_{\text{T}}^{\text{" + label + r"}})$", fontsize=20)
        # Focus on range of interest.
        min_x = -0.025
        if debug_fit:
            min_x = -0.1
        ax.set_xlim([min_x, 0.4])
        ax.legend(
            loc="upper right",
            frameon=False,
            bbox_to_anchor=(0.97, 0.765),
            # If we specify an anchor, we want to reduce an additional padding
            # to ensure that we have accurate placement.
            borderaxespad=0,
            borderpad=0,
            fontsize=20,
        )
        ax.text(
            0.97, 0.97,
            text,
            horizontalalignment="right",
            verticalalignment="top",
            multialignment="right",
            transform=ax.transAxes,
            fontsize=20,
        )

        fig.tight_layout()
        fig.savefig(output_dir / f"qt_pt_{label}_{'fit_' if do_fit else ''}pythia6_ep.pdf")
        plt.close(fig)

    return True


def plot_qt_pt_comparison(
    hists: Mapping[str, binned_data.BinnedData],
    base_plot_label: str,
    means: Mapping[Tuple[float, float], Mapping[str, float]],
    output_dir: Path,
) -> bool:
    p_ranges = [(100, 150), (150, 200), (200, 250)]
    for p_range in p_ranges:
        fig, ax = plt.subplots(figsize=(12, 9))

        # Project and normalize
        for label, hist in hists.items():
            bh_hist = hist.to_boost_histogram()
            h = binned_data.BinnedData.from_existing_data(
                bh_hist[bh.loc(p_range[0]):bh.loc(p_range[1]):bh.sum, :]
            )
            print(f"label: {label}, p_range: {p_range}, normalization: {np.sum(h.values)}")
            h /= np.sum(h.values)

            mplhep.histplot(
                H=h.values,
                bins=h.axes[0].bin_edges,
                yerr=h.errors,
                label=r"$q_{\text{T}} / p_{\text{T}}^{\text{" + label + r"}}$",
                linewidth=2,
                ax=ax,
            )

        text = base_plot_label
        text += "\n" + _mean_values_label(mean_x = means[p_range]["x"], mean_Q2 = means[p_range]["Q2"])
        text += "\n" + fr"${p_range[0]} < |\vec{{p}}| < {p_range[1]}\:\text{{GeV}}/c$"
        ax.set_xlabel(r"$q_{\text{T}} / p_{\text{T}}^{\text{X}}$", fontsize=20)
        ax.set_ylabel(r"$1/N_{\text{X}}\:\text{d}N/\text{d}(q_{\text{T}}/p_{\text{T}}^{\text{X}})$", fontsize=20)
        # Focus on range of interest.
        ax.set_xlim([-0.025, 0.4])
        ax.legend(
            loc="upper right",
            frameon=False,
            bbox_to_anchor=(0.97, 0.76),
            # If we specify an anchor, we want to reduce an additional padding
            # to ensure that we have accurate placement.
            borderaxespad=0,
            borderpad=0,
            fontsize=20,
        )
        ax.text(
            0.97, 0.97,
            text,
            horizontalalignment="right",
            verticalalignment="top",
            multialignment="right",
            transform=ax.transAxes,
            fontsize=20,
        )

        fig.tight_layout()
        fig.savefig(output_dir / f"qt_pt_p_range_{p_range[0]}_{p_range[1]}_pythia6_ep.pdf")
        plt.close(fig)

    return True


if __name__ == "__main__":
    jet_R = 0.7
    eta_limits = (1.1, 3.5)
    p_ranges = [(100, 150), (150, 200), (200, 250)]
    #min_q2 = 300
    #x_limits = (0.05, 0.8)
    base_plot_label = _base_plot_label(jet_R=jet_R, eta_limits=eta_limits)
    output_dir = Path("output") / "eic_qt_all_q2_cuts_narrow_bins"
    output_dir.mkdir(parents=True, exist_ok=True)

    y = yaml.yaml(modules_to_register=[binned_data])
    with open(output_dir / "qt.yaml", "r") as f:
        hists = y.load(f)

    # pop because it's not a hist...
    means = hists.pop("means")

    with sns.color_palette("Set2"):
        print("Plotting qt")
        plot_qt(
            hist=hists["qt"],
            base_plot_label=base_plot_label,
            means=means,
            output_dir=output_dir,
        )
        print("Plotting qt/pt jet")
        plot_qt_pt_as_function_of_p(
            hist=hists["qt_pt_jet"],
            label="jet",
            base_plot_label=base_plot_label,
            means=means,
            output_dir=output_dir,
        )
        print("Plotting qt/pt e")
        plot_qt_pt_as_function_of_p(
            hist=hists["qt_pt_electron"],
            label="e",
            base_plot_label=base_plot_label,
            means=means,
            output_dir=output_dir,
        )
        print("Plotting qt/pt parton")
        plot_qt_pt_as_function_of_p(
            hist=hists["qt_pt_parton"],
            label="parton",
            base_plot_label=base_plot_label,
            means=means,
            output_dir=output_dir,
        )
        print("Plotting qt/pt jet vs e comparison")
        plot_qt_pt_comparison(
            hists={
                "jet": hists["qt_pt_jet"],
                "e": hists["qt_pt_electron"],
                "parton": hists["qt_pt_parton"],
            },
            base_plot_label=base_plot_label,
            means=means,
            output_dir=output_dir,
        )

