"""Plot EIC qt.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch> ORNL
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import boost_histogram as bh
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep
import numpy as np
import numpy.typing as npt
import pachyderm.plot
import seaborn as sns
from pachyderm import binned_data, yaml
from scipy import optimize

import mammoth.helpers
from mammoth.eic import eic_qt

logger = logging.getLogger(__name__)
pachyderm.plot.configure()
# Enable ticks on all sides
# Unfortunately, some of this is overriding the pachyderm plotting style.
# That will have to be updated eventually...
mpl.rcParams["xtick.top"] = True
mpl.rcParams["xtick.minor.top"] = True
mpl.rcParams["ytick.right"] = True
mpl.rcParams["ytick.minor.right"] = True


def _gaussian(
    x: npt.NDArray[np.float64] | float, mean: float, sigma: float, amplitude: float
) -> npt.NDArray[np.float64] | float:
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
    return amplitude / np.sqrt(2 * np.pi * np.square(sigma)) * np.exp(-1.0 / 2.0 * np.square((x - mean) / sigma))  # type: ignore[no-any-return]


def _base_plot_label(
    eta_limits: tuple[float, float],
    jet_R: float | None = None,
    # min_q2: float, x_limits: Tuple[float, float]
) -> str:
    eta_min, eta_max = eta_limits
    text = r"PYTHIA 6, e+p 10+275 $\text{GeV}^2$"
    if jet_R is not None:
        # text += "\n" + fr"$Q^{{2}} > {min_q2} \text{{GeV}}^{{2}}$, ${x_limits[0]} < x < {x_limits[1]}$"
        text += "\n" + rf"R = {jet_R} anti-$k_{{\text{{T}}}}$ jets"
        text += "\n" + rf"${eta_min + jet_R:g} < \eta_{{\text{{jet}}}} < {eta_max - jet_R:g}$"
    else:
        text += "\n" + r"anti-$k_{\text{T}}$ jets"
        text += "\n" + rf"${eta_min} + R < \eta_{{\text{{jet}}}} < {eta_max} - R$"
    return text


def _mean_values_label(mean_x: float, mean_Q2: float) -> str:
    return rf"$\langle Q^{{2}} \rangle = {round(mean_Q2)}\:\text{{GeV}}^{{2}}$, $\langle x \rangle = {mean_x:.03g}$"


def plot_jet_p(
    hists: Mapping[str, Mapping[str, binned_data.BinnedData]],
    base_plot_label: str,
    jet_R_values: Sequence[float],
    means: Mapping[str, Mapping[tuple[float, float], Mapping[str, float]]],
    output_dir: Path,
) -> bool:
    # Setup
    p_range_for_mean_label = (0, 300)

    fig, ax = plt.subplots(figsize=(12, 9))
    for jet_R in jet_R_values:
        jet_R_str = eic_qt.jet_R_to_str(jet_R)
        h = hists[jet_R_str]["jet_p"]

        # Normalize
        h /= sum(h.values)

        # Plot
        mplhep.histplot(
            H=h.values,
            bins=h.axes[0].bin_edges,
            yerr=h.errors,
            label=rf"$R = {jet_R}$",
            linewidth=2,
            ax=ax,
        )

    text = base_plot_label
    # Assuming R = 0.7 jets here. Perhaps not entirely reasonable, but close enough for effectively a QA plot.
    text += "\n" + _mean_values_label(
        mean_x=means[eic_qt.jet_R_to_str(0.7)][p_range_for_mean_label]["x"],
        mean_Q2=means[eic_qt.jet_R_to_str(0.7)][p_range_for_mean_label]["Q2"],
    )
    ax.set_xlabel(r"$p_{\text{jet}}\:(\text{GeV}/c)$", fontsize=20)
    ax.set_yscale("log")
    ax.set_ylabel(r"$1/N_{\text{jets}}\:\text{d}N/\text{d}p_{\text{jet}}\:(\text{GeV}/c)^{-1}$", fontsize=20)
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
        0.97,
        0.97,
        text,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
        transform=ax.transAxes,
        fontsize=20,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "p_jet_pythia6_ep.pdf")
    plt.close(fig)

    return True


def plot_jet_pt(
    hist: binned_data.BinnedData,
    base_plot_label: str,
    jet_R: float,
    means: Mapping[tuple[float, float], Mapping[str, float]],
    output_dir: Path,
) -> bool:
    # Setup
    p_ranges = [(100, 150), (150, 200), (200, 250), (0, 300)]
    p_range_for_mean_label = (0, 300)
    bh_hist: bh.Histogram = hist.to_boost_histogram()

    fig, ax = plt.subplots(figsize=(12, 9))
    for p_range in p_ranges:
        h = binned_data.BinnedData.from_existing_data(
            bh_hist[bh.loc(p_range[0]) : bh.loc(p_range[1]) : bh.sum, :]  # type: ignore[misc]
        )

        # Normalize
        h /= np.sum(h.values)

        # Plot
        mplhep.histplot(
            H=h.values,
            bins=h.axes[0].bin_edges,
            yerr=h.errors,
            label=rf"${p_range[0]} < |\vec{{p}}| < {p_range[1]}\:\text{{GeV}}/c$",
            linewidth=2,
            ax=ax,
        )

    text = base_plot_label
    text += "\n" + _mean_values_label(
        mean_x=means[p_range_for_mean_label]["x"], mean_Q2=means[p_range_for_mean_label]["Q2"]
    )
    ax.set_xlabel(r"$p_{\text{T}}^{\text{jet}}\:(\text{GeV}/c)$", fontsize=20)
    ax.set_yscale("log")
    ax.set_ylabel(r"$1/N_{\text{jets}}\:\text{d}N/\text{d}p_{\text{T}}^{\text{jet}}\:(\text{GeV}/c)^{-1}$", fontsize=20)
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
        0.97,
        0.97,
        text,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
        transform=ax.transAxes,
        fontsize=20,
    )

    fig.tight_layout()
    fig.savefig(output_dir / f"pt_jet_{eic_qt.jet_R_to_str(jet_R)}_pythia6_ep.pdf")
    plt.close(fig)

    return True


def plot_jet_constituent_multiplicity(
    hist: binned_data.BinnedData,
    base_plot_label: str,
    jet_R: float,
    means: Mapping[tuple[float, float], Mapping[str, float]],
    output_dir: Path,
) -> bool:
    # Setup
    p_ranges = [(100, 150), (150, 200), (200, 250), (0, 300)]
    p_range_for_mean_label = (0, 300)
    bh_hist = hist.to_boost_histogram()

    fig, ax = plt.subplots(figsize=(12, 9))
    for p_range in p_ranges:
        h = binned_data.BinnedData.from_existing_data(
            bh_hist[bh.loc(p_range[0]) : bh.loc(p_range[1]) : bh.sum, :]  # type: ignore[misc]
        )

        # Normalize
        h /= np.sum(h.values)

        # Plot
        mplhep.histplot(
            H=h.values,
            bins=h.axes[0].bin_edges,
            yerr=h.errors,
            label=rf"${p_range[0]} < |\vec{{p}}| < {p_range[1]}\:\text{{GeV}}/c$",
            linewidth=2,
            ax=ax,
        )

    text = base_plot_label
    text += "\n" + _mean_values_label(
        mean_x=means[p_range_for_mean_label]["x"], mean_Q2=means[p_range_for_mean_label]["Q2"]
    )
    ax.set_xlabel(r"$N_{\text{const.}}^{\text{jet}}$", fontsize=20)
    ax.set_ylabel(r"$1/N_{\text{jets}}\:\text{d}N/\text{d}N_{\text{const.}}^{\text{jet}}$", fontsize=20)
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
        0.97,
        0.97,
        text,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
        transform=ax.transAxes,
        fontsize=20,
    )

    fig.tight_layout()
    fig.savefig(output_dir / f"jet_multiplicity_{eic_qt.jet_R_to_str(jet_R)}_pythia6_ep.pdf")
    plt.close(fig)

    return True


def plot_qt(
    hist: binned_data.BinnedData,
    base_plot_label: str,
    jet_R: float,
    means: Mapping[tuple[float, float], Mapping[str, float]],
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
            bh_hist[bh.loc(pt_range[0]) : bh.loc(pt_range[1]) : bh.sum, :: bh.rebin(2)]  # type: ignore[misc]
        )

        # Normalize
        logger.info(f"pt_range: {pt_range}, normalization: {np.sum(hists[pt_range].values)}")
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
            label=rf"${pt_range[0]} < p_{{\text{{T}}}} < {pt_range[1]}\:\text{{GeV}}/c$",
            linewidth=2,
            ax=ax,
        )

    text = base_plot_label
    text += "\n" + _mean_values_label(
        mean_x=means[p_range_for_mean_label]["x"], mean_Q2=means[p_range_for_mean_label]["Q2"]
    )
    ax.set_xlabel(r"$q_{\text{T}}\:(\text{GeV}/c)$", fontsize=20)
    ax.set_xlim((-0.25, max_qt * 1.05))
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
        0.97,
        0.97,
        text,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
        transform=ax.transAxes,
        fontsize=20,
    )

    fig.tight_layout()
    fig.savefig(output_dir / f"qt_{eic_qt.jet_R_to_str(jet_R)}_pythia6_ep.pdf")
    plt.close(fig)

    return True


def plot_qt_pt_as_function_of_p(
    hist: binned_data.BinnedData,
    label: str,
    base_plot_label: str,
    jet_R: float,
    means: Mapping[tuple[float, float], Mapping[str, float]],
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
            bh_hist[bh.loc(p_range[0]) : bh.loc(p_range[1]) : bh.sum, :]  # type: ignore[misc]
        )

        # Normalize
        logger.info(f"p_range: {p_range}, normalization: {np.sum(hists[p_range].values)}")
        hists[p_range] /= np.sum(hists[p_range].values)
        # hists[p_range] /= hists[p_range].axes[0].bin_widths

    for do_fit in [False, True]:
        fig, ax = plt.subplots(figsize=(12, 9))
        for i, (p_range, h) in enumerate(hists.items()):
            # Fit and plot
            width = -100
            if do_fit:
                logger.info(f"p_range: {p_range}")
                x_linspace_min_for_plotting = -0.2 if debug_fit else 0.0
                if False:
                    fixed_gaussian_mean = 0.0  # type: ignore[unreachable]
                    popt, _ = optimize.curve_fit(
                        lambda x, w, a, fixed_gaussian_mean: _gaussian(x, fixed_gaussian_mean, w, a),
                        h.axes[0].bin_centers,
                        h.values,
                        p0=[0.1, 1],
                        maxfev=2000,
                    )
                    logger.info(f"Mean: {fixed_gaussian_mean}, Width: {popt[0]:.03g}, amplitude: {popt[1]:.03g}")
                    p = ax.plot(
                        # h.axes[0].bin_centers,
                        # _gaussian(h.axes[0].bin_centers, 0.025, *popt),
                        np.linspace(x_linspace_min_for_plotting, 0.5, 100),
                        _gaussian(np.linspace(x_linspace_min_for_plotting, 0.5, 100), fixed_gaussian_mean, *popt),
                        linestyle="--",
                        linewidth=3,
                        # color=p[0].step.get_color(),
                        # We want to be on top, even if plotted first.
                        zorder=10,
                    )
                    # Store for plotting
                    width = popt[0]
                else:
                    initial_mean = [0.05, 0.025, 0.01]
                    popt, _ = optimize.curve_fit(
                        _gaussian,
                        h.axes[0].bin_centers,
                        h.values,
                        p0=[initial_mean[i], 0.05, 0.1],
                        # p0 = [0.0, 0.1],
                        maxfev=50000,
                    )
                    # logger.info(f"Mean: {popt[0]}, Width: {popt[1]}, amplitude: {popt[2]}")
                    logger.info(f"Mean: {popt[0]:.03g}, Width: {popt[1]:.03g}")
                    p = ax.plot(
                        # h.axes[0].bin_centers,
                        # _gaussian(h.axes[0].bin_centers, *popt),
                        np.linspace(x_linspace_min_for_plotting, 0.5, 100),
                        _gaussian(np.linspace(x_linspace_min_for_plotting, 0.5, 100), *popt),
                        linestyle="--",
                        linewidth=3,
                        # color=p[0].step.get_color(),
                        # We want to be on top, even if plotted first.
                        zorder=10,
                    )
                    # Store for plotting
                    width = popt[1]

                # RMS from ROOT
                try:
                    from mammoth.framework import root_utils

                    ROOT = root_utils.import_ROOT()  # noqa: F841
                    h_ROOT = h.to_ROOT()
                    # fu = ROOT.TF1("fu", "[2] * TMath::Gaus(x,[0],[1])")
                    # fu.SetParameters(0, 0.1, 0.1)
                    # res = h_ROOT.Fit("fu")
                    # import IPython; IPython.embed()
                    logger.info(f"RMS from ROOT: {h_ROOT.GetRMS():.03g}, Std Dev: {h_ROOT.GetStdDev():.03g}")
                except ImportError:
                    pass

            kwargs = {}
            plot_label = rf"${p_range[0]} < |\vec{{p}}| < {p_range[1]}\:\text{{GeV}}/c$"
            if do_fit:
                plot_label += rf", $\sigma = {width:.02g}$"
                kwargs = {"color": p[0].get_color()}

            ax.errorbar(
                h.axes[0].bin_centers,
                h.values,
                xerr=h.axes[0].bin_widths,
                yerr=h.errors,
                marker="o",
                markersize=11,
                linestyle="",
                linewidth=4,
                label=plot_label,
                **kwargs,  # type: ignore[arg-type]
            )
            # mplhep plays tricks with the legend marker, such that it doesn't increase the legend linewidth
            # when it increases in the plot. I think it's because it uses the errorbar as the marker, but
            # that linewidth isn't increased. One could do this manually perhaps, but for now, it's easier
            # to just plot it with errorbar.
            # mplhep.histplot(
            #    H=h.values,
            #    bins=h.axes[0].bin_edges,
            #    yerr=h.errors,
            #    label=plot_label,
            #    linewidth=5,
            #    ax=ax,
            #    **kwargs,
            # )

        text = base_plot_label
        text += "\n" + _mean_values_label(mean_x=means[full_p_range]["x"], mean_Q2=means[full_p_range]["Q2"])
        if do_fit:
            text += "\n" + r"Gaussian fit"
        ax.set_xlabel(r"$q_{\text{T}} / p_{\text{T}}^{\text{" + label + r"}}$", fontsize=28)
        ax.set_ylabel(
            r"$1/N_{\text{" + label + r"}}\:\text{d}N/\text{d}(q_{\text{T}}/p_{\text{T}}^{\text{" + label + r"}})$",
            fontsize=28,
        )
        # Focus on range of interest.
        min_x = -0.025
        if debug_fit:
            min_x = -0.1
        ax.set_xlim((min_x, 0.4))
        # Ensure we stop at 0, so it displays the same as a step plot.
        ax.set_ylim((0, None))  # type: ignore[arg-type]
        ax.legend(
            loc="upper right",
            frameon=False,
            bbox_to_anchor=(0.97, 0.68),
            # If we specify an anchor, we want to reduce an additional padding
            # to ensure that we have accurate placement.
            borderaxespad=0,
            borderpad=0,
            fontsize=28,
        )
        ax.text(
            0.97,
            0.97,
            text,
            horizontalalignment="right",
            verticalalignment="top",
            multialignment="right",
            transform=ax.transAxes,
            fontsize=28,
        )

        fig.tight_layout()
        fig.savefig(output_dir / f"qt_pt_{label}_{eic_qt.jet_R_to_str(jet_R)}_{'fit_' if do_fit else ''}pythia6_ep.pdf")
        plt.close(fig)

    return True


def plot_qt_pt_comparison(
    hists: Mapping[str, binned_data.BinnedData],
    base_plot_label: str,
    jet_R: float,
    means: Mapping[tuple[float, float], Mapping[str, float]],
    output_dir: Path,
) -> bool:
    p_ranges = [(100, 150), (150, 200), (200, 250)]
    for p_range in p_ranges:
        fig, ax = plt.subplots(figsize=(12, 9))

        # Project and normalize
        for label, hist in hists.items():
            bh_hist = hist.to_boost_histogram()
            h = binned_data.BinnedData.from_existing_data(
                bh_hist[bh.loc(p_range[0]) : bh.loc(p_range[1]) : bh.sum, :]  # type: ignore[misc]
            )
            logger.info(f"label: {label}, p_range: {p_range}, normalization: {np.sum(h.values)}")
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
        text += "\n" + _mean_values_label(mean_x=means[p_range]["x"], mean_Q2=means[p_range]["Q2"])
        text += "\n" + rf"${p_range[0]} < |\vec{{p}}| < {p_range[1]}\:\text{{GeV}}/c$"
        ax.set_xlabel(r"$q_{\text{T}} / p_{\text{T}}^{\text{X}}$", fontsize=20)
        ax.set_ylabel(r"$1/N_{\text{X}}\:\text{d}N/\text{d}(q_{\text{T}}/p_{\text{T}}^{\text{X}})$", fontsize=20)
        # Focus on range of interest.
        ax.set_xlim((-0.025, 0.4))
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
            0.97,
            0.97,
            text,
            horizontalalignment="right",
            verticalalignment="top",
            multialignment="right",
            transform=ax.transAxes,
            fontsize=20,
        )

        fig.tight_layout()
        fig.savefig(output_dir / f"qt_pt_{eic_qt.jet_R_to_str(jet_R)}_p_range_{p_range[0]}_{p_range[1]}_pythia6_ep.pdf")
        plt.close(fig)

    return True


if __name__ == "__main__":
    # Setup
    mammoth.helpers.setup_logging(level=logging.INFO)

    jet_R_values = [0.5, 0.7, 1.0]
    eta_limits = (1.1, 3.5)
    p_ranges = [(100, 150), (150, 200), (200, 250)]
    # min_q2 = 300
    # x_limits = (0.05, 0.8)
    output_dir = Path("output") / "eic_qt_all_q2_cuts_narrow_bins_jet_R_dependence"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading hists. One sec...")
    y = yaml.yaml(modules_to_register=[binned_data])
    with (output_dir / "qt.yaml").open() as f:
        hists = y.load(f)

    # pop because it's not a hist...
    means = hists.pop("means")

    logger.info("Beginning plotting")
    with sns.color_palette("Set2"):
        # We have no p dependence here by definition, so plot all spectra together.
        logger.info("Jet p")
        plot_jet_p(
            hists=hists,
            # Intentionally leave out jet_R from the label.
            base_plot_label=_base_plot_label(eta_limits=eta_limits),
            jet_R_values=jet_R_values,
            means=means,
            output_dir=output_dir,
        )
        for jet_R in jet_R_values:
            logger.info(f"Plotting jet R = {jet_R}")
            jet_R_str = eic_qt.jet_R_to_str(jet_R)
            base_plot_label = _base_plot_label(jet_R=jet_R, eta_limits=eta_limits)
            logger.info("Jet pt")
            plot_jet_pt(
                hist=hists[jet_R_str]["jet_pt"],
                base_plot_label=base_plot_label,
                jet_R=jet_R,
                means=means[jet_R_str],
                output_dir=output_dir,
            )
            logger.info("Jet multiplicity")
            plot_jet_constituent_multiplicity(
                hist=hists[jet_R_str]["jet_multiplicity"],
                base_plot_label=base_plot_label,
                jet_R=jet_R,
                means=means[jet_R_str],
                output_dir=output_dir,
            )
            logger.info("Plotting qt")
            plot_qt(
                hist=hists[jet_R_str]["qt"],
                base_plot_label=base_plot_label,
                jet_R=jet_R,
                means=means[jet_R_str],
                output_dir=output_dir,
            )
            logger.info("Plotting qt/pt jet")
            plot_qt_pt_as_function_of_p(
                hist=hists[jet_R_str]["qt_pt_jet"],
                label="jet",
                base_plot_label=base_plot_label,
                jet_R=jet_R,
                means=means[jet_R_str],
                output_dir=output_dir,
            )
            logger.info("Plotting qt/pt e")
            plot_qt_pt_as_function_of_p(
                hist=hists[jet_R_str]["qt_pt_electron"],
                label="e",
                base_plot_label=base_plot_label,
                jet_R=jet_R,
                means=means[jet_R_str],
                output_dir=output_dir,
            )
            logger.info("Plotting qt/pt parton")
            plot_qt_pt_as_function_of_p(
                hist=hists[jet_R_str]["qt_pt_parton"],
                label="parton",
                base_plot_label=base_plot_label,
                jet_R=jet_R,
                means=means[jet_R_str],
                output_dir=output_dir,
            )
            logger.info("Plotting qt/pt jet vs e comparison")
            plot_qt_pt_comparison(
                hists={
                    "jet": hists[jet_R_str]["qt_pt_jet"],
                    "e": hists[jet_R_str]["qt_pt_electron"],
                    "parton": hists[jet_R_str]["qt_pt_parton"],
                },
                base_plot_label=base_plot_label,
                jet_R=jet_R,
                means=means[jet_R_str],
                output_dir=output_dir,
            )
