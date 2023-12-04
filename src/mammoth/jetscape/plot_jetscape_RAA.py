"""Plot jetscape jet RAA predictions

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import boost_histogram as bh
import hist
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pachyderm.plot
import seaborn as sns
import uproot
from pachyderm import binned_data
from pachyderm import plot as pb

pachyderm.plot.configure()

_okabe_ito_colors = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    #"#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]

_jet_R_to_color_index = {0.2: 0, 0.4: 1, 0.5: 2, 0.6: 3, 0.8: 4, 1.0: 5}


def format_R(R: float) -> str:
    return f"{round(R * 100):03}"


def get_hists(filename: Path) -> dict[str, hist.Hist]:
    hists = {}
    with uproot.open(Path(filename)) as f:
        for k in f.keys(cycle=False):
            hists[k] = f[k].to_hist()

    return hists


def combine_spectra_in_cent_bins(hists: Mapping[str, hist.Hist], jet_type: str, jet_R: float, a: str, b: str) -> hist.Hist:
    name = f"{jet_type}_jetR{format_R(jet_R)}_n_events_weighted"
    #name = "n_events_weighted"

    a_n_events: hist.Hist = hists[f"PbPb_{a}"][name]  # type: ignore[assignment]
    b_n_events: hist.Hist = hists[f"PbPb_{b}"][name]  # type: ignore[assignment]
    name = f"{jet_type}_jetR{format_R(jet_R)}_jet_pt"
    a_jet_pt: hist.Hist = hists[f"PbPb_{a}"][name]  # type: ignore[assignment]
    b_jet_pt: hist.Hist = hists[f"PbPb_{b}"][name]  # type: ignore[assignment]

    # See Laura's note on adding
    #return ((a_jet_pt / a_n_events.values()[0]) + (b_jet_pt / b_n_events.values()[0])) / 2
    return (a_jet_pt + b_jet_pt) / (a_n_events.values()[0] + b_n_events.values()[0])
    #return ((a_jet_pt / np.sum(a_jet_pt.values())) + (b_jet_pt / np.sum(b_jet_pt.values()))) / 2


def _ML_jet_binning(system: str, jet_R: float) -> npt.NDArray[np.float64]:
    min_pt_values = {
        "PbPb_00_10": {
            0.2: 20,
            0.4: 30,
            0.6: 40,
        },
        "PbPb_30_50": {
            0.2: 20,
            0.4: 30,
            0.6: 30,
        },
    }
    max_pt_values = {
        "PbPb_00_10": 140,
        "PbPb_30_50": 120,
    }
    new_bins = np.concatenate([np.arange(min_pt_values[system][jet_R], 70, 10),
                               np.arange(70, 100, 15),
                               # + 0.1 to make sure that we include the end point.
                               np.arange(100, max_pt_values[system] + 0.1, 20)])
    return new_bins  # noqa: RET504


def plot(output_dir: Path,  # noqa: C901
         jet_R_values: Sequence[float] | None = None,
         jet_types: Sequence[str] | None = None,
         write_hists: bool = False,
         ) -> None:
    # Validation
    if jet_R_values is None:
        jet_R_values = [0.2, 0.4, 0.6]
    if jet_types is None:
        jet_types = ["charged", "full"]
    # Setup
    output_dir.mkdir(parents=True, exist_ok=True)

    hists = {}
    hists["pp"] = get_hists(filename=Path("jetscape_RAA_output/pp/jetscape_RAA.root"))
    hists["PbPb_00_05"] = get_hists(filename=Path("jetscape_RAA_output/PbPb/jetscape_RAA_00_05.root"))
    hists["PbPb_05_10"] = get_hists(filename=Path("jetscape_RAA_output/PbPb/jetscape_RAA_05_10.root"))
    hists["PbPb_30_40"] = get_hists(filename=Path("jetscape_RAA_output/PbPb/jetscape_RAA_30_40.root"))
    hists["PbPb_40_50"] = get_hists(filename=Path("jetscape_RAA_output/PbPb/jetscape_RAA_40_50.root"))

    labels = {
        "pp": "pp",
        "PbPb_00_05": r"0-5\% Pb-Pb",
        "PbPb_05_10": r"5-10\% Pb-Pb",
        "PbPb_30_40": r"30-40\% Pb-Pb",
        "PbPb_40_50": r"40-50\% Pb-Pb",
        # Derived hists
        "PbPb_00_10": r"0-10\% Pb-Pb",
        "PbPb_30_50": r"30-50\% Pb-Pb",
    }

    RAA_hists: dict[str, dict[str, binned_data.BinnedData]] = {
        "PbPb_00_10": {},
        "PbPb_30_50": {},
    }

    with sns.color_palette("Set2"):
        for jet_R in jet_R_values:
            for jet_type in jet_types:
                fig, ax = plt.subplots(figsize=(10, 8))
                fig_scaled, ax_scaled = plt.subplots(figsize=(10, 8))
                fig_RAA, ax_RAA = plt.subplots(figsize=(10, 8))

                text = fr"{jet_type.capitalize()} jets, $R$ = {jet_R}"
                # Just for some user feedback
                print(text)  # noqa: T201

                # Finish labeling
                text += "\n" + r"JETSCAPE Work in Progress" + "\n" + "MATTER + LBT"
                text += "\n" + r"$\alpha_{s} = 0.3$, $Q_{\text{switch}} = 2$ GeV"
                jet_eta_range = 0.9 if jet_type == "charged" else 0.7
                text += "\n" + r"anti-$k_{\text{T}}$ jets, $|\eta_{\text{jet}}| < " + str(jet_eta_range) + " - R$"

                plot_config = pb.PlotConfig(
                    name=f"jet_pt_{jet_type}_R{format_R(jet_R)}",
                    panels=[
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                                    log=True,
                                    font_size=22,
                                ),
                                pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$", font_size=22),
                            ],
                            text=pb.TextConfig(x=0.03, y=0.03, text=text, font_size=22),
                            legend=pb.LegendConfig(location="upper right", font_size=22),
                        ),
                    ],
                    figure=pb.Figure(edge_padding={"left": 0.12, "bottom": 0.08}),
                )
                plot_config_scaled = pb.PlotConfig(
                    name=f"jet_pt_{jet_type}_R{format_R(jet_R)}_scaled",
                    panels=[
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                                    log=True,
                                    font_size=22,
                                ),
                                pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$", font_size=22),
                            ],
                            text=pb.TextConfig(x=0.03, y=0.03, text=text, font_size=22),
                            legend=pb.LegendConfig(location="upper right", font_size=22),
                        ),
                    ],
                    figure=pb.Figure(edge_padding={"left": 0.12, "bottom": 0.08}),
                )
                plot_config_RAA = pb.PlotConfig(
                    name=f"jet_RAA_{jet_type}_R{format_R(jet_R)}",
                    panels=[
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label=r"$R_{\text{AA}}$",
                                    range=(0.0, 1.4),
                                    font_size=22,
                                ),
                                pb.AxisConfig("x", label=r"$p_{\text{T,jet}}\:(\text{GeV}/c)$", font_size=22),
                            ],
                            text=pb.TextConfig(x=0.97, y=0.03, text=text, font_size=22),
                            legend=pb.LegendConfig(location="upper right", font_size=22),
                        ),
                    ],
                    figure=pb.Figure(edge_padding={"left": 0.12, "bottom": 0.08}),
                )

                # Get scaled pp ref for RAA
                name = f"{jet_type}_jetR{format_R(jet_R)}_n_events_weighted"
                #name = "n_events_weighted"
                h_pp_ref_n_events = hists["pp"][name]
                name = f"{jet_type}_jetR{format_R(jet_R)}_jet_pt"
                # Scale immediately, since we're going to do it anyway
                h_pp_ref_jet_pt = hists["pp"][name] / h_pp_ref_n_events.values()[0]
                #h_pp_ref_jet_pt = hists["pp"][name] / np.sum(hists["pp"][name].values())

                for system, v in hists.items():
                    name = f"{jet_type}_jetR{format_R(jet_R)}_n_events_weighted"
                    #name = "n_events_weighted"
                    h_n_events = v[name]
                    name = f"{jet_type}_jetR{format_R(jet_R)}_jet_pt"
                    h_jet_pt = v[name]

                    (h_jet_pt[::hist.rebin(5)] / 5).plot(ax=ax, label=labels[system], linewidth=2)

                    h_jet_pt_scaled = h_jet_pt / h_n_events.values()[0]
                    #h_jet_pt_scaled = h_jet_pt / np.sum(h_jet_pt.values())
                    (h_jet_pt_scaled[::hist.rebin(5)] / 5).plot(ax=ax_scaled, label=labels[system], linewidth=2)

                    # Skip for now just to reduce the number of curves
                    if False:
                        # why da faq doesn't this work...
                        #h_RAA = h_pp_ref_jet_pt / h_jet_pt_scaled
                        #(h_RAA[::hist.rebin(5)] / 5).plot(ax=ax_RAA, label=labels[system], linewidth=2)
                        # Some bug somewhere. Not my problem... Back to binned_data...
                        h_RAA = (
                            binned_data.BinnedData.from_existing_data(h_jet_pt_scaled[::hist.rebin(5)] / 5)
                            / binned_data.BinnedData.from_existing_data(h_pp_ref_jet_pt[::hist.rebin(5)] / 5)
                        )
                        ax_RAA.errorbar(
                            h_RAA.axes[0].bin_centers,
                            h_RAA.values,
                            xerr=h_RAA.axes[0].bin_widths / 2,
                            yerr=h_RAA.errors,
                            label=labels[system],
                        )

                # Calculate 0-10%
                h_PbPb_00_10_jet_pt = combine_spectra_in_cent_bins(hists=hists, jet_type=jet_type, jet_R=jet_R, a="00_05", b="05_10")
                (h_PbPb_00_10_jet_pt[::hist.rebin(5)] / 5).plot(ax=ax_scaled, label=labels["PbPb_00_10"], linewidth=2)

                # Calculate 30-50%
                h_PbPb_30_50_jet_pt = combine_spectra_in_cent_bins(hists=hists, jet_type=jet_type, jet_R=jet_R, a="30_40", b="40_50")
                (h_PbPb_30_50_jet_pt[::hist.rebin(5)] / 5).plot(ax=ax_scaled, label=labels["PbPb_30_50"], linewidth=2)

                # Calculate RAA for calculate centralities
                # 0-10%
                #h_RAA = h_pp_ref_jet_pt / h_PbPb_00_10_jet_pt
                #(h_RAA[::hist.rebin(5)] / 5).plot(ax=ax_RAA, label=labels["PbPb_00_10"], linewidth=2)
                #h_RAA = (
                #    binned_data.BinnedData.from_existing_data((h_PbPb_00_10_jet_pt[10j::hist.rebin(10)] / 10))
                #    / binned_data.BinnedData.from_existing_data((h_pp_ref_jet_pt[10j::hist.rebin(10)] / 10))
                #)
                h_RAA = (
                    binned_data.BinnedData.from_existing_data(h_PbPb_00_10_jet_pt[10j:])
                    / binned_data.BinnedData.from_existing_data(h_pp_ref_jet_pt[10j:])
                )
                RAA_hists["PbPb_00_10"][f"{jet_type}_R{format_R(jet_R)}"] = h_RAA
                ax_RAA.errorbar(
                    h_RAA.axes[0].bin_centers,
                    h_RAA.values,
                    xerr=h_RAA.axes[0].bin_widths / 2,
                    yerr=h_RAA.errors,
                    linestyle="",
                    linewidth=2,
                    label=labels["PbPb_00_10"],
                )
                # 30-50%
                #h_RAA = h_pp_ref_jet_pt / h_PbPb_30_50_jet_pt
                #(h_RAA[::hist.rebin(5)] / 5).plot(ax=ax_RAA, label=labels["PbPb_30_50"], linewidth=2)
                #h_RAA = (
                #    binned_data.BinnedData.from_existing_data((h_PbPb_30_50_jet_pt[10j::hist.rebin(10)] / 10))
                #    / binned_data.BinnedData.from_existing_data((h_pp_ref_jet_pt[10j::hist.rebin(10)] / 10))
                #)
                h_RAA = (
                    binned_data.BinnedData.from_existing_data(h_PbPb_30_50_jet_pt[10j:])
                    / binned_data.BinnedData.from_existing_data(h_pp_ref_jet_pt[10j:])
                )
                RAA_hists["PbPb_30_50"][f"{jet_type}_R{format_R(jet_R)}"] = h_RAA
                ax_RAA.errorbar(
                    h_RAA.axes[0].bin_centers,
                    h_RAA.values,
                    xerr=h_RAA.axes[0].bin_widths / 2,
                    yerr=h_RAA.errors,
                    linestyle="",
                    linewidth=2,
                    label=labels["PbPb_30_50"],
                )

                plot_config.apply(fig=fig, ax=ax)
                filename = f"{plot_config.name}"
                fig.savefig(output_dir / f"{filename}.pdf")
                plt.close(fig)

                plot_config_scaled.apply(fig=fig_scaled, ax=ax_scaled)
                filename = f"{plot_config_scaled.name}"
                fig_scaled.savefig(output_dir / f"{filename}.pdf")
                plt.close(fig_scaled)

                plot_config_RAA.apply(fig=fig_RAA, ax=ax_RAA)
                filename = f"{plot_config_RAA.name}"
                fig_RAA.savefig(output_dir / f"{filename}.pdf")
                plt.close(fig_RAA)

    # Plot RAA as a function of R
    #with sns.color_palette("Set2"):
    for jet_type in jet_types:
        for system in ["PbPb_00_10", "PbPb_30_50"]:
            for restricted_range in [False, True]:
                text = fr"{labels[system]}, {jet_type.capitalize()} jets"
                # Just for some user feedback
                print(text)  # noqa: T201

                # Finish labeling
                text += "\n" + r"JETSCAPE Work in Progress" + "\n" + "MATTER + LBT"
                text += "\n" + r"$\alpha_{s} = 0.3$, $Q_{\text{switch}} = 2$ GeV"
                jet_eta_range = 0.9 if jet_type == "charged" else 0.7
                text += "\n" + r"anti-$k_{\text{T}}$ jets, $|\eta_{\text{jet}}| < " + str(jet_eta_range) + " - R$"

                for jet_R_label, jet_R_values_to_iterate in [("", jet_R_values), ("_alice_comparison", [0.2, 0.4]), ("_requested", [0.2, 0.4, 0.6])]:
                    x_axis_kwargs: dict[str, Any] = {}
                    if restricted_range:
                        x_axis_kwargs = {"range": (15, 145)}
                        if jet_R_label == "_alice_comparison":
                            x_axis_kwargs = {"range": (55, 145)}

                    plot_config = pb.PlotConfig(
                        name=f"jet_RAA_R_{jet_type}_{system}" + ("_zoom" if restricted_range else ""),
                        panels=[
                            pb.Panel(
                                axes=[
                                    pb.AxisConfig(
                                        "y",
                                        label=r"$R_{\text{AA}}$",
                                        range=(0.0, 1.2) if jet_R_label != "_alice_comparison" else (-0.2, 1.0),
                                        font_size=22,
                                    ),
                                    pb.AxisConfig("x", label=r"$p_{\text{T,jet}}\:(\text{GeV}/c)$", font_size=22, **x_axis_kwargs),
                                ],
                                text=pb.TextConfig(x=0.97, y=0.03, text=text, font_size=22),
                                legend=pb.LegendConfig(location="upper right", font_size=22),
                            ),
                        ],
                        figure=pb.Figure(edge_padding={"left": 0.10, "bottom": 0.09}),
                    )

                    fig, ax = plt.subplots(figsize=(10, 8))
                    for jet_R in jet_R_values_to_iterate:
                        print(f"jet_R_label: {jet_R_label}")  # noqa: T201

                        if jet_R_label != "_requested":
                            # 10 GeV wide bins up to 100 GeV, followed by 100 GeV wide bins beyond there.
                            #new_bins = np.concatenate([np.arange(10, 100, 10), np.arange(100, 1100, 100)])
                            new_bins = np.concatenate([np.arange(10, 80, 10), np.arange(80, 200, 20), np.arange(200, 500, 100), np.arange(500, 1000 + 0.1, 500)])
                        else:
                            min_pt_values = {
                                "PbPb_00_10": {
                                    0.2: 20,
                                    0.4: 30,
                                    0.6: 40,
                                },
                                "PbPb_30_50": {
                                    0.2: 20,
                                    0.4: 30,
                                    0.6: 30,
                                },
                            }
                            max_pt_values = {
                                "PbPb_00_10": 140,
                                "PbPb_30_50": 120,
                            }
                            new_bins = np.concatenate([np.arange(min_pt_values[system][jet_R], 70, 10),
                                                       np.arange(70, 100, 15),
                                                       # + 0.1 to make sure that we include the end point.
                                                       np.arange(100, max_pt_values[system] + 0.1, 20)])
                        #original_bin_width = 10
                        original_bin_width = 1

                        print(f"jet_R: {jet_R}")  # noqa: T201
                        h_RAA = RAA_hists[system][f"{jet_type}_R{format_R(jet_R)}"][:: new_bins]
                        # Normalize by bin width
                        h_RAA /= (h_RAA.axes[0].bin_widths / original_bin_width)

                        # TG3 comparison
                        #if jet_R_label == "_alice_comparison" and jet_R == 0.2:
                        #    print("TG3 comparison")
                        #    print(h_RAA[40j:140j].values)
                        #    import IPython; IPython.embed()
                        # PbPb paper comparison
                        #if jet_R_label == "_alice_comparison" and jet_R == 0.4:
                        #    print("PbPb comparison")
                        #    print(h_RAA[60j:140j].values)
                        #    import IPython; IPython.embed()

                        ax.errorbar(
                            h_RAA.axes[0].bin_centers,
                            h_RAA.values,
                            xerr=h_RAA.axes[0].bin_widths / 2,
                            yerr=h_RAA.errors,
                            label=fr"$R$ = {jet_R}",
                            alpha=0.9,
                            color=_okabe_ito_colors[_jet_R_to_color_index[jet_R]],
                        )
                        #p = ax.fill_between(
                        #    h_RAA.axes[0].bin_centers,
                        #    h_RAA.values - h_RAA.errors,
                        #    h_RAA.values + h_RAA.errors,
                        #    #h_RAA.values,
                        #    #xerr=h_RAA.axes[0].bin_widths / 2,
                        #    #yerr=h_RAA.errors,
                        #    label=fr"$R$ = {jet_R}",
                        #    alpha=0.9,
                        #    color=_okabe_ito_colors[_jet_R_to_color_index[jet_R]],
                        #)

                    plot_config.apply(fig=fig, ax=ax)
                    filename = f"{plot_config.name}{jet_R_label}"
                    fig.savefig(output_dir / f"{filename}.pdf")
                    plt.close(fig)

    # RAA ratios
    for jet_type in jet_types:
        for system in ["PbPb_00_10", "PbPb_30_50"]:
            for _ratio_R in [0.4, 0.6]:
                text = fr"{labels[system]}, {jet_type.capitalize()} jets"
                # Just for some user feedback
                print(text)  # noqa: T201

                # Finish labeling
                text += "\n" + r"JETSCAPE Work in Progress" + "\n" + "MATTER + LBT"
                text += "\n" + r"$\alpha_{s} = 0.3$, $Q_{\text{switch}} = 2$ GeV"
                jet_eta_range = 0.9 if jet_type == "charged" else 0.7
                text += "\n" + r"anti-$k_{\text{T}}$ jets, $|\eta_{\text{jet}}| < " + str(jet_eta_range) + " - R$"

                x_axis_kwargs = {"range": (15, 145)}

                plot_config = pb.PlotConfig(
                    name=f"jet_RAA_ratio_{jet_type}_{system}",
                    panels=[
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label=r"$R_{\text{AA}}(R) / R_{\text{AA}} (R=0.2)$",
                                    range=(0, 1.5),
                                    font_size=22,
                                ),
                                pb.AxisConfig("x", label=r"$p_{\text{T,jet}}\:(\text{GeV}/c)$", font_size=22, **x_axis_kwargs),
                            ],
                            text=pb.TextConfig(x=0.97, y=0.03, text=text, font_size=22),
                            legend=pb.LegendConfig(location="upper right", font_size=22),
                        ),
                    ],
                    figure=pb.Figure(edge_padding={"left": 0.10, "bottom": 0.09}),
                )

                fig, ax = plt.subplots(figsize=(10, 8))

                for jet_R in [0.4, 0.6]:
                    print(f"jet_R: {jet_R}")  # noqa: T201

                    # Get the denominator (R = 0.2), but with the bins of the larger R so that they match.
                    new_bins = _ML_jet_binning(system=system, jet_R=jet_R)
                    h_RAA_denominator = RAA_hists[system][f"{jet_type}_R{format_R(0.2)}"][:: new_bins]

                    h_RAA = RAA_hists[system][f"{jet_type}_R{format_R(jet_R)}"][:: new_bins]

                    h_ratio = h_RAA / h_RAA_denominator

                    # Normalize by bin width
                    #h_RAA /= (h_RAA.axes[0].bin_widths / original_bin_width)

                    # TG3 comparison
                    #if jet_R_label == "_alice_comparison" and jet_R == 0.2:
                    #    print("TG3 comparison")
                    #    print(h_RAA[40j:140j].values)
                    #    import IPython; IPython.embed()
                    # PbPb paper comparison
                    #if jet_R_label == "_alice_comparison" and jet_R == 0.4:
                    #    print("PbPb comparison")
                    #    print(h_RAA[60j:140j].values)
                    #    import IPython; IPython.embed()

                    ax.errorbar(
                        h_ratio.axes[0].bin_centers,
                        h_ratio.values,
                        xerr=h_ratio.axes[0].bin_widths / 2,
                        yerr=h_ratio.errors,
                        label=fr"$R$ = {jet_R} / $R$=0.2",
                        alpha=0.9,
                        color=_okabe_ito_colors[_jet_R_to_color_index[jet_R]],
                    )
                    #p = ax.fill_between(
                    #    h_RAA.axes[0].bin_centers,
                    #    h_RAA.values - h_RAA.errors,
                    #    h_RAA.values + h_RAA.errors,
                    #    #h_RAA.values,
                    #    #xerr=h_RAA.axes[0].bin_widths / 2,
                    #    #yerr=h_RAA.errors,
                    #    label=fr"$R$ = {jet_R}",
                    #    alpha=0.9,
                    #    color=_okabe_ito_colors[_jet_R_to_color_index[jet_R]],
                    #)

                plot_config.apply(fig=fig, ax=ax)
                filename = f"{plot_config.name}{jet_R_label}"
                fig.savefig(output_dir / f"{filename}.pdf")
                plt.close(fig)

    # Write hists
    if write_hists:
        with uproot.recreate(output_dir / "raa_hists.root") as f:
            for jet_type in jet_types:
                for system in ["PbPb_00_10", "PbPb_30_50"]:
                    for jet_R in [0.2, 0.4, 0.6]:
                        h_RAA = RAA_hists[system][f"{jet_type}_R{format_R(jet_R)}"]
                        # Convert to boost histogram by hand. We need to do this here because
                        # we don't hold onto overflow and underflow values, but uproot expects them
                        # So this is just a one time ugly hack (although it would be good to figure out for later...)
                        axes = []
                        for axis in h_RAA.axes:
                            # NOTE: We use Variable instead of Regular even if the bin edges are Regular because it allows us to
                            #       construct the axes just from the bin edges.
                            #bin_edges = np.zeros(len(axis.bin_edges) + 2)
                            #bin_edges[0] = -np.inf
                            #bin_edges[-1] = np.inf
                            axes.append(bh.axis.Variable(axis.bin_edges))
                        h = bh.Histogram(*axes, storage=bh.storage.Weight())
                        # Need to shape the array properly so that it will actually be able to assign to the boost histogram.
                        arr = np.zeros(shape=h.view(flow=True).shape, dtype=h.view(flow=True).dtype)
                        arr["value"][1:-1] = h_RAA.values
                        arr["variance"][1:-1] = h_RAA.variances
                        h[...] = arr

                        # Finally, write the converted hist
                        f[f"{system}_{jet_type}_R{format_R(jet_R)}"] = h


if __name__ == "__main__":
    plot(
        output_dir=Path("jetscape_RAA_output/plots"),
        write_hists=True,
        jet_R_values=[0.2, 0.4, 0.5, 0.6, 0.8],
        #jet_R_values=[0.2],
        jet_types=["charged"],
    )
