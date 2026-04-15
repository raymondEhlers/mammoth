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
#     display_name: .venv-3.13
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Second look at ALICE data for entanglement entropy
#

# %%
from __future__ import annotations

from pathlib import Path

import hist
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pachyderm.plot as pb
import polars as pl  # noqa: F401
import uproot  # noqa: F401
from pachyderm import binned_data

from mammoth.entanglement_entropy import analysis_tools, skim_to_hist
from mammoth.framework.io import output_utils  # noqa: F401

base_path = Path("projects/entanglement_entropy")

pb.configure()

# %load_ext autoreload

# %autoreload 2

# %%
import mammoth.helpers

mammoth.helpers.setup_logging()

# %%
# Load data
run_2_pp_ref_data = skim_to_hist.load_data(skim_directory=Path("trains/pp/0063/skim"), level_name="data")
run_3_pp_ref_data = skim_to_hist.load_data(skim_directory=Path("trains/pp/0006/skim"), level_name="data")
# run_2_pp_MC_pp_ref_data = skim_to_hist.load_data(skim_directory=Path("trains/pp_MC/0086/skim"), level_name="data")

# %%
# We'd like to have an option of having the same stats as the run 2 data, but at a different sqrt_s.
# We can do this by selecting the first n dijets in the Run 3 data, where n == the number of run 2 dijets
n_dijets_run_2 = len(run_2_pp_ref_data.leading_jet)
run_3_pp_ref_data_with_run_2_stats = skim_to_hist.JetData(
    leading_jet=run_3_pp_ref_data.leading_jet[:n_dijets_run_2],
    subleading_jet=run_3_pp_ref_data.subleading_jet[:n_dijets_run_2],
    scale_factors=run_3_pp_ref_data.scale_factors[:n_dijets_run_2],
)

# %%
(
    len(run_3_pp_ref_data.leading_jet),
    len(run_3_pp_ref_data_with_run_2_stats.leading_jet),
    len(run_2_pp_ref_data.leading_jet),
)

# %%
hists_per_data_source = {
    "run_2_pp_ref": skim_to_hist.define_base_histograms(levels=["data"]),
    "run_3_pp_ref": skim_to_hist.define_base_histograms(levels=["data"]),
    "run_3_pp_ref_with_run_2_stats": skim_to_hist.define_base_histograms(levels=["data"]),
    # "run_2_pp_MC_pp_ref": skim_to_hist.define_base_histograms(levels=["data"]),
}

skim_to_hist.fill_base_histograms(
    levels=["data"], hists=hists_per_data_source["run_2_pp_ref"], jet_data=run_2_pp_ref_data
)
skim_to_hist.fill_base_histograms(
    levels=["data"], hists=hists_per_data_source["run_3_pp_ref"], jet_data=run_3_pp_ref_data
)
skim_to_hist.fill_base_histograms(
    levels=["data"],
    hists=hists_per_data_source["run_3_pp_ref_with_run_2_stats"],
    jet_data=run_3_pp_ref_data_with_run_2_stats,
)
# skim_to_hist.fill_base_histograms(levels=["data"], hists=hists_per_data_source["run_2_pp_MC_pp_ref"], jet_data=run_2_pp_MC_pp_ref_data)

# %%
# Labels
data_source_to_presentation_label = {
    "run_2_pp_ref": "pp, 5.02 TeV",
    "run_2_pp_MC_pp_ref": "pythia part level, 5.02 TeV",
    "run_3_pp_ref": "pp, 5.36 TeV",
    "run_3_pp_ref_with_run_2_stats": "pp, 5.36 TeV (run 2 stat)",
}

# %%
hists_per_data_source["run_2_pp_ref"].keys()

# %%
len(hists_per_data_source["run_3_pp_ref"]["data_leading_jet_spectra"].values())


# %% [markdown]
# # Spectra
#


# %%
def plot_dijet_spectra(
    hists_per_data_source: dict[str, dict[str, hist.Hist[hist.storage.Weight]]], normalize: bool
) -> None:
    text_font_size = 22

    text = "ALICE work-in-progress"
    text += "\n" + r"charged-particle jets"
    text += "\n" + r"$R = 0.4$"
    for jet_label in ["leading", "subleading"]:
        plot_config = pb.PlotConfig(
            name=f"{jet_label}_jet_spectra",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "x",
                            label=r"$p_{\text{T, ch jet}}^{\text{" + jet_label[:7] + r"}}$",
                            font_size=text_font_size,
                        ),
                        pb.AxisConfig(
                            "y",
                            label=r"$dN/dp_{\text{T}}$",
                            font_size=text_font_size,
                            log=True,
                        ),
                    ],
                    text=pb.TextConfig(x=0.05, y=0.05, text=text, font_size=18),
                    legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.15}),
        )

        fig, ax = plt.subplots(
            1,
            1,
            figsize=(10, 6.25),
            sharex=True,
        )

        for source, hists in hists_per_data_source.items():
            print(f"{source}")
            h = hists[f"data_{jet_label}_jet_spectra"]
            h = binned_data.BinnedData.from_existing_data(h).copy()
            if normalize:
                print("True")
                h /= sum(h.values)
            else:
                print("false")
            ax.errorbar(
                h.axes[0].bin_centers,
                h.values,
                xerr=h.axes[0].bin_widths / 2,
                yerr=h.errors,
                marker="o",
                markersize=11,
                linestyle="",
                linewidth=3,
                label=f"ALICE data ({data_source_to_presentation_label[source]})",
            )

        plot_config.apply(fig=fig, ax=ax)

        _output_path = base_path / "second_look_with_skim"
        _output_path.mkdir(parents=True, exist_ok=True)
        tag = "_norm" if normalize else ""
        fig.savefig(_output_path / f"{plot_config.name}{tag}.pdf")
        plt.close(fig)


for normalize in [False, True]:
    plot_dijet_spectra(hists_per_data_source=hists_per_data_source, normalize=normalize)

# %%
hists_per_data_source["run_2_pp_ref"]["data_leading_jet_spectra"].variances()


# %% [markdown]
# # N constituents


# %%
def plot_n_constituents_lead_pt_differential(
    hists_per_data_source: dict[str, dict[str, hist.Hist[hist.storage.Weight]]],
    normalize: bool,
    lead_jet_pt_range: tuple[float, float] | None = None,
) -> None:
    text_font_size = 22

    for source, hists in hists_per_data_source.items():
        text = "ALICE work-in-progress"
        text += "\n" + r"charged-particle jets"
        text += "\n" + rf"$R = 0.4$ {data_source_to_presentation_label[source]}"
        plot_config = pb.PlotConfig(
            name="n_constituents_correlation",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "x",
                            label=r"$N_{\text{lead}}$",
                            font_size=text_font_size,
                        ),
                        pb.AxisConfig(
                            "y",
                            label=r"$N_{\text{sublead}}$",
                            font_size=text_font_size,
                        ),
                    ],
                    text=pb.TextConfig(x=0.95, y=0.81, text=text, font_size=18),
                    # legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.15}),
        )

        fig, ax = plt.subplots(
            1,
            1,
            figsize=(10, 6.25),
            sharex=True,
        )
        # Had to set axis labels by hand to avoid mplhep breaking since I had a typo in the original axis label
        # ax.set_xlabel(r"$n_{\text{const}}^{\text{lead}}$")
        # ax.set_ylabel(r"$n_{\text{const}}^{\text{sublead}}$")
        # hists["data_n_constituents_lead_jet_pt"][sum,:,:].plot(ax=ax, label="ALICE data")

        import matplotlib as mpl

        s = sum
        if lead_jet_pt_range:
            s = slice(lead_jet_pt_range[0] * 1.0j, lead_jet_pt_range[1] * 1.0j, sum)

        h = binned_data.BinnedData.from_existing_data(hists["data_n_constituents_jet_pt"][s, sum, :, :])
        if normalize:
            normalization_values = h.values.sum()
            h.values = np.divide(
                h.values, normalization_values, out=np.zeros_like(h.values), where=normalization_values != 0
            )

        z_axis_range = {
            "vmin": max(1e-4, h.values[h.values > 0].min()),
            "vmax": 1 if normalize else h.values.max(),
        }

        # Plot
        mesh = ax.pcolormesh(
            h.axes[0].bin_edges.T,
            h.axes[1].bin_edges.T,
            h.values.T,
            norm=mpl.colors.LogNorm(**z_axis_range),
        )
        fig.colorbar(mesh, pad=0.02)

        plot_config.apply(fig=fig, ax=ax)

        _output_path = base_path / "second_look_with_skim"
        _output_path.mkdir(parents=True, exist_ok=True)
        tag = "_norm" if normalize else ""
        if lead_jet_pt_range:
            tag += f"_lead_jet_pt_{lead_jet_pt_range[0]}_{lead_jet_pt_range[1]}"
        fig.savefig(_output_path / f"{plot_config.name}_{source}{tag}.pdf")
        plt.close(fig)


for normalize in [False, True]:
    for lead_jet_pt_range in [None, (10, 20), (20, 40), (40, 60), (60, 80), (80, 120)]:
        h = plot_n_constituents_lead_pt_differential(
            hists_per_data_source=hists_per_data_source, normalize=normalize, lead_jet_pt_range=lead_jet_pt_range
        )


# %%
def plot_n_constituents_1d(
    hists_per_data_source: dict[str, dict[str, hist.Hist[hist.storage.Weight]]],
    lead_jet_pt_range: tuple[float, float] | None = None,
    log_y: bool = False,
) -> None:
    text_font_size = 22

    text = "ALICE work-in-progress"
    text += "\n" + r"charged-particle jets"
    text += "\n" + r"$R = 0.4$"
    for jet_label in ["lead", "sublead"]:
        plot_config = pb.PlotConfig(
            name=f"n_constituents_{jet_label}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "x",
                            label=r"$N_{\text{" + jet_label + "}}$",
                            font_size=text_font_size,
                            range=(-2, 35),
                        ),
                        pb.AxisConfig(
                            "y",
                            label=r"$1/N_{\text{jet}}$ $\text{d}N/\text{d}n_{\text{const}}$",
                            font_size=text_font_size,
                            log=log_y,
                        ),
                    ],
                    text=pb.TextConfig(x=0.05, y=0.05, text=text, font_size=18)
                    if log_y
                    else pb.TextConfig(x=0.95, y=0.75, text=text, font_size=18),
                    legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=18),
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.13, "bottom": 0.15}),
        )

        fig, ax = plt.subplots(
            1,
            1,
            figsize=(10, 6.25),
            sharex=True,
        )

        for source, hists in hists_per_data_source.items():
            # ax.set_xlabel(r"$n_{\text{const}}^{\text{lead}}$")
            # ax.set_ylabel(r"$n_{\text{const}}^{\text{sublead}}$")

            s = sum
            if lead_jet_pt_range:
                s = slice(lead_jet_pt_range[0] * 1.0j, lead_jet_pt_range[1] * 1.0j, sum)
            if jet_label == "lead":
                h_hist = hists["data_n_constituents_jet_pt"][s, sum, :, sum]
            else:
                h_hist = hists["data_n_constituents_jet_pt"][s, sum, sum, :]

            h = binned_data.BinnedData.from_existing_data(h_hist)
            # Normalize
            if normalize:
                h /= sum(h.values)

            ax.errorbar(
                h.axes[0].bin_centers,
                h.values,
                xerr=h.axes[0].bin_widths / 2,
                yerr=h.errors,
                marker="o",
                markersize=11,
                linestyle="",
                linewidth=3,
                label=f"ALICE data ({data_source_to_presentation_label[source]})",
            )

            plot_config.apply(fig=fig, ax=ax)

            _output_path = base_path / "second_look_with_skim"
            _output_path.mkdir(parents=True, exist_ok=True)

            tag = ""
            if lead_jet_pt_range:
                tag += f"_lead_jet_pt_{lead_jet_pt_range[0]}_{lead_jet_pt_range[1]}"
            if log_y:
                tag += "_log"
            fig.savefig(_output_path / f"{plot_config.name}{tag}.pdf")
            plt.close(fig)


for lead_jet_pt_range in [None, (10, 20), (20, 40), (40, 60), (60, 80), (80, 120)]:
    for log_y in [False, True]:
        plot_n_constituents_1d(
            hists_per_data_source=hists_per_data_source, lead_jet_pt_range=lead_jet_pt_range, log_y=log_y
        )

# %% [markdown]
#

# %% [markdown]
# # Entanglement entropy

# %% [markdown]
# ## Unsummed entropy

# %%
entropies_unsummed = {
    source: analysis_tools.calculate_entropy(hists, skip_entropy_sum=True)
    for source, hists in hists_per_data_source.items()
}

# %%
np.sum(entropies_unsummed["run_3_pp_ref"].lead, axis=1)

# %%
entropies_unsummed["run_3_pp_ref"].lead[1, :]

# %%
import matplotlib as mpl

# Plot calculated entropies as a function of jet pt

# Just need the h_pt and n_const binning info, so arbitrarily pick the full precision
h_pt = binned_data.BinnedData.from_existing_data(
    hists_per_data_source["run_2_pp_ref"]["data_n_constituents_jet_pt"][:, :, sum, sum]
)
h_n_const = binned_data.BinnedData.from_existing_data(
    hists_per_data_source["run_2_pp_ref"]["data_n_constituents_jet_pt"][sum, sum, :, :]
)


def plot_unsummed_entropy(
    entropies: dict[str, hist.Hist[hist.storage.Weight]],
    lead_jet_pt_ranges: list[tuple[float, float]],
) -> None:
    text_font_size = 22

    for source, entropy in entropies.items():
        # for label in ["lead", "sublead", "joint"]:
        # NOTE: the joint has an additional dimension, so it's easier to handle separately.
        for label in ["lead", "sublead"]:
            text = "ALICE work-in-progress"
            text += "\n" + r"charged-particle jets"
            text += "\n" + r"$R = 0.4$"
            text += "\n" + f"{data_source_to_presentation_label[source]}"
            plot_config = pb.PlotConfig(
                name=f"entropy_unsummed_{label}",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "x",
                                label=r"$N_{\text{const}}$",
                                font_size=text_font_size,
                            ),
                            pb.AxisConfig(
                                "y",
                                label=r"$-p \ln{p}$" + f"({label})",
                                font_size=text_font_size,
                                range=(-0.02, 0.35),
                            ),
                        ],
                        text=pb.TextConfig(x=0.95, y=0.20, text=text, font_size=16),
                        legend=pb.LegendConfig(location="upper right", font_size=12),
                    ),
                ],
                figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.15}),
            )

            fig, ax = plt.subplots(
                1,
                1,
                figsize=(10, 6.25),
                sharex=True,
            )
            # Make the colors a bit easier to follow
            cmap = mpl.colormaps["viridis"]
            colors = cmap(np.linspace(0, 1, len(lead_jet_pt_ranges)))
            ax.set_prop_cycle(color=colors)

            # Just plot the values
            for jet_pt_range in lead_jet_pt_ranges:
                low, high = h_pt.axes[0].find_bin(jet_pt_range[0]), h_pt.axes[0].find_bin(jet_pt_range[1])
                ax.plot(
                    h_n_const.axes[0].bin_centers,
                    getattr(entropy, label)[low:high, :].squeeze(),
                    label=rf"${jet_pt_range[0]} < p_{{\text{{T, jet}}}} < {jet_pt_range[1]}\:\text{{GeV}}/c$",
                    linestyle="",
                    marker="s",
                )

            plot_config.apply(fig=fig, ax=ax)

            _output_path = base_path / "second_look_with_skim"
            _output_path.mkdir(parents=True, exist_ok=True)
            tag = f"_{source}"
            fig.savefig(_output_path / f"{plot_config.name}{tag}.pdf")
            plt.close(fig)


plot_unsummed_entropy(entropies_unsummed, lead_jet_pt_ranges=list(zip(range(10, 70, 5), range(11, 71, 5), strict=True)))

# %%
# Plot unsummed joint entropies as a function of jet pt

# Just need the h_pt and n_const binning info, so arbitrarily pick the full precision
h_pt = binned_data.BinnedData.from_existing_data(
    hists_per_data_source["run_2_pp_ref"]["data_n_constituents_jet_pt"][:, :, sum, sum]
)
h_n_const = binned_data.BinnedData.from_existing_data(
    hists_per_data_source["run_2_pp_ref"]["data_n_constituents_jet_pt"][sum, sum, :, :]
)


def plot_unsummed_entropy_joint(
    entropies: dict[str, hist.Hist[hist.storage.Weight]],
    lead_jet_pt_ranges: list[tuple[float, float]],
) -> None:
    text_font_size = 22

    for source, entropy in entropies.items():
        label = "joint"
        for jet_pt_range in lead_jet_pt_ranges:
            text = "ALICE work-in-progress"
            text += "\n" + r"charged-particle jets"
            text += (
                "\n" + rf"$R = 0.4$, ${jet_pt_range[0]} < p_{{\text{{T, jet}}}} < {jet_pt_range[1]}\:\text{{GeV}}/c$"
            )
            text += "\n" + f"{data_source_to_presentation_label[source]}"
            plot_config = pb.PlotConfig(
                name=f"entropy_unsummed_{label}",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "x",
                                label=r"$N_{\text{const, lead}}$",
                                font_size=text_font_size,
                            ),
                            pb.AxisConfig(
                                "y",
                                label=r"$N_{\text{const, sublead}}$",
                                font_size=text_font_size,
                            ),
                        ],
                        text=pb.TextConfig(x=0.95, y=0.20, text=text, font_size=16),
                    ),
                ],
                figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.15}),
            )

            fig, ax = plt.subplots(
                1,
                1,
                figsize=(10, 6.25),
                sharex=True,
            )

            # Plot
            low, high = h_pt.axes[0].find_bin(jet_pt_range[0]), h_pt.axes[0].find_bin(jet_pt_range[1])
            values = entropy.joint[low:high, :, :].squeeze()
            # Suppress values exactly at 0 for clarity
            values[values == 0] = np.nan
            mesh = ax.pcolormesh(
                h_n_const.axes[0].bin_edges.T,
                h_n_const.axes[1].bin_edges.T,
                values.T,
                norm=mpl.colors.Normalize(vmin=0.0, vmax=0.2),
            )
            fig.colorbar(mesh, pad=0.02)

            plot_config.apply(fig=fig, ax=ax)

            _output_path = base_path / "second_look_with_skim"
            _output_path.mkdir(parents=True, exist_ok=True)
            tag = f"_{source}__jet_pt_{jet_pt_range[0]}_{jet_pt_range[1]}"
            fig.savefig(_output_path / f"{plot_config.name}{tag}.pdf")
            plt.close(fig)


# plot_unsummed_entropy_joint(entropies_unsummed, lead_jet_pt_ranges=list(zip(range(10, 70, 5), range(11, 71, 5), strict=True)))
plot_unsummed_entropy_joint(
    entropies_unsummed, lead_jet_pt_ranges=list(zip(range(10, 20, 5), range(11, 21, 5), strict=True))
)

# %% [markdown]
# ## Entropy

# %%
# As a function of leading jet pt
entropies = {source: analysis_tools.calculate_entropy(hists) for source, hists in hists_per_data_source.items()}
mutual_information = {
    source: analysis_tools.calculate_mutual_information(hists, entropy=entropies[source])
    for source, hists in hists_per_data_source.items()
}

# %%
entropies["run_3_pp_ref"].sublead

# %%
# Plot calculated entropies as a function of jet pt

# Just need the pt binning info, so arbitrarily pick the full precision
h_pt = binned_data.BinnedData.from_existing_data(
    hists_per_data_source["run_2_pp_ref"]["data_n_constituents_jet_pt"][:, :, sum, sum]
)


def plot_entropy(
    entropies: dict[str, hist.Hist[hist.storage.Weight]],
    lead_jet_pt_range: tuple[float, float] | None = None,  # noqa: ARG001
    sublead_jet_pt_range: tuple[float, float] | None = None,  # noqa: ARG001
) -> None:
    text_font_size = 22

    for label in ["lead", "sublead", "joint"]:
        text = "ALICE work-in-progress"
        text += "\n" + r"charged-particle jets"
        text += "\n" + r"$R = 0.4$"
        plot_config = pb.PlotConfig(
            name=f"entropy_{label}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "x",
                            # label=r"$p_{\text{T, " + label + "}}$",
                            label=r"$p_{\text{T,lead}}$",
                            font_size=text_font_size,
                        ),
                        pb.AxisConfig(
                            "y",
                            label=r"$S_{\text{" + label + r"}}$",
                            font_size=text_font_size,
                            range=(-0.15, 2.9) if label != "joint" else (-0.15, None),
                        ),
                    ],
                    text=pb.TextConfig(x=0.05, y=0.25, text=text, font_size=16),
                    legend=pb.LegendConfig(location="lower left", font_size=16),
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.15}),
        )

        fig, ax = plt.subplots(
            1,
            1,
            figsize=(10, 6.25),
            sharex=True,
        )
        # Just plot the values
        for source, entropy in entropies.items():
            ax.plot(
                h_pt.axes[0].bin_centers,
                getattr(entropy, label),
                label=f"ALICE data ({data_source_to_presentation_label[source]})",
                linestyle="",
                marker="s",
            )
            # # Including subleading jet pt in the entropy range...
            # x_axis_selection = 0 if label == "lead" else 1
            # entropy_axis_selection = 1 if label == "lead" else 0
            # ax.plot(
            #     h_pt.axes[x_axis_selection].bin_centers,
            #     np.sum(getattr(entropy, label), axis=entropy_axis_selection),
            #     label=f"ALICE data ({data_source_to_presentation_label[source]})",
            #     linestyle="",
            #     marker="s",
            # )

        plot_config.apply(fig=fig, ax=ax)

        _output_path = base_path / "second_look_with_skim"
        _output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(_output_path / f"{plot_config.name}.pdf")
        plt.close(fig)


plot_entropy(entropies)


# %%
# Plot mutual information
def plot_mutual_information(
    mutual_information: dict[str, npt.NDArray[np.floating]],
    entropies: dict[str, npt.NDArray[np.floating]],
    normalization: str = "",
) -> None:
    # Just need the binning info, so pick the full precision
    h_pt = binned_data.BinnedData.from_existing_data(
        hists_per_data_source["run_3_pp_ref"]["data_n_constituents_jet_pt"][:, :, sum, sum]
    )

    text_font_size = 22

    text = "ALICE work-in-progress"
    text += "\n" + r"charged-particle jets"
    text += "\n" + r"$R = 0.4$, \text{" + normalization + "}"
    plot_config = pb.PlotConfig(
        name="mutual_information_jet_pt",
        panels=[
            # Main panel
            pb.Panel(
                axes=[
                    pb.AxisConfig(
                        "x",
                        label=r"$p_{\text{T, lead}}$",
                        font_size=text_font_size,
                    ),
                    pb.AxisConfig(
                        "y",
                        label=r"$S_1 + S_2 - S_{12}$",
                        font_size=text_font_size,
                        range=(-0.1, 2.5 if not normalization else 1.6),
                    ),
                ],
                text=pb.TextConfig(x=0.05, y=0.71, text=text, font_size=18),
                legend=pb.LegendConfig(location="upper left", anchor=(0.05, 0.95), font_size=22),
            ),
        ],
        figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.15}),
    )

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, 6.25),
        sharex=True,
    )
    # Just plot the values
    for source, info in mutual_information.items():
        values = info.copy()
        match normalization:
            case "s1+s2":
                values /= entropies[source].lead + entropies[source].sublead
            case "min(s1,s2)":
                values /= np.minimum(entropies[source].lead, entropies[source].sublead)
            case _:
                # Nothing to be done
                ...
        ax.plot(
            h_pt.axes[0].bin_centers,
            values,
            label=f"ALICE data {data_source_to_presentation_label[source]}",
            linestyle="",
            marker="o",
        )

    plot_config.apply(fig=fig, ax=ax)

    _output_path = base_path / "second_look_with_skim"
    _output_path.mkdir(parents=True, exist_ok=True)
    tag = f"_{normalization}" if normalization else ""
    fig.savefig(_output_path / f"{plot_config.name}{tag}.pdf")
    plt.close(fig)


# temp_mutual_information = {
#     "run_2_pp_ref": mutual_information["run_2_pp_ref"],
#     "run_3_pp_ref": mutual_information["run_3_pp_ref"],
# }
# for normalization in [""]:
for normalization in ["", "s1+s2", "min(s1,s2)"]:
    plot_mutual_information(mutual_information=mutual_information, entropies=entropies, normalization=normalization)

# %%
