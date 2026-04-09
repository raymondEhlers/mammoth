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

import itertools
from pathlib import Path
from typing import Any

import hist
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pachyderm.plot as pb
import polars as pl  # noqa: F401
import uproot
from pachyderm import binned_data

from mammoth.entanglement_entropy import skim_to_hist
from mammoth.framework.io import output_utils

base_path = Path("projects/entanglement_entropy")

pb.configure()

# %%
# %load_ext autoreload

# %autoreload 2

from mammoth.entanglement_entropy import skim_to_hist

# %%
# Load data
run_2_pp_ref_data = skim_to_hist.load_data(skim_directory=Path("trains/pp/0063/skim"), level_name="data")
run_3_pp_ref_data = skim_to_hist.load_data(skim_directory=Path("trains/pp/0006/skim"), level_name="data")
#run_2_pp_MC_pp_ref_data = skim_to_hist.load_data(skim_directory=Path("trains/pp_MC/0086/skim"), level_name="data")

# %%
run_2_pp_ref_data

# %%
hists_per_data_source = {
    "run_2_pp_ref": skim_to_hist.define_base_histograms(levels=["data"]),
    "run_3_pp_ref": skim_to_hist.define_base_histograms(levels=["data"]),
    # "run_2_pp_MC_pp_ref": skim_to_hist.define_base_histograms(levels=["data"]),
}

skim_to_hist.fill_base_histograms(levels=["data"], hists=hists_per_data_source["run_2_pp_ref"], jet_data=run_2_pp_ref_data)
skim_to_hist.fill_base_histograms(levels=["data"], hists=hists_per_data_source["run_3_pp_ref"], jet_data=run_3_pp_ref_data)
# skim_to_hist.fill_base_histograms(levels=["data"], hists=hists_per_data_source["run_2_pp_MC_pp_ref"], jet_data=run_2_pp_MC_pp_ref_data)

# %%
# Labels
data_source_to_presentation_label = {
    "run_2_pp_ref": "pp, 5.02 TeV",
    "run_2_pp_MC_pp_ref": "pythia part level, 5.02 TeV",
    "run_3_pp_ref": "pp, 5.36 TeV",
}

# %%
hists_per_data_source["run_2_pp_ref"].keys()


# %%
def read_data(filename: Path) -> dict[str, Any]:
    output = {}
    with uproot.open(filename) as f:
        for k in f.keys(cycle=False):
            output[k] = f[k].to_hist()

    return output


def split_input_hists_into_chunks(directory_containing_hists: Path, n_chunks: int) -> list[list[Path]]:
    """Read data into N chunks, along with the merged hist"""
    all_hists = list(directory_containing_hists.glob("*.root"))

    # First, shuffle up the list
    rng = np.random.default_rng()
    # NOTE: shuffle is in place
    rng.shuffle(all_hists)

    # 2. Calculate the base size and remainder
    list_len = len(all_hists)
    base_size = list_len // n_chunks
    remainder = list_len % n_chunks

    # 3. Create an iterator from the shuffled list
    it = iter(all_hists)

    # 4. Use a list comprehension with itertools.islice to create chunks
    chunks = []
    for i in range(n_chunks):
        # Determine the size of the current chunk
        chunk_size = base_size + (1 if i < remainder else 0)

        # Use islice to grab the next 'chunk_size' items from the iterator
        # and convert the resulting iterator to a list
        chunk = list(itertools.islice(it, chunk_size))
        chunks.append(chunk)

    return chunks


def merge_hist_chunks(hists_in_chunks: list[list[Path]], merged_analysis_hists_path: Path) -> None:
    # Merge each chunk
    for i, input_hists in enumerate(hists_in_chunks):
        output_utils.shit_hadd(input_filenames=input_hists, output_filename=merged_analysis_hists_path / f"{i}.root")

    # And then the full merge
    output_utils.shit_hadd(input_filenames=list(itertools.chain.from_iterable(hists_in_chunks)), output_filename=merged_analysis_hists_path / "full_merged.root")


def read_merged_chunks(
    merged_analysis_hists_path: Path, n_chunks: int
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    hist_chunks = {}
    for i in range(n_chunks):
        hist_chunks[i] = read_data(merged_analysis_hists_path / f"{i}.root")

    hists_merged = read_data(merged_analysis_hists_path / "full_merged.root")

    return hist_chunks, hists_merged


def read_hists_in_chunks(
    directory_containing_hists: Path, n_chunks: int
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    # Setup
    merged_analysis_hists_path = directory_containing_hists / "merged" / "analysis_hists" / f"{n_chunks}_chunks"
    merged_analysis_hists_path.mkdir(parents=True, exist_ok=True)

    # Check if they already exist. If so, no need to recreate them
    def merged_files_exist(merged_analysis_hists_path: Path, n_chunks: int) -> bool:
        files_exist = [(merged_analysis_hists_path / f"{i}.root").exists() for i in range(n_chunks)] + [
            (merged_analysis_hists_path / "full_merged.root").exists()
        ]

        print(files_exist)

        return all(files_exist)

    if not merged_files_exist(merged_analysis_hists_path=merged_analysis_hists_path, n_chunks=n_chunks):
        print("Creating files...")
        # Split up the hists in chunk chunks
        hist_filenames_in_chunks = split_input_hists_into_chunks(
            directory_containing_hists=directory_containing_hists, n_chunks=n_chunks
        )

        # Merge them in the chunks (and write them to file)
        merge_hist_chunks(
            hists_in_chunks=hist_filenames_in_chunks, merged_analysis_hists_path=merged_analysis_hists_path
        )
    else:
        print("Skipping file creation and using existing ones...")

    # And then read them back so we can use them for analysis
    return read_merged_chunks(merged_analysis_hists_path=merged_analysis_hists_path, n_chunks=n_chunks)


# hists = read_data(base_path / "test_hiccup_0004" / "LHC22o__test_partial_merge__BerkeleyTree__dijet_trigger_pt_leading_20_140_subleading_10_140.root")
# hists = read_data(
#     Path("trains")
#     / "pp"
#     / "0004"
#     / "skim"
#     / "hists"
#     / "merged"
#     / "LHC22o__full_merge__BerkeleyTree__dijet_trigger_pt_leading_20_140_subleading_10_140.root"
# )

hists_in_chunks, merged_hists = read_hists_in_chunks(
    directory_containing_hists=Path("trains") / "pp" / "0004" / "skim" / "hists",
    # n_chunks=20,
    # n_chunks=10,
    n_chunks=8,
)

# %%
hists = merged_hists
hists


# %% [markdown]
# # Spectra
#

# %%
def plot_inclusive_spectra(hists: dict[str, hist.Hist[hist.storage.Weight]], label: str) -> None:
    msg = "We don't have fully inclusive spectra from the skim. Skip this..."
    raise ValueError(msg)
    text_font_size = 22

    text = "ALICE work-in-progress"
    text += "\n" + r"charged-particle jets"
    text += "\n" + r"$R = 0.4$ $\sqrt{s_{\text{NN}}} = 13.6\:\text{TeV}$"
    plot_config = pb.PlotConfig(
        name="inclusive_jet_spectra",
        panels=[
            # Main panel
            pb.Panel(
                axes=[
                    pb.AxisConfig(
                        "x",
                        label=r"$p_{\text{T, ch jet}}$",
                        font_size=text_font_size,
                    ),
                    pb.AxisConfig(
                        "y",
                        label=r"$dN/dp_{\text{T}}$",
                        font_size=text_font_size,
                        log=True,
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
    hists["data_inclusive_jet_spectra"].plot(ax=ax, label="ALICE data")
    # for sub_hists in hists_in_chunks.values():
    #     sub_hists["data_inclusive_jet_spectra"].plot(ax=ax)

    plot_config.apply(fig=fig, ax=ax)

    _output_path = base_path / "second_look_with_skim" / label
    _output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(_output_path / f"{plot_config.name}.pdf")
    plt.close(fig)


# %%
def plot_dijet_spectra(hists_per_data_source: dict[str, dict[str, hist.Hist[hist.storage.Weight]]], normalize: bool) -> None:
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
                    text=pb.TextConfig(x=0.95, y=0.81, text=text, font_size=18),
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
            h = hists[f"data_{jet_label}_jet_spectra"]
            h = binned_data.BinnedData.from_existing_data(h)
            if normalize:
                h /= sum(h.values)
            ax.errorbar(
                h.axes[0].bin_centers,
                h.values,
                xerr=h.axes[0].bin_widths / 2,
                # TODO(RJE): It's not working for some reason - not sure why...
                #            variances seem way too big compared to the values...
                # yerr=h.variances,
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
def plot_n_constituents_lead_pt_differential(hists_per_data_source: dict[str, dict[str, hist.Hist[hist.storage.Weight]]], normalize: bool, lead_jet_pt_range: tuple[float, float] | None = None) -> None:
    text_font_size = 22

    for source, hists in hists_per_data_source.items():
        text = "ALICE work-in-progress"
        text += "\n" + r"charged-particle jets"
        text += "\n" + fr"$R = 0.4$ {data_source_to_presentation_label[source]}"
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
            s = slice(lead_jet_pt_range[0] * 1.j, lead_jet_pt_range[1] * 1.j, sum)

        h = binned_data.BinnedData.from_existing_data(hists["data_n_constituents_jet_pt"][s, sum, :, :])
        if normalize:
            normalization_values = h.values.sum()
            h.values = np.divide(h.values, normalization_values, out=np.zeros_like(h.values), where=normalization_values != 0)

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
        h = plot_n_constituents_lead_pt_differential(hists_per_data_source=hists_per_data_source, normalize=normalize, lead_jet_pt_range=lead_jet_pt_range)


# %%
def plot_n_constituents_1d(hists_per_data_source: dict[str, dict[str, hist.Hist[hist.storage.Weight]]], lead_jet_pt_range: tuple[float, float] | None = None) -> None:
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
                            log=True,
                        ),
                    ],
                    text=pb.TextConfig(x=0.95, y=0.81, text=text, font_size=18),
                    legend=pb.LegendConfig(location="upper right", anchor=(0.95, 0.95), font_size=22),
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
                s = slice(lead_jet_pt_range[0] * 1.j, lead_jet_pt_range[1] * 1.j, sum)
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
                # TODO(RJE): It's not working for some reason - not sure why...
                #            variances seem way too big compared to the values...
                # yerr=h.variances,
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
            fig.savefig(_output_path / f"{plot_config.name}{tag}.pdf")
            plt.close(fig)

for lead_jet_pt_range in [None, (10, 20), (20, 40), (40, 60), (60, 80), (80, 120)]:
    plot_n_constituents_1d(hists_per_data_source=hists_per_data_source, lead_jet_pt_range=lead_jet_pt_range)


# %% [markdown]
# # Entanglement entropy

# %%
def vectorized_entropy(dist: npt.NDArray[np.floating], sum_axes: int | tuple[int]) -> npt.NDArray[np.floating]:
    """Calculate Shannon entropy for each jet_pt bin (vectorized).

    Args:
        dist: numpy array with jet_pt as the first dimension
        sum_axes: int or tuple of axes to sum over for normalization and entropy

    Returns:
        entropy: 1D array of entropies for each jet_pt bin (in nats)
    """
    # Normalize along the specified axes (keepdims for proper broadcasting)
    norm = np.sum(dist, axis=sum_axes, keepdims=True)
    # print(f"{norm=}")
    # if np.isclose(norm, 0):
    #    return np.zeros(dist.shape[0])
    # prob_dist = dist / norm
    # prob_dist = np.where(norm > 0, dist / norm, 0)
    # prob_dist = np.divdide(norm > 0, dist / norm, 0)
    prob_dist = np.divide(dist, norm, out=np.zeros_like(dist), where=norm > 0)

    # Handle zeros: use np.where to avoid log(0)
    # safe_log = np.where(prob_dist > 0, np.log(prob_dist), 0)
    safe_log = np.log(prob_dist, out=np.zeros_like(prob_dist), where=prob_dist > 0)

    # Calculate entropy: -sum(p * log(p))
    entropy = -np.sum(prob_dist * safe_log, axis=sum_axes)

    return entropy.squeeze()


import attrs


@attrs.define(frozen=True)
class Result:
    lead: npt.NDArray[np.float64]
    sublead: npt.NDArray[np.float64]
    joint: npt.NDArray[np.float64]


#def calculate_entropy(input_hists: dict[str, hist.Hist]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
def calculate_entropy(input_hists: dict[str, hist.Hist]) -> Result:
    input_hist = input_hists["data_n_constituents_jet_pt"]
    # Axes: (lead jet pt, sublead jet pt, lead n_const, sublead n_const)
    h_joint = binned_data.BinnedData.from_existing_data(input_hist)
    # Axes: (lead jet pt, sublead jet pt, lead n_const)
    h_lead = binned_data.BinnedData.from_existing_data(input_hist[:, :, :, sum])
    # Axes: (lead jet pt, sublead jet pt, sublead n_const)
    h_sublead = binned_data.BinnedData.from_existing_data(input_hist[:, :, sum, :])

    entropy_lead = vectorized_entropy(h_lead.values, sum_axes=2)
    # NOTE: This is the same axis as for the lead because we've summed over the lead above
    entropy_sublead = vectorized_entropy(h_sublead.values, sum_axes=2)
    entropy_joint = vectorized_entropy(h_joint.values, sum_axes=(2, 3))

    # Returned values are the entropy values as a function of (lead jet pt, sublead jet pt)
    return Result(lead=entropy_lead, sublead=entropy_sublead, joint=entropy_joint)

def calculate_mutual_information(input_hists: dict[str, hist.Hist]) -> npt.NDArray[np.float64]:
    entropy = calculate_entropy(input_hists=input_hists)

    mutual_information = entropy.lead + entropy.sublead - entropy.joint
    return mutual_information  # noqa: RET504

# As a function of leading jet pt
entropies = {
    source: calculate_entropy(hists)
    for source, hists in hists_per_data_source.items()
}
# entropies_chunks = {}
# for k, v in hists_in_chunks.items():
#     entropies_chunks[k] = calculate_entropy(v)

mutual_information = {
    source: calculate_mutual_information(hists)
    for source, hists in hists_per_data_source.items()
}
# mutual_information_chunks = {}
# for k, v in hists_in_chunks.items():
#     mutual_information_chunks[k] = calculate_mutual_information(v)

# %%
entropies.sublead.shape

# %%
# Plot entropies as a function of jet pt

# Just need the binning info, so pick the full precision
h_pt = binned_data.BinnedData.from_existing_data(hists_per_data_source["run_2_pp_ref"]["data_n_constituents_jet_pt"][:, :, sum, sum])

def plot_entropy(
    entropies: dict[str, hist.Hist[hist.storage.Weight]],
    lead_jet_pt_range: tuple[float, float] | None = None,
    sublead_jet_pt_range: tuple[float, float] | None = None
) -> None:
    text_font_size = 22

    for label in ["lead", "sublead"]:
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
                            label=r"$p_{\text{T, " + label + "}}$",
                            font_size=text_font_size,
                        ),
                        pb.AxisConfig(
                            "y",
                            label=r"$S_{\text{" + label + r"}}$",
                            font_size=text_font_size,
                        ),
                    ],
                    text=pb.TextConfig(x=0.05, y=0.95, text=text, font_size=18),
                    legend=pb.LegendConfig(location="upper right", font_size=14),
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
        selection = (sum, slice(None, None, None)) if label == "lead" else (slice(None, None, None), sum)
        for source, entropy in entropies.items():
            x_axis_selection = 0 if label == "lead" else 1
            entropy_axis_selection = 1 if label == "lead" else 0

            ax.plot(
                h_pt.axes[x_axis_selection].bin_centers,
                np.sum(getattr(entropy, label), axis=entropy_axis_selection),
                label=f"ALICE data ({data_source_to_presentation_label[source]})",
                linestyle="",
                marker="s",
            )

        plot_config.apply(fig=fig, ax=ax)

        _output_path = base_path / "second_look_with_skim"
        _output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(_output_path / f"{plot_config.name}.pdf")
        plt.close(fig)

plot_entropy(entropies)

# %%
mutual_information, mutual_information_chunks[0]

# %%
# We'll define the "covariance", although I think it's really not a covariance.
covariance = -1 * mutual_information
covariance

covariance_chunks = {}
for k, v in mutual_information_chunks.items():
    covariance_chunks[k] = -1 * v


# %%
def calculate_uncertainties(values: dict[str, npt.NDArray[np.float64]], n_exclude: int = 0) -> npt.NDArray[np.float64]:
    all_values = np.vstack([v for v in values.values()])  # noqa: C416
    print(all_values[:, 32])
    all_values = np.sort(all_values, axis=0)
    limited = all_values
    if n_exclude > 0:
        limited = all_values[n_exclude:-n_exclude]
    print(limited.shape)
    print(limited[:, 32])
    return np.min(limited, axis=0), np.max(limited, axis=0)


calculate_uncertainties(mutual_information_chunks, n_exclude=1)

# %%
# Plot mutual information

# Just need the binning info, so pick the full precision
h_lead = binned_data.BinnedData.from_existing_data(hists["data_n_constituents_lead_jet_pt"][:, :, sum])

text_font_size = 22

text = "ALICE work-in-progress"
text += "\n" + r"charged-particle jets"
text += "\n" + r"$R = 0.4$ $\sqrt{s_{\text{NN}}} = 13.6\:\text{TeV}$"
plot_config = pb.PlotConfig(
    name="mutual_information_lead_jet_pt",
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
                ),
            ],
            text=pb.TextConfig(x=0.05, y=0.81, text=text, font_size=18),
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
ax.plot(
    h_lead.axes[0].bin_centers,
    mutual_information,
    label="ALICE data",
    linestyle="",
    marker="o",
)
for v in mutual_information_chunks.values():
    ax.plot(
        h_lead.axes[0].bin_centers,
        v,
        #linestyle="-",
        marker="o",
        color="red",
    )
ax.plot(
    h_lead.axes[0].bin_centers,
    np.mean(list(mutual_information_chunks.values()), axis=0),
    label="Mean of chunks",
    linestyle="",
    marker="o",
)

# Plot with uncertainties
mutual_information_low, mutual_information_high = calculate_uncertainties(mutual_information_chunks, n_exclude=0)
# TODO(RJE): If I exclude, sometimes I'm getting negative errors since I'm excluding the largest value.
#            I thought I could require at least 0, but somehow it doesn't work. Figure this out tomorrow
low = np.maximum(mutual_information - mutual_information_low, np.zeros_like(mutual_information))
high = np.maximum(mutual_information_high - mutual_information, np.zeros_like(mutual_information))
low = mutual_information - mutual_information_low
high = mutual_information_high - mutual_information
print(f"{low=}, {high=}")
# ax.errorbar(
#     x=h_lead.axes[0].bin_centers,
#     y=mutual_information,
#     yerr=np.array(
#         [
#             mutual_information - mutual_information_low,
#             mutual_information_high - mutual_information,
#         ]
#     ),
#     label="ALICE data",
# #     linestyle="",
#     marker="o",
# )
# Just trying this out for visibility
ax.fill_between(
    x=h_lead.axes[0].bin_centers,
    y1=mutual_information_low,
    y2=mutual_information_high,
    alpha=0.3,
    zorder=5,
)

plot_config.apply(fig=fig, ax=ax)

_output_path = base_path / "second_look_with_skim"
_output_path.mkdir(parents=True, exist_ok=True)
fig.savefig(_output_path / f"{plot_config.name}.pdf")
plt.close(fig)

# %%
# A vibe coded check of the entropy bias
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


def check_entropy_bias(hists):
    """
    Perform a scaling check to estimate the bias-free Mutual Information.
    """
    input_hist = hists["data_n_constituents_lead_jet_pt"]

    # Let's say the full histogram has N events
    # We want to subsample it at different fractions to see the trend
    # Note: For histograms, 'subsampling' usually means scaling the weights
    # or multinomial sampling, but if you have the raw data chunks,
    # it is better to aggregate them.

    # Assuming 'hists' is your full aggregate, and you can recreate
    # smaller aggregates (e.g., from your chunks).

    # Let's define fractions of the dataset to test
    # e.g., 1/20, 1/10, 1/5, 1/2, 1/1
    # You already have chunks of 1/20.

    fractions = []
    mi_values = []

    # 1. Calculate for single chunks (1/20th statistics)
    # Average the MI across all your chunks to get a stable point
    mi_chunks_avg = np.mean(list(mutual_information_chunks.values()), axis=0)
    fractions.append(1/len(mutual_information_chunks))
    mi_values.append(mi_chunks_avg)

    # 2. Calculate for combined chunks (e.g., pairs -> 1/10th statistics)
    # (Pseudo-code: combine chunk1+chunk2, chunk3+chunk4...)
    # mi_1_10_avg = ...
    # fractions.append(1/10)
    # mi_values.append(mi_1_10_avg)

    # 3. Add the Full dataset (1/1 statistics)
    mi_full = mutual_information # Your existing calculation
    fractions.append(1.0)
    mi_values.append(mi_full)

    # Convert to arrays for easy slicing per pt-bin
    fractions = np.array(fractions)
    mi_values = np.array(mi_values) # Shape (n_fractions, n_pt_bins)

    # --- Extrapolation Step ---

    # The bias scales with 1/N.
    # fractions represent N/N_total.
    # So we plot MI vs (1/fraction).
    inv_fractions = 1.0 / fractions

    estimated_true_mi = []

    # Loop over each pt bin
    for i in range(mi_values.shape[1]):
        y = mi_values[:, i]
        x = inv_fractions

        # Fit line: MI_obs = MI_true + (Bias_Const * 1/N)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # The intercept is the value at 1/N = 0 (Infinite Statistics)
        estimated_true_mi.append(intercept)

        # Optional: Plotting for a specific bin to check linearity
        if i == 80: # Plot first bin as sanity check
            plt.scatter(x, y, label='Observed MI at diff stats')
            plt.plot(x, intercept + slope*x, 'r--', label='Fit')
            plt.scatter([0], [intercept], color='green', marker='*', s=100, label='Extrapolated True MI')
            plt.xlabel(r"Inverse Fraction of Data (1/f) $\propto 1/N$")
            plt.ylabel("Calculated Mutual Information")
            plt.legend()
            plt.title(f"Bias Check for pT bin {i}")
            plt.show()

    return np.array(estimated_true_mi)

check_entropy_bias(hists)

# %% [markdown]
# ## Linfoot coefficient

# %%
# Another possible definition - using the Linfoot coefficient
# $\rho = \sqrt{1 - \exp{2\text{Cov}(1, 2)}}$
# We artificially introduce the sign to keep track of that information

# linfoot = np.sign(covariance) * np.sqrt(1 - np.exp(2 * covariance))
linfoot_coefficient = np.sqrt(1 - np.exp(2 * covariance))
linfoot_coefficient_chunks = {k: np.sqrt(1 - np.exp(2 * _v)) for k, _v in covariance_chunks.items()}

# Plot linfoot coefficient

text_font_size = 22

text = "ALICE work-in-progress"
text += "\n" + r"charged-particle jets"
text += "\n" + r"$R = 0.4$ $\sqrt{s_{\text{NN}}} = 13.6\:\text{TeV}$"
text += "\n" + r"Linfoot coefficient w/ mutual info"
text += "\n" + r"$\rho = \sqrt{1 - \exp(-2 I(n_1, n_2))}$"
plot_config = pb.PlotConfig(
    name="linfoot_coefficient_mutual_information_lead_jet_pt",
    panels=[
        # Main panel
        pb.Panel(
            axes=[
                pb.AxisConfig(
                    "x",
                    label=r"$p_{\text{T, lead}}$",
                    font_size=text_font_size,
                ),
                pb.AxisConfig("y", label=r"Linfoot $\rho$", font_size=text_font_size, range=(-0.1, 1.3)),
            ],
            text=pb.TextConfig(x=0.03, y=0.85, text=text, font_size=18),
            legend=pb.LegendConfig(location="upper left", anchor=(0.03, 0.95), font_size=22),
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
ax.plot(
    h_lead.axes[0].bin_centers,
    linfoot_coefficient,
    label="ALICE data",
    linestyle="",
    marker="o",
)

# Plot with uncertainties
linfoot_coefficient_low, linfoot_coefficient_high = calculate_uncertainties(linfoot_coefficient_chunks, n_exclude=0)
# TODO(RJE): If I exclude, sometimes I'm getting negative errors since I'm excluding the largest value.
#            I thought I could require at least 0, but somehow it doesn't work. Figure this out tomorrow
low = np.maximum(linfoot_coefficient - linfoot_coefficient_low, np.zeros_like(linfoot_coefficient))
high = np.maximum(linfoot_coefficient_high - linfoot_coefficient, np.zeros_like(linfoot_coefficient))
low = linfoot_coefficient - linfoot_coefficient_low
high = linfoot_coefficient_high - linfoot_coefficient
print(f"{low=}, {high=}")
# ax.errorbar(
#     x=h_lead.axes[0].bin_centers,
#     y=linfoot_coefficient,
#     yerr=np.array(
#         [
#             linfoot_coefficient - linfoot_coefficient_low,
#             linfoot_coefficient_high - linfoot_coefficient,
#         ]
#     ),
#     label="ALICE data",
# #     linestyle="",
#     marker="o",
# )
# Just trying this out for visibility
ax.fill_between(
    x=h_lead.axes[0].bin_centers,
    y1=linfoot_coefficient_low,
    y2=linfoot_coefficient_high,
    alpha=0.3,
)

plot_config.apply(fig=fig, ax=ax)

_output_path = base_path / "second_look_with_skim"
_output_path.mkdir(parents=True, exist_ok=True)
fig.savefig(_output_path / f"{plot_config.name}.pdf")
plt.close(fig)

# %% [markdown]
# ## Plot entropies from different chunks

# %%
# Plot entropies from different chunks

# Just need the binning info, so pick the full precision
h_lead = binned_data.BinnedData.from_existing_data(hists["data_n_constituents_lead_jet_pt"][:, :, sum])

text_font_size = 22

for label in ["lead", "sublead"]:
    text = "ALICE work-in-progress"
    text += "\n" + r"charged-particle jets"
    text += "\n" + r"$R = 0.4$ $\sqrt{s_{\text{NN}}} = 13.6\:\text{TeV}$"
    plot_config = pb.PlotConfig(
        name=f"entropy_{label}_lead_jet_pt",
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
                        label=r"$S_{\text{" + label + r"}}$",
                        font_size=text_font_size,
                    ),
                ],
                text=pb.TextConfig(x=0.05, y=0.95, text=text, font_size=18),
                legend=pb.LegendConfig(location="lower center", anchor=(0.5, 0.05), font_size=14),
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
    for i, v in enumerate(entropies_chunks.values()):
        ax.plot(
            h_lead.axes[0].bin_centers,
            getattr(v, label),
            #linestyle="-",
            marker="o",
            label=f"Chunk {i}"
            #color="red",
        )
    ax.plot(
        h_lead.axes[0].bin_centers,
        getattr(entropies, label),
        label="ALICE data",
        linestyle="",
        marker="s",
        color="blue",
    )
    # ax.plot(
    #     h_lead.axes[0].bin_centers,
    #     np.mean(list(mutual_information_chunks.values()), axis=0),
    #     label="Mean of chunks",
    #     linestyle="",
    #     marker="o",
    # )

    plot_config.apply(fig=fig, ax=ax)

    _output_path = base_path / "second_look_with_skim"
    _output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(_output_path / f"{plot_config.name}.pdf")
    plt.close(fig)


# %%
# Entropy test - can I not sum over any axes when calculating it?

def calculate_entropy_test(input_hists: dict[str, hist.Hist]) -> Result:
    input_hist = input_hists["data_n_constituents_lead_jet_pt"]
    h_joint = binned_data.BinnedData.from_existing_data(input_hist)
    h_lead = binned_data.BinnedData.from_existing_data(input_hist[:, :, sum])
    h_sublead = binned_data.BinnedData.from_existing_data(input_hist[:, sum, :])

    entropy_lead = vectorized_entropy(h_lead.values, sum_axes=())
    entropy_sublead = vectorized_entropy(h_sublead.values, sum_axes=())
    entropy_joint = vectorized_entropy(h_joint.values, sum_axes=(1, 2))

    return Result(lead=entropy_lead, sublead=entropy_sublead, joint=entropy_joint)

# As a function of leading jet pt
entropies_test = calculate_entropy_test(hists)
entropies_test_chunks = {}
for k, v in hists_in_chunks.items():
    entropies_test_chunks[k] = calculate_entropy_test(v)

# %%
# Doesn't work as formulated...
np.where(entropies_test.lead != 0)


# %% [markdown]
# # Older tests

# %%
def shannon_entropy(P: npt.NDArray[np.float64]) -> np.float64:
    return -np.sum(P[P > 0] * np.log(P[P > 0]), axis=1)


def shannon_entropy_on_hist(h: binned_data.BinnedData, n_dijets: int) -> binned_data.BinnedData:
    h /= n_dijets
    h_out = h.copy()
    values = h_out.values()
    values = values[values > 0] * np.log(values[values > 0])
    h_out.values = values
    return h_out


def mutual_information(h: hist.Hist):
    P_AB = h / h.values().sum(axis=None)
    # P_AB = joint_hist / len(n_A)
    P_A = P_AB[:, :, sum]
    P_B = P_AB[:, sum, :]
    S_A = shannon_entropy(P_A.values())
    S_B = shannon_entropy(P_B.values())
    S_AB = shannon_entropy(P_AB.values())
    return S_A, S_B, S_AB, S_A + S_B - S_AB


# %%
print(hists["data_n_constituents_lead_jet_pt"].values().sum(axis=None))
mutual_information(hists["data_n_constituents_lead_jet_pt"])

# %%
h = hists["data_n_constituents_lead_jet_pt"]
P_AB = h / h.values().sum(axis=None)
# P_AB = joint_hist / len(n_A)
P_A = P_AB[:, :, sum]
P_B = P_AB[:, sum, :]


# %%
def test(P):
    shape = P.shape
    return P[P > 0] * np.log(P[P > 0])


proj_onto_jet_pt = P_B[:, sum]
mask_reshaped = (P_B.values() > 0).reshape(P_B.shape)
# test(P_B.values())
# P_B.values()[mask_reshaped[:, ]].reshape(P_B.shape)
ma = np.ma.masked_less_equal(P_B.values(), 0)
probabilities = ma / ma.sum(axis=1, keepdims=True)
epsilon = 1e-10
probabilities = np.maximum(probabilities, epsilon)
res = (probabilities * np.log(probabilities)).sum(axis=1)

# %%
# TODO(RJE): Just need to do this bin-by-bin for each and then construct the cov
res.data

# %%
P_B.values()[:, P_B.values() > 0]


# %%
def calculate(h: binned_data.BinnedData) -> binned_data.BinnedData:
    # Normalize
    h /= np.sum(h.values)

    # Calculate probability
    values = h.values.copy()
    shape = values.shape
    print(values.shape)
    values = (values[values > 0] * np.log(values[values > 0])).reshape(shape)
    # And store in a new histogram
    return binned_data.BinnedData(
        axes=[ax.copy() for ax in h.axes],
        values=values,
        variances=np.ones_like(values),
    )


calculate(binned_data.BinnedData.from_existing_data(hists["data_n_constituents_lead_jet_pt"]))
