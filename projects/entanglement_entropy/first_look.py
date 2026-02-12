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

# %% [markdown]
# # First look at ALICE data for entanglement entropy
#

# %%
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pachyderm.plot as pb
import uproot

base_path = Path("projects/entanglement_entropy")

pb.configure()


# %%
def read_data(filename: Path) -> dict[str, Any]:
    output = {}
    with uproot.open(filename) as f:
        for k in f.keys(cycle=False):
            output[k] = f[k].to_hist()

    return output


# hists = read_data(base_path / "test_hiccup_0004" / "LHC22o__test_partial_merge__BerkeleyTree__dijet_trigger_pt_leading_20_140_subleading_10_140.root")
hists = read_data(
    Path("trains")
    / "pp"
    / "0004"
    / "skim"
    / "hists"
    / "merged"
    / "LHC22o__full_merge__BerkeleyTree__dijet_trigger_pt_leading_20_140_subleading_10_140.root"
)

# %%
hists

# %% [markdown]
# # Spectra
#

# %%
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

plot_config.apply(fig=fig, ax=ax)

_output_path = base_path / "first_look_figures"
_output_path.mkdir(parents=True, exist_ok=True)
fig.savefig(_output_path / f"{plot_config.name}.pdf")
plt.close(fig)

# %%
text_font_size = 22

text = "ALICE work-in-progress"
text += "\n" + r"charged-particle jets"
text += "\n" + r"$R = 0.4$ $\sqrt{s_{\text{NN}}} = 13.6\:\text{TeV}$"
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
    hists[f"data_{jet_label}_jet_spectra"].plot(ax=ax, label="ALICE data")

    plot_config.apply(fig=fig, ax=ax)

    _output_path = base_path / "first_look_figures"
    _output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(_output_path / f"{plot_config.name}.pdf")
    plt.close(fig)

# %% [markdown]
# # N constituents

# %%
import hist

text_font_size = 22

text = "ALICE work-in-progress"
text += "\n" + r"charged-particle jets"
text += "\n" + r"$R = 0.4$ $\sqrt{s_{\text{NN}}} = 13.6\:\text{TeV}$"
plot_config = pb.PlotConfig(
    name="n_constituents_pt_integrated",
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

h = hists["data_n_constituents_lead_jet_pt"][sum, :, :]
z_axis_range = {
    "vmin": max(1e-4, h.values()[h.values() > 0].min()),
    "vmax": h.values().max(),
    # "vmax": 1,
}

# Plot
mesh = ax.pcolormesh(
    h.axes[0].edges.T,
    h.axes[1].edges.T,
    h.values().T,
    norm=mpl.colors.LogNorm(**z_axis_range),
)
fig.colorbar(mesh, pad=0.02)

plot_config.apply(fig=fig, ax=ax)

_output_path = base_path / "first_look_figures"
_output_path.mkdir(parents=True, exist_ok=True)
fig.savefig(_output_path / f"{plot_config.name}.pdf")
plt.close(fig)

# %%
text_font_size = 22

text = "ALICE work-in-progress"
text += "\n" + r"charged-particle jets"
text += "\n" + r"$R = 0.4$ $\sqrt{s_{\text{NN}}} = 13.6\:\text{TeV}$"
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
                    ),
                    pb.AxisConfig(
                        "y",
                        label=r"$\text{d}N/\text{d}n_{\text{const}}$",
                        font_size=text_font_size,
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
    ax.set_xlabel(r"$n_{\text{const}}^{\text{lead}}$")
    ax.set_ylabel(r"$n_{\text{const}}^{\text{sublead}}$")
    if jet_label == "lead":
        hists["data_n_constituents_lead_jet_pt"][sum, :, sum].plot(ax=ax, label="ALICE data")
    else:
        hists["data_n_constituents_lead_jet_pt"][sum, sum, :].plot(ax=ax, label="ALICE data")

    plot_config.apply(fig=fig, ax=ax)

    _output_path = base_path / "first_look_figures"
    _output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(_output_path / f"{plot_config.name}.pdf")
    plt.close(fig)

# %% [markdown]
# # Entanglement entropy

# %%
import numpy as np
import numpy.typing as npt
from pachyderm import binned_data


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
# 2nd try
import numpy as np


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

# %%
# 3rd try....
import numpy as np
from pachyderm import binned_data


def vectorized_entropy(dist: npt.NDArray[np.floating], sum_axes: int | tuple[int]):
    """
    Calculate Shannon entropy for each jet_pt bin (vectorized).

    Parameters:
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


input_hist = hists["data_n_constituents_lead_jet_pt"]
h_joint = binned_data.BinnedData.from_existing_data(input_hist)
h_lead = binned_data.BinnedData.from_existing_data(input_hist[:, :, sum])
h_sublead = binned_data.BinnedData.from_existing_data(input_hist[:, sum, :])

entropy_lead = vectorized_entropy(h_lead.values, sum_axes=1)
entropy_sublead = vectorized_entropy(h_sublead.values, sum_axes=1)
entropy_joint = vectorized_entropy(h_joint.values, sum_axes=(1, 2))

mutual_information = entropy_lead + entropy_sublead - entropy_joint

# %%
mutual_information

# %%
# Plot mutual information

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
                    label=r"$S_1 + S_2 - S_12$",
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
ax.plot(
    h_lead.axes[0].bin_centers,
    mutual_information,
    label="ALICE data",
    linestyle="",
    marker="o",
)

plot_config.apply(fig=fig, ax=ax)

_output_path = base_path / "first_look_figures"
_output_path.mkdir(parents=True, exist_ok=True)
fig.savefig(_output_path / f"{plot_config.name}.pdf")
plt.close(fig)

# %%
