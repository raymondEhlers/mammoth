# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# # Exploration plots for EECs
#
#

# +
from __future__ import annotations

# Setup
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pachyderm.plot as pb
from mammoth import helpers as mammoth_helpers
from mammoth.eec.plot import exploration

# %load_ext autoreload
# %autoreload 2

# %matplotlib widget
# #%config InlineBackend.figure_formats = ["png", "pdf"]
# Don't show mpl images inline. We'll handle displaying them separately.
#plt.ioff()
# Ensure the axes are legible on a dark background
#mpl.rcParams['figure.facecolor'] = 'w'

mammoth_helpers.setup_logging(level=logging.DEBUG)
# Quiet down the matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("pachyderm.histogram").setLevel(logging.INFO)
logging.getLogger("boost_histogram").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# General settings
embed_images = False
base_dir = Path("projects/EECs")
output_dir = base_dir / Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# +
from typing import Mapping, Sequence

import hist
import seaborn as sns
from pachyderm import binned_data
# -

# ## Load data

# +
import uproot

f = uproot.open(base_dir / "test_eec_thermal_model.root")
#hists = {key.replace(";1", ""): binned_data.BinnedData.from_existing_data(f.get(key)) for key in f}
hists = {key.replace(";1", ""): f.get(key).to_hist() for key in f}
# -

list(hists.keys())

# +
import cycler
import numpy as np

def _plot_RL(
    hists: Mapping[str, hist.Hist],
    names_and_labels: Mapping[str, str],
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    # TODO: Make this an argument
    trigger_name_to_range = {
        "reference": (5, 9),
        "signal": (20, 50),
    }
    # TODO: Collect this in the task...
    trigger_name_to_fraction = {
        "reference": 0.2,
        "signal": 0.8,
    }

    _palette_6_mod = {
        "purple": "#7e459e",
        "green": "#85aa55",
        "blue": "#7385d9",
        "magenta": "#b84c7d",
        "teal": "#4cab98",
        "orange": "#FF8301",
    }
    _extended_colors = {
        "alt_purple": "#c09cd3",
        # Generated
        #"alt_green": "#3f591d",
        "alt_green": "#517225",
        # Already existing green
        #"alt_green": "#55a270",
        "alt_blue": "#4bafd0",
    }

    #_colors_for_assignments = []
    #for _method in grooming_methods:
    _method_to_color = {
        "dynamical_core": _palette_6_mod["purple"],
        "dynamical_kt": _palette_6_mod["green"],
        "dynamical_time": _palette_6_mod["blue"],
        "soft_drop_z_cut_02": _palette_6_mod["magenta"],
        "dynamical_core_z_cut_02": _extended_colors["alt_purple"],
        "dynamical_kt_z_cut_02": _extended_colors["alt_green"],
        "dynamical_time_z_cut_02": _extended_colors["alt_blue"],
        "soft_drop_z_cut_04": _palette_6_mod["orange"],
    }
    #_colors_for_assignments.append(_method_to_color[_method])

    with sns.color_palette("Set2"):
        # fig, ax = plt.subplots(figsize=(9, 10))
        # Size is specified to make it convenient to compare against Hard Probes plots.
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(10, 8),
        )

        ax.set_prop_cycle(cycler.cycler(color=list(_method_to_color.values())))

        for name, (level, label, legend_entry) in names_and_labels.items():
            is_signal = "signal" == label
            h = hists[name]
            # Project by trigger range
            trigger_range = trigger_name_to_range[label]
            # Project to range
            h = h[:, hist.loc(trigger_range[0]):hist.loc(trigger_range[1]):hist.sum]
            # Convert
            h = binned_data.BinnedData.from_existing_data(h)

            # Normalize
            # TODO: Collect n_trig without being reliant on fraction...
            n_trig = hists[f"{level}_inclusive_trigger_spectra"][
                hist.loc(trigger_range[0]): hist.loc(trigger_range[1]):hist.sum
            ].value * trigger_name_to_fraction[label]
            h /= n_trig
            # Bin widths
            h /= h.axes[0].bin_widths

            p = ax.errorbar(
                h.axes[0].bin_centers,
                h.values,
                yerr=h.errors,
                xerr=h.axes[0].bin_widths / 2,
                marker="o",
                markersize=11,
                linestyle="",
                linewidth=3,
                label=legend_entry,
            )

    # Labeling and presentation
    plot_config.apply(fig=fig, ax=ax)
    # A few additional tweaks.
    #ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_RL(
    hists: dict[str, binned_data.BinnedData],
    names_and_labels: dict[str, str],
    output_dir: Path,
    text_font_size: int = 31,
) -> bool:
    collision_system = "embed_thermal_model"
    text = "ALICE Work in Progress"
    text += "\n" + r"PYTHIA8 $\bigotimes$ thermal model"
    text+= "\n" + r"$\sqrt{s_{\text{NN}}} = 5.02$ TeV"
    _plot_RL(
        hists=hists,
        names_and_labels=names_and_labels,
        plot_config=pb.PlotConfig(
            name=f"raw_EEC__{collision_system}__{'_'.join(names_and_labels)}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "x",
                            label=r"$R_{\text{L}}$",
                            log=True,
                            font_size=text_font_size,
                        ),
                        pb.AxisConfig(
                            "y",
                            label=r"$1/N_{\text{trig}}\:\text{d}N/\text{d}R_{\text{L}}$",
                            font_size=text_font_size,
                            #log=True,
                            #range=(1e2, 1e7),
                            range=(0, 10),
                        ),
                    ],
                    text=pb.TextConfig(x=0.02, y=0.98, text=text, font_size=text_font_size),
                    legend=pb.LegendConfig(location="upper right", font_size=round(text_font_size*0.8), anchor=(0.98, 0.98), marker_label_spacing=0.075),
                ),
            ],
            figure=pb.Figure(edge_padding={"left": 0.15, "bottom": 0.095, "top": 0.975}),
        ),
        output_dir=output_dir,
    )



# -

for names_and_labels in [
    {
        "det_level_signal_eec_log": ("det_level", "signal", "Det. level"),
        "hybrid_signal_eec_log": ("hybrid", "signal", "Hybrid"),
    },
    {
        "det_level_reference_eec_log": ("det_level", "reference", "Det. level ref."),
        "det_level_signal_eec_log": ("det_level", "signal", "Det. level signal"),
    },
    {
        "hybrid_reference_eec_log": ("hybrid", "reference", "Hybrid ref."),
        "hybrid_signal_eec_log": ("hybrid", "signal", "Hybrid signal"),
    },
]:
    plot_RL(
        hists=hists,
        names_and_labels=names_and_labels,
        output_dir=output_dir,
    )

list(hists.keys())


