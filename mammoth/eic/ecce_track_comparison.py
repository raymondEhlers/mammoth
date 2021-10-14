#!/usr/bin/env python3

"""Tracking comparison

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import itertools
import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import attr
import cycler
import hist
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot as pb
import seaborn as sns
import uproot
from mammoth import helpers
from mammoth.eic import base as ecce_base
from mammoth.eic import run_ecce_analysis
from pachyderm import binned_data

pb.configure()

logger = logging.getLogger(__name__)

_okabe_ito_colors = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]


def _load_results(input_specs: Sequence[run_ecce_analysis.DatasetSpec], input_dir: Path, filename: str) -> Dict[str, Dict[str, hist.Hist]]:
    output_hists = {}
    for spec in input_specs:
        logger.info(f"Loading hists from {input_dir / str(spec) / filename}")
        output_hists[str(spec)] = ecce_base.load_hists(input_dir / str(spec) / filename, filter="hist")

        for k, v in output_hists[str(spec)].items():
            output_hists[str(spec)][k] = v.to_hist()

    return output_hists


def _plot_tracking_comparison(input_specs: Sequence[run_ecce_analysis.DatasetSpec], hists: Mapping[str, Mapping[str, hist.Hist]],
                              all_regions: Sequence[Tuple[float, float]], regions_index: Sequence[int],
                              plot_config: pb.PlotConfig, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_prop_cycle(cycler.cycler(color=_okabe_ito_colors))

    for input_spec in input_specs:
        input_spec_hists = hists[str(input_spec)]
        for eta_index, h in input_spec_hists.items():
            m = h.values() > -1e-4
            indices_with_values = np.where(m)[0]
            if len(indices_with_values) == 0:
                logger.info(f"No valid values. Skipping {eta_index}, {str(input_spec)}")
                continue
            else:
                # Inefficient, but I don't really care here - it doesn't need to be that fast.
                groups = [(k, sum(1 for i in g)) for k,g in itertools.groupby(m)]
                # Max is three groups: Falses, followed by Trues, followed by Falses
                if len(groups) > 3:
                    logger.warning(f"Can't slice in a continuous range for {eta_index}, {str(input_spec)}. Groups: {groups}")
                # NOTE: Have to explicitly convert to int because they have an explicit isinstance on int, and apparently np.int64 doesn't count...
                s = slice(int(indices_with_values[0]), int(indices_with_values[-1] + 1))
            h_sliced = h[s]
            logger.info(f"plotting eta_index: {eta_index}, {str(input_spec)}")
            ax.errorbar(
                h_sliced.axes[0].centers, h_sliced.values(),
                yerr=np.sqrt(h_sliced.variances()),
                label=str(input_spec)
            )

    # Labeling and presentation
    plot_config.apply(fig=fig, ax=ax)
    # A few additional tweaks.
    #ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    # ax_ratio.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.2))

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_tracking_comparison(input_specs: Sequence[run_ecce_analysis.DatasetSpec], output_hists: Dict[str, Dict[str, hist.Hist]], hist_name_template: str, plot_name: str,
                             all_regions: Sequence[Tuple[float, float]],
                             regions_index: Sequence[int], regions_label: str,
                             output_dir: Path) -> None:
    text = "ECCE"
    text += "\n" + r"$R=0.5$ anti-$k_{\text{T}}$ jets"
    _plot_tracking_comparison(
        input_specs=input_specs,
        hists={
            str(input_spec): {
                str(index): output_hists[str(input_spec)][hist_name_template.format(particle=input_spec.particle.capitalize(), eta_region_index=index)]
                for index in regions_index
            }
            for input_spec in input_specs
        },
        all_regions=all_regions,
        regions_index=regions_index,
        plot_config=pb.PlotConfig(
            name=f"{plot_name}_{regions_label}_{'_'.join([str(v) for v in regions_index])}",
            panels=pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$p\:(\text{GeV}/c)$", font_size=22, range=(0, 30)),
                        pb.AxisConfig(
                            "y",
                            label=r"Mean",
                            #range=(0, 1.4),
                            font_size=22,
                        ),
                    ],
                    text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                    legend=pb.LegendConfig(location="lower left", font_size=22),
                ),
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.08)),
        ),
        output_dir=output_dir,
    )


if __name__ == "__main__":
    helpers.setup_logging()

    input_dir = Path("/Volumes/data/eic/trackingComparison/2021-10-13")
    output_dir = Path("/Volumes/data/eic/trackingComparison/2021-10-13/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_specs = [
        run_ecce_analysis.DatasetSpecSingleParticle(
            site="production",
            particle="electron",
            momentum_selection=[0.0, 20],
            label="",
        ),
        run_ecce_analysis.DatasetSpecSingleParticle(
            site="production",
            particle="pion",
            momentum_selection=[0.0, 20],
            label="",
        ),
        run_ecce_analysis.DatasetSpecSingleParticle(
            site="cades",
            particle="electron",
            momentum_selection=[0.3, 20],
            label="geoOption5",
        ),
        run_ecce_analysis.DatasetSpecSingleParticle(
            site="cades",
            particle="pion",
            momentum_selection=[0.3, 20],
            label="geoOption5",
        ),
        run_ecce_analysis.DatasetSpecSingleParticle(
            site="cades",
            particle="electron",
            momentum_selection=[0.3, 20],
            label="geoOption6",
        ),
        run_ecce_analysis.DatasetSpecSingleParticle(
            site="cades",
            particle="pion",
            momentum_selection=[0.3, 20],
            label="geoOption6",
        ),
    ]

    output_hists = _load_results(
        input_specs=input_specs,
        input_dir=input_dir,
        filename="output_TRKRS.root",
    )

    hist_name = "h_tracks_reso_p_{primary_track_source}_{name_reso_add}_{mc_particles_selection}_{eta_region}"
    # Example:   h_tracks_reso_p_0_All_All_6

    ...

    #int nEta = 15
    #Double_t partEta[nEta+1]        = { -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.2, -0.4, 0.4, 1.2,
    #                                   1.5, 2.0, 2.5, 3.0, 3.5, 4.0};

    # NOTE: 15 == - 4.0 - 4.0
    eta_bins = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.2, -0.4, 0.4, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    eta_ranges = [(a, b) for a, b in zip(eta_bins[:-1], eta_bins[1:])]
    eta_ranges.append((-4.0, 4.0))
    #backward_regions = eta_ranges[0:5]
    #barrel_regions = eta_ranges[5:10]
    #forward_regions = eta_ranges[10:14]
    backward_regions = list(range(0, 5))
    barrel_regions = list(range(5, 11))
    forward_regions = list(range(11, 15))

    #hist_name = "histPResol_Electron_FitMean_15"
    hist_name_template = "histPResol_{particle}_FitMean_{eta_region_index}"

    #for label, regions in [("backward", backward_regions), ("barrel", barrel_regions), ("forward", forward_regions)]:
    for i in range(0, len(eta_ranges)):
        if i in backward_regions:
            label = "backward"
        elif i in barrel_regions:
            label = "barrel"
        elif i in forward_regions:
            label = "forward"
        else:
            label = "all"

        plot_tracking_comparison(
            input_specs=input_specs,
            output_hists=output_hists,
            hist_name_template=hist_name_template,
            plot_name="p_mean",
            all_regions=eta_ranges, regions_label=label, regions_index=[i],
            output_dir=output_dir,
        )

    import IPython; IPython.embed()

    hist_name_template = "histPResol_{particle}_FitSigma_15"

    plot_tracking_comparison(
        input_specs=input_specs, output_hists=output_hists, hist_name_template=hist_name_template
    )


