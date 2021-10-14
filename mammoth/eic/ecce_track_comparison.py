#!/usr/bin/env python3

"""Tracking comparison

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import itertools
import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import attr
import hist
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot as pb
import seaborn as sns
import uproot
from mammoth import helpers
from mammoth.eic import base as ecce_base
from mammoth.eic import plot_ecce_track_comparison, run_ecce_analysis
from pachyderm import binned_data


pb.configure()

logger = logging.getLogger(__name__)


def _load_results(input_specs: Sequence[run_ecce_analysis.DatasetSpec], input_dir: Path, filename: str) -> Dict[str, Dict[str, hist.Hist]]:
    output_hists = {}
    for spec in input_specs:
        logger.info(f"Loading hists from {input_dir / str(spec) / filename}")
        output_hists[str(spec)] = ecce_base.load_hists(input_dir / str(spec) / filename, filter="hist")

        for k, v in output_hists[str(spec)].items():
            output_hists[str(spec)][k] = v.to_hist()

    return output_hists



def run() -> None:
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
        #run_ecce_analysis.DatasetSpecSingleParticle(
        #    site="production",
        #    particle="pion",
        #    momentum_selection=[0.0, 20],
        #    label="",
        #),
        run_ecce_analysis.DatasetSpecSingleParticle(
            site="cades",
            particle="electron",
            momentum_selection=[0.3, 20],
            label="geoOption5",
        ),
        #run_ecce_analysis.DatasetSpecSingleParticle(
        #    site="cades",
        #    particle="pion",
        #    momentum_selection=[0.3, 20],
        #    label="geoOption5",
        #),
        run_ecce_analysis.DatasetSpecSingleParticle(
            site="cades",
            particle="electron",
            momentum_selection=[0.3, 20],
            label="geoOption6",
        ),
        #run_ecce_analysis.DatasetSpecSingleParticle(
        #    site="cades",
        #    particle="pion",
        #    momentum_selection=[0.3, 20],
        #    label="geoOption6",
        #),
    ]

    # Setup
    _input_spec_labels = {}
    for particle in ["Pion", "Electron"]:
        _input_spec_labels.update({
            f"production-single{particle}-p-0-to-20": "2 LGAD layers",
            f"cades-single{particle}-p-0.3-to-20-geoOption5": "1 LGAD layer, $30 \mu$m",
            f"cades-single{particle}-p-0.3-to-20-geoOption6": "1 LGAD layer, $55 \mu$m",
        })

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
    backward_regions = list(range(1, 5))
    barrel_regions = list(range(5, 10))
    forward_regions = list(range(10, 14))
    all = [15]

    #hist_name = "histPResol_Electron_FitMean_15"
    hist_name_template = "histPResol_{particle}_FitMean_{eta_region_index}"

    # Plots:
    # - [x] Single pion in four forward regions for each config
    # - [x] Single electron in four forward regions for each config
    # - Pythia in four forward regions for each config
    # - Comparison between systems for each forward eta region
    # - Same as above, but switch forward -> backward

    #for selected_particle in ["pion"]:
    for selected_particle, latex_label in [("pion", "$\pi$"), ("electron", "$e^{\pm}$")]:
        for input_spec in input_specs:
            if input_spec.particle != selected_particle:
                continue
            output_dir_for_input_spec = output_dir / str(input_spec)
            output_dir_for_input_spec.mkdir(parents=True, exist_ok=True)
            text = "ECCE Simulation"
            text += "\n" + _input_spec_labels[str(input_spec)]
            text += "\n" + "Single " + latex_label + fr", ${input_spec.momentum_selection[0]:g} < p_{{\text{{T}}}} < {input_spec.momentum_selection[1]:g}$"

            hist_name_template = "histPResol_{particle}_FitMean_{eta_region_index}"
            plot_ecce_track_comparison.plot_tracking_comparison(
                input_specs=[input_spec],
                output_hists=output_hists,
                hist_name_template=hist_name_template,
                plot_name="p_mean",
                all_regions=eta_ranges, regions_label="forward", regions_index=forward_regions,
                text=text,
                y_range=(-0.1, 0.1), y_label=r"$\langle (p_{\text{T}}^{\text{rec}} - p_{\text{T}}^{\text{MC}}) / p_{\text{T}}^{\text{MC}} \rangle$",
                output_dir=output_dir_for_input_spec,
            )

            hist_name_template = "histPResol_{particle}_FitSigma_{eta_region_index}"
            plot_ecce_track_comparison.plot_tracking_comparison(
                input_specs=[input_spec],
                output_hists=output_hists,
                hist_name_template=hist_name_template,
                plot_name="p_width",
                all_regions=eta_ranges, regions_label="forward", regions_index=forward_regions,
                text=text,
                y_range=(0.0, 0.17), y_label=r"$\sigma((p_{\text{T}}^{\text{rec}} - p_{\text{T}}^{\text{MC}}) / p_{\text{T}}^{\text{MC}})$",
                output_dir=output_dir_for_input_spec,
            )

    from importlib import reload
    import IPython; IPython.embed()

    raise RuntimeError("Stahp")

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

        # TODO: Need to split pions and electrons. Because duh.
        hist_name_template = "histPResol_{particle}_FitMean_{eta_region_index}"
        plot_ecce_track_comparison.plot_tracking_comparison(
            input_specs=input_specs,
            output_hists=output_hists,
            hist_name_template=hist_name_template,
            plot_name="p_mean",
            all_regions=eta_ranges, regions_label=label, regions_index=[i],
            y_range=(-0.5, 0.5),
            output_dir=output_dir,
        )

        hist_name_template = "histPResol_{particle}_FitSigma_{eta_region_index}"
        plot_ecce_track_comparison.plot_tracking_comparison(
            input_specs=input_specs,
            output_hists=output_hists,
            hist_name_template=hist_name_template,
            plot_name="p_width",
            all_regions=eta_ranges, regions_label=label, regions_index=[i],
            y_range=(-0.05, 0.1),
            output_dir=output_dir,
        )

    import IPython; IPython.embed()


if __name__ == "__main__":
    run()
