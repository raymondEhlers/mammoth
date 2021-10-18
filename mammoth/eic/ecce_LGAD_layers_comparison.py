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


def _load_results(input_specs: Sequence[run_ecce_analysis.DatasetSpec], input_dir: Path, filename: str, filter: str ="hist") -> Dict[str, Dict[str, hist.Hist]]:
    output_hists = {}
    for spec in input_specs:
        logger.info(f"Loading hists from {input_dir / str(spec) / filename}")
        output_hists[str(spec)] = ecce_base.load_hists(input_dir / str(spec) / filename, filter=filter)

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
            particle="pion",
            momentum_selection=[0.0, 20],
            label="0layer",
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
            site="production",
            particle="electron",
            momentum_selection=[0.0, 20],
            label="0layer",
        ),
    ]
    pythia_input_specs = [
        # CADES geoOption5
        run_ecce_analysis.DatasetSpecPythia(
            site="cades",
            generator="pythia8",
            electron_beam_energy=10, proton_beam_energy=100,
            q2_selection=[1, 100],
            label="geoOption5",
        ),
        run_ecce_analysis.DatasetSpecPythia(
            site="cades",
            generator="pythia8",
            electron_beam_energy=10, proton_beam_energy=100,
            q2_selection=[100],
            label="geoOption5",
        ),
        # CADES geoOption6
        run_ecce_analysis.DatasetSpecPythia(
            site="cades",
            generator="pythia8",
            electron_beam_energy=10, proton_beam_energy=100,
            q2_selection=[1, 100],
            label="geoOption6",
        ),
        run_ecce_analysis.DatasetSpecPythia(
            site="cades",
            generator="pythia8",
            electron_beam_energy=10, proton_beam_energy=100,
            q2_selection=[100],
            label="geoOption6",
        ),
        # Pythia8
        run_ecce_analysis.DatasetSpecPythia(
            site="production",
            generator="pythia8",
            electron_beam_energy=10, proton_beam_energy=100,
            q2_selection=[1, 100],
            label="",
        ),
        run_ecce_analysis.DatasetSpecPythia(
            site="production",
            generator="pythia8",
            electron_beam_energy=10, proton_beam_energy=100,
            q2_selection=[100],
            label="",
        ),
        run_ecce_analysis.DatasetSpecPythia(
            site="production",
            generator="pythia8",
            electron_beam_energy=10, proton_beam_energy=100,
            q2_selection=[100],
            label="0layer",
        ),
    ]
    # These are the ones which contain the JRH_extra outputs
    pythia_high_q2_input_specs = [s for s in pythia_input_specs if s._q2_selection == [100]]

    # Setup
    _input_spec_labels = {
        f"production-pythia8-10x100-q2-1-to-100": "2 LGAD layers",
        f"production-pythia8-10x100-q2-100": "2 LGAD layers",
        f"production-pythia8-10x100-q2-1-to-100-0layer": "No LGADs",
        f"production-pythia8-10x100-q2-100-0layer": "No LGADs",
        f"cades-pythia8-10x100-q2-1-to-100-geoOption5": "1 LGAD layer, $30 \mu$m",
        f"cades-pythia8-10x100-q2-100-geoOption5": "1 LGAD layer, $30 \mu$m",
        f"cades-pythia8-10x100-q2-1-to-100-geoOption6": "1 LGAD layer, $55 \mu$m",
        f"cades-pythia8-10x100-q2-100-geoOption6": "1 LGAD layer, $55 \mu$m",
    }
    for particle in ["Pion", "Electron"]:
        _input_spec_labels.update({
            f"production-single{particle}-p-0-to-20": "2 LGAD layers",
            f"production-single{particle}-p-0-to-20-0layer": "No LGADs",
            f"cades-single{particle}-p-0.3-to-20-geoOption5": "1 LGAD layer, $30 \mu$m",
            f"cades-single{particle}-p-0.3-to-20-geoOption6": "1 LGAD layer, $55 \mu$m",
        })

    output_hists = _load_results(
        input_specs=input_specs,
        input_dir=input_dir,
        filename="output_TRKRS.root",
    )

    output_hists_pythia = _load_results(
        input_specs=pythia_input_specs,
        input_dir=input_dir,
        filename="output_TRKRS.root",
    )

    hist_name = "h_tracks_reso_p_{primary_track_source}_{name_reso_add}_{mc_particles_selection}_{eta_region}"
    # Example:   h_tracks_reso_p_0_All_All_6

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
    # - [x] Comparison between systems for each forward eta region
    # - [x] Same as above, but switch forward -> backward
    # - [x] Pythia in four forward regions for each config

    from importlib import reload
    import IPython; IPython.embed()

    # Single particle productions
    for regions_label, region_indices in [("forward", forward_regions), ("barrel", barrel_regions), ("backward", backward_regions)]:
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
                    input_spec_labels=_input_spec_labels,
                    output_hists=output_hists,
                    hist_name_template=hist_name_template,
                    plot_name=f"p_mean_{selected_particle}",
                    all_regions=eta_ranges, regions_label=regions_label, regions_index=region_indices,
                    text=text,
                    selected_particle=input_spec.particle,
                    y_range=(-0.1, 0.1), y_label=r"$\langle (p_{\text{T}}^{\text{rec}} - p_{\text{T}}^{\text{MC}}) / p_{\text{T}}^{\text{MC}} \rangle$",
                    output_dir=output_dir_for_input_spec,
                )

                hist_name_template = "histPResol_{particle}_FitSigma_{eta_region_index}"
                plot_ecce_track_comparison.plot_tracking_comparison(
                    input_specs=[input_spec],
                    input_spec_labels=_input_spec_labels,
                    output_hists=output_hists,
                    hist_name_template=hist_name_template,
                    plot_name=f"p_width_{selected_particle}",
                    all_regions=eta_ranges, regions_label=regions_label, regions_index=region_indices,
                    text=text,
                    selected_particle=input_spec.particle,
                    y_range=(0.0, 0.17), y_label=r"$\sigma((p_{\text{T}}^{\text{rec}} - p_{\text{T}}^{\text{MC}}) / p_{\text{T}}^{\text{MC}})$",
                    output_dir=output_dir_for_input_spec,
                )

    for regions_label, region_indices in [("forward", forward_regions), ("barrel", barrel_regions), ("backward", backward_regions)]:
        # Now, the comparison for each eta region
        for selected_particle, latex_label in [("pion", "$\pi$"), ("electron", "$e^{\pm}$")]:
            #plot_input_specs = [input_spec for input_spec in input_specs if input_spec.particle == selected_particle]
            plot_input_specs = []
            for input_spec in input_specs:
                if input_spec.particle == selected_particle:
                    plot_input_specs.append(input_spec)
            logger.info(f"plot_input_specs: {plot_input_specs}")

            # Skip if we didn't load the data
            if not len(plot_input_specs):
                continue

            for i in region_indices:
                # Labels
                text = "ECCE Simulation"
                text += "\n" + "Single " + latex_label
                text += "\n" + plot_ecce_track_comparison.get_eta_label(eta_ranges[i])

                hist_name_template = "histPResol_{particle}_FitMean_{eta_region_index}"
                plot_ecce_track_comparison.plot_tracking_comparison(
                    input_specs=plot_input_specs,
                    input_spec_labels=_input_spec_labels,
                    output_hists=output_hists,
                    hist_name_template=hist_name_template,
                    plot_name=f"p_mean_comparison_{selected_particle}",
                    all_regions=eta_ranges, regions_label=regions_label, regions_index=[i],
                    text=text,
                    selected_particle=selected_particle,
                    y_range=(-0.1, 0.1), y_label=r"$\langle (p_{\text{T}}^{\text{rec}} - p_{\text{T}}^{\text{MC}}) / p_{\text{T}}^{\text{MC}} \rangle$",
                    output_dir=output_dir,
                )

                hist_name_template = "histPResol_{particle}_FitSigma_{eta_region_index}"
                plot_ecce_track_comparison.plot_tracking_comparison(
                    input_specs=plot_input_specs,
                    input_spec_labels=_input_spec_labels,
                    output_hists=output_hists,
                    hist_name_template=hist_name_template,
                    plot_name=f"p_width_comparison_{selected_particle}",
                    all_regions=eta_ranges, regions_label=regions_label, regions_index=[i],
                    text=text,
                    selected_particle=selected_particle,
                    y_range=(0.0, 0.17), y_label=r"$\sigma((p_{\text{T}}^{\text{rec}} - p_{\text{T}}^{\text{MC}}) / p_{\text{T}}^{\text{MC}})$",
                    output_dir=output_dir,
                )


    # Pythia
    for regions_label, region_indices in [("forward", forward_regions), ("barrel", barrel_regions), ("backward", backward_regions)]:
        for selected_particle, latex_label in [("all", ""), ("pion", "$\pi$"), ("electron", "$e^{\pm}$")]:
            for input_spec in pythia_input_specs:
                output_dir_for_input_spec = output_dir / str(input_spec)
                output_dir_for_input_spec.mkdir(parents=True, exist_ok=True)
                text = "ECCE Simulation"
                text += "\n" + _input_spec_labels[str(input_spec)]
                text += "\n" + "PYTHIA 8 10x100, " +  (latex_label + "," if latex_label else "") + f"${input_spec.q2_display}$"

                hist_name_template = "histPResol_{particle}_FitMean_{eta_region_index}"
                plot_ecce_track_comparison.plot_tracking_comparison(
                    input_specs=[input_spec],
                    input_spec_labels=_input_spec_labels,
                    output_hists=output_hists_pythia,
                    hist_name_template=hist_name_template,
                    plot_name=f"p_mean_{selected_particle}",
                    all_regions=eta_ranges, regions_label=regions_label, regions_index=region_indices,
                    text=text,
                    selected_particle=selected_particle,
                    y_range=(-0.1, 0.1), y_label=r"$\langle (p_{\text{T}}^{\text{rec}} - p_{\text{T}}^{\text{MC}}) / p_{\text{T}}^{\text{MC}} \rangle$",
                    output_dir=output_dir_for_input_spec,
                )

                hist_name_template = "histPResol_{particle}_FitSigma_{eta_region_index}"
                plot_ecce_track_comparison.plot_tracking_comparison(
                    input_specs=[input_spec],
                    input_spec_labels=_input_spec_labels,
                    output_hists=output_hists_pythia,
                    hist_name_template=hist_name_template,
                    plot_name=f"p_width_{selected_particle}",
                    all_regions=eta_ranges, regions_label=regions_label, regions_index=region_indices,
                    text=text,
                    selected_particle=selected_particle,
                    y_range=(0.0, 0.17), y_label=r"$\sigma((p_{\text{T}}^{\text{rec}} - p_{\text{T}}^{\text{MC}}) / p_{\text{T}}^{\text{MC}})$",
                    output_dir=output_dir_for_input_spec,
                )

        # Now, the comparison for each eta region
        for selected_particle, latex_label in [("all", ""), ("pion", "$\pi$"), ("electron", "$e^{\pm}$")]:
            #plot_input_specs = [input_spec for input_spec in input_specs if input_spec.particle == selected_particle]
            for q2_selection in ["q2-1-to-100", "q2-100"]:
                plot_input_specs = []
                for input_spec in pythia_input_specs:
                    if input_spec.q2 == q2_selection:
                        plot_input_specs.append(input_spec)

                # Skip if we didn't load the data
                if not len(plot_input_specs):
                    continue

                label_input_spec = plot_input_specs[0]

                for i in region_indices:
                    # Labels
                    text = "ECCE Simulation"
                    text += "\n" + "PYTHIA 8 10x100, " +  (latex_label + "," if latex_label else "") + f"${input_spec.q2_display}$"
                    text += "\n" + plot_ecce_track_comparison.get_eta_label(eta_ranges[i])

                    hist_name_template = "histPResol_{particle}_FitMean_{eta_region_index}"
                    plot_ecce_track_comparison.plot_tracking_comparison(
                        input_specs=plot_input_specs,
                        input_spec_labels=_input_spec_labels,
                        output_hists=output_hists_pythia,
                        hist_name_template=hist_name_template,
                        plot_name=f"p_mean_comparison_{selected_particle}_pythia_{label_input_spec.q2}",
                        all_regions=eta_ranges, regions_label=regions_label, regions_index=[i],
                        text=text,
                        selected_particle=selected_particle,
                        y_range=(-0.1, 0.1), y_label=r"$\langle (p_{\text{T}}^{\text{rec}} - p_{\text{T}}^{\text{MC}}) / p_{\text{T}}^{\text{MC}} \rangle$",
                        x_range=(0.1, 100),
                        output_dir=output_dir,
                    )

                    hist_name_template = "histPResol_{particle}_FitSigma_{eta_region_index}"
                    plot_ecce_track_comparison.plot_tracking_comparison(
                        input_specs=plot_input_specs,
                        input_spec_labels=_input_spec_labels,
                        output_hists=output_hists_pythia,
                        hist_name_template=hist_name_template,
                        plot_name=f"p_width_comparison_{selected_particle}_pythia_{label_input_spec.q2}",
                        all_regions=eta_ranges, regions_label=regions_label, regions_index=[i],
                        text=text,
                        selected_particle=selected_particle,
                        y_range=(0.0, 0.17), y_label=r"$\sigma((p_{\text{T}}^{\text{rec}} - p_{\text{T}}^{\text{MC}}) / p_{\text{T}}^{\text{MC}})$",
                        x_range=(0.1, 100),
                        output_dir=output_dir,
                    )

    # Single particle jets aren't really meaningful, so we just load pythia
    output_hists_jets_pythia = _load_results(
        input_specs=pythia_high_q2_input_specs,
        input_dir=input_dir,
        filename="output_JRH_extra.root",
        filter="",
    )

    for regions_label, region_indices in [("forward", forward_regions), ("barrel", barrel_regions), ("backward", backward_regions)]:
        for jet_type in ["track"]:
            for variable in ["p", "E", "pT"]:
                for input_spec in pythia_high_q2_input_specs:
                    output_dir_for_input_spec = output_dir / "jets" / str(input_spec)
                    output_dir_for_input_spec.mkdir(parents=True, exist_ok=True)
                    text = "ECCE Simulation"
                    text += "\n" + _input_spec_labels[str(input_spec)]
                    text += "\n" + "PYTHIA 8 10x100, " + f"${input_spec.q2_display}$"
                    text += "\n" + r"anti-$k_{\text{T}}$ $R$=0.5 jets"

                    x_label = r"$p^{\text{jet}}\:(\text{GeV}/c)$"
                    y_label_var = r"p"
                    if variable == "pT":
                        x_label = r"$p^{\text{T,jet}}\:(\text{GeV}/c)$"
                        y_label_var = r"p_{\text{T}}"
                    elif variable == "E":
                        x_label = r"$E^{\text{jet}}\:(\text{GeV})$"
                        y_label_var = r"E"

                    hist_name_template = f"h_JES_{jet_type}_{variable}_{{eta_region_index}}"
                    plot_ecce_track_comparison.plot_tracking_comparison(
                        input_specs=[input_spec],
                        input_spec_labels=_input_spec_labels,
                        output_hists=output_hists_jets_pythia,
                        hist_name_template=hist_name_template,
                        plot_name=f"JES_{jet_type}_{variable}",
                        all_regions=eta_ranges, regions_label=regions_label, regions_index=region_indices,
                        text=text,
                        selected_particle="",
                        y_range=(-0.2, 0.2), y_label=r"$\langle (" + y_label_var + r"^{\text{rec}} - " + y_label_var + r"^{\text{MC}}) / " + y_label_var + r"^{\text{MC}} \rangle$",
                        plot_jets=True, x_label=x_label, x_range=(0.1, 30),
                        output_dir=output_dir_for_input_spec,
                    )

                    hist_name_template = f"h_JER_{jet_type}_{variable}_{{eta_region_index}}"
                    plot_ecce_track_comparison.plot_tracking_comparison(
                        input_specs=[input_spec],
                        input_spec_labels=_input_spec_labels,
                        output_hists=output_hists_jets_pythia,
                        hist_name_template=hist_name_template,
                        plot_name=f"JER_{jet_type}_{variable}",
                        all_regions=eta_ranges, regions_label=regions_label, regions_index=region_indices,
                        text=text,
                        selected_particle="",
                        y_range=(0.0, 0.45), y_label=r"$\sigma((" + y_label_var + r"^{\text{rec}} - " + y_label_var + r"^{\text{MC}}) / " + y_label_var + r"^{\text{MC}})$",
                        plot_jets=True, x_label=x_label, x_range=(0.1, 30),
                        output_dir=output_dir_for_input_spec,
                    )

    for regions_label, region_indices in [("forward", forward_regions), ("barrel", barrel_regions), ("backward", backward_regions)]:
        for jet_type in ["track"]:
            for variable in ["p", "E", "pT"]:
                #plot_input_specs = [input_spec for input_spec in input_specs if input_spec.particle == selected_particle]
                for q2_selection in ["q2-100"]:
                    plot_input_specs = []
                    for input_spec in pythia_high_q2_input_specs:
                        if input_spec.q2 == q2_selection:
                            plot_input_specs.append(input_spec)

                    # Skip if we didn't load the data
                    if not len(plot_input_specs):
                        continue
                    _jet_output_dir = output_dir / "jets"
                    _jet_output_dir.mkdir(parents=True, exist_ok=True)
                    label_input_spec = plot_input_specs[0]

                    for i in region_indices:
                        # Labels
                        text = "ECCE Simulation"
                        text += "\n" + "PYTHIA 8 10x100, " + f"${input_spec.q2_display}$"
                        text += "\n" + r"anti-$k_{\text{T}}$ $R$=0.5 jets"
                        text += "\n" + plot_ecce_track_comparison.get_eta_label(eta_ranges[i])

                        x_label = r"$p^{\text{jet}}\:(\text{GeV}/c)$"
                        y_label_var = r"p"
                        if variable == "pT":
                            x_label = r"$p^{\text{T,jet}}\:(\text{GeV}/c)$"
                            y_label_var = r"p_{\text{T}}"
                        elif variable == "E":
                            x_label = r"$E^{\text{jet}}\:(\text{GeV})$"
                            y_label_var = r"E"

                        hist_name_template = f"h_JES_{jet_type}_{variable}_{{eta_region_index}}"
                        plot_ecce_track_comparison.plot_tracking_comparison(
                            input_specs=plot_input_specs,
                            input_spec_labels=_input_spec_labels,
                            output_hists=output_hists_jets_pythia,
                            hist_name_template=hist_name_template,
                            plot_name=f"JES_{jet_type}_{variable}",
                            all_regions=eta_ranges, regions_label=regions_label, regions_index=[i],
                            text=text,
                            selected_particle="",
                            y_range=(-0.2, 0.2), y_label=r"$\langle (" + y_label_var + r"^{\text{rec}} - " + y_label_var + r"^{\text{MC}}) / " + y_label_var + r"^{\text{MC}} \rangle$",
                            plot_jets=True, x_label=x_label, x_range=(0.1, 50),
                            output_dir=_jet_output_dir,
                        )

                        hist_name_template = f"h_JER_{jet_type}_{variable}_{{eta_region_index}}"
                        plot_ecce_track_comparison.plot_tracking_comparison(
                            input_specs=plot_input_specs,
                            input_spec_labels=_input_spec_labels,
                            output_hists=output_hists_jets_pythia,
                            hist_name_template=hist_name_template,
                            plot_name=f"JER_{jet_type}_{variable}",
                            all_regions=eta_ranges, regions_label=regions_label, regions_index=[i],
                            text=text,
                            selected_particle="",
                            y_range=(0.0, 0.45), y_label=r"$\sigma((" + y_label_var + r"^{\text{rec}} - " + y_label_var + r"^{\text{MC}}) / " + y_label_var + r"^{\text{MC}})$",
                            plot_jets=True, x_label=x_label, x_range=(0.1, 50),
                            output_dir=_jet_output_dir,
                        )


                        #hist_name_template = "histPResol_{particle}_FitMean_{eta_region_index}"
                        #plot_ecce_track_comparison.plot_tracking_comparison(
                        #    input_specs=plot_input_specs,
                        #    input_spec_labels=_input_spec_labels,
                        #    output_hists=output_hists_pythia,
                        #    hist_name_template=hist_name_template,
                        #    plot_name=f"p_mean_comparison_{selected_particle}_pythia_{label_input_spec.q2}",
                        #    all_regions=eta_ranges, regions_label=regions_label, regions_index=[i],
                        #    text=text,
                        #    selected_particle=selected_particle,
                        #    #x_range=(0.1, 100),
                        #    y_range=(-0.1, 0.1), y_label=r"$\langle (p_{\text{T}}^{\text{rec}} - p_{\text{T}}^{\text{MC}}) / p_{\text{T}}^{\text{MC}} \rangle$",
                        #    x_label=r"$p^{\text{jet}}\:(\text{GeV}/c)$",
                        #    output_dir=output_dir,
                        #)

                        #hist_name_template = "histPResol_{particle}_FitSigma_{eta_region_index}"
                        #plot_ecce_track_comparison.plot_tracking_comparison(
                        #    input_specs=plot_input_specs,
                        #    input_spec_labels=_input_spec_labels,
                        #    output_hists=output_hists_pythia,
                        #    hist_name_template=hist_name_template,
                        #    plot_name=f"p_width_comparison_{selected_particle}_pythia_{label_input_spec.q2}",
                        #    all_regions=eta_ranges, regions_label=regions_label, regions_index=[i],
                        #    text=text,
                        #    selected_particle=selected_particle,
                        #    #x_range=(0.1, 100),
                        #    y_range=(0.0, 0.17), y_label=r"$\sigma((p_{\text{T}}^{\text{rec}} - p_{\text{T}}^{\text{MC}}) / p_{\text{T}}^{\text{MC}})$",
                        #    x_label=r"$p^{\text{jet}}\:(\text{GeV}/c)$",
                        #    output_dir=_jet_output_dir,
                        #)

    import IPython; IPython.embed()

    #for label, regions in [("backward", backward_regions), ("barrel", barrel_regions), ("forward", forward_regions)]:
    #for i in range(0, len(eta_ranges)):
    #    if i in backward_regions:
    #        label = "backward"
    #    elif i in barrel_regions:
    #        label = "barrel"
    #    elif i in forward_regions:
    #        label = "forward"
    #    else:
    #        label = "all"

    #    # TODO: Need to split pions and electrons. Because duh.
    #    hist_name_template = "histPResol_{particle}_FitMean_{eta_region_index}"
    #    plot_ecce_track_comparison.plot_tracking_comparison(
    #        input_specs=input_specs,
    #        output_hists=output_hists,
    #        hist_name_template=hist_name_template,
    #        plot_name="p_mean",
    #        all_regions=eta_ranges, regions_label=label, regions_index=[i],
    #        y_range=(-0.5, 0.5),
    #        output_dir=output_dir,
    #    )

    #    hist_name_template = "histPResol_{particle}_FitSigma_{eta_region_index}"
    #    plot_ecce_track_comparison.plot_tracking_comparison(
    #        input_specs=input_specs,
    #        output_hists=output_hists,
    #        hist_name_template=hist_name_template,
    #        plot_name="p_width",
    #        all_regions=eta_ranges, regions_label=label, regions_index=[i],
    #        y_range=(-0.05, 0.1),
    #        output_dir=output_dir,
    #    )


if __name__ == "__main__":
    run()
