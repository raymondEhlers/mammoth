#!/usr/bin/env python3

""" Jet RAA for comparison with the ALICE jet background ML analysis

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import attr
import awkward as ak
import hist
import numpy as np

from mammoth import analysis_base, helpers
from mammoth.framework import jet_finding, sources, transform
from mammoth.jetscape import utils


logger = logging.getLogger(__name__)


@attr.s(frozen=True)
class JetLabel:
    jet_R: float = attr.ib()
    label: str = attr.ib()

    def __str__(self) -> str:
        return f"{self.label}_jetR{round(self.jet_R * 100):03}"


def load_data(filename: Path) -> ak.Array:
    logger.info("Loading data")
    source = sources.ParquetSource(
        filename=filename,
    )
    arrays = source.data()
    logger.info("Transforming data")
    return transform.data(arrays=arrays, rename_prefix={"data": "particles"})


def find_jets_for_analysis(arrays: ak.Array, jet_R_values: Sequence[float], particle_column_name: str = "data", min_jet_pt: float = 30) -> Dict[JetLabel, ak.Array]:
    logger.info("Start analyzing")
    # Event selection
    # None for jetscape
    # Track cuts
    logger.info("Track level cuts")
    # Data track cuts:
    # - min: 150 MeV
    data_track_pt_mask = arrays[particle_column_name].pt >= 0.150
    arrays[particle_column_name] = arrays[particle_column_name][data_track_pt_mask]

    # Track selections:
    # - Signal particles vs holes
    signal_particles_mask = arrays[particle_column_name, "status"] == 0
    holes_mask = ~signal_particles_mask
    # - Charged particles only for charged-particle jets
    _charged_hadron_PIDs = [11, 13, 211, 321, 2212, 3222, 3112, 3312, 3334]
    charged_particles_mask = analysis_base.build_PID_selection_mask(
        arrays[particle_column_name], absolute_pids=_charged_hadron_PIDs
    )

    # We want to analyze both charged and full jets
    particles_signal = arrays[particle_column_name][signal_particles_mask]
    particles_signal_charged = arrays[particle_column_name][signal_particles_mask & charged_particles_mask]
    particles_holes = arrays[particle_column_name][holes_mask]

    # Finally, require that we have particles for each event
    # NOTE: We have to do it in a separate mask because the above is masked as the particle level,
    #       but here we need to mask at the event level. (If you try to mask at the particle, you'll
    #       end up with empty events)
    # NOTE: We store the mask because we need to apply it to the holes when we perform the subtraction below
    event_has_particles_signal = ak.num(particles_signal, axis=1) > 0
    particles_signal = particles_signal[event_has_particles_signal]
    event_has_particles_signal_charged = ak.num(particles_signal_charged, axis=1) > 0
    particles_signal_charged = particles_signal_charged[event_has_particles_signal_charged]

    # Jet finding
    logger.info("Find jets")
    # Always use the pp jet area because we aren't going to do subtraction via fastjet
    area_settings = jet_finding.AREA_PP
    jets = {}
    # NOTE: The dict comprehension that was here previously was cute, but it made it harder to
    #       debug issues, so we use a standard set of for loops here instead
    for jet_R in jet_R_values:
        #for particles, label in zip([particles_signal, particles_signal_charged], ["full", "charged"]):
        for particles, label in zip([particles_signal_charged, particles_signal], ["charged", "full"]):
            tag = JetLabel(jet_R=jet_R, label=label)
            logger.info(f"label: {tag}")
            jets[tag] = jet_finding.find_jets(
                particles=particles,
                algorithm="anti-kt",
                jet_R=jet_R,
                area_settings=area_settings,
                min_jet_pt=min_jet_pt,
            )

    # Calculated the subtracted pt due to the holes.
    for jet_label, jet_collection in jets.items():
        jets[jet_label]["pt_subtracted"] = utils.subtract_holes_from_jet_pt(
            jets=jet_collection,
            # NOTE: There can be different number of events for full vs charged jets, so we need
            #       to apply the appropriate event mask to the holes
            particles_holes=particles_holes[
                event_has_particles_signal_charged if jet_label.label == "charged" else event_has_particles_signal
            ],
            jet_R=jet_label.jet_R,
            builder=ak.ArrayBuilder(),
        ).snapshot()

    # Store the cross section with each jet. This way, we can flatten from events -> jets
    for jet_label, jet_collection in jets.items():
        # Before any jets cuts, add in cross section
        # NOTE: There can be different number of events for full vs charged jets, so we need
        #       to apply the appropriate event mask to the holes
        jets[jet_label]["cross_section"] = arrays["cross_section"][
                event_has_particles_signal_charged if jet_label.label == "charged" else event_has_particles_signal
            ]

    # Apply jet level cuts.
    # None for now

    return jets


def analyze_jets(arrays: ak.Array, jets: Mapping[JetLabel, ak.Array]) -> Dict[str, hist.Hist]:
    hists = {}
    hists["n_events"] = hist.Hist(hist.axis.Regular(1, -0.5, 0.5))
    hists["n_events_weighted"] = hist.Hist(hist.axis.Regular(1, -0.5, 0.5))
    for jet_label in jets:
        hists[f"{jet_label}_jet_pt"] = hist.Hist(hist.axis.Regular(200, 0, 200, label="jet_pt"), storage=hist.storage.Weight())
        # Try a coarser binning to reduce outliers
        hists[f"{jet_label}_jet_pt_coarse_binned"] = hist.Hist(hist.axis.Regular(40, 0, 200, label="jet_pt"), storage=hist.storage.Weight())
        # This is assuredly overkill, but it hopefully means that I won't need to mess with it anymore
        hists[f"{jet_label}_n_events"] = hist.Hist(hist.axis.Regular(1, -0.5, 0.5))
        hists[f"{jet_label}_n_events_weighted"] = hist.Hist(hist.axis.Regular(1, -0.5, 0.5))
        hists[f"{jet_label}_n_jets"] = hist.Hist(hist.axis.Regular(1, -0.5, 0.5))
        hists[f"{jet_label}_n_jets_weighted"] = hist.Hist(hist.axis.Regular(1, -0.5, 0.5))

    hists["n_events"].fill(0, weight=len(arrays))
    # Just need it to get the first cross section value. It should be the same for all cases
    first_jet_label = next(iter(jets))
    # NOTE: Apparently this can be empty, so we have to retrieve the value carefully
    first_cross_section = ak.flatten(jets[first_jet_label].cross_section)
    if len(first_cross_section) == 0:
        _cross_section_weight_factor = 1
    else:
        _cross_section_weight_factor = first_cross_section[0]
    hists["n_events_weighted"].fill(0, weight=len(arrays) * _cross_section_weight_factor)
    for jet_label, jet_collection in jets.items():
        hists[f"{jet_label}_jet_pt"].fill(
            ak.flatten(jet_collection.pt_subtracted), weight=ak.flatten(jet_collection.cross_section)
        )
        hists[f"{jet_label}_jet_pt_coarse_binned"].fill(
            ak.flatten(jet_collection.pt_subtracted), weight=ak.flatten(jet_collection.cross_section)
        )
        hists[f"{jet_label}_n_events"].fill(0, weight=len(jet_collection))
        hists[f"{jet_label}_n_events_weighted"].fill(0, weight=len(jet_collection) * _cross_section_weight_factor)
        hists[f"{jet_label}_n_jets"].fill(0, weight=len(ak.flatten(jet_collection.pt_subtracted)))
        #hists[f"{jet_label}_n_jets_weighted"].fill(0, weight=len(ak.flatten(jet_collection.pt_subtracted)) * ak.flatten(jet_collection.cross_section)[0])
        hists[f"{jet_label}_n_jets_weighted"].fill(0, weight=len(ak.flatten(jet_collection.pt_subtracted)) * _cross_section_weight_factor)

    return hists


def run(arrays: ak.Array, min_jet_pt: float = 5, jet_R_values: Optional[Sequence[float]] = None) -> Dict[str, hist.Hist]:
    # Validation
    if jet_R_values is None:
        jet_R_values = [0.2, 0.4, 0.6]

    # Find jets
    jets = find_jets_for_analysis(
        arrays=arrays,
        jet_R_values=jet_R_values,
        min_jet_pt=min_jet_pt,
    )

    # Analyze the jets
    hists = analyze_jets(arrays=arrays, jets=jets)

    return hists


if __name__ == "__main__":
    # Basic setup
    helpers.setup_logging(level=logging.INFO)

    hists = run(
        arrays=load_data(
            #Path(f"/alf/data/rehlers/jetscape/osiris/AAPaperData/5020_PP_Colorless/skim/test/JetscapeHadronListBin7_9_00.parquet")
            #Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/5020_PP_Colorless/skim/JetscapeHadronListBin270_280_01.parquet")
            Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_40-50_0.30_2.0_1/skim/JetscapeHadronListBin7_9_01.parquet"),
        ),
        # Low for testing
        min_jet_pt=3,
        # Jet one R for faster testing
        jet_R_values=[0.4],
    )

    import IPython
    
    IPython.embed()