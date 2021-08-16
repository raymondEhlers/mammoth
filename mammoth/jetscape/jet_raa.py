#!/usr/bin/env python3

""" Jet RAA for comparison with the ALICE jet background ML analysis

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Dict, Sequence

import awkward as ak
import attr

from mammoth import analysis_base, helpers
from mammoth.framework import jet_finding, sources, transform
from mammoth.jetscape import utils


logger = logging.getLogger(__name__)


@attr.s(frozen=True)
class JetLabel:
    jet_R: float = attr.ib()
    label: str = attr.ib()


def load_data(filename: Path) -> ak.Array:
    logger.info("Loading data")
    source = sources.ParquetSource(
        filename=filename,
    )
    arrays = source.data()
    logger.info("Transforming data")
    return transform.data(arrays=arrays, rename_prefix={"data": "particles"})


def analysis(arrays: ak.Array, jet_R_values: Sequence[float], particle_column_name: str = "data", min_jet_pt: float = 30) -> Dict[JetLabel, ak.Array]:
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

    # Jet finding
    logger.info("Find jets")
    area_settings = jet_finding.AREA_PP
    jets = {
        JetLabel(jet_R=jet_R, label=label): jet_finding.find_jets(
            particles=particles,
            algorithm="anti-kt",
            jet_R=jet_R,
            area_settings=area_settings,
            min_jet_pt=min_jet_pt,
        )
        for jet_R in jet_R_values
        for particles, label in zip([particles_signal, particles_signal_charged], ["full", "charged"])
    }

    # Calculated the subtracted pt due to the holes.
    for jet_label, jet_collection in jets.items():
        jets[jet_label]["pt_subtracted"] = utils.subtract_holes_from_jet_pt(
            jets=jet_collection,
            particles_holes=particles_holes,
            jet_R=jet_label.jet_R,
            builder=ak.ArrayBuilder(),
        ).snapshot()

    # Store the cross section with each jet. This way, we can flatten from events -> jets
    for jet_label, jet_collection in jets.items():
        # Before any jets cuts, add in cross section
        jets[jet_label]["cross_section"] = arrays["cross_section"]

    # Apply jet level cuts.
    # None for now

    return jets


if __name__ == "__main__":
    # Basic setup
    helpers.setup_logging(level=logging.INFO)

    # Find jets
    jets = analysis(
        arrays=load_data(
            Path(f"/alf/data/rehlers/jetscape/osiris/AAPaperData/5020_PP_Colorless/skim/test/JetscapeHadronListBin7_9_00.parquet")
        ),
        jet_R_values=[0.2, 0.4, 0.6],
        min_jet_pt=3,
    )

    import IPython
    
    IPython.embed()