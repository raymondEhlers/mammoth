#!/usr/bin/env python3

""" Jet RAA for comparison with the ALICE jet background ML analysis

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Sequence

import awkward as ak
import numba as nb

import mammoth.base
from mammoth.framework import jet_finding, sources, transform


logger = logging.getLogger(__name__)

@nb.njit  # type: ignore
def subtract_holes_from_jet_pt(jets: ak.Array, particles_holes: ak.Array, jet_R: float, builder: ak.ArrayBuilder) -> ak.ArrayBuilder:
    """Subtract holes from the jet pt
    
    TODO: Centralize this for jetscape, given that each analysis almost certainly needs to do this.
    """
    for jets_in_event, holes_in_event in zip(jets, particles_holes):
        builder.begin_list()
        for jet in jets_in_event:
            jet_pt = jet.pt
            for hole in holes_in_event:
                if jet.deltaR(hole) < jet_R:
                    jet_pt -= hole.pt
            builder.append(jet_pt)
        builder.end_list()

    return builder


def load_data(filename: Path) -> ak.Array:
    logger.info("Loading data")
    source = sources.ParquetSource(
        filename=filename,
    )
    arrays = source.data()
    logger.info("Transforming data")
    return transform.data(arrays=arrays, rename_prefix={"data": "particles"})


def analysis(arrays: ak.Array, jet_R_values: Sequence[float], particle_column_name: str = "data", min_jet_pt: float = 30) -> ak.Array:
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
    charged_particles_mask = mammoth.base.build_PID_selection_mask(
        arrays[particle_column_name], absolute_pids=_charged_hadron_PIDs
    )

    # We want to analyze both charged and full jets
    particles_signal = arrays[particle_column_name][signal_particles_mask]
    particles_signal_charged = arrays[particle_column_name][signal_particles_mask & charged_particles_mask]
    particles_holes = arrays[particle_column_name][holes_mask]

    # Jet finding
    logger.info("Find jets")
    area_settings = jet_finding.AREA_PP
    jets = ak.zip(
        {
            f"{label}_jetR{round(jet_R * 100):03}_{particle_column_name}": jet_finding.find_jets(
                particles=particles,
                algorithm="anti-kt",
                jet_R=jet_R,
                area_settings=area_settings,
                min_jet_pt=min_jet_pt,
            )
            for jet_R in jet_R_values
            for particles, label in zip([particles_signal, particles_signal_charged], ["full", "charged"])
        },
        depth_limit=1,
    )

    # Calculated the subtracted pt due to the holes.
    for jet_collection_name, jet_collection in zip(ak.fields(jets), ak.unzip(jets)):
        jet_R_location = jet_collection_name.find("jetR") + 4
        jet_R = float(jet_collection_name[jet_R_location:jet_R_location+3]) / 100
        jets[jet_collection_name, "pt_subtracted"] = subtract_holes_from_jet_pt(
            jets=jet_collection,
            particles_holes=particles_holes,
            jet_R=jet_R,
            builder=ak.ArrayBuilder(),
        ).snapshot()

    # Store the cross section with each jet. This way, we can flatten from events -> jets
    for jet_collection_name, jet_collection in zip(ak.fields(jets), ak.unzip(jets)):
        # Before any jets cuts, add in cross section
        jets[jet_collection_name, "cross_section"] = arrays["cross_section"]
        #jets["cross_section"] = 

    # Apply jet level cuts.
    # None for now

    return jets

def ignore() -> ak.Array:

    # Check for any jets. If there are none, we probably want to bail out.
    # We need some variable to avoid flattening into a record, so select px arbitrarily.
    if len(ak.flatten(jets[ak.fields(jets)[0]].px, axis=None)) == 0:
        raise ValueError(f"No jets left for {ak.fields(jets)[0]}. Are your settings correct?")

    import IPython; IPython.embed()

    # Next step for using existing skimming:
    # Flatten from events -> jets
    # NOTE: Apparently it's takes issues with flattening the jets directly, so we have to do it
    #       separately for the different collections and then zip them together. This should keep
    #       matching together as appropriate.
    jets = ak.zip(
        {k: ak.flatten(v, axis=1) for k, v in zip(ak.fields(jets), ak.unzip(jets))},
        depth_limit=1,
    )

    logger.warning(f"n jets: {len(jets)}")

    # Now, the final transformation into a form that can be used to skim into a flat tree.
    return jets


if __name__ == "__main__":
    # Basic setup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s")
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    # For sanity when using IPython
    logging.getLogger("parso").setLevel(logging.INFO)
    # Quiet down BinndData copy warnings
    logging.getLogger("pachyderm.binned_data").setLevel(logging.INFO)
    # Quiet down numba
    logging.getLogger("numba").setLevel(logging.INFO)

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