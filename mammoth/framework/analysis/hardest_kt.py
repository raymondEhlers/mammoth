"""Run analysis using PYTHIA + thermal model.

.. codeuathor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import awkward as ak
import numpy as np
import numpy.typing as npt
import vector

from mammoth.framework import jet_finding, sources, transform


logger = logging.getLogger(__name__)
vector.register_awkward()


def analysis_MC(pythia_filename: Path, jet_R: float, min_pythia_jet_pt: float) -> ak.Array:
    logger.info("Start")
    pythia_source = sources.ParquetSource(
        filename=pythia_filename,
    )
    arrays = pythia_source.data()
    logger.info("Transform")
    arrays = transform.mc(arrays=arrays)

    # Event selection
    arrays = arrays[
        (arrays["is_ev_rej"] == 0)
        & (np.abs(arrays["z_vtx_reco"]) < 10)
    ]

    # Track cuts
    logger.info("Track level cuts")
    # Particle level track cuts:
    # - min: 150 MeV (from the EMCal container)
    part_track_pt_mask = (arrays["part_level"].pt >= 0.150)
    arrays["part_level"] = arrays["part_level"][part_track_pt_mask]
    # Detector level track cuts:
    # - min: 150 MeV
    # NOTE: Since the HF Tree uses track containers, the min is usually applied by default
    det_track_pt_mask = (arrays["det_level"].pt >= 0.150)
    arrays["det_level"] = arrays["det_level"][det_track_pt_mask]

    # Jet finding
    logger.info("Find jets")
    jets = ak.zip(
        {
            "part_level": jet_finding.find_jets(
                particles=arrays["part_level"],
                algorithm="anti-kt",
                jet_R=jet_R,
                area_settings=jet_finding.AREA_PP,
                # NOTE: We only want the minimum pt to apply to the detector level.
                #       Otherwise, we'll bias our particle level jets.
                min_jet_pt=1,
            ),
            "det_level": jet_finding.find_jets(
                particles=arrays["det_level"],
                algorithm="anti-kt",
                jet_R=jet_R,
                area_settings=jet_finding.AREA_PP,
                min_jet_pt=min_pythia_jet_pt,
            ),
        },
        depth_limit=1,
    )

    # Apply jet level cuts.
    # **************
    # Remove detector level jets with constituents with pt > 100 GeV
    # Those tracks are almost certainly fake at detector level.
    # NOTE: We skip at part level because it doesn't share these detector effects.
    # NOTE: We need to do it after jet finding to avoid a z bias.
    # **************
    det_level_max_track_pt_cut = ~ak.any(jets["det_level"].constituents.pt > 100, axis=-1)
    # **************
    # Apply area cut
    # Requires at least 60% of possible area.
    # **************
    min_area = jet_finding.area_percentage(60, jet_R),
    part_level_min_area_mask = jets["part_level", "area"] > min_area
    det_level_min_area_mask = jets["det_level", "area"] > min_area
    # *************
    # Require more than one constituent at detector level if we're not in PbPb
    # Matches a cut in AliAnalysisTaskJetDynamicalGrooming
    # *************
    det_level_min_n_constituents_mask = ak.num(jets["det_level", "constituents"], axis = 2) > 1

    # Apply the cuts
    jets["part_level"] = jets["part_level"][
        part_level_min_area_mask
    ]
    jets["det_level"] = jets["det_level"][
        det_level_max_track_pt_cut
        & det_level_min_area_mask
        & det_level_min_n_constituents_mask
    ]

    logger.info("Matching jets")
    jets["part_level", "matching"], jets["det_level", "matching"] = jet_finding.jet_matching(
        jets_base=jets["part_level"],
        jets_tag=jets["det_level"],
        max_matching_distance=0.3,
    )

    # Now, use matching info
    # First, require that there are jets in an event. If there are jets, and require that there
    # is a valid match.
    # NOTE: These can't be combined into one mask because they operate at different levels: events and jets
    logger.info("Using matching info")
    jets_present_mask = (ak.num(jets["part_level"], axis=1) > 0) & (ak.num(jets["det_level"], axis=1) > 0)
    jets = jets[jets_present_mask]

    # Now, onto the individual jet collections
    # We want to require valid matched jet indices. The strategy here is to lead via the detector
    # level jets. The procedure is as follows:
    #
    # 1. Identify all of the detector level jets with valid matches.
    # 2. Apply that mask to the detector level jets.
    # 3. Use the matching indices from the detector level jets to index the particle level jets.
    # 4. We should be done and ready to flatten. Note that the matching indices will refer to
    #    the original arrays, not the masked ones. In principle, this should be updated, but
    #    I'm unsure if they'll be used again, so we wait to update them until it's clear that
    #    it's required.
    #
    # The other benefit to this approach is that it should reorder the particle level matches
    # to be the same shape as the detector level jets, so in principle they are paired together.
    # TODO: Check this is truly the case.
    det_level_matched_jets_mask = jets["det_level"]["matching"] > -1
    jets["det_level"] = jets["det_level"][det_level_matched_jets_mask]
    jets["part_level"] = jets["part_level"][jets["det_level", "matching"]]

    logger.info("Reclustering jets...")
    for level in ["part_level", "det_level"]:
        logger.info(f"Reclustering {level}")
        jets[level, "reclustering"] = jet_finding.recluster_jets(
            jets=jets[level]
        )
    logger.info("Done with reclustering")

    logger.warning(f"n events: {len(jets)}")

    # Next step for using existing skimming:
    # Flatten from events -> jets
    # NOTE: Apparently it's takes issues with flattening the jets directly, so we have to do it
    #       separately for the different collections and then zip them together. This should keep
    #       matching together as appropriate.
    jets = ak.zip(
        {
            k: ak.flatten(v, axis=1)
            for k, v in zip(ak.fields(jets), ak.unzip(jets))
        },
        depth_limit=1,
    )

    # Now, the final transformation into a form that can be used to skim into a flat tree.
    return jets

def analysis_data(filename: Path, jet_R: float, min_pythia_jet_pt: float) -> ak.Array:
    logger.info("Start")
    source = sources.ParquetSource(
        filename=filename,
    )
    arrays = source.data()
    logger.info("Transform")
    arrays = _transform_inputs(arrays=arrays)

    # Event selection
    arrays = arrays[
        (arrays["is_ev_rej"] == 0)
        & (np.abs(arrays["z_vtx_reco"]) < 10)
    ]

    # Track cuts
    logger.info("Track level cuts")
    # Particle level track cuts:
    # - min: 150 MeV (from the EMCal container)
    part_track_pt_mask = (arrays["part_level"].pt >= 0.150)
    arrays["part_level"] = arrays["part_level"][part_track_pt_mask]
    # Detector level track cuts:
    # - min: 150 MeV
    # NOTE: Since the HF Tree uses track containers, the min is usually applied by default
    det_track_pt_mask = (arrays["det_level"].pt >= 0.150)
    arrays["det_level"] = arrays["det_level"][det_track_pt_mask]

    # Jet finding
    logger.info("Find jets")
    jets = ak.zip(
        {
            "part_level": jet_finding.find_jets(
                particles=arrays["part_level"],
                algorithm="anti-kt",
                jet_R=jet_R,
                area_settings=jet_finding.AREA_PP,
                # NOTE: We only want the minimum pt to apply to the detector level.
                #       Otherwise, we'll bias our particle level jets.
                min_jet_pt=1,
            ),
            "det_level": jet_finding.find_jets(
                particles=arrays["det_level"],
                algorithm="anti-kt",
                jet_R=jet_R,
                area_settings=jet_finding.AREA_PP,
                min_jet_pt=min_pythia_jet_pt,
            ),
        },
        depth_limit=1,
    )

    # Apply jet level cuts.
    # **************
    # Remove detector level jets with constituents with pt > 100 GeV
    # Those tracks are almost certainly fake at detector level.
    # NOTE: We skip at part level because it doesn't share these detector effects.
    # NOTE: We need to do it after jet finding to avoid a z bias.
    # **************
    det_level_max_track_pt_cut = ~ak.any(jets["det_level"].constituents.pt > 100, axis=-1)
    # **************
    # Apply area cut
    # Requires at least 60% of possible area.
    # **************
    min_area = jet_finding.area_percentage(60, jet_R),
    part_level_min_area_mask = jets["part_level", "area"] > min_area
    det_level_min_area_mask = jets["det_level", "area"] > min_area
    # *************
    # Require more than one constituent at detector level if we're not in PbPb
    # Matches a cut in AliAnalysisTaskJetDynamicalGrooming
    # *************
    det_level_min_n_constituents_mask = ak.num(jets["det_level", "constituents"], axis = 2) > 1

    # Apply the cuts
    jets["part_level"] = jets["part_level"][
        part_level_min_area_mask
    ]
    jets["det_level"] = jets["det_level"][
        det_level_max_track_pt_cut
        & det_level_min_area_mask
        & det_level_min_n_constituents_mask
    ]

    logger.info("Matching jets")
    jets["part_level", "matching"], jets["det_level", "matching"] = jet_finding.jet_matching(
        jets_base=jets["part_level"],
        jets_tag=jets["det_level"],
        max_matching_distance=0.3,
    )

    # Now, use matching info
    # First, require that there are jets in an event. If there are jets, and require that there
    # is a valid match.
    # NOTE: These can't be combined into one mask because they operate at different levels: events and jets
    logger.info("Using matching info")
    jets_present_mask = (ak.num(jets["part_level"], axis=1) > 0) & (ak.num(jets["det_level"], axis=1) > 0)
    jets = jets[jets_present_mask]

    # Now, onto the individual jet collections
    # We want to require valid matched jet indices. The strategy here is to lead via the detector
    # level jets. The procedure is as follows:
    #
    # 1. Identify all of the detector level jets with valid matches.
    # 2. Apply that mask to the detector level jets.
    # 3. Use the matching indices from the detector level jets to index the particle level jets.
    # 4. We should be done and ready to flatten. Note that the matching indices will refer to
    #    the original arrays, not the masked ones. In principle, this should be updated, but
    #    I'm unsure if they'll be used again, so we wait to update them until it's clear that
    #    it's required.
    #
    # The other benefit to this approach is that it should reorder the particle level matches
    # to be the same shape as the detector level jets, so in principle they are paired together.
    # TODO: Check this is truly the case.
    det_level_matched_jets_mask = jets["det_level"]["matching"] > -1
    jets["det_level"] = jets["det_level"][det_level_matched_jets_mask]
    jets["part_level"] = jets["part_level"][jets["det_level", "matching"]]

    logger.info("Reclustering jets...")
    for level in ["part_level", "det_level"]:
        logger.info(f"Reclustering {level}")
        jets[level, "reclustering"] = jet_finding.recluster_jets(
            jets=jets[level]
        )
    logger.info("Done with reclustering")

    logger.warning(f"n events: {len(jets)}")

    # Next step for using existing skimming:
    # Flatten from events -> jets
    # NOTE: Apparently it's takes issues with flattening the jets directly, so we have to do it
    #       separately for the different collections and then zip them together. This should keep
    #       matching together as appropriate.
    jets = ak.zip(
        {
            k: ak.flatten(v, axis=1)
            for k, v in zip(ak.fields(jets), ak.unzip(jets))
        },
        depth_limit=1,
    )

    # Now, the final transformation into a form that can be used to skim into a flat tree.
    return jets


def analysis_embedding() -> ak.Array:
    ...


def setup_logging(level: int = logging.DEBUG) -> None:
    # Basic setup
    logging.basicConfig(level=level, format="%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s")
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    # For sanity when using IPython
    logging.getLogger("parso").setLevel(logging.INFO)
    # Quiet down BinndData copy warnings
    logging.getLogger("pachyderm.binned_data").setLevel(logging.INFO)
    # Quiet down numba
    logging.getLogger("numba").setLevel(logging.INFO)


if __name__ == "__main__":
    setup_logging(level=logging.INFO)
    jets = analysis_MC(pythia_filename=Path("/software/rehlers/dev/mammoth/projects/framework/pythia/AnalysisResults.parquet"), jet_R=0.4, min_pythia_jet_pt = 20)

    import IPython

    IPython.embed()
