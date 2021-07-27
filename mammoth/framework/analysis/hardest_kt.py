"""Run analysis using PYTHIA + thermal model.

.. codeuathor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import awkward as ak
import numpy as np
import vector

from mammoth.framework import jet_finding, sources, transform


logger = logging.getLogger(__name__)
vector.register_awkward()


def load_MC(filename: Path) -> ak.Array:
    logger.info("Loading MC")
    pythia_source = sources.ParquetSource(
        filename=filename,
    )
    arrays = pythia_source.data()
    logger.info("Transforming MC")
    return transform.mc(arrays=arrays)


def analysis_MC(arrays: ak.Array, jet_R: float, min_jet_pt: Mapping[str, float]) -> ak.Array:
    # Event selection
    arrays = arrays[(arrays["is_ev_rej"] == 0) & (np.abs(arrays["z_vtx_reco"]) < 10)]

    # Track cuts
    logger.info("Track level cuts")
    # Particle level track cuts:
    # - min: 150 MeV (from the EMCal container)
    part_track_pt_mask = arrays["part_level"].pt >= 0.150
    arrays["part_level"] = arrays["part_level"][part_track_pt_mask]
    # Detector level track cuts:
    # - min: 150 MeV
    # NOTE: Since the HF Tree uses track containers, the min is usually applied by default
    det_track_pt_mask = arrays["det_level"].pt >= 0.150
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
                min_jet_pt=min_jet_pt.get("part_level", 1),
            ),
            "det_level": jet_finding.find_jets(
                particles=arrays["det_level"],
                algorithm="anti-kt",
                jet_R=jet_R,
                area_settings=jet_finding.AREA_PP,
                min_jet_pt=min_jet_pt["det_level"],
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
    det_level_mask = ~ak.any(jets["det_level"].constituents.pt > 100, axis=-1)
    # **************
    # Apply area cut
    # Requires at least 60% of possible area.
    # **************
    min_area = (jet_finding.area_percentage(60, jet_R),)
    part_level_mask = jets["part_level", "area"] > min_area
    det_level_mask = det_level_mask & (jets["det_level", "area"] > min_area)
    # *************
    # Require more than one constituent at detector level if we're not in PbPb
    # Matches a cut in AliAnalysisTaskJetDynamicalGrooming
    # *************
    det_level_mask = det_level_mask & (ak.num(jets["det_level", "constituents"], axis=2) > 1)

    # Apply the cuts
    jets["part_level"] = jets["part_level"][part_level_mask]
    jets["det_level"] = jets["det_level"][det_level_mask]

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
        jets[level, "reclustering"] = jet_finding.recluster_jets(jets=jets[level])
    logger.info("Done with reclustering")

    logger.warning(f"n events: {len(jets)}")

    # Next step for using existing skimming:
    # Flatten from events -> jets
    # NOTE: Apparently it's takes issues with flattening the jets directly, so we have to do it
    #       separately for the different collections and then zip them together. This should keep
    #       matching together as appropriate.
    jets = ak.zip(
        {k: ak.flatten(v, axis=1) for k, v in zip(ak.fields(jets), ak.unzip(jets))},
        depth_limit=1,
    )

    # Now, the final transformation into a form that can be used to skim into a flat tree.
    return jets


def load_data(
    filename: Path,
    rename_prefix: Mapping[str, str],
) -> ak.Array:
    logger.info("Loading data")
    source = sources.ParquetSource(
        filename=filename,
    )
    arrays = source.data()
    logger.info("Transforming data")
    return transform.data(arrays=arrays, rename_prefix=rename_prefix)


def analysis_data(
    collision_system: str, arrays: ak.Array, jet_R: float, min_jet_pt: float, particle_column_name: str = "data"
) -> ak.Array:
    logger.info("Start analyzing")
    # Event selection
    logger.warning(f"pre event sel n events: {len(arrays)}")
    arrays = arrays[(arrays["is_ev_rej"] == 0) & (np.abs(arrays["z_vtx_reco"]) < 10)]
    logger.warning(f"post event sel n events: {len(arrays)}")

    # Track cuts
    logger.info("Track level cuts")
    # Data track cuts:
    # - min: 150 MeV (from the EMCal container)
    data_track_pt_mask = arrays[particle_column_name].pt >= 0.150
    arrays[particle_column_name] = arrays[particle_column_name][data_track_pt_mask]

    # Jet finding
    logger.info("Find jets")
    area_settings = jet_finding.AREA_PP
    additional_kwargs: Dict[str, Any] = {}
    if collision_system in ["PbPb", "embedPythia"]:
        area_settings = jet_finding.AREA_AA
        additional_kwargs["constituent_subtraction"] = jet_finding.ConstituentSubtractionSettings(
            r_max=0.25,
        )
    jets = ak.zip(
        {
            particle_column_name: jet_finding.find_jets(
                particles=arrays[particle_column_name],
                algorithm="anti-kt",
                jet_R=jet_R,
                area_settings=area_settings,
                min_jet_pt=min_jet_pt,
                **additional_kwargs,
            ),
        },
        depth_limit=1,
    )

    # Apply jet level cuts.
    # **************
    # Remove jets with constituents with pt > 100 GeV
    # Those tracks are almost certainly fake at detector level.
    # NOTE: We need to do it after jet finding to avoid a z bias.
    # **************
    mask = ~ak.any(jets[particle_column_name].constituents.pt > 100, axis=-1)
    # **************
    # Apply area cut
    # Requires at least 60% of possible area.
    # **************
    min_area = (jet_finding.area_percentage(60, jet_R),)
    mask = mask & (jets[particle_column_name, "area"] > min_area)
    # *************
    # Require more than one constituent at detector level if we're not in PbPb
    # Matches a cut in AliAnalysisTaskJetDynamicalGrooming
    # *************
    if collision_system not in ["PbPb", "embedPythia"]:
        mask = mask & (ak.num(jets[particle_column_name].constituents, axis=2) > 1)

    # Apply the cuts
    jets[particle_column_name] = jets[particle_column_name][mask]

    # Check for any jets. If there are none, we probably want to bail out.
    # We need some variable to avoid flattening into a record, so select px arbitrarily.
    if len(ak.flatten(jets[particle_column_name].px, axis=None)) == 0:
        raise ValueError(f"No jets left for {particle_column_name}. Are your settings correct?")

    logger.info(f"Reclustering {particle_column_name} jets...")
    jets[particle_column_name, "reclustering"] = jet_finding.recluster_jets(jets=jets[particle_column_name])
    logger.info("Done with reclustering")
    logger.warning(f"n events: {len(jets)}")

    # Next step for using existing skimming:
    # Flatten from events -> jets
    # NOTE: Apparently it's takes issues with flattening the jets directly, so we have to do it
    #       separately for the different collections and then zip them together. This should keep
    #       matching together as appropriate.
    jets = ak.zip(
        {k: ak.flatten(v, axis=1) for k, v in zip(ak.fields(jets), ak.unzip(jets))},
        depth_limit=1,
    )

    # Now, the final transformation into a form that can be used to skim into a flat tree.
    return jets


def load_embedding(signal_filename: Path, background_filename: Path) -> Tuple[Dict[str, int], ak.Array]:
    # Setup
    logger.info("Loading embedded data")
    source_index_identifiers = {"signal": 0, "background": 100_000}

    # Signal
    pythia_source = sources.ChunkSource(
        chunk_size=-1,
        sources=sources.ParquetSource(filename=signal_filename),
    )
    pbpb_source = sources.ParquetSource(
        filename=background_filename,
    )

    # Now, just zip them together, effectively.
    combined_source = sources.MultipleSources(
        fixed_size_sources={"background": pbpb_source},
        chunked_sources={"signal": pythia_source},
        source_index_identifiers=source_index_identifiers,
    )

    logger.info("Transforming embedded")
    arrays = combined_source.data()
    # Apply background event selection
    # We have to apply this here because we don't keep track of the background associated quantities.
    background_event_selection = (arrays["background", "is_ev_rej"] == 0) & (np.abs(arrays["background", "z_vtx_reco"]) < 10)
    arrays = arrays[background_event_selection]
    return source_index_identifiers, transform.embedding(
        arrays=arrays, source_index_identifiers=source_index_identifiers
    )


def analysis_embedding(source_index_identifiers: Mapping[str, int], arrays: ak.Array, jet_R: float, min_jet_pt: Mapping[str, float]) -> ak.Array:
    # Event selection
    arrays = arrays[(arrays["is_ev_rej"] == 0) & (np.abs(arrays["z_vtx_reco"]) < 10)]
    # TODO: Ensure event selection applies to be the signal and the background.

    # Track cuts
    logger.info("Track level cuts")
    # Particle level track cuts:
    # - min: 150 MeV (from the EMCal container)
    part_track_pt_mask = arrays["part_level"].pt >= 0.150
    arrays["part_level"] = arrays["part_level"][part_track_pt_mask]
    # Detector level track cuts:
    # - min: 150 MeV
    # NOTE: Since the HF Tree uses track containers, the min is usually applied by default
    det_track_pt_mask = arrays["det_level"].pt >= 0.150
    arrays["det_level"] = arrays["det_level"][det_track_pt_mask]
    # Hybrid level track cuts:
    # - min: 150 MeV
    # NOTE: Since the HF Tree uses track containers, the min is usually applied by default
    hybrid_track_pt_mask = arrays["hybrid"].pt >= 0.150
    arrays["hybrid"] = arrays["hybrid"][hybrid_track_pt_mask]

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
                min_jet_pt=min_jet_pt.get("det_level", 5.0),
            ),
            "hybrid": jet_finding.find_jets(
                particles=arrays["hybrid"],
                algorithm="anti-kt",
                jet_R=jet_R,
                area_settings=jet_finding.AREA_AA,
                min_jet_pt=min_jet_pt["hybrid"],
                constituent_subtraction=jet_finding.ConstituentSubtractionSettings(
                    r_max=0.25,
                ),
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
    det_level_mask = ~ak.any(jets["det_level"].constituents.pt > 100, axis=-1)
    hybrid_mask = ~ak.any(jets["hybrid"].constituents.pt > 100, axis=-1)
    # **************
    # Apply area cut
    # Requires at least 60% of possible area.
    # **************
    min_area = (jet_finding.area_percentage(60, jet_R),)
    part_level_mask = jets["part_level", "area"] > min_area
    det_level_mask = det_level_mask & (jets["det_level", "area"] > min_area)
    hybrid_mask = hybrid_mask & (jets["hybrid", "area"] > min_area)

    # Apply the cuts
    jets["part_level"] = jets["part_level"][part_level_mask]
    jets["det_level"] = jets["det_level"][det_level_mask]

    logger.info("Matching jets")
    # TODO: For better matching, need to use the hybrid sub -> hybrid sub info
    jets["det_level", "matching"], jets["hybrid", "matching"] = jet_finding.jet_matching(
        jets_base=jets["det_level"],
        jets_tag=jets["hybrid"],
        max_matching_distance=0.3,
    )
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
    jets_present_mask = (
        (ak.num(jets["part_level"], axis=1) > 0)
        & (ak.num(jets["det_level"], axis=1) > 0)
        & (ak.num(jets["hybrid"], axis=1) > 0)
    )
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
    hybrid_to_det_level_valid_matches = jets["hybrid", "matching"] > -1
    det_to_part_level_valid_matches = jets["det_level", "matching"] > -1
    hybrid_to_det_level_including_det_to_part_level_valid_matches = det_to_part_level_valid_matches[jets["hybrid", "matching"][hybrid_to_det_level_valid_matches]]
    # First, restrict the hybrid level, requiring hybrid to det_level valid matches and
    # det_level to part_level valid matches.
    jets["hybrid"] = jets["hybrid"][hybrid_to_det_level_valid_matches][hybrid_to_det_level_including_det_to_part_level_valid_matches]
    # Next, restrict the det_level. Since we've restricted the hybrid to only valid matches, we should be able
    # to directly apply the masking indices.
    jets["det_level"] = jets["det_level"][jets["hybrid", "matching"]]
    # Same reasoning here.
    jets["part_level"] = jets["part_level"][jets["det_level", "matching"]]

    # After all of these gymnastics, we may not have jets at all levels, so require there to a jet of each type.
    # In principle, we've done this twice now, but logically this seems to be clearest.
    jets_present_mask = (
        (ak.num(jets["part_level"], axis=1) > 0)
        & (ak.num(jets["det_level"], axis=1) > 0)
        & (ak.num(jets["hybrid"], axis=1) > 0)
    )
    jets = jets[jets_present_mask]

    logger.info("Reclustering jets...")
    for level in ["hybrid", "det_level", "part_level"]:
        logger.info(f"Reclustering {level}")
        jets[level, "reclustering"] = jet_finding.recluster_jets(jets=jets[level])
    logger.info("Done with reclustering")

    logger.warning(f"n events: {len(jets)}")

    # Next step for using existing skimming:
    # Flatten from events -> jets
    # NOTE: Apparently it's takes issues with flattening the jets directly, so we have to do it
    #       separately for the different collections and then zip them together. This should keep
    #       matching together as appropriate.
    jets = ak.zip(
        {k: ak.flatten(v, axis=1) for k, v in zip(ak.fields(jets), ak.unzip(jets))},
        depth_limit=1,
    )

    # Now, the final transformation into a form that can be used to skim into a flat tree.
    return jets


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

    # Some tests:
    #######
    # Data:
    ######
    # pp: needs min_jet_pt = 5 to have any jets
    # collision_system = "pp"
    # pythia: Can test both "part_level" and "det_level" in the rename map.
    # collision_system = "pythia"
    # PbPb:
    # collision_system = "PbPb"
    # jets = analysis_data(
    #     collision_system=collision_system,
    #     arrays=load_data(
    #         filename=Path(
    #             f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults.parquet"
    #         ),
    #         # rename_prefix={"data": "det_level"},
    #         rename_prefix={"data": "data"},
    #     ),
    #     jet_R=0.4,
    #     min_jet_pt=20,
    # )
    ######
    # MC
    ######
    # jets = analysis_MC(
    #     arrays=load_MC(filename=Path("/software/rehlers/dev/mammoth/projects/framework/pythia/AnalysisResults.parquet")),
    #     jet_R=0.4,
    #     min_jet_pt={
    #         "det_level": 20,
    #     },
    # )
    ###########
    # Embedding
    ###########
    jets = analysis_embedding(
        *load_embedding(
            signal_filename=Path("/software/rehlers/dev/mammoth/projects/framework/pythia/AnalysisResults.parquet"),
            background_filename=Path("/software/rehlers/dev/mammoth/projects/framework/PbPb/AnalysisResults.parquet"),
        ),
        jet_R=0.4,
        min_jet_pt={
            "hybrid": 20,
            "det_level": 1,
            "part_level": 1,
        },
    )

    import IPython

    IPython.embed()
