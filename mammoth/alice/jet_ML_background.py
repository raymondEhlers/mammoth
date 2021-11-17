"""Analysis code related to the jet ML background analysis.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Tuple

import awkward as ak
import numpy as np
import uproot

from mammoth.framework import jet_finding, models, sources, transform
from mammoth.framework.normalize_data import jet_extractor, track_skim

logger = logging.getLogger(__name__)


def load_embedding(signal_filename: Path, background_filename: Path, background_collision_system_tag: str) -> Tuple[Dict[str, int], ak.Array]:
    # Setup
    logger.info("Loading embedded data")
    source_index_identifiers = {"signal": 0, "background": 100_000}

    # Signal
    signal_source = jet_extractor.JEWELSource(
        filename=signal_filename,
    )
    fast_sim_source = sources.ChunkSource(
        # The chunk size will be determined by the number of background events
        chunk_size=-1,
        sources=sources.ALICEFastSimTrackingEfficiency(
            particle_level_data=signal_source.data(),
            fast_sim_parameters=models.ALICEFastSimParameters(
                event_activity=models.ALICETrackingEfficiencyEventActivity.central_00_10,
                period=models.ALICETrackingEfficiencyPeriod.LHC15o,
            )
        ),
        repeat=True,
    )
    # Background
    # For embedding, we will always be embedding into PbPb
    background_source = track_skim.FileSource(filename=background_filename, collision_system=background_collision_system_tag)

    # Now, just zip them together, effectively.
    combined_source = sources.MultipleSources(
        fixed_size_sources={"background": background_source},
        chunked_sources={"signal": fast_sim_source},
        source_index_identifiers=source_index_identifiers,
    )

    logger.info("Transforming embedded")
    arrays = combined_source.data()
    # Empty mask
    mask = np.ones(len(arrays)) > 0
    # First, require that there are particles for each event at part_level, det_level, and hybrid
    # NOTE: We store the mask because we need to apply it to the holes when we perform the subtraction below
    mask = mask & (ak.num(arrays["signal", "part_level"], axis=1) > 0)
    mask = mask & (ak.num(arrays["signal", "det_level"], axis=1) > 0)
    mask = mask & (ak.num(arrays["background", "data"], axis=1) > 0)

    # Selections on signal or background individually
    # Signal
    # NOTE: We can apply the signal selections in the analysis task later since we propagate those fields,
    #       so we skip it for now
    # Background
    # Only apply background event selection if applicable
    background_fields = ak.fields(arrays["background"])
    if "is_ev_rej" in background_fields:
        mask = mask & (arrays["background", "is_ev_rej"] == 0)
    if "z_vtx_reco" in background_fields:
        mask = mask & (np.abs(arrays["background", "z_vtx_reco"]) < 10)

    # Finally, apply selection
    n_events_removed = len(arrays) - np.count_nonzero(mask)
    logger.info(f"Removing {n_events_removed} events out of {len(arrays)} total events ({round(n_events_removed / len(arrays) * 100, 2)}%) due to event selection")
    arrays = arrays[mask]

    return source_index_identifiers, transform.embedding(
        arrays=arrays, source_index_identifiers=source_index_identifiers,
        # NOTE: We set a fixed background index value because it's required for the ML framework.
        #       In principle, we could do this later during the analysis, but assignment gets tricky
        #       due to awkward arrays, so this is a simpler approach.
        fixed_background_index_value=-1,
    )


def analysis_embedding(source_index_identifiers: Mapping[str, int],
                       arrays: ak.Array,
                       jet_R: float,
                       min_jet_pt: Mapping[str, float],
                       use_standard_rho_subtract: bool = True,
                       use_constituent_subtraction: bool = False,
                       ) -> ak.Array:
    # Validation
    if use_standard_rho_subtract and use_constituent_subtraction:
        raise ValueError("Selected both rho subtraction and constituent subtraction. Select only one.")

    # Event selection
    # This would apply to the signal events, because this is what we propagate from the embedding transform
    event_level_mask = np.ones(len(arrays)) > 0
    if "is_ev_rej" in ak.fields(arrays):
        event_level_mask = event_level_mask & (arrays["is_ev_rej"] == 0)
    if "z_vtx_reco" in ak.fields(arrays):
        event_level_mask = event_level_mask & (np.abs(arrays["z_vtx_reco"]) < 10)
    arrays = arrays[event_level_mask]

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
    hybrid_kwargs: Dict[str, Any] = {}
    if use_standard_rho_subtract:
        hybrid_kwargs = {"background_subtraction": True}
    if use_constituent_subtraction:
        hybrid_kwargs = {"constituent_subtraction": jet_finding.ConstituentSubtractionSettings(r_max=0.25)}
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
                **hybrid_kwargs,
            ),
        },
        depth_limit=1,
    )

    #logger.info("Right after jet finding...")
    #import IPython; IPython.embed()

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
    min_area = jet_finding.area_percentage(60, jet_R)
    part_level_mask = jets["part_level", "area"] > min_area
    det_level_mask = det_level_mask & (jets["det_level", "area"] > min_area)
    hybrid_mask = hybrid_mask & (jets["hybrid", "area"] > min_area)

    # Apply the cuts
    jets["part_level"] = jets["part_level"][part_level_mask]
    jets["det_level"] = jets["det_level"][det_level_mask]
    jets["hybrid"] = jets["hybrid"][hybrid_mask]

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
    #import IPython; IPython.embed()

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

    logger.warning(f"n events: {len(jets)}")

    #import IPython; IPython.embed()

    # Next step for using existing skimming:
    # Flatten from events -> jets
    # NOTE: Apparently it's takes issues with flattening the jets directly, so we have to do it
    #       separately for the different collections and then zip them together. This should keep
    #       matching together as appropriate.
    jets = ak.zip(
        {k: ak.flatten(v, axis=1) for k, v in zip(ak.fields(jets), ak.unzip(jets))},
        depth_limit=1,
    )

    # Calculate angularity
    # We do this after flatten the jets because it's simpler, and we don't actually care about the =
    # event structure for this calculation
    for label in ["part_level", "det_level", "hybrid"]:
        jets[label, "angularity"] = ak.sum(
            jets[label].constituents.pt * jets[label].constituents.deltaR(jets[label]),
            axis=1
        ) / jets[label].pt

    # Now, the final transformation into a form that can be used to skim into a flat tree.
    return jets


def write_skim(jets: ak.Array, filename: Path) -> None:
    # Rename according to the expected conventions
    jets_renamed = {
        "Jet_Pt": jets["hybrid"].pt,
        "Jet_NumTracks": ak.num(jets["hybrid"].constituents, axis=1),
        "Jet_Track_Pt": jets["hybrid"].constituents.pt,
        "Jet_Track_Label": jets["hybrid"].constituents["index"],
        "Jet_Shape_Angularity": jets["hybrid", "angularity"],
        "Jet_MC_MatchedDetLevelJet_Pt": jets["det_level"].pt,
        "Jet_MC_MatchedPartLevelJet_Pt": jets["part_level"].pt,
        # TODO: Update this when the shared momentum fraction is calculated
        "Jet_MC_TruePtFraction": jets["hybrid"].pt * 0 + 1,
    }

    logger.info(f"Writing to root file: {filename}")
    # Write with uproot
    try:
        with uproot.recreate(filename) as f:
            f["tree"] = jets_renamed
    except Exception as e:
        logger.exception(e)
        raise e from ValueError(
            f"{jets.type}, {jets}"
        )



if __name__ == "__main__":
    import mammoth.helpers
    mammoth.helpers.setup_logging()

    JEWEL_identifier = "NoToy_PbPb"
    pt_hat_bin = "80_140"
    index = "000"
    for background_index in range(820, 830):
        signal_filename = Path(f"/alf/data/rehlers/skims/JEWEL_PbPb_no_recoil/skim/JEWEL_{JEWEL_identifier}_PtHard{pt_hat_bin}_{index}.parquet")
        background_filename = Path(f"/alf/data/rehlers/substructure/trains/PbPb/7666/run_by_run/LHC15o/246087/AnalysisResults.15o.{background_index}.root")
        logger.info(f"Processing {background_filename}")
        source_index_identifiers, arrays = load_embedding(
            signal_filename=signal_filename,
            background_filename=background_filename,
            background_collision_system_tag="PbPb_central",
        )

        #jet_R = 0.6
        jet_R = 0.4
        jets = analysis_embedding(
            source_index_identifiers=source_index_identifiers,
            arrays=arrays,
            jet_R=jet_R,
            min_jet_pt={"hybrid": 10},
            use_standard_rho_subtract=True,
        )

        output_filename = Path(f"/alf/data/rehlers/skims/JEWEL_PbPb_no_recoil/skim/ML/jetR{round(jet_R * 100):03}_{signal_filename.stem}_{background_filename.parent.stem}_{background_filename.stem}.root")
        output_filename.parent.mkdir(exist_ok=True, parents=True)
        write_skim(jets=jets, filename=output_filename)

    import IPython; IPython.start_ipython(user_ns={**globals(),**locals()})