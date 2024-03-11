"""Analysis code related to the jet ML background analysis.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import awkward as ak
import numpy as np
import numpy.typing as npt
import uproot

from mammoth.framework import jet_finding, load_data, models, sources
from mammoth.framework.io import jet_extractor, track_skim

logger = logging.getLogger(__name__)


def load_embedding(
    signal_filename: Path,
    background_filename: Path,
    background_collision_system_tag: str,
) -> tuple[dict[str, int], ak.Array]:
    # Setup
    logger.info("Loading embedded data")
    source_index_identifiers = {"signal": 0, "background": 100_000}

    # Signal
    # We'll have the JEWEL source repeat as necessary, but the fast sim can vary from
    # one iteration to the next
    signal_source = sources.MultiSource(
        sources=jet_extractor.JEWELFileSource(
            filename=signal_filename,
        ),
        repeat=True,
    )
    fast_sim_source = sources.ALICEFastSimTrackingEfficiency(
        particle_level_source=signal_source,
        fast_sim_parameters=models.ALICEFastSimParameters(
            event_activity=models.ALICETrackingEfficiencyEventActivity.central_00_10,
            period=models.ALICETrackingEfficiencyPeriod.LHC15o,
        ),
    )
    # Background
    # For embedding, we will always be embedding into PbPb
    background_source = track_skim.FileSource(
        filename=background_filename, collision_system=background_collision_system_tag
    )

    # Now, just zip them together, effectively.
    combined_source = sources.CombineSources(
        constrained_size_source={"background": background_source},
        unconstrained_size_sources={"signal": fast_sim_source},
        source_index_identifiers=source_index_identifiers,
    )

    logger.info("Transforming embedded")
    arrays = next(combined_source.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE))
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
    logger.info(
        f"Removing {n_events_removed} events out of {len(arrays)} total events ({round(n_events_removed / len(arrays) * 100, 2)}%) due to event selection"
    )
    arrays = arrays[mask]

    return source_index_identifiers, load_data.normalize_for_three_input_level(
        arrays=arrays,
        source_index_identifiers=source_index_identifiers,
        # NOTE: We set a fixed background index value because it's required for the ML framework.
        #       In principle, we could do this later during the analysis, but assignment gets tricky
        #       due to awkward arrays, so this is a simpler approach.
        fixed_background_index_value=-1,
    )


def analysis_embedding(
    source_index_identifiers: Mapping[str, int],
    arrays: ak.Array,
    jet_R: float,
    min_jet_pt: Mapping[str, float],
    use_standard_rho_subtract: bool = True,
    use_constituent_subtraction: bool = False,
) -> ak.Array:
    # Validation
    if use_standard_rho_subtract and use_constituent_subtraction:
        _msg = "Selected both rho subtraction and constituent subtraction. Select only one."
        raise ValueError(_msg)

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
    # We usually calculate rho only using the PbPb particles (ie. not including the embedded det_level),
    # so we need to select only them.
    # NOTE: The most general approach would be some divisor argument to select the signal source indexed
    #       particles, but since the background has the higher source index, we can just select particles
    #       with an index smaller than that offset.
    background_only_particles_mask = ~(arrays["hybrid", "source_index"] < source_index_identifiers["background"])

    # Since rho subtraction is the default, we start with that
    subtractor: jet_finding.RhoSubtractor | jet_finding.ConstituentSubtractor = jet_finding.RhoSubtractor()
    if use_constituent_subtraction:
        subtractor = jet_finding.ConstituentSubtractor(r_max=0.25)
    jets = ak.zip(
        {
            "part_level": jet_finding.find_jets(
                particles=arrays["part_level"],
                jet_finding_settings=jet_finding.JetFindingSettings(
                    R=jet_R,
                    algorithm="anti-kt",
                    # NOTE: We only want the minimum pt to apply to the detector level.
                    #       Otherwise, we'll bias our particle level jets.
                    pt_range=jet_finding.pt_range(pt_min=min_jet_pt.get("part_level", 1.0)),
                    # NOTE: We only want fiducial acceptance at the "data" level (ie. hybrid)
                    eta_range=jet_finding.eta_range(jet_R=jet_R, fiducial_acceptance=False),
                    area_settings=jet_finding.AreaPP(),
                ),
            ),
            "det_level": jet_finding.find_jets(
                particles=arrays["det_level"],
                jet_finding_settings=jet_finding.JetFindingSettings(
                    R=jet_R,
                    algorithm="anti-kt",
                    pt_range=jet_finding.pt_range(pt_min=min_jet_pt.get("det_level", 5.0)),
                    # NOTE: We only want fiducial acceptance at the "data" level (ie. hybrid)
                    eta_range=jet_finding.eta_range(jet_R=jet_R, fiducial_acceptance=False),
                    area_settings=jet_finding.AreaPP(),
                ),
            ),
            "hybrid": jet_finding.find_jets(
                particles=arrays["hybrid"],
                jet_finding_settings=jet_finding.JetFindingSettings(
                    R=jet_R,
                    algorithm="anti-kt",
                    pt_range=jet_finding.pt_range(pt_min=min_jet_pt["hybrid"]),
                    eta_range=jet_finding.eta_range(jet_R=jet_R, fiducial_acceptance=True),
                    area_settings=jet_finding.AreaAA(),
                ),
                background_particles=arrays["hybrid"][background_only_particles_mask],
                background_subtraction=jet_finding.BackgroundSubtraction(
                    type=jet_finding.BackgroundSubtractionType.event_wise_constituent_subtraction,
                    estimator=jet_finding.JetMedianBackgroundEstimator(
                        jet_finding_settings=jet_finding.JetMedianJetFindingSettings()
                    ),
                    subtractor=subtractor,
                ),
            ),
        },
        depth_limit=1,
    )

    # We need to keep track of the event level cuts on the jets from here until flattening.
    # This enables us to store event level quantities by projecting them along with the jets.
    # Unfortunately, this takes a good deal of book keeping
    # event_level_mask_for_jets = np.ones(len(arrays)) > 0
    event_level_mask_for_jets = []

    # Add some event level quantities.
    # Unfortunately, because we often mask each jet level separately, the bookkeeping can be
    # quite a pain in the ass.
    # NOTE: We need these event level quantities to follow the shape of the jets, so we take
    #       the pt as a proxy for the shape. Since the eventual result will be that jets at
    #       each level will match up, we can arbitrarily select "part_level", and it will
    #       match up at the end.
    #       Adding the desired values will project them to the right shapes.

    # logger.info("Right after jet finding...")
    # import IPython; IPython.embed()

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
    # det_level <-> hybrid
    jets["det_level", "matching"], jets["hybrid", "matching"] = jet_finding.jet_matching_geometrical(
        jets_base=jets["det_level"],
        jets_tag=jets["hybrid"],
        max_matching_distance=0.3,
    )
    # part_level <-> det_level
    jets["part_level", "matching"], jets["det_level", "matching"] = jet_finding.jet_matching_geometrical(
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
    # event_level_mask_for_jets = event_level_mask_for_jets & jets_present_mask
    event_level_mask_for_jets.append(jets_present_mask)

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
    hybrid_to_det_level_valid_matches = jets["hybrid", "matching"] > -1
    det_to_part_level_valid_matches = jets["det_level", "matching"] > -1
    hybrid_to_det_level_including_det_to_part_level_valid_matches = det_to_part_level_valid_matches[
        jets["hybrid", "matching"][hybrid_to_det_level_valid_matches]
    ]
    # First, restrict the hybrid level, requiring hybrid to det_level valid matches and
    # det_level to part_level valid matches.
    jets["hybrid"] = jets["hybrid"][hybrid_to_det_level_valid_matches][
        hybrid_to_det_level_including_det_to_part_level_valid_matches
    ]
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
    # event_level_mask_for_jets = event_level_mask_for_jets & jets_present_mask
    event_level_mask_for_jets.append(jets_present_mask)

    logger.warning(f"n events: {len(jets)}")

    event_level_fields = [
        # Event weight
        "event_weight",
        # Store the original jet pt for the extractor bins
        "jet_pt_original",
    ]
    event_level_arrays = arrays[event_level_fields]
    for m in event_level_mask_for_jets:
        event_level_arrays = event_level_arrays[m]
    event_level_following_jets_shape = ak.zip(
        {k: jets["part_level"].pt * 0 + event_level_arrays[k] for k in event_level_fields}
    )

    # Next step for using existing skimming:
    # Flatten from events -> jets
    # NOTE: Apparently it's takes issues with flattening the jets directly, so we have to do it
    #       separately for the different collections and then zip them together. This should keep
    #       matching together as appropriate.
    jets = ak.zip(
        {
            k: ak.flatten(v, axis=1)
            for k, v in zip(
                ak.fields(jets) + ak.fields(event_level_following_jets_shape),
                ak.unzip(jets) + ak.unzip(event_level_following_jets_shape),
                strict=True,
            )
        },
        depth_limit=1,
    )

    logger.warning(f"n jets: {len(jets)}")

    # Now, calculate some properties based on the final matched jets
    # We do this after flatten the jets because it's simpler, and we don't actually care about
    # the event structure for many calculations
    # Angularity
    for label in ["part_level", "det_level", "hybrid"]:
        jets[label, "angularity"] = (
            ak.sum(jets[label].constituents.pt * jets[label].constituents.deltaR(jets[label]), axis=1) / jets[label].pt
        )
    # Shared momentum fraction
    try:
        # import pickle
        # with open("projects/exploration/repro.pkl", "wb") as f:
        #    pickle.dump(ak.to_buffers(jets["det_level"]), f)
        # ak.to_parquet(array=jets["det_level"], where="projects/exploration/repro.parquet")
        # res = jet_finding.repro(jets["det_level"].constituents)
        # Take slice for test...
        # res = jet_finding.shared_momentum_fraction_for_flat_array(
        #    generator_like_jet_pts=jets["det_level"][:1].pt,
        #    generator_like_jet_constituents=jets["det_level"][:1].constituents,
        #    measured_like_jet_constituents=jets["hybrid"][:1].constituents,
        # )
        # logger.info("Success")
        # import IPython; IPython.embed()
        jets["det_level", "shared_momentum_fraction"] = jet_finding.shared_momentum_fraction_for_flat_array(
            generator_like_jet_pts=jets["det_level"].pt,
            # NOTE: Here, there was once a bug which required a to `ak.to_packed` call on both constituent fields.
            #       I haven't test it, but it's supposed to be long fixed, so I removed the calls here. If there's
            #       a problem, they can be easily added back.
            generator_like_jet_constituents=jets["det_level"].constituents,
            measured_like_jet_constituents=jets["hybrid"].constituents,
        )

        # Require a shared momentum fraction of > 50%
        shared_momentum_fraction_mask = jets["det_level", "shared_momentum_fraction"] >= 0.5
        n_jets_removed = len(jets) - np.count_nonzero(shared_momentum_fraction_mask)
        logger.info(
            f"Removing {n_jets_removed} events out of {len(jets)} total jets ({round(n_jets_removed / len(jets) * 100, 2)}%) due to shared momentum fraction"
        )
        jets = jets[shared_momentum_fraction_mask]
    except Exception as e:
        print(e)  # noqa: T201
        import IPython

        IPython.embed()  # type: ignore[no-untyped-call]

    # Now, the final transformation into a form that can be used to skim into a flat tree.
    return jets


def write_skim(jets: ak.Array, filename: Path) -> None:
    # Rename according to the expected conventions
    jets_renamed = {
        # JEWEL specific
        "Jet_Pt_NoToy": jets["jet_pt_original"],
        # Generic
        "Jet_Pt": jets["hybrid"].pt,
        "Jet_NumTracks": ak.num(jets["hybrid"].constituents, axis=1),
        "Jet_Track_Pt": jets["hybrid"].constituents.pt,
        "Jet_Track_Label": jets["hybrid"].constituents["identifier"],
        "Jet_Shape_Angularity": jets["hybrid", "angularity"],
        "Jet_MC_MatchedDetLevelJet_Pt": jets["det_level"].pt,
        "Jet_MC_MatchedPartLevelJet_Pt": jets["part_level"].pt,
        "Jet_MC_TruePtFraction": jets["det_level"]["shared_momentum_fraction"],
        "Event_Weight": jets["event_weight"],
    }

    logger.info(f"Writing to root file: {filename}")
    # Write with uproot
    try:
        with uproot.recreate(filename) as f:
            f["tree"] = jets_renamed
    except Exception as e:
        logger.exception(e)
        raise e from ValueError(f"{jets.type}, {jets}")


def run_embedding_analysis(
    signal_filename: Path,
    background_filename: Path,
    background_collision_system_tag: str,
    jet_R: float,
    min_jet_pt: Mapping[str, float],
    output_filename: Path,
    use_standard_rho_subtraction: bool = True,
    use_constituent_subtraction: bool = False,
) -> bool:
    # Keep the calls separate to help out when debugging. It shouldn't cost anything in terms of resources
    source_index_identifiers, arrays = load_embedding(
        signal_filename=signal_filename,
        background_filename=background_filename,
        background_collision_system_tag=background_collision_system_tag,
    )

    jets = analysis_embedding(
        source_index_identifiers=source_index_identifiers,
        arrays=arrays,
        jet_R=jet_R,
        min_jet_pt=min_jet_pt,
        use_standard_rho_subtract=use_standard_rho_subtraction,
        use_constituent_subtraction=use_constituent_subtraction,
    )

    output_filename.parent.mkdir(exist_ok=True, parents=True)
    write_skim(jets=jets, filename=output_filename)

    return True


if __name__ == "__main__":
    import mammoth.helpers

    mammoth.helpers.setup_logging()

    JEWEL_identifier = "NoToy_PbPb"
    pt_hat_bin = "80_140"
    index = "000"
    signal_filename = Path(
        f"/alf/data/rehlers/skims/JEWEL_PbPb_no_recoil/skim/central_00_10/JEWEL_{JEWEL_identifier}_PtHard{pt_hat_bin}_{index}.parquet"
    )

    background_collision_system_tag = "PbPb_central"
    # jet_R = 0.6
    jet_R = 0.4

    use_standard_rho_subtraction = True
    use_constituent_subtraction = False

    # for background_index in range(820, 830):
    for background_index in range(2, 3):
        background_filename = Path(
            f"/alf/data/rehlers/substructure/trains/PbPb/7666/run_by_run/LHC15o/246087/AnalysisResults.15o.{background_index:03}.root"
        )
        output_filename = Path(
            f"/alf/data/rehlers/skims/JEWEL_PbPb_no_recoil/skim/ML/jetR{round(jet_R * 100):03}_{signal_filename.stem}_{background_filename.parent.stem}_{background_filename.stem}.root"
        )

        logger.info(f"Processing {background_filename}")
        run_embedding_analysis(
            signal_filename=signal_filename,
            background_filename=background_filename,
            background_collision_system_tag=background_collision_system_tag,
            jet_R=jet_R,
            min_jet_pt={"hybrid": 10},
            output_filename=output_filename,
            use_standard_rho_subtraction=use_standard_rho_subtraction,
            use_constituent_subtraction=use_constituent_subtraction,
        )

    import IPython

    IPython.start_ipython(user_ns={**globals(), **locals()})  # type: ignore[no-untyped-call]


def mask_for_flat_distribution(
    jet_pt: npt.NDArray[np.float64],
    target_number_of_jets: int,
) -> npt.NDArray[np.bool_]:
    """A mask to create a flat pt distribution for ML training.

    Wrote this function for Hannah to use for ML training.

    Note:
        This is not especially fast. We're basically finding the mapping for a histogram.
        But it gets the job done, which should be enough here since it's not called repeatedly.
        Could accelerate it with numba if ever needed.

    Note:
        Assumes 1 GeV wide bins!!!

    Args:
        jet_pt: Column of jet pt
        target_number_of_jets: Number of jets we want in each jet pt bin.
    Returns:
        Mask of True if row should be kept
    """
    # Setup
    rng = np.random.default_rng()

    # Determine the jet pt bins. By taking the floor, we assume that the bins are 1 GeV wide!
    jet_pt_bins = np.floor(jet_pt).astype(np.int64)
    # Start with all false mask
    mask = np.zeros(len(jet_pt_bins)) > 0

    # Loop over all jet pt values, finding their indices, and then randomly selecting the number that we want.
    for i in np.arange(0, np.max(jet_pt_bins) + 1):
        # We need to know the indices of the current bin that we're investigating
        current_jet_pt_bin_indices = np.where(jet_pt_bins == i)[0]
        # If there are no entries, then there's nothing to be done
        if len(current_jet_pt_bin_indices):
            # If there are not enough jets, we just need to take them all. Otherwise, this will cause
            # an issue for `choice`.
            if len(current_jet_pt_bin_indices) < target_number_of_jets:
                keep_jets = current_jet_pt_bin_indices
            else:
                keep_jets = rng.choice(current_jet_pt_bin_indices, size=target_number_of_jets, replace=False)
            # If we've selected the indices, keep them around!
            mask[keep_jets] = True
    return mask
