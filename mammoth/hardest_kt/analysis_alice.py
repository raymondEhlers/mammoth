"""Run analysis using PYTHIA + thermal model.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import awkward as ak
import numpy as np
import vector

from mammoth import helpers
from mammoth.framework import jet_finding, sources, transform
from mammoth.framework.normalize_data import track_skim


logger = logging.getLogger(__name__)
vector.register_awkward()


def load_data(
    filename: Path,
    collision_system: str,
    rename_prefix: Mapping[str, str],
) -> ak.Array:
    """ Load data for ALICE analysis from the track skim task output.

    Could come from a ROOT file or a converted parquet file.

    Args:
        filename: Filename containing the data.
        collision_system: Collision system corresponding to the data to load.
        rename_prefix: Prefix to label the data, and any mapping that we might need to perform. Note: the mapping
            goes from value -> key!
    Returns:
        The loaded data, transformed as appropriate based on the collision system
    """
    # Validation
    if "embed" in collision_system:
        raise ValueError("This function doesn't handle embedding. Please call the dedicated functions.")
    logger.info(f"Loading \"{collision_system}\" data")

    source = track_skim.FileSource(
        filename=filename,
        collision_system=collision_system,
    )
    arrays = source.data()

    # If we are renaming one of the prefixes to "data", that means that we want to treat it
    # as if it were standard data rather than pythia.
    if collision_system in ["pythia"] and "data" not in list(rename_prefix.keys()):
        logger.info("Transforming as MC")
        return transform.mc(arrays=arrays, rename_prefix=rename_prefix)

    # If not pythia, we don't need to handle it separately - it's all just data
    # All the rest of the collision systems would be embedded together separately by other functions
    logger.info("Transforming as data")
    return transform.data(arrays=arrays, rename_prefix=rename_prefix)


def analysis_MC(arrays: ak.Array, jet_R: float, min_jet_pt: Mapping[str, float], validation_mode: bool = False) -> ak.Array:
    logger.info("Start analyzing")
    # Event selection
    logger.warning(f"pre event sel n events: {len(arrays)}")
    event_level_mask = np.ones(len(arrays)) > 0
    if "is_ev_rej" in ak.fields(arrays):
        event_level_mask = event_level_mask & (arrays["is_ev_rej"] == 0)
    if "z_vtx_reco" in ak.fields(arrays):
        event_level_mask = event_level_mask & (np.abs(arrays["z_vtx_reco"]) < 10)
    arrays = arrays[event_level_mask]
    logger.warning(f"post event sel n events: {len(arrays)}")

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
    # NOTE: We don't need to explicitly select charged particles here because when creating
    #       the alice track skims, we explicitly select charged particles there.
    #       However, this would be required for separate MC (eg. jetscape, herwig, etc) productions.
    #       This is probably best served by some switch somewhere.
    logger.warning(f"post track cuts n events: {len(arrays)}")

    # Finally, require that we have part and det level particles for each event
    # NOTE: We have to do it in a separate mask because the above is masked as the particle level,
    #       but here we need to mask at the event level. (If you try to mask at the particle, you'll
    #       end up with empty events)
    # NOTE: Remember that the lengths of det and particle level need to match up, so be careful with the mask!
    event_has_particles_mask = (ak.num(arrays["part_level"], axis=1) > 0) & (ak.num(arrays["det_level"], axis=1) > 0)
    arrays = arrays[event_has_particles_mask]
    logger.warning(f"post requiring a particle in every event n events: {len(arrays)}")

    # Jet finding
    logger.info("Find jets")
    # First, setup what is needed for validation mode if enabled
    area_kwargs = {}
    if validation_mode:
        area_kwargs["random_seed"] = [12345, 67890]
    jets = ak.zip(
        {
            "part_level": jet_finding.find_jets_new(
                particles=arrays["part_level"],
                jet_finding_settings=jet_finding.JetFindingSettings(
                    R=jet_R,
                    algorithm="anti-kt",
                    # NOTE: We only want the minimum pt to apply to the detector level.
                    #       Otherwise, we'll bias our particle level jets.
                    pt_range=jet_finding.pt_range(pt_min=min_jet_pt.get("part_level", 1)),
                    eta_range=jet_finding.eta_range(
                        jet_R=jet_R,
                        # We will require the det level jets to be in the fiducial acceptance, which means
                        # that the part level jets don't need to be within the fiducial acceptance - just
                        # within the overall acceptance.
                        # NOTE: This is rounding a little bit since we still have an eta cut at particle
                        #       level which will then cut into the particle level jets. However, this is
                        #       what we've done in the past, so we continue it here.
                        fiducial_acceptance=False
                    ),
                    area_settings=jet_finding.AreaPP(**area_kwargs),
                ),
            ),
            "det_level": jet_finding.find_jets_new(
                particles=arrays["det_level"],
                jet_finding_settings=jet_finding.JetFindingSettings(
                    R=jet_R,
                    algorithm="anti-kt",
                    pt_range=jet_finding.pt_range(pt_min=min_jet_pt["det_level"]),
                    eta_range=jet_finding.eta_range(jet_R=jet_R, fiducial_acceptance=True),
                    area_settings=jet_finding.AreaPP(**area_kwargs),
                )
            ),
        },
        depth_limit=1,
    )
    logger.warning(f"Found det_level n jets: {np.count_nonzero(np.asarray(ak.flatten(jets['det_level'].px, axis=None)))}")

    import IPython; IPython.embed()

    # Apply jet level cuts.
    # **************
    # Remove detector level jets with constituents with pt > 100 GeV
    # Those tracks are almost certainly fake at detector level.
    # NOTE: We skip at part level because it doesn't share these detector effects.
    # NOTE: We need to do it after jet finding to avoid a z bias.
    # **************
    det_level_mask = ~ak.any(jets["det_level"].constituents.pt > 100, axis=-1)
    logger.warning(f"max track constituent max accepted: {np.count_nonzero(np.asarray(ak.flatten(det_level_mask == True, axis=None)))}")
    # **************
    # Apply area cut
    # Requires at least 60% of possible area.
    # **************
    min_area = jet_finding.area_percentage(60, jet_R)
    part_level_mask = jets["part_level", "area"] > min_area
    det_level_mask = det_level_mask & (jets["det_level", "area"] > min_area)
    logger.warning(f"add area cut n accepted: {np.count_nonzero(np.asarray(ak.flatten(det_level_mask == True, axis=None)))}")
    # *************
    # Require more than one constituent at detector level if we're not in PbPb
    # Matches a cut in AliAnalysisTaskJetDynamicalGrooming
    # *************
    det_level_mask = det_level_mask & (ak.num(jets["det_level", "constituents"], axis=2) > 1)
    logger.warning(f"more than one constituent n accepted: {np.count_nonzero(np.asarray(ak.flatten(det_level_mask == True, axis=None)))}")

    # Apply the cuts
    jets["part_level"] = jets["part_level"][part_level_mask]
    jets["det_level"] = jets["det_level"][det_level_mask]
    logger.warning(f"all jet cuts n accepted: {np.count_nonzero(np.asarray(ak.flatten(jets['det_level'].px, axis=None)))}")

    # Jet matching
    logger.info("Matching jets")
    jets = _jet_matching_MC(
        jets=jets,
        # NOTE: This is larger than the matching distance that I would usually use (where we usually use 0.3 =
        #       in embedding), but this is apparently what we use in pythia. So just go with it.
        part_level_det_level_max_matching_distance=1.0,
    )

    # Reclustering
    logger.info("Reclustering jets...")
    for level in ["part_level", "det_level"]:
        logger.info(f"Reclustering {level}")
        # We only do the area calculation for data.
        reclustering_kwargs = {}
        if level != "part_level":
            reclustering_kwargs["area_settings"] = jet_finding.AreaSubstructure(**area_kwargs)
        jets[level, "reclustering"] = jet_finding.recluster_jets_new(
            jets=jets[level],
            jet_finding_settings=jet_finding.ReclusteringJetFindingSettings(**reclustering_kwargs),
            store_recursive_splittings=True,
        )
    logger.info("Done with reclustering")

    logger.warning(f"n events: {len(jets)}")
    logger.warning(f"n jets accepted: {np.count_nonzero(np.asarray(ak.flatten(jets['det_level'].px, axis=None)))}")

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

    # Now, the final transformation into a form that can be used to skim into a flat tree.
    return jets


def _jet_matching_MC(jets: ak.Array,
                     part_level_det_level_max_matching_distance: float = 1.0,
                     ) -> ak.Array:
    """Geometrical jet matching for MC

    Note:
        The default matching distance is larger than the matching distance that I would have
        expected to use (eg. we usually use 0.3 = in embedding), but this is apparently what
        we use in pythia. So just go with it.

    Args:
        jets: Array containing the jets to match
        part_level_det_level_max_matching distance: Maximum matching distance
            between part and det level. Default: 1.0
    Returns:
        jets array containing only the matched jets
    """
    jets["part_level", "matching"], jets["det_level", "matching"] = jet_finding.jet_matching_geometrical(
        jets_base=jets["part_level"],
        jets_tag=jets["det_level"],
        max_matching_distance=part_level_det_level_max_matching_distance,
    )

    # Now, use matching info
    # First, require that there are jets in an event. If there are jets, further require that there
    # is a valid match.
    # NOTE: These can't be combined into one mask because they operate at different levels: events and jets
    logger.info("Using matching info")
    jets_present_mask = (ak.num(jets["part_level"], axis=1) > 0) & (ak.num(jets["det_level"], axis=1) > 0)
    jets = jets[jets_present_mask]
    logger.warning(f"post jets present mask n accepted: {np.count_nonzero(np.asarray(ak.flatten(jets['det_level'].px, axis=None)))}")

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
    # Semi-validated result for det <-> part w/ thermal model:
    # det <-> part for the thermal model looks like:
    # part: ([[0, 3, 1, 2, 4, 5], [0, 1, -1], [], [0], [1, 0, -1]],
    # det:   [[0, 2, 3, 1, 4, 5], [0, 1], [], [0], [1, 0]])
    # Semi-validated by pythia validation vs standard AliPhysics task.
    # TODO: Check this is truly the case by looking at both track (ie. det and true) collections -> This works!
    det_level_matched_jets_mask = jets["det_level"]["matching"] > -1
    jets["det_level"] = jets["det_level"][det_level_matched_jets_mask]
    jets["part_level"] = jets["part_level"][jets["det_level", "matching"]]
    logger.warning(f"post requiring valid matches n accepted: {np.count_nonzero(np.asarray(ak.flatten(jets['det_level'].px, axis=None)))}")

    return jets


def analysis_data(
    collision_system: str, arrays: ak.Array, jet_R: float, min_jet_pt: Mapping[str, float], particle_column_name: str = "data",
    validation_mode: bool = False,
) -> ak.Array:
    logger.info("Start analyzing")
    # Event selection
    logger.warning(f"pre event sel n events: {len(arrays)}")
    event_level_mask = np.ones(len(arrays)) > 0
    if "is_ev_rej" in ak.fields(arrays):
        event_level_mask = event_level_mask & (arrays["is_ev_rej"] == 0)
    if "z_vtx_reco" in ak.fields(arrays):
        event_level_mask = event_level_mask & (np.abs(arrays["z_vtx_reco"]) < 10)
    arrays = arrays[event_level_mask]
    logger.warning(f"post event sel n events: {len(arrays)}")

    # Track cuts
    logger.info("Track level cuts")
    # Data track cuts:
    # - min: 150 MeV (default from the EMCal container)
    data_track_pt_mask = arrays[particle_column_name].pt >= 0.150
    arrays[particle_column_name] = arrays[particle_column_name][data_track_pt_mask]

    # Jet finding
    logger.info("Find jets")
    # First, setup what is needed for validation mode if enabled
    area_kwargs = {}
    if validation_mode:
        area_kwargs["random_seed"] = [12345, 67890]

    area_settings = jet_finding.AreaPP(**area_kwargs)
    additional_kwargs: Dict[str, Any] = {}
    if collision_system in ["PbPb", "embedPythia", "thermal_model"]:
        area_settings = jet_finding.AreaAA(**area_kwargs)
        additional_kwargs["background_subtraction"] = jet_finding.BackgroundSubtraction(
            type=jet_finding.BackgroundSubtractionType.event_wise_constituent_subtraction,
            estimator=jet_finding.JetMedianBackgroundEstimator(
                jet_finding_settings=jet_finding.JetMedianJetFindingSettings(
                    area_settings=jet_finding.AreaAA(**area_kwargs)
                )
            ),
            subtractor=jet_finding.ConstituentSubtractor(
                r_max=0.25,
            ),
        )

    jets = ak.zip(
        {
            particle_column_name: jet_finding.find_jets_new(
                particles=arrays[particle_column_name],
                jet_finding_settings=jet_finding.JetFindingSettings(
                    R=jet_R,
                    algorithm="anti-kt",
                    pt_range=jet_finding.pt_range(pt_min=min_jet_pt.get(particle_column_name, 1.)),
                    eta_range=jet_finding.eta_range(jet_R=jet_R, fiducial_acceptance=True),
                    area_settings=area_settings,
                ),
                **additional_kwargs,
            ),
        },
        depth_limit=1,
    )
    logger.warning(f"Found n jets: {np.count_nonzero(np.asarray(ak.flatten(jets[particle_column_name].px, axis=None)))}")

    import IPython; IPython.embed()

    # Apply jet level cuts.
    # **************
    # Remove jets with constituents with pt > 100 GeV
    # Those tracks are almost certainly fake at detector level.
    # NOTE: We need to do it after jet finding to avoid a z bias.
    # **************
    mask = ~ak.any(jets[particle_column_name].constituents.pt > 100, axis=-1)
    logger.warning(f"max track constituent max accepted: {np.count_nonzero(np.asarray(ak.flatten(mask == True, axis=None)))}")
    # **************
    # Apply area cut
    # Requires at least 60% of possible area.
    # **************
    min_area = jet_finding.area_percentage(60, jet_R)
    #logger.warning(f"min area: {np.count_nonzero(np.asarray(ak.flatten((jets[particle_column_name, 'area'] > min_area) == True, axis=None)))}")
    mask = mask & (jets[particle_column_name, "area"] > min_area)
    logger.warning(f"add area cut accepted: {np.count_nonzero(np.asarray(ak.flatten(mask == True, axis=None)))}")
    # *************
    # Require more than one constituent at detector level if we're not in PbPb
    # Matches a cut in AliAnalysisTaskJetDynamicalGrooming
    # *************
    if collision_system not in ["PbPb", "embedPythia", "thermal_model"]:
        mask = mask & (ak.num(jets[particle_column_name].constituents, axis=2) > 1)

    # Apply the cuts
    jets[particle_column_name] = jets[particle_column_name][mask]

    # Check for any jets. If there are none, we probably want to bail out.
    # We need some variable to avoid flattening into a record, so select px arbitrarily.
    if len(ak.flatten(jets[particle_column_name].px, axis=None)) == 0:
        raise ValueError(f"No jets left for {particle_column_name}. Are your settings correct?")

    logger.warning(f"jet pt: {ak.flatten(jets[particle_column_name].pt).to_list()}")
    #import IPython; IPython.embed()
    #raise RuntimeError("Stahp!")

    logger.info(f"Reclustering {particle_column_name} jets...")
    jets[particle_column_name, "reclustering"] = jet_finding.recluster_jets_new(
        jets=jets[particle_column_name],
        jet_finding_settings=jet_finding.ReclusteringJetFindingSettings(
            # We perform the area calculation here since we're dealing with data, as is done in the AliPhysics DyG task
            area_settings=jet_finding.AreaSubstructure(**area_kwargs)
        ),
        store_recursive_splittings=True,
    )
    logger.info("Done with reclustering")
    logger.warning(f"n events: {len(jets)}")

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


def load_embedding(signal_filename: Path, background_filename: Path) -> Tuple[Dict[str, int], ak.Array]:
    # Setup
    logger.info("Loading embedded data")
    source_index_identifiers = {"signal": 0, "background": 100_000}

    # Signal
    pythia_source = sources.ChunkSource(
        # Chunk size will be set when combining the sources.
        chunk_size=-1,
        sources=track_skim.FileSource(
            filename=signal_filename,
            collision_system="pythia",
        )
    )
    # Background
    pbpb_source=track_skim.FileSource(
        filename=background_filename,
        collision_system="PbPb",
    )

    # Now, just zip them together, effectively.
    combined_source = sources.MultipleSources(
        fixed_size_sources={"background": pbpb_source},
        chunked_sources={"signal": pythia_source},
        source_index_identifiers=source_index_identifiers,
    )

    logger.info("Transforming embedded")
    arrays = combined_source.data()

    # Apply some basic requirements on the data
    mask = np.ones(len(arrays)) > 0
    # Require there to be particles for each level of particle collection for each event.
    # TODO: This select is repeated below. I think it's better below.
    #       Two things to confirm:
    #       - Is it actually better below? Or does it matter (eg. if here, could we forget it in the analysis?)
    #       - Is it the right thing to do? (I think so, but confirm that this is consistent with the AliPhysics analysis)
    mask = mask & (ak.num(arrays["signal", "part_level"], axis=1) > 0)
    mask = mask & (ak.num(arrays["signal", "det_level"], axis=1) > 0)
    mask = mask & (ak.num(arrays["background", "data"], axis=1) > 0)

    # Apply background event selection
    # We have to apply this here because we don't keep track of the background associated quantities.
    # NOTE: In principle, we're wasting pythia events here since there could be good pythia events which
    #       are rejected just because of the background. However, the background is our constraint, so it's fine.
    background_event_selection = np.ones(len(arrays)) > 0
    background_fields = ak.fields(arrays["background"])
    if "is_ev_rej" in background_fields:
        background_event_selection = background_event_selection & (arrays["background", "is_ev_rej"] == 0)
    if "z_vtx_reco" in background_fields:
        background_event_selection = background_event_selection & (np.abs(arrays["background", "z_vtx_reco"]) < 10)

    # Finally, apply the masks
    arrays = arrays[(mask & background_event_selection)]

    return source_index_identifiers, transform.embedding(
        arrays=arrays, source_index_identifiers=source_index_identifiers
    )


def load_thermal_model(
    signal_filename: Path,
    thermal_model_parameters: sources.ThermalModelParameters,
) -> Tuple[Dict[str, int], ak.Array]:
    # Setup
    source_index_identifiers = {"signal": 0, "background": 100_000}

    # Signal
    pythia_source = track_skim.FileSource(
        filename=signal_filename,
        collision_system="pythia",
    )
    # Background
    thermal_source = sources.ThermalModelExponential(
        # Chunk size will be set when combining the sources.
        chunk_size=-1,
        thermal_model_parameters=thermal_model_parameters,
    )

    # Now, just zip them together, effectively.
    combined_source = sources.MultipleSources(
        fixed_size_sources={"signal": pythia_source},
        chunked_sources={"background": thermal_source},
        source_index_identifiers=source_index_identifiers,
    )

    arrays = combined_source.data()
    signal_fields = ak.fields(arrays["signal"])
    # Empty mask
    # TODO: Confirm that this masks as expected...
    #mask = arrays["signal"][signal_fields[0]] * 0 >= 0
    mask = np.ones(len(arrays)) > 0
    # NOTE: We can apply the signal selections in the analysis task below
    # TODO: Refactor
    #if "is_ev_rej" in signal_fields:
    #    mask = mask & (arrays["signal", "is_ev_rej"] == 0)
    #if "z_vtx_reco" in signal_fields:
    #    mask = mask & (np.abs(arrays["signal", "z_vtx_reco"]) < 10)

    # Not necessary since there's no event selection for the thermal model
    background_fields = ak.fields(arrays["background"])
    if "is_ev_rej" in background_fields:
        mask = mask & (arrays["background", "is_ev_rej"] == 0)
    if "z_vtx_reco" in background_fields:
        mask = mask & (np.abs(arrays["background", "z_vtx_reco"]) < 10)

    arrays = arrays[mask]

    return source_index_identifiers, transform.embedding(
        arrays=arrays, source_index_identifiers=source_index_identifiers
    )


def analysis_embedding(
    source_index_identifiers: Mapping[str, int],
    arrays: ak.Array,
    jet_R: float,
    min_jet_pt: Mapping[str, float],
    r_max: float = 0.25,
    validation_mode: bool = False,
    shared_momentum_fraction_min: float = 0.5,
) -> ak.Array:
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

    # Finally, require that particles for all particle collections for each event
    # NOTE: We have to do it in a separate mask because the above is masked as the particle level,
    #       but here we need to mask at the event level. (If you try to mask at the particle, you'll
    #       end up with empty events)
    # NOTE: Remember that the lengths of det and particle level need to match up, so be careful with the mask!
    # TODO: Double check that this requirement is kosher. I think it's totally reasonable, but needs that check...
    logger.warning(f"pre requiring a particle in every event n events: {len(arrays)}")
    event_has_particles_mask = (
        (ak.num(arrays["part_level"], axis=1) > 0)
        & (ak.num(arrays["det_level"], axis=1) > 0)
        & (ak.num(arrays["hybrid"], axis=1) > 0)
    )
    arrays = arrays[event_has_particles_mask]
    logger.warning(f"post requiring a particle in every event n events: {len(arrays)}")

    # Jet finding
    logger.info("Find jets")
    # First, setup what is needed for validation mode if enabled
    area_kwargs = {}
    if validation_mode:
        area_kwargs["random_seed"] = [12345, 67890]
    # We usually calculate rho only using the PbPb particles (ie. not including the embedded det_level),
    # so we need to select only them.
    # NOTE: The most general approach would be some divisor argument to select the signal source indexed
    #       particles, but since the background has the higher source index, we can just select particles
    #       with an index smaller than that offset.
    background_only_particles_mask = ~(arrays["hybrid", "index"] < source_index_identifiers["background"])

    jets = ak.zip(
        {
            "part_level": jet_finding.find_jets_new(
                particles=arrays["part_level"],
                jet_finding_settings=jet_finding.JetFindingSettings(
                    R=jet_R,
                    algorithm="anti-kt",
                    # NOTE: We only want the minimum pt to apply to the detector level.
                    #       Otherwise, we'll bias our particle level jets.
                    pt_range=jet_finding.pt_range(pt_min=min_jet_pt.get("part_level", 1.)),
                    # NOTE: We only want fiducial acceptance at the "data" level (ie. hybrid)
                    eta_range=jet_finding.eta_range(jet_R=jet_R, fiducial_acceptance=False),
                    area_settings=jet_finding.AreaPP(**area_kwargs),
                )
            ),
            "det_level": jet_finding.find_jets_new(
                particles=arrays["det_level"],
                jet_finding_settings=jet_finding.JetFindingSettings(
                    R=jet_R,
                    algorithm="anti-kt",
                    pt_range=jet_finding.pt_range(pt_min=min_jet_pt.get("det_level", 5.)),
                    # NOTE: We only want fiducial acceptance at the "data" level (ie. hybrid)
                    eta_range=jet_finding.eta_range(jet_R=jet_R, fiducial_acceptance=False),
                    area_settings=jet_finding.AreaPP(**area_kwargs),
                )
            ),
            "hybrid": jet_finding.find_jets_new(
                particles=arrays["hybrid"],
                jet_finding_settings=jet_finding.JetFindingSettings(
                    R=jet_R,
                    algorithm="anti-kt",
                    pt_range=jet_finding.pt_range(pt_min=min_jet_pt["hybrid"]),
                    eta_range=jet_finding.eta_range(jet_R=jet_R, fiducial_acceptance=True),
                    area_settings=jet_finding.AreaAA(**area_kwargs),
                ),
                background_particles=arrays["hybrid"][background_only_particles_mask],
                background_subtraction=jet_finding.BackgroundSubtraction(
                    type=jet_finding.BackgroundSubtractionType.event_wise_constituent_subtraction,
                    estimator=jet_finding.JetMedianBackgroundEstimator(
                        jet_finding_settings=jet_finding.JetMedianJetFindingSettings(
                            area_settings=jet_finding.AreaAA(**area_kwargs)
                        )
                    ),
                    subtractor=jet_finding.ConstituentSubtractor(
                        r_max=r_max,
                    ),
                ),
            ),
        },
        depth_limit=1,
    )

    import IPython; IPython.embed()

    # Apply jet level cuts.
    # **************
    # Remove detector level jets with constituents with pt > 100 GeV
    # Those tracks are almost certainly fake at detector level.
    # NOTE: We skip at part level because it doesn't share these detector effects.
    # NOTE: We need to do it after jet finding to avoid a z bias.
    # **************
    det_level_mask = ~ak.any(jets["det_level"].constituents.pt > 100., axis=-1)
    hybrid_mask = ~ak.any(jets["hybrid"].constituents.pt > 100., axis=-1)
    # For part level, we set a cut of 1000. It should be quite rare that it has an effect, but included for consistency
    part_level_mask = ~ak.any(jets["part_level"].constituents.pt > 1000., axis=-1)
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

    # Jet matching
    logger.info("Matching jets")
    jets = _jet_matching_embedding(
        jets=jets,
        det_level_hybrid_max_matching_distance=0.3,
        part_level_det_level_max_matching_distance=0.3,
    )

    # Reclustering
    # If we're out of jets, reclustering will fail. So if we're out of jets, then skip this step
    if len(jets) == 0:
        logger.warning("No jets left for reclustering. Skipping reclustering...")
    else:
        logger.info("Reclustering jets...")
        for level in ["hybrid", "det_level", "part_level"]:
            logger.info(f"Reclustering {level}")
            # We only do the area calculation for data.
            reclustering_kwargs = {}
            if level != "part_level":
                reclustering_kwargs["area_settings"] = jet_finding.AreaSubstructure(**area_kwargs)
            jets[level, "reclustering"] = jet_finding.recluster_jets_new(
                jets=jets[level],
                jet_finding_settings=jet_finding.ReclusteringJetFindingSettings(**reclustering_kwargs),
                store_recursive_splittings=True,
            )

        logger.info("Done with reclustering")

    logger.warning(f"n events: {len(jets)}")

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

    # Shared momentum fraction
    try:
        jets["det_level", "shared_momentum_fraction"] = jet_finding.shared_momentum_fraction_for_flat_array(
            generator_like_jet_pts=jets["det_level"].pt,
            generator_like_jet_constituents=jets["det_level"].constituents,
            measured_like_jet_constituents=jets["hybrid"].constituents,
        )

        # Require a shared momentum fraction (default >= 0.5)
        shared_momentum_fraction_mask = (jets["det_level", "shared_momentum_fraction"] >= shared_momentum_fraction_min)
        n_jets_removed = len(jets) - np.count_nonzero(shared_momentum_fraction_mask)
        logger.info(f"Removing {n_jets_removed} events out of {len(jets)} total jets ({round(n_jets_removed / len(jets) * 100, 2)}%) due to shared momentum fraction")
        jets = jets[shared_momentum_fraction_mask]
    except Exception as e:
        print(e)
        import IPython

        IPython.embed()

    # Now, the final transformation into a form that can be used to skim into a flat tree.
    return jets


def _jet_matching_embedding(jets: ak.Array,
                            det_level_hybrid_max_matching_distance: float = 0.3,
                            part_level_det_level_max_matching_distance: float = 0.3,
                            ) -> ak.Array:
    """Jet matching for embedding."""

    # TODO: For better matching, need to use the hybrid sub -> hybrid sub info
    #       However, this may not be necessary given the additional jet indexing info that
    #       we have available here.
    jets["det_level", "matching"], jets["hybrid", "matching"] = jet_finding.jet_matching_geometrical(
        jets_base=jets["det_level"],
        jets_tag=jets["hybrid"],
        max_matching_distance=det_level_hybrid_max_matching_distance,
    )
    jets["part_level", "matching"], jets["det_level", "matching"] = jet_finding.jet_matching_geometrical(
        jets_base=jets["part_level"],
        jets_tag=jets["det_level"],
        max_matching_distance=part_level_det_level_max_matching_distance,
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

    return jets


if __name__ == "__main__":
    helpers.setup_logging(level=logging.INFO)

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
    #for collision_system in ["pp", "pythia", "PbPb"]:
    for collision_system in ["pp"]:
        logger.info(f"Analyzing \"{collision_system}\"")
        jets = analysis_data(
            collision_system=collision_system,
            arrays=load_data(
                filename=Path(
                    f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults_track_skim.parquet"
                ),
                collision_system=collision_system,
                rename_prefix={"data": "data"} if collision_system != "pythia" else {"data": "det_level"},
            ),
            jet_R=0.4,
            min_jet_pt={"data": 5. if collision_system == "pp" else 20.},
        )

        #import IPython; IPython.embed()
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
    # jets = analysis_embedding(
    #     *load_embedding(
    #         signal_filename=Path("/software/rehlers/dev/mammoth/projects/framework/pythia/AnalysisResults.parquet"),
    #         background_filename=Path("/software/rehlers/dev/mammoth/projects/framework/PbPb/AnalysisResults.parquet"),
    #     ),
    #     jet_R=0.4,
    #     min_jet_pt={
    #         "hybrid": 20,
    #         "det_level": 1,
    #         "part_level": 1,
    #     },
    # )
    ###############
    # Thermal model
    ###############
    # jets = analysis_embedding(
    #     *load_thermal_model(
    #         signal_filename=Path("/software/rehlers/dev/substructure/trains/pythia/641/run_by_run/LHC20g4/295612/1/AnalysisResults.20g4.001.root"),
    #         thermal_model_parameters=sources.THERMAL_MODEL_SETTINGS["central"],
    #     ),
    #     jet_R=0.2,
    #     min_jet_pt={
    #         "hybrid": 20,
    #         #"det_level": 1,
    #         #"part_level": 1,
    #     },
    #     r_max=0.25,
    # )

    import IPython

    IPython.embed()
