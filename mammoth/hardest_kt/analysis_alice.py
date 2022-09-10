"""Run analysis using PYTHIA + thermal model.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import collections
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import awkward as ak
import numpy as np
import vector

from mammoth import helpers
from mammoth.alice import helpers as alice_helpers
from mammoth.framework import jet_finding, load_data, sources
from mammoth.framework.analysis import jets as analysis_jets
from mammoth.framework.io import track_skim


logger = logging.getLogger(__name__)
vector.register_awkward()


def _transform_data(
    gen_data: sources.T_GenData,
    collision_system: str,
    rename_prefix: Mapping[str, str],
) -> sources.T_GenData:
    for arrays in gen_data:
        # If we are renaming one of the prefixes to "data", that means that we want to treat it
        # as if it were standard data rather than pythia.
        if collision_system in ["pythia"] and "data" not in list(rename_prefix.keys()):
            logger.info("Transforming as MC")
            yield load_data.normalize_for_MC(arrays=arrays, rename_prefix=rename_prefix)

        # If not pythia, we don't need to handle it separately - it's all just data
        # All the rest of the collision systems would be embedded together separately by other functions
        logger.info("Transforming as data")
        yield load_data.normalize_for_data(arrays=arrays, rename_prefix=rename_prefix)


def load_data_old(
    filename: Path,
    collision_system: str,
    rename_prefix: Mapping[str, str],
    chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.FULL_SOURCE,
) -> Union[ak.Array, Iterable[ak.Array]]:
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

    _transform_data_iter = _transform_data(
        gen_data=source.gen_data(chunk_size=chunk_size),
        collision_system=collision_system,
        rename_prefix=rename_prefix,
    )
    return _transform_data_iter if chunk_size is not sources.ChunkSizeSentinel.FULL_SOURCE else next(_transform_data_iter)


def analysis_MC(arrays: ak.Array, jet_R: float, min_jet_pt: Mapping[str, float], validation_mode: bool = False) -> ak.Array:
    logger.info("Start analyzing")
    # Event selection
    arrays = alice_helpers.standard_event_selection(arrays=arrays)

    # Track cuts
    arrays = alice_helpers.standard_track_selection(
        arrays=arrays,
        require_at_least_one_particle_in_each_collection_per_event=True
    )

    # Jet finding
    logger.info("Find jets")
    # First, setup what is needed for validation mode if enabled
    area_kwargs = {}
    if validation_mode:
        area_kwargs["random_seed"] = jet_finding.VALIDATION_MODE_RANDOM_SEED
    jets = ak.zip(
        {
            "part_level": jet_finding.find_jets(
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
            "det_level": jet_finding.find_jets(
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

    # Apply jet level cuts.
    jets = alice_helpers.standard_jet_selection(
        jets=jets,
        jet_R=jet_R,
        collision_system=collision_system,
        substructure_constituent_requirements=True,
    )
    logger.warning(f"all jet cuts n accepted: {np.count_nonzero(np.asarray(ak.flatten(jets['det_level'].px, axis=None)))}")

    # Jet matching
    logger.info("Matching jets")
    jets = analysis_jets.jet_matching_MC(
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
        jets[level, "reclustering"] = jet_finding.recluster_jets(
            jets=jets[level],
            jet_finding_settings=jet_finding.ReclusteringJetFindingSettings(**reclustering_kwargs),
            store_recursive_splittings=True,
        )
    logger.info("Done with reclustering")

    logger.warning(f"n events: {len(jets)}")
    logger.warning(f"n jets accepted: {np.count_nonzero(np.asarray(ak.flatten(jets['det_level'].px, axis=None)))}")

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


def analysis_data(
    collision_system: str, arrays: ak.Array, jet_R: float, min_jet_pt: Mapping[str, float],
    particle_column_name: str = "data",
    validation_mode: bool = False,
    background_subtraction_settings: Optional[Mapping[str, Any]] = None,
) -> ak.Array:
    # Validation
    if background_subtraction_settings is None:
        background_subtraction_settings = {}

    logger.info("Start analyzing")
    # Event selection
    arrays = alice_helpers.standard_event_selection(arrays=arrays)

    # Track cuts
    arrays = alice_helpers.standard_track_selection(
        arrays=arrays,
        require_at_least_one_particle_in_each_collection_per_event=False,
        selected_particle_column_name=particle_column_name,
    )

    # Jet finding
    logger.info("Find jets")
    # First, setup what is needed for validation mode if enabled
    area_kwargs = {}
    if validation_mode:
        area_kwargs["random_seed"] = jet_finding.VALIDATION_MODE_RANDOM_SEED

    area_settings = jet_finding.AreaPP(**area_kwargs)
    additional_kwargs: Dict[str, Any] = {}
    if collision_system in ["PbPb", "embedPythia", "embed_pythia", "embed_thermal_model"]:
        area_settings = jet_finding.AreaAA(**area_kwargs)
        additional_kwargs["background_subtraction"] = jet_finding.BackgroundSubtraction(
            type=jet_finding.BackgroundSubtractionType.event_wise_constituent_subtraction,
            estimator=jet_finding.JetMedianBackgroundEstimator(
                jet_finding_settings=jet_finding.JetMedianJetFindingSettings(
                    area_settings=jet_finding.AreaAA(**area_kwargs)
                )
            ),
            subtractor=jet_finding.ConstituentSubtractor(
                r_max=background_subtraction_settings.get("r_max", 0.25),
            ),
        )

    logger.warning(f"For particle column '{particle_column_name}', additional_kwargs: {additional_kwargs}")
    jets = ak.zip(
        {
            particle_column_name: jet_finding.find_jets(
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

    # Apply jet level cuts.
    jets = alice_helpers.standard_jet_selection(
        jets=jets,
        jet_R=jet_R,
        collision_system=collision_system,
        substructure_constituent_requirements=True,
        selected_particle_column_name=particle_column_name,
    )
    # Check for any jets. If there are none, we probably want to bail out.
    # We need some variable to avoid flattening into a record, so select px arbitrarily.
    if len(ak.flatten(jets[particle_column_name].px, axis=None)) == 0:
        raise ValueError(f"No jets left for {particle_column_name}. Are your settings correct?")

    logger.info(f"Reclustering {particle_column_name} jets...")
    jets[particle_column_name, "reclustering"] = jet_finding.recluster_jets(
        jets=jets[particle_column_name],
        jet_finding_settings=jet_finding.ReclusteringJetFindingSettings(
            # We perform the area calculation here since we're dealing with data, as is done in the AliPhysics DyG task
            area_settings=jet_finding.AreaSubstructure(**area_kwargs)
        ),
        store_recursive_splittings=True,
    )
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

    logger.warning(f"n jets: {len(jets)}")

    # Now, the final transformation into a form that can be used to skim into a flat tree.
    return jets


def _event_select_and_transform_embedding(
    gen_data: sources.T_GenData,
    source_index_identifiers: Mapping[str, int],
) -> sources.T_GenData:
    for arrays in gen_data:
        # Apply some basic requirements on the data
        mask = np.ones(len(arrays)) > 0
        # Require there to be particles for each level of particle collection for each event.
        # Although this will need to be repeated after the track cuts, it's good to start here since
        # it will avoid wasting signal or background events on events which aren't going to succeed anyway.
        mask = mask & (ak.num(arrays["signal", "part_level"], axis=1) > 0)
        mask = mask & (ak.num(arrays["signal", "det_level"], axis=1) > 0)
        mask = mask & (ak.num(arrays["background", "data"], axis=1) > 0)

        # Signal event selection
        # NOTE: We can apply the signal selections in the analysis task below, so we don't apply it here

        # Apply background event selection
        # We have to apply this here because we don't keep track of the background associated quantities.
        background_event_selection = np.ones(len(arrays)) > 0
        background_fields = ak.fields(arrays["background"])
        if "is_ev_rej" in background_fields:
            background_event_selection = background_event_selection & (arrays["background", "is_ev_rej"] == 0)
        if "z_vtx_reco" in background_fields:
            background_event_selection = background_event_selection & (np.abs(arrays["background", "z_vtx_reco"]) < 10)

        # Finally, apply the masks
        arrays = arrays[(mask & background_event_selection)]

        logger.info("Transforming embedded")
        yield load_data.normalize_for_embedding(
            arrays=arrays, source_index_identifiers=source_index_identifiers
        )


def _validate_potential_list_of_inputs(inputs: Union[Path, Sequence[Path]]) -> List[Path]:
    filenames = []
    if not isinstance(inputs, collections.abc.Iterable):
        filenames = [inputs]
    else:
        filenames = list(inputs)
    return filenames


def load_embedding_multiple_sources(
    signal_input: Union[Path, Sequence[Path]],
    background_input: Union[Path, Sequence[Path]],
    chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.FULL_SOURCE,
    repeat_unconstrained_when_needed_for_statistics: bool = True,
    background_is_constrained_source: bool = True,
) -> Union[Tuple[Dict[str, int], ak.Array], Tuple[Dict[str, int], Iterable[ak.Array]]]:
    # Validation
    # We allow for multiple signal filenames
    signal_filenames = _validate_potential_list_of_inputs(signal_input)
    # And also for background
    background_filenames = _validate_potential_list_of_inputs(background_input)

    # Setup
    logger.info("Loading embedded data")
    source_index_identifiers = {"signal": 0, "background": 100_000}

    # We only want to pass this to the unconstrained kwargs
    unconstrained_source_kwargs = {"repeat": repeat_unconstrained_when_needed_for_statistics}
    pythia_source_kwargs: Dict[str, Any] = {}
    pbpb_source_kwargs: Dict[str, Any] = {}
    if background_is_constrained_source:
        pythia_source_kwargs = unconstrained_source_kwargs
    else:
        pbpb_source_kwargs = unconstrained_source_kwargs

    # Signal
    pythia_source = sources.MultiSource(
        sources=[
            track_skim.FileSource(
                filename=_filename,
                collision_system="pythia",
            )
            for _filename in signal_filenames
        ],
        **pythia_source_kwargs,
    )
    # Background
    pbpb_source = sources.MultiSource(
        sources=[
            track_skim.FileSource(
                filename=_filename,
                collision_system="PbPb",
            )
            for _filename in background_filenames
        ],
        **pbpb_source_kwargs,
    )
    # By default the background is the constrained source
    constrained_size_source = {"background": pbpb_source}
    unconstrained_size_source = {"signal": pythia_source}
    # Swap when the signal is the constrained source
    if not background_is_constrained_source:
        unconstrained_size_source, constrained_size_source = constrained_size_source, unconstrained_size_source

    # Now, just zip them together, effectively.
    combined_source = sources.CombineSources(
        constrained_size_source=constrained_size_source,
        unconstrained_size_sources=unconstrained_size_source,
        source_index_identifiers=source_index_identifiers,
    )

    _transform_data_iter = _event_select_and_transform_embedding(
        gen_data=combined_source.gen_data(chunk_size=chunk_size),
        source_index_identifiers=source_index_identifiers
    )
    return source_index_identifiers, _transform_data_iter if chunk_size is not sources.ChunkSizeSentinel.FULL_SOURCE else next(_transform_data_iter)


def load_embedding(
    signal_input: Union[Path, Sequence[Path]],
    background_filename: Path,
    chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.FULL_SOURCE,
    repeat_signal_when_needed_for_statistics: bool = True,
) -> Union[Tuple[Dict[str, int], ak.Array], Tuple[Dict[str, int], Iterable[ak.Array]]]:
    # Validation
    # We allow for multiple signal filenames
    signal_filenames = []
    if not isinstance(signal_input, collections.abc.Iterable):
        signal_filenames = [signal_input]
    else:
        signal_filenames = list(signal_input)
    # Setup
    logger.info("Loading embedded data")
    source_index_identifiers = {"signal": 0, "background": 100_000}

    # Signal
    pythia_source = sources.MultiSource(
        sources=[
            track_skim.FileSource(
                filename=_filename,
                collision_system="pythia",
            )
            for _filename in signal_filenames
        ],
        repeat=repeat_signal_when_needed_for_statistics,
    )
    # Background
    pbpb_source = track_skim.FileSource(
        filename=background_filename,
        collision_system="PbPb",
    )

    # Now, just zip them together, effectively.
    combined_source = sources.CombineSources(
        constrained_size_source={"background": pbpb_source},
        unconstrained_size_sources={"signal": pythia_source},
        source_index_identifiers=source_index_identifiers,
    )

    _transform_data_iter = _event_select_and_transform_embedding(
        gen_data=combined_source.gen_data(chunk_size=chunk_size),
        source_index_identifiers=source_index_identifiers
    )
    return source_index_identifiers, _transform_data_iter if chunk_size is not sources.ChunkSizeSentinel.FULL_SOURCE else next(_transform_data_iter)


def load_embed_thermal_model(
    signal_input: Union[Path, Sequence[Path]],
    thermal_model_parameters: sources.ThermalModelParameters,
    chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.FULL_SOURCE,
    signal_collision_system: str = "pythia",
) -> Union[Tuple[Dict[str, int], ak.Array], Tuple[Dict[str, int], Iterable[ak.Array]]]:
    """Load data for embedding into a thermal model.

    This is somewhat different than the other load_* functions because we have added the ability
    to return data in chunks for processing.

    Args:
        signal_input: Signal input filenames.
        thermal_model_parameters: Thermal model parameters.
        chunk_size: Chunk size. Default: Everything in one chunk.
        signal_collision_system: Name of the collision system of the input. Default: "pythia".
    """
    # Validation
    # We allow for multiple signal filenames
    signal_filenames = []
    if not isinstance(signal_input, collections.abc.Iterable):
        signal_filenames = [signal_input]
    else:
        signal_filenames = list(signal_input)
    # Setup
    logger.info(f"Loading embed thermal model with processing chunk size {chunk_size}")
    source_index_identifiers = {"signal": 0, "background": 100_000}

    # Signal
    pythia_source = sources.MultiSource(
        sources=[
            track_skim.FileSource(
                filename=_filename,
                collision_system=signal_collision_system,
            )
            for _filename in signal_filenames
        ],
        repeat=False,
    )
    # Background
    thermal_source = sources.ThermalModelExponential(
        thermal_model_parameters=thermal_model_parameters,
    )

    # Now, just zip them together, effectively.
    combined_source = sources.CombineSources(
        constrained_size_source={"signal": pythia_source},
        unconstrained_size_sources={"background": thermal_source},
        source_index_identifiers=source_index_identifiers,
    )

    _transform_data_iter = _event_select_and_transform_embedding(
        gen_data=combined_source.gen_data(chunk_size=chunk_size),
        source_index_identifiers=source_index_identifiers
    )
    return source_index_identifiers, _transform_data_iter if chunk_size is not sources.ChunkSizeSentinel.FULL_SOURCE else next(_transform_data_iter)


def analysis_embedding(
    source_index_identifiers: Mapping[str, int],
    arrays: ak.Array,
    jet_R: float,
    min_jet_pt: Mapping[str, float],
    background_subtraction_settings: Optional[Mapping[str, Any]] = None,
    validation_mode: bool = False,
    shared_momentum_fraction_min: float = 0.5,
    det_level_artificial_tracking_efficiency: float = 1.0,
) -> ak.Array:
    # Validation
    if background_subtraction_settings is None:
        background_subtraction_settings = {}

    # Event selection
    # This would apply to the signal events, because this is what we propagate from the embedding transform
    arrays = alice_helpers.standard_event_selection(arrays=arrays)

    # Track cuts
    arrays = alice_helpers.standard_track_selection(
        arrays=arrays,
        require_at_least_one_particle_in_each_collection_per_event=True
    )

    # Jet finding
    logger.info("Find jets")
    # First, setup what is needed for validation mode if enabled
    area_kwargs = {}
    if validation_mode:
        area_kwargs["random_seed"] = jet_finding.VALIDATION_MODE_RANDOM_SEED

    # Calculate the relevant masks for hybrid level particles:
    # 1. We may need to mask the hybrid level particles to apply an artificial tracking inefficiency
    # 2. We usually calculate rho only using the PbPb particles (ie. not including the embedded det_level),
    #    so we need to select only them.
    hybrid_level_mask, background_only_particles_mask = analysis_jets.hybrid_level_particles_mask_for_jet_finding(
        arrays=arrays,
        det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
        source_index_identifiers=source_index_identifiers,
        validation_mode=validation_mode
    )

    # Finally setup and run the jet finders
    jets = ak.zip(
        {
            "part_level": jet_finding.find_jets(
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
            "det_level": jet_finding.find_jets(
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
            "hybrid": jet_finding.find_jets(
                particles=arrays["hybrid"][hybrid_level_mask],
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
                        r_max=background_subtraction_settings.get("r_max", 0.25),
                    ),
                ),
            ),
        },
        depth_limit=1,
    )

    # Apply jet level cuts.
    jets = alice_helpers.standard_jet_selection(
        jets=jets,
        jet_R=jet_R,
        collision_system=collision_system,
        substructure_constituent_requirements=True,
    )

    # Jet matching
    logger.info("Matching jets")
    jets = analysis_jets.jet_matching_embedding(
        jets=jets,
        # Values match those used in previous substructure analyses
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
            jets[level, "reclustering"] = jet_finding.recluster_jets(
                jets=jets[level],
                jet_finding_settings=jet_finding.ReclusteringJetFindingSettings(**reclustering_kwargs),
                store_recursive_splittings=True,
            )

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

    # I'm not sure if these work when there are no jets, so better to skip any calculations if there are no jets.
    if len(jets) > 0:
        # Shared momentum fraction
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

    # Now, the final transformation into a form that can be used to skim into a flat tree.
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
            arrays=load_data.data(
                data_input=Path(
                    f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults_track_skim.parquet"
                ),
                data_source=track_skim.FileSource.create_deferred_source(collision_system=collision_system),
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
    #     *load_embed_thermal_model(
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
