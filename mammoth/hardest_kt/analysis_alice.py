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
from mammoth.framework import analysis_tools, jet_finding, load_data, sources
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


def load_data(
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
        area_kwargs["random_seed"] = [12345, 67890]
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
    jets = analysis_tools.jet_matching_MC(
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
    if collision_system in ["PbPb", "embedPythia", "embed_thermal_model"]:
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
    if collision_system not in ["PbPb", "embedPythia", "embed_thermal_model"]:
        mask = mask & (ak.num(jets[particle_column_name].constituents, axis=2) > 1)

    # Apply the cuts
    jets[particle_column_name] = jets[particle_column_name][mask]

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
    # Swap when the signal is the contrained source
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
    # NOTE: Remember that the lengths of particle collections need to match up, so be careful with the mask!
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

    # Setup an artificial tracking efficiency for detector level particles
    # To apply this, we want to select all background tracks + the subset of det level particles to keep
    # First, start with an all True mask
    hybrid_level_mask = (arrays["hybrid"].pt >= 0)
    if det_level_artificial_tracking_efficiency < 1.0:
        if validation_mode:
            raise ValueError("Cannot apply artificial tracking efficiency during validation mode. The randomness will surely break the validation.")

        # Here, we focus in on the detector level particles.
        # We want to select only them, determine whether they will be rejected, and then assign back
        # to the full hybrid mask. However, since awkward arrays are immutable, we need to do all of
        # this in numpy, and then unflatten.
        # First, we determine the total number of det_level particles to determine how many random
        # numbers to generate (plus, the info needed to unflatten later)
        _n_det_level_particles_per_event = ak.num(arrays["hybrid"][~background_only_particles_mask], axis=1)
        _total_n_det_level_particles = ak.sum(_n_det_level_particles_per_event)

        # Next, drop particles if their random values that are higher than the tracking efficiency
        _rng = np.random.default_rng()
        random_values = _rng.uniform(low=0.0, high=1.0, size=_total_n_det_level_particles)
        _drop_particles_mask = random_values > det_level_artificial_tracking_efficiency
        # NOTE: The check above will assign `True` when the random value is higher than the tracking efficiency.
        #       However, since we to remove those particles and keep ones below, we need to invert this selection.
        _det_level_particles_to_keep_mask = ~_drop_particles_mask

        # Now, we need to integrate it into the hybrid_level_mask
        # First, flatten that hybrid list into a numpy array, as well as the mask that selects det_level particles only
        _hybrid_level_mask_np = ak.to_numpy(ak.flatten(hybrid_level_mask))
        _det_level_particles_mask_np = ak.to_numpy(ak.flatten(~background_only_particles_mask))
        # Then, we can use the mask selecting det level particles only to assign whether to keep each det level
        # particle due to the tracking efficiency. Since the mask does the assignment
        _hybrid_level_mask_np[_det_level_particles_mask_np] = _det_level_particles_to_keep_mask

        # Unflatten so we can apply the mask to the existing particles
        hybrid_level_mask = ak.unflatten(_hybrid_level_mask_np, ak.num(arrays["hybrid"]))

        # Cross check that it worked.
        # If the entire hybrid mask is True, then it means that no particles were removed.
        # NOTE: I don't have this as an assert because if there aren't _that_ many particles and the efficiency
        #       is high, I suppose it's possible that this fails, and I don't want to kill jobs for that reason.
        if ak.all(hybrid_level_mask == True):
            logger.warning(
                "No particles were removed in the artificial tracking efficiency."
                " This is possible, but not super likely. Please check your settings!"
            )

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
    part_level_mask = part_level_mask & (jets["part_level", "area"] > min_area)
    det_level_mask = det_level_mask & (jets["det_level", "area"] > min_area)
    hybrid_mask = hybrid_mask & (jets["hybrid", "area"] > min_area)

    # Apply the cuts
    jets["part_level"] = jets["part_level"][part_level_mask]
    jets["det_level"] = jets["det_level"][det_level_mask]
    jets["hybrid"] = jets["hybrid"][hybrid_mask]

    # Jet matching
    logger.info("Matching jets")
    jets = analysis_tools.jet_matching_embedding(
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
