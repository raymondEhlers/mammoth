"""Collection of transforms for sources.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import collections.abc
import logging
from collections.abc import Iterator, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import numpy.typing as npt
import vector

from mammoth.framework import particle_ID, sources, task

logger = logging.getLogger(__name__)


_default_particle_columns = {
    "px": np.float32,
    "py": np.float32,
    "pz": np.float32,
    "E": np.float32,
    "source_index": np.int64,
    "identifier": np.int64,
}


def _validate_potential_list_of_inputs(inputs: Path | Sequence[Path]) -> list[Path]:
    return [inputs] if not isinstance(inputs, collections.abc.Iterable) else list(inputs)


def normalize_for_one_input_level(
    arrays: ak.Array,
    rename_levels: Mapping[str, str] | None = None,
    mass_hypothesis: float | Mapping[str, float] = 0.139,
    particle_columns: Mapping[str, npt.DTypeLike] | None = None,
) -> ak.Array:
    """Transform into a form appropriate for analysis of one input level.

    In practice, this means that we treat this as data.

    Args:
        arrays: Input arrays
        rename_levels: Prefix to label the data, and any mapping that we might need to perform. Note: the mapping
            goes from value -> key!  Default: "data" -> "data".
        mass_hypothesis: Mass hypothesis for the prefixes, or individually. Default: 0.139 GeV
            for all particle collections. (Same interface here even though we expect only one particle collection).
        particle_columns: Dtypes for particle columns (unused as of July 2021).

    Returns:
        Fully transformed arrays, with all particle collections and event level info.
    """
    # Setup
    if not particle_columns:
        particle_columns = _default_particle_columns
    # Validation
    if rename_levels is None or not rename_levels:
        rename_levels = {"data": "data"}
    _prefixes = list(rename_levels.keys())
    if isinstance(mass_hypothesis, int | float):
        mass_hypotheses = {p: float(mass_hypothesis) for p in _prefixes}
    else:
        mass_hypotheses = dict(mass_hypothesis)

    # Transform various track collections.
    # 1) Add a source index, to identify where the particles came from.
    # 2) Add identifier column, to identify relationships between particles. May be the source index if not otherwise specified.
    # 3) Complete the four vectors (as necessary).
    data = arrays[rename_levels["data"]]
    data["source_index"] = ak.local_index(data)
    if "identifier" not in ak.fields(data):
        data["identifier"] = data["source_index"]
    # Only add the mass if either mass or energy aren't already present
    if "m" not in ak.fields(data) and "E" not in ak.fields(data):
        # The HFTreeCreator FastSim may not have the particle_ID information available, so we need to be
        # to workaround this case. The simplest thing we can do is just use the a fixed mass hypothesis as
        # we do at detector level.
        if "particle_ID" not in ak.fields(data):
            # NOTE: There's no point in throwing this warning unless we would actually care about
            #       the particle ID namely, if there's particle level data involved. We can check
            #       by looking for "part_level" on either side of the rename fields. Otherwise,
            #       it's entirely expected to use the mass hypothesis e.g. for real data.
            if "part_level" in rename_levels or "part_level" in list(rename_levels.values()):
                logger.warning("No particle ID info is available, so using mass hypothesis")
            # NOTE: This value can be customized if desired!
            data["m"] = data["pt"] * 0 + mass_hypotheses["data"]
        else:
            # Since we have truth level info, construct the part level mass based on the particle_ID
            # rather than a fixed mass hypothesis.
            # NOTE: At this point, the input data should have been normalized to use "particle_ID" for
            #       the particle ID column name, so we shouldn't need to change the column name here.
            data["m"] = particle_ID.particle_masses_from_particle_ID(arrays=data)
    # NOTE: This is fully equivalent because we registered vector:
    #       >>> data = ak.with_name(data, name="Momentum4D")
    data = vector.Array(data)

    # Combine inputs
    return ak.Array(
        {
            "data": data,
            # Include the rest of the non particle related fields (ie. event level info)
            # NOTE: We also intentionally skip the name of the value associated with "data" in the rename
            #       prefix to avoid copying both the original and the renamed into the same array.
            **{
                k: v
                for k, v in zip(ak.fields(arrays), ak.unzip(arrays), strict=True)
                if k not in [*_prefixes, rename_levels["data"]]
            },
        }
    )


def normalize_for_two_input_level(
    arrays: ak.Array,
    rename_levels: Mapping[str, str] | None = None,
    mass_hypothesis: float | Mapping[str, float] = 0.139,
    particle_columns: Mapping[str, npt.DTypeLike] | None = None,
) -> ak.Array:
    """Transform into a form appropriate for analysis with two input levels.

    In practice, this usually means that we're treating the inputs as MC. This assumes columns of:
    - part_level
    - det_level

    Args:
        arrays: Input arrays
        rename_levels: Prefix to label the data, and any mapping that we might need to perform. Note: the mapping
            goes from value -> key!  Default: "part_level" -> "part_level", "det_level" -> "det_level".
        mass_hypothesis: Mass hypothesis for either all three prefixes, or individually. Default: 0.139 GeV
            for all particle collections.
        particle_columns: Dtypes for particle columns (unused as of July 2021).

    Returns:
        Fully transformed arrays, with all particle collections and event level info.
    """
    # Setup
    if not particle_columns:
        particle_columns = _default_particle_columns
    # Validation
    # Since we require the rename_levels to define what prefixes to work with, if it's passed as an
    # empty mapping, we should treat it as if None was actually passed.
    if rename_levels is None or not rename_levels:
        rename_levels = {
            "part_level": "part_level",
            "det_level": "det_level",
        }
    _prefixes = list(rename_levels.keys())
    if isinstance(mass_hypothesis, int | float):
        mass_hypotheses = {p: float(mass_hypothesis) for p in _prefixes}
    else:
        mass_hypotheses = dict(mass_hypothesis)

    # Transform various track collections.
    # 1) Add a source index, to identify where the particles came from.
    # 2) Add identifier column, to identify relationships between particles. May be the source index if not otherwise specified.
    # 2) Complete the four vectors (as necessary).
    det_level = arrays[rename_levels["det_level"]]
    det_level["source_index"] = ak.local_index(det_level)
    if "identifier" not in ak.fields(det_level):
        det_level["identifier"] = det_level["source_index"]
    if "m" not in ak.fields(det_level) and "E" not in ak.fields(det_level):
        det_level["m"] = det_level["pt"] * 0 + mass_hypotheses["det_level"]
    # NOTE: This is fully equivalent because we registered vector:
    #       >>> det_level = ak.with_name(det_level, name="Momentum4D")
    det_level = vector.Array(det_level)
    # Part level
    part_level = arrays[rename_levels["part_level"]]
    part_level["source_index"] = ak.local_index(part_level)
    if "identifier" not in ak.fields(part_level):
        part_level["identifier"] = part_level["source_index"]
    if "m" not in ak.fields(part_level) and "E" not in ak.fields(part_level):
        # The HFTreeCreator FastSim may not have the particle_ID information available, so we need to be
        # to workaround this case. The simplest thing we can do is just use the a fixed mass hypothesis as
        # we do at detector level.
        if "particle_ID" not in ak.fields(part_level):
            # NOTE: This value can be customized if desired!
            logger.warning("No particle ID info is available, so using mass hypothesis for particle level!")
            part_level["m"] = part_level["pt"] * 0 + mass_hypotheses["part_level"]
        else:
            # Since we have truth level info, construct the part level mass based on the particle_ID
            # rather than a fixed mass hypothesis.
            # NOTE: At this point, the input data should have been normalized to use "particle_ID" for
            #       the particle ID column name, so we shouldn't need to change the column name here.
            part_level["m"] = particle_ID.particle_masses_from_particle_ID(arrays=part_level)
    part_level = vector.Array(part_level)

    # Combine inputs
    return ak.Array(
        {
            "part_level": part_level,
            "det_level": det_level,
            # Include the rest of the non particle related fields (ie. event level info)
            **{k: v for k, v in zip(ak.fields(arrays), ak.unzip(arrays), strict=True) if k not in _prefixes},
        }
    )


def _transform_data(
    gen_data: sources.T_GenData,
    collision_system: str,
    rename_levels: Mapping[str, str],
) -> sources.T_GenData:
    """Perform normalization for data and MC

    Throws:
        sources.NoDataAvailableError: Raised if the array is empty, since we can't normalize empty data
    """
    for arrays in gen_data:
        # Validation
        if len(arrays) == 0:
            _msg = "There's no data available in the source!"
            raise sources.NoDataAvailableError(_msg)

        # Handle MC vs data.
        # We treat MC separately unless we rename one of the prefixes to "data". In that case,
        # it means that we want to treat the MC as if it were standard data.
        if collision_system in ["pythia", "pp_MC", "PbPb_MC"] and "data" not in list(rename_levels.keys()):
            logger.info("Transforming as two input levels")
            yield normalize_for_two_input_level(arrays=arrays, rename_levels=rename_levels)
        else:
            # If not pythia, we don't need any special handling - it's all just data
            # All the rest of the collision systems would be embedded together separately by other functions
            logger.info("Transforming as one input level")
            yield normalize_for_one_input_level(arrays=arrays, rename_levels=rename_levels)


def data(
    data_input: Path | Sequence[Path],
    data_source: sources.SourceFromFilename,
    collision_system: str,
    rename_levels: Mapping[str, str],
    chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.FULL_SOURCE,
) -> ak.Array | Iterator[ak.Array]:
    """Load data for ALICE analysis from the track skim task output.

    Could come from a ROOT file or a converted parquet file.

    Args:
        data_input: Filenames containing the data.
        data_source: Data source to be used to load the data stored at the filenames.
        collision_system: Collision system corresponding to the data to load.
        rename_levels: Prefix to label the data, and any mapping that we might need to perform. Note: the mapping
            goes from value -> key!
        chunk_size: Chunk size to use when loading the data. Default: Full source.
    Returns:
        The loaded data, transformed as appropriate based on the collision system
    """
    # Validation
    if "embed" in collision_system:
        _msg = "This function doesn't handle embedding. Please call the dedicated functions."
        raise ValueError(_msg)
    logger.info(f'Loading "{collision_system}" data')
    # We allow for multiple filenames
    filenames = _validate_potential_list_of_inputs(data_input)

    source = sources.MultiSource(
        sources=[
            data_source(
                filename=_filename,
            )
            for _filename in filenames
        ],
    )

    _transform_data_iter = _transform_data(
        gen_data=source.gen_data(chunk_size=chunk_size),
        collision_system=collision_system,
        rename_levels=rename_levels,
    )
    return (
        _transform_data_iter if chunk_size is not sources.ChunkSizeSentinel.FULL_SOURCE else next(_transform_data_iter)
    )


def normalize_for_three_input_level(  # noqa: C901
    arrays: ak.Array,
    source_index_identifiers: Mapping[str, int],
    post_embedding_rename_levels: dict[str, str] | None = None,
    mass_hypothesis: float | Mapping[str, float] = 0.139,
    particle_columns: Mapping[str, npt.DTypeLike] | None = None,
    fixed_background_index_value: int | None = None,
) -> ak.Array:
    """Transform into a form appropriate for analysis of three input levels.

    In practice, this means setting up the input collections for embedding. This assumes columns of:
    - part_level
    - det_level
    - hybrid_level

    Note:
        This performs embedding in the process of transforming.

    Args:
        arrays: Input arrays
        source_index_identifiers: Index offset map for each source.
        post_embedding_rename_levels: Prefix to map the standard embedding level names to custom names.
           Note: the mapping goes from key -> value, unlike the loading_data rename prefix!
           Default: "part_level" -> "part_level", "det_level" -> "det_level", "hybrid_level" -> "hybrid_level".
           NOTE: This is different than for the other two normalize functions because we already
           have full control of the standard names. So here we rename **after** loading the data,
           rather than as part of loading the data. Since they are distinct purposes, we select different names.
        mass_hypothesis: Mass hypothesis for either all three prefixes, or individually. Default: 0.139 GeV
            for all particle collections.
        particle_columns: dtypes for particle columns (unused as of July 2021).
        fixed_background_index_value: If an integer is passed, fix the background index for all particles
            to that value. This reduces the information propagated, but is required for some applications
            (namely, the jet background ML studies). Default: None.

    Returns:
        Fully transformed arrays, with all particle collections and event level info.
    """
    # Setup
    if not particle_columns:
        particle_columns = _default_particle_columns
    # Validation
    _mass_hypothesis_prefixes = ["part_level", "det_level", "background"]
    if isinstance(mass_hypothesis, int | float):
        mass_hypotheses = {p: float(mass_hypothesis) for p in _mass_hypothesis_prefixes}
    else:
        mass_hypotheses = dict(mass_hypothesis)
    # Since we require the post_embedding_rename_levels to define what prefixes to work with,
    # if it's passed as an empty mapping, we should treat it as if None was actually passed.
    if post_embedding_rename_levels is None or not post_embedding_rename_levels:
        post_embedding_rename_levels = {
            "part_level": "part_level",
            "det_level": "det_level",
            "hybrid_level": "hybrid_level",
        }
    # Ensure that every level is defined if the user only passed a subset
    # NOTE: This can be a bit dangerous, since it avoid a ValueError if the user forgets.
    #       However, it's really convenient, so better to include it...
    # NOTE: This will modify this dict in place! This is actually kind of nice here because
    #       this step will only do something on the first time through and then it will have the
    #       correct values going forward. But just be aware - this may not be what is expected.
    #       If this becomes an issue, we can always just copy the dict here.
    rename_level_warnings = ""
    for k in ["part_level", "det_level", "hybrid_level"]:
        if k not in post_embedding_rename_levels:
            post_embedding_rename_levels[k] = k
            if rename_level_warnings:
                rename_level_warnings += "\n"
            rename_level_warnings += (
                f"Adding missing post embedding level '{k}' to the rename levels - ensure this is what you want!"
            )
    if rename_level_warnings:
        logger.warning(rename_level_warnings)

    logger.warning(f"{post_embedding_rename_levels=}")

    # Transform various track collections.
    # 1) Add a source index, to identify where the particles came from.
    # 2) Add identifier column, to identify relationships between particles. May be the source index if not otherwise specified.
    # 2) Complete the four vectors (as necessary).
    # Detector level
    det_level = arrays["signal"]["det_level"]
    det_level["source_index"] = ak.local_index(det_level) + source_index_identifiers["signal"]
    if "identifier" not in ak.fields(det_level):
        det_level["identifier"] = det_level["source_index"]
    if "m" not in ak.fields(det_level) and "E" not in ak.fields(det_level):
        det_level["m"] = det_level["pt"] * 0 + mass_hypotheses["det_level"]
    # NOTE: This is fully equivalent because we registered vector:
    #       >>> det_level = ak.with_name(det_level, name="Momentum4D")
    det_level = vector.Array(det_level)
    # Part level
    part_level = arrays["signal"]["part_level"]
    # NOTE: The particle level and detector level index values overlap. However, I think (as of Feb 2022)
    #       that this should be fine since they're unlikely to be clustered together. That being said,
    #       if we're looking at things like the shared momentum fraction, it's critical that they're _not_
    #       matched by this source index, but rather by `identifier` (which for this to work, most likely
    #       needs to be provided by the user).
    part_level["source_index"] = ak.local_index(part_level) + source_index_identifiers["signal"]
    if "identifier" not in ak.fields(part_level):
        part_level["identifier"] = part_level["source_index"]
    if "m" not in ak.fields(part_level) and "E" not in ak.fields(part_level):
        # The HFTreeCreator FastSim may not have the particle_ID information available, so we need to be
        # to workaround this case. The simplest thing we can do is just use the a fixed mass hypothesis as
        # we do at detector level.
        if "particle_ID" not in ak.fields(part_level):
            # NOTE: This value can be customized if desired!
            logger.warning("No particle ID info is available, so using mass hypothesis for particle level!")
            part_level["m"] = part_level["pt"] * 0 + mass_hypotheses["part_level"]
        else:
            # Since we have truth level info, construct the part level mass based on the particle_ID
            # rather than a fixed mass hypothesis.
            # NOTE: At this point, the input data should have been normalized to use "particle_ID" for
            #       the particle ID column name, so we shouldn't need to change the column name here.
            part_level["m"] = particle_ID.particle_masses_from_particle_ID(arrays=part_level)
    part_level = vector.Array(part_level)
    background = arrays["background"]["data"]
    if fixed_background_index_value is not None:
        background["source_index"] = ak.local_index(background) * 0 + fixed_background_index_value
    else:
        background["source_index"] = ak.local_index(background) + source_index_identifiers["background"]
    if "identifier" not in ak.fields(background):
        background["identifier"] = background["source_index"]
    if "m" not in ak.fields(background) and "E" not in ak.fields(background):
        background["m"] = background["pt"] * 0 + mass_hypotheses["background"]
    background = vector.Array(background)

    # Combine inputs
    logger.debug("Embedding...")
    return ak.Array(
        {
            post_embedding_rename_levels["part_level"]: part_level,
            post_embedding_rename_levels["det_level"]: det_level,
            # Practically, this is where we are performing the embedding
            # Need to re-zip so it applies the vector at the same level as the other collections
            # (ie. we want `var * Momentum4D[...]`, but without the zip, we have `Momentum4D[var * ...]`)
            # NOTE: For some reason, ak.concatenate returns float64 here. I'm not sure why, but for now
            #       it's not diving into.
            post_embedding_rename_levels["hybrid_level"]: vector.zip(
                dict(
                    zip(
                        particle_columns.keys(),
                        ak.unzip(
                            ak.concatenate(
                                [
                                    ak.Array({k: getattr(det_level, k) for k in particle_columns}),
                                    ak.Array({k: getattr(background, k) for k in particle_columns}),
                                ],
                                axis=1,
                            )
                        ),
                        strict=True,
                    )
                )
            ),
            # Include the rest of the non particle related fields (ie. event level info)
            # NOTE: We don't need to exclude the "hybrid_level" key from below because we're
            #       generating it here for the first time.
            # NOTE: We also intentionally skip the name of the keys associated with post_embedding_rename_level
            #       to avoid copying both the original and the renamed into the same array.
            **{
                k: v
                for k, v in zip(ak.fields(arrays["signal"]), ak.unzip(arrays["signal"]), strict=True)
                if k not in ["det_level", "part_level", *list(post_embedding_rename_levels.keys())]
            },
        }
    )


def _event_select_and_transform_embedding(
    gen_data: sources.T_GenData,
    source_index_identifiers: Mapping[str, int],
    post_embedding_rename_levels: dict[str, str],
    use_alice_standard_event_selection_on_background: bool = True,
) -> sources.T_GenData:
    """Perform event selection and normalization for embedding

    Args:
        gen_data: Generator of arrays to process.
        source_index_identifiers: Index offset map for each source.
        post_embedding_rename_levels: Prefix to map the standard embedding level names to custom names.
            See: `normalize_for_three_input_level`.
        use_alice_standard_event_selection_on_background: Whether to use the ALICE standard event
            selection on the background source. Default: True.

    Throws:
        sources.NoDataAvailableError: Raised if the array is empty
    """
    for arrays in gen_data:
        # Validation
        if len(arrays) == 0:
            _msg = "There's no data available in the source!"
            raise sources.NoDataAvailableError(_msg)

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
        if use_alice_standard_event_selection_on_background:
            # Use delayed import here. It's admittedly odd to import from the alice module, but it's super
            # convenient here, so we just run with it.
            from mammoth.alice import helpers as alice_helpers

            background_event_selection = alice_helpers.standard_event_selection(arrays["background"], return_mask=True)
        else:
            background_event_selection = np.ones(len(arrays)) > 0

        # Finally, apply the masks
        arrays = arrays[(mask & background_event_selection)]  # noqa: PLW2901

        logger.info("Transforming embedded")
        yield normalize_for_three_input_level(
            arrays=arrays,
            source_index_identifiers=source_index_identifiers,
            post_embedding_rename_levels=post_embedding_rename_levels,
        )


def embedding(
    signal_input: Path | Sequence[Path],
    signal_source: sources.SourceFromFilename,
    background_input: Path | Sequence[Path],
    background_source: sources.SourceFromFilename,
    post_embedding_rename_levels: dict[str, str],
    chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.FULL_SOURCE,
    repeat_unconstrained_when_needed_for_statistics: bool = True,
    background_is_constrained_source: bool = True,
    use_alice_standard_event_selection_on_background: bool = True,
) -> tuple[dict[str, int], ak.Array] | tuple[dict[str, int], Iterator[ak.Array]]:
    """Load data for embedding.

    Note:
        The signal and background sources are only constructed with the Path. If you need
        to pass additional arguments, you can do so by defining a closure around the source.
        For the simplest examples, it could be something like:

        ```python
        signal_source = functools.partial(track_skim.FileSource, collision_system="pythia")
        background_source = functools.partial(track_skim.FileSource, collision_system="PbPb")
        ```

    Args:
        signal_input: Path to the input file(s) for the signal.
        signal_source: File source to load the signal data.
        background_input: Path to the input file(s) for the background.
        background_source: File source to load the background data.
        chunk_size: Chunk size to use when loading the data. Default: Full source.
        repeated_unconstrained_when_needed_for_statistics: Whether to repeat unconstrained events source
            when the unconstrained has fewer events than the constrained. Default: True
        background_is_constrained_source: Whether the background is a constrained source. Default: True
        use_alice_standard_event_selection_on_background: Whether to use the ALICE standard event selection
            on the background source.
    Returns:
        A tuple of the source index identifiers and the data. The data is an iterator if we don't ask
            for the full source via the chunk size.
    """
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
    pythia_source_kwargs: dict[str, Any] = {}
    pbpb_source_kwargs: dict[str, Any] = {}
    if background_is_constrained_source:
        pythia_source_kwargs = unconstrained_source_kwargs
    else:
        pbpb_source_kwargs = unconstrained_source_kwargs

    # Signal
    pythia_source = sources.MultiSource(
        sources=[
            signal_source(
                filename=_filename,
            )
            for _filename in signal_filenames
        ],
        **pythia_source_kwargs,
    )
    # Background
    pbpb_source = sources.MultiSource(
        sources=[
            background_source(
                filename=_filename,
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
        source_index_identifiers=source_index_identifiers,
        post_embedding_rename_levels=post_embedding_rename_levels,
        use_alice_standard_event_selection_on_background=use_alice_standard_event_selection_on_background,
    )
    return (
        source_index_identifiers,
        _transform_data_iter if chunk_size is not sources.ChunkSizeSentinel.FULL_SOURCE else next(_transform_data_iter),
    )


def _accept_filename_no_op_for_thermal_model(*, filename: str, **kwargs: Any) -> sources.ThermalModelExponential:  # noqa: ARG001
    return sources.ThermalModelExponential(**kwargs)


def embedding_thermal_model(
    signal_input: Path | Sequence[Path],
    signal_source: sources.SourceFromFilename,
    thermal_model_parameters: sources.ThermalModelParameters,
    post_embedding_rename_levels: dict[str, str],
    chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.FULL_SOURCE,
) -> tuple[dict[str, int], ak.Array] | tuple[dict[str, int], Iterator[ak.Array]]:
    # Setup
    logger.info("Loading thermal model for embedding")

    return embedding(
        signal_input=signal_input,
        signal_source=signal_source,
        # NOTE: We can't just pass an empty list because we loop over this list to create the background source.
        #       However, since we don't actually care about the value, we just pass a dummy value.
        background_input=[Path("dummy")],
        background_source=partial(
            # NOTE We don't actually need to do anything with the filename, so we just need to accept and ignore it.
            #      We did that previously by just accepting arbitrary kwargs (implicitly), but that's not ideal.
            #      This is admittedly a hack, but it's slightly more precise.
            _accept_filename_no_op_for_thermal_model,
            thermal_model_parameters=thermal_model_parameters,
        ),
        post_embedding_rename_levels=post_embedding_rename_levels,
        chunk_size=chunk_size,
        # Since we will set the chunk size for the thermal model, there's no need to repeat
        # the background thermal model for more statistics - it will always be the right size.
        repeat_unconstrained_when_needed_for_statistics=False,
        # Background is generated, so the constraint is the signal
        background_is_constrained_source=False,
        # Not meaningful for thermal model, so disable
        use_alice_standard_event_selection_on_background=False,
    )


###############
# Setup sources
###############


def setup_source_for_data_or_MC_task(
    *,
    # Task settings
    task_settings: task.Settings,
    task_metadata: task.Metadata,
    # General IO
    minimize_IO_as_possible: bool,
    # Inputs
    signal_input: Path | Sequence[Path],
    signal_source: sources.DelayedSource,
    loading_data_rename_levels: Mapping[str, str],
    # Outputs
    output_settings: task.OutputSettings,
) -> Iterator[ak.Array]:
    """Setup data source for a data analysis task.

    Note:
        This is a lot like `data(...)`, but it integrates better with our task input, and
        it checks for existing inputs. We don't want such a check when we just define the =
        inputs and normalize the source, so we keep this as a bit of a wrapper.

    Note:
        This is also works for MC.

    Args:
        task_settings: Task settings.
        task_metadata: Task metadata.
        signal_input: Input signal file(s).
        signal_source: Source for the signal.
        loading_data_rename_levels: Rename existing level to another level in the array. Eg. "detLevel" -> "det_level".
        minimize_IO_as_possible: Whether to minimize IO as much as possible.
        output_settings: Output settings.
    Returns:
        iter_arrays: Iterator over the arrays to process.
    Raises:
        FailedToSetupSourceError: If the source could not be setup.
    """
    # Validation
    input_filenames = _validate_potential_list_of_inputs(signal_input)

    # Description parameters
    task_metadata.update(
        {
            "input_filenames": str([str(_filename) for _filename in input_filenames]),
        }
    )

    if minimize_IO_as_possible:
        logger.info("Skipping check for task output to minimize IO.")
    else:
        res = task.check_for_task_output(output_settings=output_settings, chunk_size=task_settings.chunk_size)
        if res[0]:
            raise task.FailedToSetupSourceError(result_success=res[0], result_message=res[1])

    # Setup iteration over the input files
    try:
        iter_arrays = data(
            data_input=input_filenames,
            data_source=partial(signal_source, collision_system=task_settings.collision_system),
            collision_system=task_settings.collision_system,
            rename_levels=loading_data_rename_levels,
            chunk_size=task_settings.chunk_size,
        )
    except sources.NoDataAvailableError as e:
        # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
        # Remember that this depends heavily on the analysis cuts!
        # NOTE: We need to make sure this directory exists - it may not if we're doing file staging and the first
        #      chunk doesn't work.
        output_settings.output_filename.parent.mkdir(parents=True, exist_ok=True)
        output_settings.output_filename.with_suffix(".empty").touch()
        raise task.FailedToSetupSourceError(
            result_success=True,
            result_message=f"Done - no data available (reason: {e}), so not trying to skim",
        ) from e

    # Validate that the arrays are in an a format that we can iterate over
    if isinstance(iter_arrays, ak.Array):
        iter_arrays = iter([iter_arrays])
    assert not isinstance(iter_arrays, ak.Array), "Check configuration. This should be an iterable, not an ak.Array!"

    return iter_arrays


def setup_source_for_embedding_task(
    *,
    # Task settings
    task_settings: task.Settings,
    task_metadata: task.Metadata,
    # General IO
    minimize_IO_as_possible: bool,
    # Inputs
    signal_input: Path | Sequence[Path],
    signal_source: sources.DelayedSource,
    signal_source_collision_system: str,
    background_input: Path | Sequence[Path],
    background_source: sources.DelayedSource,
    background_source_collision_system: str,
    background_is_constrained_source: bool,
    post_embedding_rename_levels: dict[str, str],
    # Outputs
    output_settings: task.OutputSettings,
    # Repeat categories as above, but with default arguments (as needed)
) -> tuple[dict[str, int], Iterator[ak.Array]]:
    """Setup embed MC source for a analysis task.

    Note:
        This is a lot like `embedding(...)`, but it integrates better with our task input, and
        it checks for existing inputs. We don't want such a check when we just define the inputs and normalize
        the source, so we keep this as a bit of a wrapper.

    Args:
        task_settings: Task settings.
        task_metadata: Task metadata.
        signal_input: Input signal file(s).
        signal_source: Source for the signal.
        signal_source_collision_system: Collision system of the signal source.
        background_input: Input background file(s).
        background_source: Source for the background.
        background_source_collision_system: Collision system of the background source.
        background_is_constrained_source: Whether the background is the constrained source.
        minimize_IO_as_possible: Whether to minimize IO as much as possible.
        output_settings: Output settings.
    Returns:
        (source_index_identifiers, iter_arrays), where:
            source_index_identifiers: Mapping of source index to identifier.
            iter_arrays: Iterator over the arrays to process.
    Raises:
        FailedToSetupSourceError: If the source could not be setup.
    """
    # Validation
    signal_input_filenames = _validate_potential_list_of_inputs(signal_input)
    background_input_filenames = _validate_potential_list_of_inputs(background_input)

    # Description parameters
    task_metadata.update(
        {
            "signal_input_filenames": str([str(_filename) for _filename in signal_input_filenames]),
            "background_input_filename": str([str(_filename) for _filename in background_input_filenames]),
        }
    )

    if minimize_IO_as_possible:
        logger.info("Skipping check for task output to minimize IO.")
    else:
        res = task.check_for_task_output(output_settings=output_settings, chunk_size=task_settings.chunk_size)
        if res[0]:
            raise task.FailedToSetupSourceError(result_success=res[0], result_message=res[1])

    # Setup iteration over the input files
    # If we don't use a processing chunk size, it should all be done in one chunk by default.
    # However, the memory usage often gets too large if the signal is the constrained source,
    # so this allows us to control the overall memory size by breaking it up into chunks,
    # such that we only load the data chunks that's currently needed for processing.
    # This is a bit idealistic because we often need to load the full file, but at least it sets
    # for potential improvements
    try:
        source_index_identifiers, iter_arrays = embedding(
            signal_input=signal_input_filenames,
            signal_source=partial(signal_source, collision_system=signal_source_collision_system),
            background_input=background_input_filenames,
            background_source=partial(background_source, collision_system=background_source_collision_system),
            background_is_constrained_source=background_is_constrained_source,
            post_embedding_rename_levels=post_embedding_rename_levels,
            chunk_size=task_settings.chunk_size,
        )
    except sources.NoDataAvailableError as e:
        # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
        # Remember that this depends heavily on the analysis cuts!
        # NOTE: We need to make sure this directory exists - it may not if we're doing file staging and the first
        #      chunk doesn't work.
        output_settings.output_filename.parent.mkdir(parents=True, exist_ok=True)
        output_settings.output_filename.with_suffix(".empty").touch()
        raise task.FailedToSetupSourceError(
            result_success=True,
            result_message=f"Done - no data available (reason: {e}), so not trying to skim",
        ) from e

    # Since we have the source index identifier, might as well add it into the metadata
    task_metadata.update({"source_index_identifiers": source_index_identifiers})

    # Validate that the arrays are in an a format that we can iterate over
    if isinstance(iter_arrays, ak.Array):
        iter_arrays = iter([iter_arrays])
    assert not isinstance(iter_arrays, ak.Array), "Check configuration. This should be an iterable, not an ak.Array!"

    return source_index_identifiers, iter_arrays


def setup_source_for_embedding_thermal_model_task(
    *,
    # Task settings
    task_settings: task.Settings,
    task_metadata: task.Metadata,
    # General IO
    minimize_IO_as_possible: bool,
    # Inputs
    signal_input: Path | Sequence[Path],
    signal_source: sources.DelayedSource,
    signal_source_collision_system: str,
    thermal_model_parameters: sources.ThermalModelParameters,
    post_embedding_rename_levels: dict[str, str],
    # Outputs
    output_settings: task.OutputSettings,
    # Repeat categories as above, but with default arguments (as needed)
) -> tuple[dict[str, int], Iterator[ak.Array]]:
    """Setup embed MC into thermal model source for a analysis task.

    Args:
        task_settings: Task settings.
        task_metadata: Task metadata.
        minimize_IO_as_possible: Whether to minimize IO as much as possible.
        signal_input: Input signal file(s).
        signal_source: Source for the signal.
        signal_source_collision_system: Collision system of the signal source.
        thermal_model_parameters: Parameters for the thermal model.
        output_settings: Output settings.
    Returns:
        (source_index_identifiers, iter_arrays), where:
            source_index_identifiers: Mapping of source index to identifier.
            iter_arrays: Iterator over the arrays to process.
    Raises:
        FailedToSetupSourceError: If the source could not be setup.
    """
    # Validation
    signal_input_filenames = _validate_potential_list_of_inputs(signal_input)

    # Description parameters
    task_metadata.update(
        {
            "signal_input_filenames": str([str(_filename) for _filename in signal_input_filenames]),
        }
    )

    if minimize_IO_as_possible:
        logger.info("Skipping check for task output to minimize IO.")
    else:
        res = task.check_for_task_output(output_settings=output_settings, chunk_size=task_settings.chunk_size)
        if res[0]:
            raise task.FailedToSetupSourceError(result_success=res[0], result_message=res[1])

    # Setup iteration over the input files
    # If we don't use a processing chunk size, it should all be done in one chunk by default.
    # However, the memory usage often gets too large, so this allows us to control the overall memory
    # size by breaking it up into chunks, such that we only generate the thermal model chunk
    # that's currently needed for processing
    try:
        source_index_identifiers, iter_arrays = embedding_thermal_model(
            signal_input=signal_input_filenames,
            signal_source=partial(signal_source, collision_system=signal_source_collision_system),
            thermal_model_parameters=thermal_model_parameters,
            post_embedding_rename_levels=post_embedding_rename_levels,
            chunk_size=task_settings.chunk_size,
        )
    except sources.NoDataAvailableError as e:
        # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
        # Remember that this depends heavily on the analysis cuts!
        # NOTE: We need to make sure this directory exists - it may not if we're doing file staging and the first
        #      chunk doesn't work.
        output_settings.output_filename.parent.mkdir(parents=True, exist_ok=True)
        output_settings.output_filename.with_suffix(".empty").touch()
        raise task.FailedToSetupSourceError(
            result_success=True,
            result_message=f"Done - no data available (reason: {e}), so not trying to skim",
        ) from e

    # Since we have the source index identifier, might as well add it into the metadata
    task_metadata.update({"source_index_identifiers": source_index_identifiers})

    # Validate that the arrays are in an a format that we can iterate over
    if isinstance(iter_arrays, ak.Array):
        iter_arrays = iter([iter_arrays])
    assert not isinstance(iter_arrays, ak.Array), "Check configuration. This should be an iterable, not an ak.Array!"

    return source_index_identifiers, iter_arrays
