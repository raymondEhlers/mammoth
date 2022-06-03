""" Collection of transforms for sources.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import collections
import functools
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Mapping, Optional, Tuple, Union

import awkward as ak
import numpy as np
import numpy.typing as npt
import vector

from mammoth.framework import particle_ID, sources

logger = logging.getLogger(__name__)


_default_particle_columns = {
    "px": np.float32,
    "py": np.float32,
    "pz": np.float32,
    "E": np.float32,
    "index": np.int64,
}


def _validate_potential_list_of_inputs(inputs: Union[Path, Sequence[Path]]) -> List[Path]:
    filenames = []
    if not isinstance(inputs, collections.abc.Iterable):
        filenames = [inputs]
    else:
        filenames = list(inputs)
    return filenames


def normalize_for_data(
    arrays: ak.Array,
    rename_prefix: Optional[Mapping[str, str]] = None,
    mass_hypothesis: Union[float, Mapping[str, float]] = 0.139,
    particle_columns: Optional[Mapping[str, npt.DTypeLike]] = None,
) -> ak.Array:
    """Transform into a form appropriate for data analysis.

    Args:
        arrays: Input arrays
        rename_prefix: Prefix to label the data, and any mapping that we might need to perform. Note: the mapping
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
    if rename_prefix is None:
        rename_prefix = {"data": "data"}
    _prefixes = list(rename_prefix.keys())
    if isinstance(mass_hypothesis, (int, float)):
        mass_hypotheses = {p: float(mass_hypothesis) for p in _prefixes}
    else:
        mass_hypotheses = dict(mass_hypothesis)

    # Transform various track collections.
    # 1) Add indices.
    # 2) Complete the four vectors (as necessary).
    data = arrays[rename_prefix["data"]]
    data["index"] = ak.local_index(data)
    # Only add the mass if either mass or energy aren't already present
    if "m" not in ak.fields(data) and "E" not in ak.fields(data):
        data["m"] = data["pt"] * 0 + mass_hypotheses["data"]
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
                for k, v in zip(ak.fields(arrays), ak.unzip(arrays))
                if k not in _prefixes + [rename_prefix["data"]]
            },
        }
    )


def normalize_for_MC(
    arrays: ak.Array,
    rename_prefix: Optional[Mapping[str, str]] = None,
    mass_hypothesis: Union[float, Mapping[str, float]] = 0.139,
    particle_columns: Optional[Mapping[str, npt.DTypeLike]] = None,
) -> ak.Array:
    """Transform into a form appropriate for MC analysis.

    Args:
        arrays: Input arrays
        rename_prefix: Prefix to label the data, and any mapping that we might need to perform. Note: the mapping
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
    # Since we require the rename_prefix to define what prefixes to work with, if it's passed as an
    # empty mapping, we should treat it as is None was actually passed.
    if rename_prefix is None or not rename_prefix:
        rename_prefix = {
            "part_level": "part_level",
            "det_level": "det_level",
        }
    _prefixes = list(rename_prefix.keys())
    if isinstance(mass_hypothesis, (int, float)):
        mass_hypotheses = {p: float(mass_hypothesis) for p in _prefixes}
    else:
        mass_hypotheses = dict(mass_hypothesis)

    # Transform various track collections.
    # 1) Add indices.
    # 2) Complete the four vectors (as necessary).
    det_level = arrays[rename_prefix["det_level"]]
    det_level["index"] = ak.local_index(det_level)
    if "m" not in ak.fields(det_level) and "E" not in ak.fields(det_level):
        det_level["m"] = det_level["pt"] * 0 + mass_hypotheses["det_level"]
    # NOTE: This is fully equivalent because we registered vector:
    #       >>> det_level = ak.with_name(det_level, name="Momentum4D")
    det_level = vector.Array(det_level)
    # Part level
    part_level = arrays[rename_prefix["part_level"]]
    part_level["index"] = ak.local_index(part_level)
    if "m" not in ak.fields(part_level) and "E" not in ak.fields(part_level):
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
            **{k: v for k, v in zip(ak.fields(arrays), ak.unzip(arrays)) if k not in _prefixes},
        }
    )


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
            yield normalize_for_MC(arrays=arrays, rename_prefix=rename_prefix)

        # If not pythia, we don't need to handle it separately - it's all just data
        # All the rest of the collision systems would be embedded together separately by other functions
        logger.info("Transforming as data")
        yield normalize_for_data(arrays=arrays, rename_prefix=rename_prefix)


def data(
    data_input: Union[Path, Sequence[Path]],
    data_source: sources.SourceFromFilename,
    collision_system: str,
    rename_prefix: Mapping[str, str],
    chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.FULL_SOURCE,
) -> Union[ak.Array, Iterable[ak.Array]]:
    """Load data for ALICE analysis from the track skim task output.

    Could come from a ROOT file or a converted parquet file.

    Args:
        data_input: Filenames containing the data.
        data_source: Data source to be used to load the data stored at the filenames.
        collision_system: Collision system corresponding to the data to load.
        rename_prefix: Prefix to label the data, and any mapping that we might need to perform. Note: the mapping
            goes from value -> key!
        chunk_size: Chunk size to use when loading the data. Default: Full source.
    Returns:
        The loaded data, transformed as appropriate based on the collision system
    """
    # Validation
    if "embed" in collision_system:
        raise ValueError("This function doesn't handle embedding. Please call the dedicated functions.")
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
        rename_prefix=rename_prefix,
    )
    return (
        _transform_data_iter if chunk_size is not sources.ChunkSizeSentinel.FULL_SOURCE else next(_transform_data_iter)
    )


def normalize_for_embedding(
    arrays: ak.Array,
    source_index_identifiers: Mapping[str, int],
    mass_hypothesis: Union[float, Mapping[str, float]] = 0.139,
    particle_columns: Optional[Mapping[str, npt.DTypeLike]] = None,
    fixed_background_index_value: Optional[int] = None,
) -> ak.Array:
    """Transform into a form appropriate for embedding.

    Note:
        This performs embedding in the process of transforming.

    Args:
        arrays: Input arrays
        source_index_identifiers: Index offset map for each source.
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
    if isinstance(mass_hypothesis, (int, float)):
        mass_hypotheses = {p: float(mass_hypothesis) for p in _mass_hypothesis_prefixes}
    else:
        mass_hypotheses = dict(mass_hypothesis)

    # Transform various track collections.
    # 1) Add indices.
    # 2) Complete the four vectors (as necessary).
    det_level = arrays["signal"]["det_level"]
    det_level["index"] = ak.local_index(det_level) + source_index_identifiers["signal"]
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
    #       matched by this index, but rather by `label`.
    part_level["index"] = ak.local_index(part_level) + source_index_identifiers["signal"]
    if "m" not in ak.fields(part_level) and "E" not in ak.fields(part_level):
        # Since we have truth level info, construct the part level mass based on the particle_ID
        # rather than a fixed mass hypothesis.
        # NOTE: At this point, the input data should have been normalized to use "particle_ID" for
        #       the particle ID column name, so we shouldn't need to change the column name here.
        part_level["m"] = particle_ID.particle_masses_from_particle_ID(arrays=part_level)
    part_level = vector.Array(part_level)
    background = arrays["background"]["data"]
    if fixed_background_index_value is not None:
        background["index"] = ak.local_index(background) * 0 + fixed_background_index_value
    else:
        background["index"] = ak.local_index(background) + source_index_identifiers["background"]
    if "m" not in ak.fields(background) and "E" not in ak.fields(background):
        background["m"] = background["pt"] * 0 + mass_hypotheses["background"]
    background = vector.Array(background)

    # Combine inputs
    logger.debug("Embedding...")
    return ak.Array(
        {
            "part_level": part_level,
            "det_level": det_level,
            # Practically, this is where we are performing the embedding
            # Need to re-zip so it applies the vector at the same level as the other collections
            # (ie. we want `var * Momentum4D[...]`, but without the zip, we have `Momentum4D[var * ...]`)
            # NOTE: For some reason, ak.concatenate returns float64 here. I'm not sure why, but for now
            #       it's not diving into.
            "hybrid": vector.zip(
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
                    )
                )
            ),
            # Include the rest of the non particle related fields (ie. event level info)
            **{
                k: v
                for k, v in zip(ak.fields(arrays["signal"]), ak.unzip(arrays["signal"]))
                if k not in ["det_level", "part_level"]
            },
        }
    )


def _event_select_and_transform_embedding(
    gen_data: sources.T_GenData,
    source_index_identifiers: Mapping[str, int],
    use_alice_standard_event_selection_on_background: bool = True,
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
        if use_alice_standard_event_selection_on_background:
            # Use delayed import here. It's admittedly odd to import from the alice module, but it's super
            # convenient here, so we just run with it.
            from mammoth.alice import helpers as alice_helpers

            background_event_selection = alice_helpers.standard_event_selection(arrays["background"], return_mask=True)
        else:
            background_event_selection = np.ones(len(arrays)) > 0

        # Finally, apply the masks
        arrays = arrays[(mask & background_event_selection)]

        logger.info("Transforming embedded")
        yield normalize_for_embedding(arrays=arrays, source_index_identifiers=source_index_identifiers)


def embedding(
    signal_input: Union[Path, Sequence[Path]],
    signal_source: sources.SourceFromFilename,
    background_input: Union[Path, Sequence[Path]],
    background_source: sources.SourceFromFilename,
    chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.FULL_SOURCE,
    repeat_unconstrained_when_needed_for_statistics: bool = True,
    background_is_constrained_source: bool = True,
    use_alice_standard_event_selection_on_background: bool = True,
) -> Union[Tuple[Dict[str, int], ak.Array], Tuple[Dict[str, int], Iterable[ak.Array]]]:
    """Load data for embedding.

    Note:
        The signal and background sources are only constructed with the Path. If you need
        to pass additional arguments, you can do so by defining a closure around the source.
        For the simplest examples, it could be something like:

        ```python
        signal_source=functools.partial(track_skim.FileSource, collision_system="pythia")
        background_source=functools.partial(track_skim.FileSource, collision_system="PbPb")
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
    pythia_source_kwargs: Dict[str, Any] = {}
    pbpb_source_kwargs: Dict[str, Any] = {}
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
        use_alice_standard_event_selection_on_background=use_alice_standard_event_selection_on_background,
    )
    return (
        source_index_identifiers,
        _transform_data_iter if chunk_size is not sources.ChunkSizeSentinel.FULL_SOURCE else next(_transform_data_iter),
    )
