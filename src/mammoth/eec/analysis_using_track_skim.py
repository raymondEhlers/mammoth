"""Running EEC analysis on track skim

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import collections.abc
import logging
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import awkward as ak
import hist
import uproot

from mammoth.eec import analysis_alice
from mammoth.framework import load_data, sources
from mammoth.framework import task as framework_task
from mammoth.framework.io import output_utils, track_skim

logger = logging.getLogger(__name__)


# TODO: Delete this entire file after confirming that our new version works!

def eec_embed_thermal_model_analysis(  # noqa: C901
    production_identifier: str,
    collision_system: str,
    signal_input: Path | Sequence[Path],
    trigger_pt_ranges: dict[str, tuple[float, float]],
    min_track_pt: dict[str, float],
    momentum_weight_exponent: int | float,
    det_level_artificial_tracking_efficiency: float,
    thermal_model_parameters: sources.ThermalModelParameters,
    output_filename: Path,
    scale_factor: float,
    chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.SINGLE_FILE,
    output_trigger_skim: bool = False,
    return_hists: bool = False,
    validation_mode: bool = False,
) -> framework_task.Output:
    # Validation
    signal_input_filenames = []
    if not isinstance(signal_input, collections.abc.Iterable):
        signal_input_filenames = [signal_input]
    else:
        signal_input_filenames = list(signal_input)

    # Setup
    _parameters: dict[str, Any] = {
        "collision_system": collision_system,
        "signal_input_filenames": str([str(_filename) for _filename in signal_input_filenames]),
    }
    if chunk_size is not sources.ChunkSizeSentinel.FULL_SOURCE:
        _parameters["chunk_size"] = chunk_size
    _description = framework_task.description_from_metadata(metadata=_parameters)
    output_metadata = {
        # Useful to have the summary as a string
        "description": _description,
        # but also useful to have programmatic access
        "parameters": _parameters,
    }

    # Try to bail out early to avoid reprocessing if possible.
    if (chunk_size == sources.ChunkSizeSentinel.SINGLE_FILE or chunk_size == sources.ChunkSizeSentinel.FULL_SOURCE) and output_trigger_skim:
        # We need to exercise a bit of care here in the case that have chunk sizes smaller than an individual file.
        # In that case, the first file could be empty, but later chunks may not be so. To avoid that case, we only
        # perform this check if we are using a single file or the full source.
        # NOTE: Use "hybrid_reference" as a proxy. Better to use reference than signal since we're more likely
        #       to have reference triggers.
        res = output_utils.check_for_task_parquet_skim_output_file(
            output_filename=output_utils.task_output_path_skim(output_filename=output_filename, skim_name="hybrid_reference"),
            reference_array_name="triggers",
        )
    else:
        res = output_utils.check_for_task_hist_output_file(
            output_filename=output_utils.task_output_path_hist(output_filename=output_filename),
            reference_hist_name="hybrid_reference_eec",
        )
    if res[0]:
        return framework_task.Output(
            production_identifier,
            collision_system,
            *res,
            metadata=output_metadata,
        )

    # Setup iteration over the input files
    # If we don't use a processing chunk size, it should all be done in one chunk by default.
    # However, the memory usage often gets too large, so this allows us to control the overall memory
    # size by breaking it up into chunks, such that we only generate the thermal model chunk
    # that's currently needed for processing
    try:
        source_index_identifiers, iter_arrays = load_data.embedding_thermal_model(
            signal_input=signal_input_filenames,
            signal_source=partial(track_skim.FileSource, collision_system="pythia"),
            thermal_model_parameters=thermal_model_parameters,
            chunk_size=chunk_size,
        )
    except sources.NoDataAvailableError as e:
        # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
        # Remember that this depends heavily on the jet pt cuts!
        output_filename.with_suffix(".empty").touch()
        return framework_task.Output(
            production_identifier, collision_system,
            True, f"Done - no data available (reason: {e}), so not trying to skim",
            metadata=output_metadata,
        )

    # Validate that the arrays are in an a format that we can iterate over
    if isinstance(iter_arrays, ak.Array):
        iter_arrays = iter([iter_arrays])
    assert not isinstance(iter_arrays, ak.Array), "Check configuration. This should be an iterable, not an ak.Array!"

    _nonstandard_results = []
    hists: dict[str, hist.Hist] = {}
    for i_chunk, arrays in enumerate(iter_arrays):
        # Setup
        # We need to identify the chunk in the output name
        # NOTE: To be consistent with expectations for a single chunk, the output name should only append the suffix
        #       if it's more than the first chunk
        if i_chunk > 0:
            _output_filename = (
                output_filename.parent / f"{output_filename.stem}_chunk_{i_chunk:03}{output_filename.suffix}"
            )
        else:
            _output_filename = output_filename

        # Try to bail out as early to avoid reprocessing if possible.
        if output_trigger_skim:
            res = output_utils.check_for_task_parquet_skim_output_file(
                output_filename=output_utils.task_output_path_skim(output_filename=_output_filename, skim_name="hybrid_reference"),
                reference_array_name="hybrid_reference",
            )
        else:
            res = output_utils.check_for_task_hist_output_file(
                output_filename=output_utils.task_output_path_hist(output_filename=_output_filename),
                reference_hist_name="hybrid_reference_eec",
            )
        if res[0]:
            _nonstandard_results.append(res)
            logger.info(f"Skipping already processed chunk {i_chunk}: {res}")
            continue

        try:
            analysis_output = analysis_alice.analysis_embedding(
                source_index_identifiers=source_index_identifiers,
                arrays=arrays,
                trigger_pt_ranges=trigger_pt_ranges,
                min_track_pt=min_track_pt,
                momentum_weight_exponent=momentum_weight_exponent,
                scale_factor=scale_factor,
                det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
                return_skim=output_trigger_skim,
                validation_mode=validation_mode,
            )
        except sources.NoDataAvailableError as e:
            # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
            # Remember that this depends heavily on the jet pt cuts!
            _output_filename.with_suffix(".empty").touch()
            _message = (
                True,
                f"Chunk {i_chunk}: Done - no data available (reason: {e}), so not trying to skim",
            )
            _nonstandard_results.append(_message)
            logger.info(_message)
            continue

        # Merge and store the output hists
        analysis_output.merge_hists(task_hists=hists)
        analysis_output.write(
            output_filename=_output_filename,
            write_hists=False,
            write_skim=output_trigger_skim,
        )

        # Cleanup (may not be necessary, but it doesn't hurt)
        del arrays
        if not output_trigger_skim:
            del analysis_output

    # Write hists
    if hists:
        output_hist_filename = output_utils.task_output_path_hist(output_filename=output_filename)
        output_hist_filename.parent.mkdir(parents=True, exist_ok=True)
        with uproot.recreate(output_hist_filename) as f:
            output_utils.write_hists_to_file(hists=hists, f=f)

    # Cleanup
    if not return_hists:
        del hists

    return framework_task.Output(
        production_identifier=production_identifier,
        collision_system=collision_system,
        success=True,
        message=f"success for {_description}"
        + (f". Additional non-standard results: {_nonstandard_results}" if _nonstandard_results else ""),
        hists=hists if return_hists else {},
        metadata=output_metadata,
    )



#########################
# Original/reference code
#########################

def steer_embed_thermal_model_analysis(  # noqa: C901
    production_identifier: str,
    collision_system: str,
    #####
    # I/O arguments
    file_source_with_args: sources.SourceFromFilename | sources.DelayedSource,
    signal_input: Path | Sequence[Path],
    thermal_model_parameters: sources.ThermalModelParameters,
    output_filename: Path,
    # More args, to be filled in
    # Bind this with a partial...
    # Analysis arguments:
    trigger_pt_ranges: dict[str, tuple[float, float]],
    min_track_pt: dict[str, float],
    momentum_weight_exponent: int | float,
    det_level_artificial_tracking_efficiency: float,
    scale_factor: float,
    ###
    chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.SINGLE_FILE,
    output_skim: bool = False,
    return_hists: bool = False,
    validation_mode: bool = False,
) -> framework_task.Output:
    # Validation
    signal_input_filenames = []
    if not isinstance(signal_input, collections.abc.Iterable):
        signal_input_filenames = [signal_input]
    else:
        signal_input_filenames = list(signal_input)

    # Setup
    _parameters: dict[str, Any] = {
        "collision_system": collision_system,
        "signal_input_filenames": str([str(_filename) for _filename in signal_input_filenames]),
    }
    if chunk_size is not sources.ChunkSizeSentinel.FULL_SOURCE:
        _parameters["chunk_size"] = chunk_size
    _description = framework_task.description_from_metadata(metadata=_parameters)
    output_metadata = {
        # Useful to have the summary as a string
        "description": _description,
        # but also useful to have programmatic access
        "parameters": _parameters,
    }

    # Try to bail out early to avoid reprocessing if possible.
    if (chunk_size == sources.ChunkSizeSentinel.SINGLE_FILE or chunk_size == sources.ChunkSizeSentinel.FULL_SOURCE) and output_skim:
        # We need to exercise a bit of care here in the case that have chunk sizes smaller than an individual file.
        # In that case, the first file could be empty, but later chunks may not be so. To avoid that case, we only
        # perform this check if we are using a single file or the full source.
        # NOTE: Use "hybrid_reference" as a proxy. Better to use reference than signal since we're more likely
        #       to have reference triggers.
        res = output_utils.check_for_task_parquet_skim_output_file(
            output_filename=output_utils.task_output_path_skim(output_filename=output_filename, skim_name="hybrid_reference"),
            reference_array_name="triggers",
        )
    else:
        res = output_utils.check_for_task_hist_output_file(
            output_filename=output_utils.task_output_path_hist(output_filename=output_filename),
            reference_hist_name="hybrid_reference_eec",
        )
    if res[0]:
        return framework_task.Output(
            production_identifier,
            collision_system,
            *res,
            metadata=output_metadata,
        )

    # Setup iteration over the input files
    # If we don't use a processing chunk size, it should all be done in one chunk by default.
    # However, the memory usage often gets too large, so this allows us to control the overall memory
    # size by breaking it up into chunks, such that we only generate the thermal model chunk
    # that's currently needed for processing
    try:
        source_index_identifiers, iter_arrays = load_data.embedding_thermal_model(
            signal_input=signal_input_filenames,
            signal_source=partial(file_source_with_args, collision_system="pythia"),
            thermal_model_parameters=thermal_model_parameters,
            chunk_size=chunk_size,
        )
    except sources.NoDataAvailableError as e:
        # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
        # Remember that this depends heavily on the jet pt cuts!
        output_filename.with_suffix(".empty").touch()
        return framework_task.Output(
            production_identifier, collision_system,
            True, f"Done - no data available (reason: {e}), so not trying to skim",
            metadata=output_metadata,
        )

    # Validate that the arrays are in an a format that we can iterate over
    if isinstance(iter_arrays, ak.Array):
        iter_arrays = iter([iter_arrays])
    assert not isinstance(iter_arrays, ak.Array), "Check configuration. This should be an iterable, not an ak.Array!"

    _nonstandard_results = []
    hists: dict[str, hist.Hist] = {}
    for i_chunk, arrays in enumerate(iter_arrays):
        # Setup
        # We need to identify the chunk in the output name
        # NOTE: To be consistent with expectations for a single chunk, the output name should only append the suffix
        #       if it's more than the first chunk
        if i_chunk > 0:
            _output_filename = (
                output_filename.parent / f"{output_filename.stem}_chunk_{i_chunk:03}{output_filename.suffix}"
            )
        else:
            _output_filename = output_filename

        # Try to bail out as early to avoid reprocessing if possible.
        if output_skim:
            res = output_utils.check_for_task_parquet_skim_output_file(
                output_filename=output_utils.task_output_path_skim(output_filename=_output_filename, skim_name="hybrid_reference"),
                reference_array_name="hybrid_reference",
            )
        else:
            res = output_utils.check_for_task_hist_output_file(
                output_filename=output_utils.task_output_path_hist(output_filename=_output_filename),
                reference_hist_name="hybrid_reference_eec",
            )
        if res[0]:
            _nonstandard_results.append(res)
            logger.info(f"Skipping already processed chunk {i_chunk}: {res}")
            continue

        try:
            analysis_output = analysis_alice.analysis_embedding(
                source_index_identifiers=source_index_identifiers,
                arrays=arrays,
                trigger_pt_ranges=trigger_pt_ranges,
                min_track_pt=min_track_pt,
                momentum_weight_exponent=momentum_weight_exponent,
                scale_factor=scale_factor,
                det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
                return_skim=output_skim,
                validation_mode=validation_mode,
            )
        except sources.NoDataAvailableError as e:
            # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
            # Remember that this depends heavily on the jet pt cuts!
            _output_filename.with_suffix(".empty").touch()
            _message = (
                True,
                f"Chunk {i_chunk}: Done - no data available (reason: {e}), so not trying to skim",
            )
            _nonstandard_results.append(_message)
            logger.info(_message)
            continue

        # Merge and store the output hists
        analysis_output.merge_hists(task_hists=hists)
        analysis_output.write(
            output_filename=_output_filename,
            write_hists=False,
            write_skim=output_skim,
        )

        # Cleanup (may not be necessary, but it doesn't hurt)
        del arrays
        if not output_skim:
            del analysis_output

    # Write hists
    if hists:
        output_hist_filename = output_utils.task_output_path_hist(output_filename=output_filename)
        output_hist_filename.parent.mkdir(parents=True, exist_ok=True)
        with uproot.recreate(output_hist_filename) as f:
            output_utils.write_hists_to_file(hists=hists, f=f)

    # Cleanup
    if not return_hists:
        del hists

    return framework_task.Output(
        production_identifier=production_identifier,
        collision_system=collision_system,
        success=True,
        message=f"success for {_description}"
        + (f". Additional non-standard results: {_nonstandard_results}" if _nonstandard_results else ""),
        hists=hists if return_hists else {},
        metadata=output_metadata,
    )
