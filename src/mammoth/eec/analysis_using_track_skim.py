"""Running EEC analysis on track skim

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Any, Iterator, Sequence

import awkward as ak
import hist
import uproot

import mammoth.helpers
from mammoth.eec import analysis_alice
from mammoth.framework import load_data, sources
from mammoth.framework import task as framework_task
from mammoth.framework.analysis import conventions as analysis_conventions
from mammoth.framework.io import output_utils
from mammoth.framework.io import track_skim

logger = logging.getLogger(__name__)


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
    _description = analysis_conventions.description_from_parameters(parameters=_parameters)
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

        trigger_skim: dict[str, ak.Array] | None = None
        if output_trigger_skim:
            assert not isinstance(analysis_output, dict)
            analysis_hists, trigger_skim = analysis_output
        else:
            assert isinstance(analysis_output, dict)
            analysis_hists = analysis_output

        # Merge the output hists
        if analysis_hists:
            from mammoth import job_utils

            hists = output_utils.merge_results(hists, analysis_hists)

        if trigger_skim:
            for skim_name, skim_array in trigger_skim.items():
                _skim_output_filename = output_utils.task_output_path_skim(output_filename=_output_filename, skim_name=skim_name)
                _skim_output_filename.parent.mkdir(parents=True, exist_ok=True)
                if ak.num(skim_array, axis=0) == 0:
                    # Skip the skim if it's empty
                    _skim_output_filename.with_suffix(".empty").touch()
                else:
                    # Write the skim
                    ak.to_parquet(
                        array=skim_array,
                        destination=str(_skim_output_filename),
                        compression="zstd",
                        # Optimize for columns with anything other than floats
                        parquet_dictionary_encoding=True,
                        # Optimize for columns with floats
                        parquet_byte_stream_split=True,
                    )

        # Cleanup (may not be necessary, but it doesn't hurt)
        del arrays
        del analysis_hists
        del analysis_output
        if trigger_skim:
            del trigger_skim

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


# New code starting from here


#
#
#def test_func(test_f: SetupSource) -> None:
#
#    a, b, c = test_f(task_settings=task_settings, output_options=output_options)

"""

Concept:
    task: Corresponds to one unit (eg. file)
    analysis: Corresponds to analysis of one chunk (eg. one file, or one chunk of a file)

"""

from functools import partial

# Copied from load_data - it's probably redundant...
def _validate_potential_list_of_inputs(inputs: Path | Sequence[Path]) -> list[Path]:
    return [inputs] if not isinstance(inputs, collections.abc.Iterable) else list(inputs)



import uproot

from mammoth.framework.io import output_utils


from typing import Protocol


def steer_embed_task_execution(
    *,
    task_settings: framework_task.Settings,
    task_metadata: framework_task.Metadata,
    #####
    # I/O arguments
    # Inputs
    source_index_identifiers: dict[str, int],
    iter_arrays: Iterator[ak.Array],
    # Outputs
    output_options: framework_task.OutputSettings,
    # Analysis arguments
    analysis_function: framework_task.EmbeddingAnalysis,
    # ...
    # trigger_pt_ranges: dict[str, tuple[float, float]],
    # min_track_pt: dict[str, float],
    # momentum_weight_exponent: int | float,
    # det_level_artificial_tracking_efficiency: float,
    # scale_factor: float,
    # ...
    # Default analysis parameters
    validation_mode: bool,
) -> framework_task.Output:
    # Validation
    if output_options.return_skim and not (task_settings.chunk_size == sources.ChunkSizeSentinel.SINGLE_FILE or task_settings.chunk_size == sources.ChunkSizeSentinel.FULL_SOURCE):
        # NOTE: Returning the skim is only supported for the case where we're processing a single file or the full source
        msg = "Cannot return skim if processing in chunks. Update your output options."
        raise ValueError(msg)

    _nonstandard_processing_outcome = []
    task_hists: dict[str, hist.Hist] = {}
    i_chunk = 0
    # NOTE: We use a while loop here so that we can bail out on processing before we even read the data if the file already exists.
    try:
        while True:
            # Setup
            # We need to identify the chunk in the output name
            # NOTE: To be consistent with expectations for a single chunk, the output name should only append the suffix
            #       if it's more than the first chunk
            if i_chunk > 0:
                _output_filename = (
                    output_options.output_filename.parent / f"{output_options.output_filename.stem}_chunk_{i_chunk:03}{output_options.output_filename.suffix}"
                )
            else:
                _output_filename = output_options.output_filename
            local_output_options = output_options.with_new_output_filename(_output_filename)

            # Try to bail out as early to avoid reprocessing if possible.
            res = framework_task.check_for_task_output(
                output_options=local_output_options,
            )
            if res[0]:
                _nonstandard_processing_outcome.append(res)
                logger.info(f"Skipping already processed chunk {i_chunk}: {res}")
                continue

            # We know we need to process, so now it's time to actually grab the data!
            arrays = next(iter_arrays)

            try:
                analysis_output = analysis_function(
                    arrays=arrays,
                    source_index_identifiers=source_index_identifiers,
                    validation_mode=validation_mode,
                    # Although return_skim is an output option that we try to abstract away, it can be quite costly in terms of memory.
                    # Consequently, we break the abstraction and pass it, since many tasks are under memory pressure.
                    # NOTE: Need to specify a concrete value here, so we default to False if nothing is specified.
                    return_skim=False if output_options.return_skim is None else output_options.return_skim,
                )
            except sources.NoDataAvailableError as e:
                # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
                # Remember that this depends heavily on the jet pt cuts!
                _output_filename.with_suffix(".empty").touch()
                _message = (
                    True,
                    f"Chunk {i_chunk}: Done - no data available (reason: {e}), so not trying to skim",
                )
                _nonstandard_processing_outcome.append(_message)
                logger.info(_message)
                continue

            analysis_output.merge_hists(task_hists=task_hists)
            analysis_output.write(
                output_filename=local_output_options.output_filename,
                write_hists=local_output_options.write_chunk_hists,
                write_skim=local_output_options.write_chunk_skim,
            )

            # Cleanup (may not be necessary, but it doesn't hurt)
            del arrays
            # We can't delete the analysis output if we're going to return the skim
            # (again, we can only do this if we analyze in one chunk)
            if not output_options.return_skim:
                del analysis_output

            # Setup for next loop
            i_chunk += 1
    except StopIteration:
        # All done!
        ...

    # Write hists
    if output_options.write_merged_hists:
        # Cross check that we have something to write
        assert task_hists

        # And then actually write it
        output_hist_filename = output_utils.task_output_path_hist(output_filename=output_options.output_filename)
        output_hist_filename.parent.mkdir(parents=True, exist_ok=True)
        with uproot.recreate(output_hist_filename) as f:
            output_utils.write_hists_to_file(hists=task_hists, f=f)

    # Cleanup
    if not output_options.return_merged_hists:
        del task_hists

    description, output_metadata = framework_task.description_and_output_metadata(task_metadata=task_metadata)
    return framework_task.Output(
        production_identifier=task_settings.production_identifier,
        collision_system=task_settings.collision_system,
        success=True,
        message=f"success for {description}"
        + (f". Additional non-standard processing outcomes: {_nonstandard_processing_outcome}" if _nonstandard_processing_outcome else ""),
        hists=task_hists if output_options.return_merged_hists else {},
        results=analysis_output.skim if output_options.return_skim else {},
        metadata=output_metadata,
    )

python_app_embed_MC_into_thermal_model = framework_task.python_app_embed_MC_into_thermal_model(
    analysis=analysis_alice.analysis_embedding,
)

# TODO: Double wrap - we want to be able to pass the full set of arguments to the function, and have it pass on the analysis arguments
#@embedding_task(preprocess_arguments=_my_preprocessing_func, parameters=_my_parameters_func)
#def embed_analysis(
#    analysis_arguments,
#) -> ...:
#    ...


def steer_embed_task(
    *,
    # Task settings
    task_settings: framework_task.Settings,
    # I/O
    setup_input_source: framework_task.SetupEmbeddingSource,
    output_options: framework_task.OutputSettings,
    # Analysis
    # NOTE: The analysis arguments are bound to both of these functions before passing here
    analysis_function: framework_task.EmbeddingAnalysis,
    analysis_metadata_function: framework_task.CustomizeAnalysisMetadata,
    # We split these argument out to ensure that they're explicitly supported
    validation_mode: bool,
) -> framework_task.Output:
    # Validation
    if "embed" not in task_settings.collision_system:
        msg = f"Trying to use embedding steering with wrong collision system {task_settings.collision_system}"
        raise RuntimeError(msg)

    # Description parameters
    task_metadata: framework_task.Metadata = {
        "collision_system": task_settings.collision_system,
    }
    if task_settings.chunk_size is not sources.ChunkSizeSentinel.FULL_SOURCE:
        task_metadata["chunk_size"] = task_settings.chunk_size
    # Add in the customized analysis parameters
    task_metadata.update(analysis_metadata_function(task_settings=task_settings))

    # NOTE: Always call `description_and_output_metadata` right before they're needed to ensure they
    #       up to date since they may change

    try:
        source_index_identifiers, iter_arrays = setup_input_source(
            task_settings=task_settings,
            output_options=output_options,
            task_metadata=task_metadata,
        )
    except framework_task.FailedToSetupSourceError as e:
        # Source wasn't suitable for some reason - bail out.
        _, output_metadata = framework_task.description_and_output_metadata(task_metadata=task_metadata)
        return framework_task.Output(
            production_identifier=task_settings.production_identifier,
            collision_system=task_settings.collision_system,
            success=e.result_success,
            message=e.result_message,
            metadata=output_metadata,
        )

    return steer_embed_task_execution(
        task_settings=task_settings,
        task_metadata=task_metadata,
        source_index_identifiers=source_index_identifiers,
        iter_arrays=iter_arrays,
        output_options=output_options,
        analysis_function=analysis_function,
        validation_mode=validation_mode,
    )


########################
# All good through here!
########################



#def run_embedding(
#    #prod: production.ProductionSettings,
#    #job_framework: job_utils.JobFramework,
#    #debug_mode: bool,
#) -> :
#    ...
#    return source = partial







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
    _description = analysis_conventions.description_from_parameters(parameters=_parameters)
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

        trigger_skim: dict[str, ak.Array] | None = None
        if output_skim:
            assert not isinstance(analysis_output, dict)
            analysis_hists, trigger_skim = analysis_output
        else:
            assert isinstance(analysis_output, dict)
            analysis_hists = analysis_output

        # Merge the output hists
        if analysis_hists:
            from mammoth import job_utils

            hists = output_utils.merge_results(hists, analysis_hists)

        if trigger_skim:
            for skim_name, skim_array in trigger_skim.items():
                _skim_output_filename = output_utils.task_output_path_skim(output_filename=_output_filename, skim_name=skim_name)
                _skim_output_filename.parent.mkdir(parents=True, exist_ok=True)
                if ak.num(skim_array, axis=0) == 0:
                    # Skip the skim if it's empty
                    _skim_output_filename.with_suffix(".empty").touch()
                else:
                    # Write the skim
                    ak.to_parquet(
                        array=skim_array,
                        destination=str(_skim_output_filename),
                        compression="zstd",
                        # Optimize for columns with anything other than floats
                        parquet_dictionary_encoding=True,
                        # Optimize for columns with floats
                        parquet_byte_stream_split=True,
                    )

        # Cleanup (may not be necessary, but it doesn't hurt)
        del arrays
        del analysis_hists
        del analysis_output
        if trigger_skim:
            del trigger_skim

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
