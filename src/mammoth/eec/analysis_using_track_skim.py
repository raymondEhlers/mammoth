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


def output_path_hist(output_filename: Path) -> Path:
    return output_filename.parent / "hists" / output_filename.with_suffix(".root").name


def output_path_skim(output_filename: Path, skim_name: str) -> Path:
    return output_filename.parent / skim_name / output_filename.name


def check_for_root_skim_output_file(output_filename: Path, reference_tree_name: str = "tree") -> tuple[bool, str]:
    # Try to bail out as early to avoid reprocessing if possible.
    # First, check for the empty filename
    empty_filename = output_filename.with_suffix(".empty")
    if empty_filename.exists():
        # It will be empty, so there's nothing to check. Just return
        return (True, "Done - found empty file indicating that there are no tree outputs after analysis")

    # Next, check the contents of the output file
    if output_filename.exists():
        if reference_tree_name:
            try:
                with uproot.open(output_filename) as f:
                    # If the tree exists, can be read, and has more than 0 entries, we should be good
                    if f[reference_tree_name].num_entries > 0:
                        # Return immediately to indicate that we're done.
                        return (True, f"already processed (confirmed)")
            except Exception:
                # If it fails for some reason, give up - we want to try again
                pass
        else:
            return (True, "already processed (no reference tree name provided, but file exists)")

    return (False, "")


def check_for_parquet_skim_output_file(output_filename: Path, reference_array_name: str = "") -> tuple[bool, str]:
    # Try to bail out as early to avoid reprocessing if possible.
    # First, check for the empty filename
    empty_filename = output_filename.with_suffix(".empty")
    if empty_filename.exists():
        # It will be empty, so there's nothing to check. Just return
        return (True, "Done - found empty file indicating that there are no array outputs after analysis")

    # Next, check the contents of the output file
    if output_filename.exists():
        if reference_array_name:
            try:
                arrays = ak.from_parquet(output_filename)
                # If the reference array exists, can be read, and has more than 0 entries, we have analyzed successfully
                # and don't need to do it again
                if ak.num(arrays[reference_array_name], axis=0) > 0:
                    # Return immediately to indicate that we're done.
                    return (True, "already processed (confirmed)")
            except Exception:
                # If it fails for some reason, give up - we want to try again
                pass
        else:
            return (True, "already processed (no reference array name provided, but file exists)")

    return (False, "")


def check_for_hist_output_file(output_filename: Path, reference_hist_name: str = "") -> tuple[bool, str]:
    # Try to bail out as early to avoid reprocessing if possible.
    # First, check for the empty filename
    empty_filename = output_filename.with_suffix(".empty")
    if empty_filename.exists():
        # It will be empty, so there's nothing to check. Just return
        return (True, "Done - found empty file indicating that there are no hists after analysis")

    # Next, check the contents of the output file
    if output_filename.exists():
        if reference_hist_name:
            try:
                with uproot.open(output_filename) as f:
                    # If the tree exists, can be read, and has more than 0 entries, we should be good
                    if ak.any(f[reference_hist_name].values() > 0):
                        # Return immediately to indicate that we're done.
                        return (True, "already processed (confirmed)")
            except Exception:
                # If it fails for some reason, give up - we want to try again
                pass
        else:
            return (True, "already processed (no reference hist name provided, but file exists)")

    return (False, "")


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
        res = check_for_parquet_skim_output_file(
            output_filename=output_path_skim(output_filename=output_filename, skim_name="hybrid_reference"),
            reference_array_name="triggers",
        )
    else:
        res = check_for_hist_output_file(
            output_filename=output_path_hist(output_filename=output_filename),
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
            res = check_for_parquet_skim_output_file(
                output_filename=output_path_skim(output_filename=_output_filename, skim_name="hybrid_reference"),
                reference_array_name="hybrid_reference",
            )
        else:
            res = check_for_hist_output_file(
                output_filename=output_path_hist(output_filename=_output_filename),
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
                output_trigger_skim=output_trigger_skim,
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
                _skim_output_filename = output_path_skim(output_filename=_output_filename, skim_name=skim_name)
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
        output_hist_filename = output_path_hist(output_filename=output_filename)
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


# TODO: Need to finish prototyping how I'm actually going to do this...

class FailedToSetupSourceError(Exception):
    """Failed to setup the input source for the task.

    Attributes:
        result_success: Whether the task was successful.
        result_message: Message describing the result.
    """
    def __init__(self, result_success: bool, result_message: str):
        self.result_success = result_success
        self.result_message = result_message

    def __iter__(self) -> tuple[bool, str]:
        return (self.result_success, self.result_message)

    def __str__(self) -> str:
        return f"{type(self).__name__}(result_success={self.result_success}, result_message={self.result_message})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(result_success={self.result_success}, result_message={self.result_message})"


def check_for_task_output(
    output_options: OutputOptions,
    chunk_size: sources.T_ChunkSize | None = None,
) -> tuple[bool, str]:
    """ Check for outputs to skip processing early if possible.

    Args:
        output_options: Output options.
        chunk_size: Chunk size for processing.

    Returns:
        Tuple of (skip_processing, message).
    """
    # Default to assuming that there's no output, and therefore we'll need to process again
    res = (False, "")

    # Try to bail out early to avoid reprocessing if possible.
    # Conditions for when we'll want to check here:
    #   1. We're processing in a single chunk (ie. we're using a single file input or the full input source)
    #   OR
    #   2. We write merged hists, which means we have a meaningful output without a chunk index in the filename
    #   OR
    #   3. Chunk size was not provided, so we can't figure out what state we're in, so we just go ahead and always check.
    # NOTE: We have to exercise a bit of care here in the case that have chunk sizes smaller than an individual file.
    #       In that case, the first file could be empty, but later chunks may not be empty. To avoid that case, we only
    #       check when there is meaningful output, as defined by the two conditions above.
    if chunk_size is None or (chunk_size == sources.ChunkSizeSentinel.SINGLE_FILE or chunk_size == sources.ChunkSizeSentinel.FULL_SOURCE) or output_options.write_merged_hists:
        if output_options.primary_output.type == "skim":
            if output_options.output_filename.suffix == ".root":
                res = check_for_root_skim_output_file(
                    output_filename=output_path_skim(output_filename=output_options.output_filename, skim_name=output_options.primary_output.name),
                    reference_tree_name=output_options.primary_output.reference_name,
                )
            else:
                res = check_for_parquet_skim_output_file(
                    output_filename=output_path_skim(output_filename=output_options.output_filename, skim_name=output_options.primary_output.name),
                    reference_array_name=output_options.primary_output.reference_name,
                )
        if output_options.primary_output.type == "hists":
            res = check_for_hist_output_file(
                output_filename=output_path_hist(output_filename=output_options.output_filename),
                reference_hist_name=output_options.primary_output.reference_name,
            )

    return res


"""

Concept:
    task: Corresponds to one unit (eg. file)
    analysis: Corresponds to analysis of one chunk (eg. one file, or one chunk of a file)

"""

from functools import partial

def setup_source_for_embedding(
    *,
    # Task settings
    task_settings: TaskSettings,
    # Inputs
    signal_input: Path | Sequence[Path],
    signal_source: sources.SourceFromFilename | sources.DelayedSource,
    background_input: Path | Sequence[Path],
    background_source: sources.SourceFromFilename | sources.DelayedSource,
    background_is_constrained_source: bool,
    # Outputs
    output_options: OutputOptions,
) -> tuple[dict[str, int], Iterator[ak.Array]]:
    """ Setup embed MC source for a analysis task.

    Raises:
        FailedToSetupSourceError: If the source could not be setup.
    """
    # Validation
    signal_input_filenames = []
    if not isinstance(signal_input, collections.abc.Iterable):
        signal_input_filenames = [signal_input]
    else:
        signal_input_filenames = list(signal_input)
    background_input_filenames = []
    if not isinstance(background_input, collections.abc.Iterable):
        background_input_filenames = [background_input]
    else:
        background_input_filenames = list(background_input)

    res = check_for_task_output(
        output_options=output_options,
        chunk_size=task_settings.chunk_size
    )
    if res[0]:
        raise FailedToSetupSourceError(result_success=res[0], result_message=res[1])

    # Setup iteration over the input files
    # If we don't use a processing chunk size, it should all be done in one chunk by default.
    # However, the memory usage often gets too large if the signal is the constrained source,
    # so this allows us to control the overall memory size by breaking it up into chunks,
    # such that we only load the data chunks that's currently needed for processing.
    # This is a bit idealistic because we often need to load the full file, but at least it sets
    # for potential improvements
    try:
        source_index_identifiers, iter_arrays = load_data.embedding(
            signal_input=signal_input_filenames,
            signal_source=partial(signal_source, collision_system="pythia"),
            background_input=background_input_filenames,
            background_source=partial(background_source, collision_system="PbPb"),
            background_is_constrained_source=background_is_constrained_source,
            chunk_size=task_settings.chunk_size,
        )
    except sources.NoDataAvailableError as e:
        # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
        # Remember that this depends heavily on the jet pt cuts!
        output_options.output_filename.with_suffix(".empty").touch()
        raise FailedToSetupSourceError(
            result_success=True,
            result_message=f"Done - no data available (reason: {e}), so not trying to skim",
        ) from None

    # Validate that the arrays are in an a format that we can iterate over
    if isinstance(iter_arrays, ak.Array):
        iter_arrays = iter([iter_arrays])
    assert not isinstance(iter_arrays, ak.Array), "Check configuration. This should be an iterable, not an ak.Array!"

    return source_index_identifiers, iter_arrays


def setup_source_for_embedding_thermal_model(
    *,
    # Task settings
    task_settings: TaskSettings,
    # Inputs
    signal_input: Path | Sequence[Path],
    signal_source: sources.SourceFromFilename | sources.DelayedSource,
    thermal_model_parameters: sources.ThermalModelParameters,
    # Outputs
    output_options: OutputOptions,
) -> tuple[dict[str, int], Iterator[ak.Array]]:
    """ Setup embed MC source for a analysis task.

    Raises:
        FailedToSetupSourceError: If the source could not be setup.
    """
    # Validation
    signal_input_filenames = []
    if not isinstance(signal_input, collections.abc.Iterable):
        signal_input_filenames = [signal_input]
    else:
        signal_input_filenames = list(signal_input)

    res = check_for_task_output(
        output_options=output_options,
        chunk_size=task_settings.chunk_size
    )
    if res[0]:
        raise FailedToSetupSourceError(result_success=res[0], result_message=res[1])

    # Setup iteration over the input files
    # If we don't use a processing chunk size, it should all be done in one chunk by default.
    # However, the memory usage often gets too large, so this allows us to control the overall memory
    # size by breaking it up into chunks, such that we only generate the thermal model chunk
    # that's currently needed for processing
    try:
        source_index_identifiers, iter_arrays = load_data.embedding_thermal_model(
            signal_input=signal_input_filenames,
            signal_source=partial(signal_source, collision_system="pythia"),
            thermal_model_parameters=thermal_model_parameters,
            chunk_size=task_settings.chunk_size,
        )
    except sources.NoDataAvailableError as e:
        # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
        # Remember that this depends heavily on the jet pt cuts!
        output_options.output_filename.with_suffix(".empty").touch()
        raise FailedToSetupSourceError(
            result_success=True,
            result_message=f"Done - no data available (reason: {e}), so not trying to skim",
        )

    # Validate that the arrays are in an a format that we can iterate over
    if isinstance(iter_arrays, ak.Array):
        iter_arrays = iter([iter_arrays])
    assert not isinstance(iter_arrays, ak.Array), "Check configuration. This should be an iterable, not an ak.Array!"

    return source_index_identifiers, iter_arrays


# TODO: Double wrap - we want to be able to pass the full set of arguments to the function, and have it pass on the analysis arguments
#@embedding_task(preprocess_arguemnts=_my_preprocessing_func, parameters=_my_parameters_func)
#def embed_analysis(
#    analysis_arguments,
#) -> ...:
#    ...

def steer_embed_task(
    task_settings: TaskSettings,
    #####
) -> framework_task.Output:
    # Validation
    if "embed" not in task_settings.collision_system:
        msg = f"Trying to use embedding steering with wrong collision system {task_settings.collision_system}"
        raise RuntimeError(msg)

    # TODO: Parametrize this
    # Setup
    _parameters = {
        "collision_system": task_settings.collision_system,
        "R": jet_R,
        "signal_input_filenames": str([str(_filename) for _filename in signal_input_filenames]),
        "background_input_filename": str([str(_filename) for _filename in background_input_filenames]),
    }
    if chunk_size is not sources.ChunkSizeSentinel.FULL_SOURCE:
        _parameters["chunk_size"] = chunk_size
    _description = analysis_conventions.description_from_parameters(parameters=_parameters)
    # ENDTODO

    source_index_identifiers, iter_arrays = setup_source_for_embedding(task_settings=task_settings)

    return steer_embed_task_exeuction(
        task_settings=task_settings,
        source_index_identifiers=source_index_identifiers,
        iter_arrays=iter_arrays,
        output_options=output_options,
    )


from typing import Protocol

class EmbeddingAnalysis(Protocol):

    def __call__(self, source_index_identifiers: dict[str, int], arrays: ak.Array) -> AnalysisOutput:
        ...


def steer_embed_task_execution(
    *,
    task_settings: TaskSettings,
    #####
    # I/O arguments
    # Inputs
    source_index_identifiers: dict[str, int],
    iter_arrays: Iterator[ak.Array],
    # Outputs
    output_options: OutputOptions,
    # Analysis arguments
    analysis_func: EmbeddingAnalysis,
    # ...
    # trigger_pt_ranges: dict[str, tuple[float, float]],
    # min_track_pt: dict[str, float],
    # momentum_weight_exponent: int | float,
    # det_level_artificial_tracking_efficiency: float,
    # scale_factor: float,
    # Default analysis parameters
    validation_mode: bool = False,
) -> framework_task.Output:
    # Validation
    if output_options.return_skim and not (task_settings.chunk_size == sources.ChunkSizeSentinel.SINGLE_FILE or task_settings.chunk_size == sources.ChunkSizeSentinel.FULL_SOURCE):
        # NOTE: Returning the skim is only supported for the case where we're processing a single file or the full source
        msg = "Cannot return skim if processing in chunks. Update your output options."
        raise ValueError(msg)

    _nonstandard_results = []
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
            res = check_for_task_output(
                output_options=local_output_options,
            )
            if res[0]:
                _nonstandard_results.append(res)
                logger.info(f"Skipping already processed chunk {i_chunk}: {res}")
                continue

            # We know we need to process, so now it's time to actually grab the data!
            arrays = next(iter_arrays)

            try:
                analysis_output = analysis_func(
                    source_index_identifiers=source_index_identifiers,
                    arrays=arrays,
                )
                #analysis_output = analysis_alice.analysis_embedding(
                #    source_index_identifiers=source_index_identifiers,
                #    arrays=arrays,
                #    trigger_pt_ranges=trigger_pt_ranges,
                #    min_track_pt=min_track_pt,
                #    momentum_weight_exponent=momentum_weight_exponent,
                #    scale_factor=scale_factor,
                #    det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
                #    output_trigger_skim=output_skim,
                #    validation_mode=validation_mode,
                #)
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

            analysis_output.merge_hists(task_hists=task_hists)
            analysis_output.write(output_filename=local_output_options.output_filename)

            # Cleanup (may not be necessary, but it doesn't hurt)
            del arrays
            # We can't delete the analysis output if we're going to return the skim
            # (again, we can only do this if we analyze in one chunk)
            if not output_options.return_skim:
                del analysis_output

            # Setup for next loop
            i_chunk += 1
    except StopIteration:
        ...

    # Write hists
    if task_hists:
        output_hist_filename = output_path_hist(output_filename=output_options.output_filename)
        output_hist_filename.parent.mkdir(parents=True, exist_ok=True)
        with uproot.recreate(output_hist_filename) as f:
            output_utils.write_hists_to_file(hists=task_hists, f=f)

    # Cleanup
    if not output_options.return_merged_hists:
        del task_hists

    return framework_task.Output(
        production_identifier=task_settings.production_identifier,
        collision_system=task_settings.collision_system,
        success=True,
        message=f"success for {_description}"
        + (f". Additional non-standard results: {_nonstandard_results}" if _nonstandard_results else ""),
        hists=task_hists if output_options.return_merged_hists else {},
        results=analysis_output.skim if output_options.return_skim else {},
        metadata=output_metadata,
    )


#@attrs.frozen(kw_only=True)
#class InputOptions:
#    background_is_constrained_source: bool = True

@attrs.frozen(kw_only=True)
class OutputOptions:
    output_filename: Path
    primary_output: PrimaryOutput
    return_skim: bool | None = attrs.field(default=None)
    write_chunk_skim: bool | None = attrs.field(default=None)
    return_merged_hists: bool | None = attrs.field(default=None)
    write_chunk_hists: bool | None = attrs.field(default=None)
    write_merged_hists: bool | None = attrs.field(default=None)

    def with_new_output_filename(self, new_output_filename: Path) -> OutputOptions:
        return type(self)(
            output_filename=new_output_filename,
            primary_output=self.primary_output,
            return_skim=self.return_skim,
            write_chunk_skim=self.write_chunk_skim,
            return_merged_hists=self.return_merged_hists,
            write_chunk_hists=self.write_chunk_hists,
            write_merged_hists=self.write_merged_hists,
        )

@attrs.frozen(kw_only=True)
class PrimaryOutput:
    type: str = attrs.field(validator=[attrs.validators.in_(["hists", "skim"])])
    reference_name: str
    name: str = attrs.field(default="")

@attrs.frozen(kw_only=True)
class TaskSettings:
    production_identifier: str
    collision_system: str
    chunk_size: sources.T_ChunkSize

import uproot

from mammoth.framework.io import output_utils

import attrs
@attrs.frozen(kw_only=True)
class AnalysisOutput:
    hists: dict[str, hist.Hist] = attrs.field(factory=dict)
    write_hists: bool | None = attrs.field(default=None)
    skim: ak.Array | dict[str, ak.Array] = attrs.field(factory=dict)
    write_skim: bool | None = attrs.field(default=None)

    def _write_hists(self, output_filename: Path) -> None:
        output_hist_filename = output_path_hist(output_filename=output_filename)
        output_hist_filename.parent.mkdir(parents=True, exist_ok=True)
        with uproot.recreate(output_hist_filename) as f:
            output_utils.write_hists_to_file(hists=self.hists, f=f)

    def _write_skim(self, output_filename: Path, skim: dict[str, ak.Array]) -> None:
        for skim_name, skim_array in skim.items():
            # Enables the possibility of writing a single standard file (just wrap in a dict with an empty string as key).
            if skim_name != "":
                _skim_output_filename = output_path_skim(output_filename=output_filename, skim_name=skim_name)
            else:
                _skim_output_filename = output_filename

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

    def write(self, output_filename: Path) -> None:
        # If not specified, fall back to the default, which is to write the analysis outputs if they're provided
        write_hists = self.write_hists
        if write_hists is None:
            write_hists = bool(self.hists)
        write_skim = self.write_skim
        if write_skim is None:
            write_skim = bool(self.skim)

        # Validation
        if isinstance(self.skim, ak.Array):
            skim = {"": self.skim}
        else:
            skim = self.skim

        # Write if requested
        if write_hists:
            self._write_hists(output_filename=output_filename)

        if write_skim:
            self._write_skim(output_filename=output_filename, skim=skim)

    def merge_hists(self, task_hists: dict[str, hist.Hist]) -> None:
        # No point in trying to merge if there are no hists!
        if self.hists:
            task_hists = output_utils.merge_results(task_hists, self.hists)


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
        res = check_for_parquet_skim_output_file(
            output_filename=output_path_skim(output_filename=output_filename, skim_name="hybrid_reference"),
            reference_array_name="triggers",
        )
    else:
        res = check_for_hist_output_file(
            output_filename=output_path_hist(output_filename=output_filename),
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
            res = check_for_parquet_skim_output_file(
                output_filename=output_path_skim(output_filename=_output_filename, skim_name="hybrid_reference"),
                reference_array_name="hybrid_reference",
            )
        else:
            res = check_for_hist_output_file(
                output_filename=output_path_hist(output_filename=_output_filename),
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
                output_trigger_skim=output_skim,
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
                _skim_output_filename = output_path_skim(output_filename=_output_filename, skim_name=skim_name)
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
        output_hist_filename = output_path_hist(output_filename=output_filename)
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
