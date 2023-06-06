"""Steering for tasks

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import functools
import logging
from typing import Iterator

import awkward as ak
import hist
import uproot

from mammoth.framework import sources
from mammoth.framework import task as framework_task
from mammoth.framework.io import output_utils

logger = logging.getLogger(__name__)


def steer_task_execution(
    *,
    task_settings: framework_task.Settings,
    task_metadata: framework_task.Metadata,
    #####
    # I/O arguments
    # Inputs
    iter_arrays: Iterator[ak.Array],
    # Outputs
    output_settings: framework_task.OutputSettings,
    #####
    # Analysis arguments
    analysis_function: framework_task.AnalysisBound,
    # Default (ie. one's which must be implemented) analysis parameters
    validation_mode: bool,
) -> framework_task.Output:
    # Validation
    if output_settings.return_skim and not (task_settings.chunk_size == sources.ChunkSizeSentinel.SINGLE_FILE or task_settings.chunk_size == sources.ChunkSizeSentinel.FULL_SOURCE):
        # NOTE: Returning the skim is only supported for the case where we're processing a single file or the full source
        msg = "Cannot return skim if processing in chunks. Update your Output settings."
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
                    output_settings.output_filename.parent / f"{output_settings.output_filename.stem}_chunk_{i_chunk:03}{output_settings.output_filename.suffix}"
                )
            else:
                _output_filename = output_settings.output_filename
            local_output_settings = output_settings.with_new_output_filename(_output_filename)

            # Try to bail out as early to avoid reprocessing if possible.
            res = framework_task.check_for_task_output(
                output_settings=local_output_settings,
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
                    validation_mode=validation_mode,
                    # Although return_skim is an output option that we try to abstract away, it can be quite costly in terms of memory.
                    # Consequently, we break the abstraction and pass it, since many tasks are under memory pressure.
                    # NOTE: Need to specify a concrete value here, so we default to False if nothing is specified.
                    return_skim=False if output_settings.return_skim is None else output_settings.return_skim,
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
                output_filename=local_output_settings.output_filename,
                write_hists=local_output_settings.write_chunk_hists,
                write_skim=local_output_settings.write_chunk_skim,
            )

            # Cleanup (may not be necessary, but it doesn't hurt)
            del arrays
            # We can't delete the analysis output if we're going to return the skim
            # (again, we can only do this if we analyze in one chunk)
            if not output_settings.return_skim:
                del analysis_output

            # Setup for next loop
            i_chunk += 1
    except StopIteration:
        # All done!
        ...

    # Write hists
    if output_settings.write_merged_hists:
        # Cross check that we have something to write
        assert task_hists

        # And then actually write it
        output_hist_filename = output_utils.task_output_path_hist(output_filename=output_settings.output_filename)
        output_hist_filename.parent.mkdir(parents=True, exist_ok=True)
        with uproot.recreate(output_hist_filename) as f:
            output_utils.write_hists_to_file(hists=task_hists, f=f)

    # Cleanup
    if not output_settings.return_merged_hists:
        del task_hists

    description, output_metadata = framework_task.description_and_output_metadata(task_metadata=task_metadata)
    return framework_task.Output(
        production_identifier=task_settings.production_identifier,
        collision_system=task_settings.collision_system,
        success=True,
        message=f"success for {description}"
        + (f". Additional non-standard processing outcomes: {_nonstandard_processing_outcome}" if _nonstandard_processing_outcome else ""),
        hists=task_hists if output_settings.return_merged_hists else {},
        results=analysis_output.skim if output_settings.return_skim else {},
        metadata=output_metadata,
    )


def _default_task_metadata(task_settings: framework_task.Settings) -> framework_task.Metadata:
    """Default task metadata.

    Args:
        task_settings: Task settings.
    Returns:
        Default task metadata.
    """
    task_metadata: framework_task.Metadata = {
        "collision_system": task_settings.collision_system,
    }
    if task_settings.chunk_size is not sources.ChunkSizeSentinel.FULL_SOURCE:
        task_metadata["chunk_size"] = task_settings.chunk_size

    return task_metadata


def steer_data_task(
    *,
    # Task settings
    task_settings: framework_task.Settings,
    # I/O
    setup_input_source: framework_task.SetupSource,
    output_settings: framework_task.OutputSettings,
    # Analysis
    # NOTE: The analysis arguments are bound to both of these functions before passing here
    analysis_function: framework_task.AnalysisBound,
    analysis_metadata_function: framework_task.CustomizeAnalysisMetadata,
    # We split these argument out to ensure that they're explicitly supported
    validation_mode: bool,
) -> framework_task.Output:
    """ Steering for a data task.

    Note:
        This doesn't vary significantly from the embedding case. Mostly, we just pass through the
        source index identifier and update the typing. This is a possible candidate for combining
        together in the future, although I appreciate the explicitness of the current configuration.

    Args:
        task_settings: Task settings.
        setup_input_source: Function to setup the input source.
        output_settings: Output settings.
        analysis_function: Analysis function.
        analysis_metadata_function: Function to customize the analysis metadata.
        validation_mode: Whether or not to run in validation mode.

    Returns:
        Output from the task.
    """
    # Validation
    if "embed" in task_settings.collision_system:
        msg = f"Trying to use embedding collision system '{task_settings.collision_system}' with data steering. You probably want the dedicated function."
        raise RuntimeError(msg)

    # Task metadata
    task_metadata = _default_task_metadata(task_settings=task_settings)
    # Add in the customized analysis parameters
    task_metadata.update(analysis_metadata_function(task_settings=task_settings))

    # NOTE: Always call `description_and_output_metadata` right before they're needed to ensure they
    #       up to date since they may change

    try:
        iter_arrays = setup_input_source(
            task_settings=task_settings,
            output_settings=output_settings,
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

    return steer_task_execution(
        task_settings=task_settings,
        task_metadata=task_metadata,
        iter_arrays=iter_arrays,
        output_settings=output_settings,
        analysis_function=analysis_function,
        validation_mode=validation_mode,
    )


def steer_embed_task(
    *,
    # Task settings
    task_settings: framework_task.Settings,
    # I/O
    setup_input_source: framework_task.SetupEmbeddingSource,
    output_settings: framework_task.OutputSettings,
    # Analysis
    # NOTE: The analysis arguments are bound to both of these functions before passing here
    analysis_function: framework_task.EmbeddingAnalysisBound,
    analysis_metadata_function: framework_task.CustomizeAnalysisMetadata,
    # We split these argument out to ensure that they're explicitly supported
    validation_mode: bool,
) -> framework_task.Output:
    """ Steering for an embedding task.

    Note:
        This doesn't vary significantly from the data case. Mostly, we just pass through the
        source index identifier and update the typing. This is a possible candidate for combining
        together in the future, although I appreciate the explicitness of the current configuration.

    Args:
        task_settings: Task settings.
        setup_input_source: Function to setup the input source.
        output_settings: Output settings.
        analysis_function: Analysis function.
        analysis_metadata_function: Function to customize the analysis metadata.
        validation_mode: Whether or not to run in validation mode.

    Returns:
        Output from the task.
    """
    # Validation
    if "embed" not in task_settings.collision_system:
        msg = f"Trying to use embedding steering with wrong collision system {task_settings.collision_system}"
        raise RuntimeError(msg)

    # Task metadata
    task_metadata = _default_task_metadata(task_settings=task_settings)
    # Add in the customized analysis parameters
    task_metadata.update(analysis_metadata_function(task_settings=task_settings))

    # NOTE: Always call `description_and_output_metadata` right before they're needed to ensure they
    #       up to date since they may change

    try:
        source_index_identifiers, iter_arrays = setup_input_source(
            task_settings=task_settings,
            output_settings=output_settings,
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

    # Add the source index identifiers to the analysis task arguments.
    # This way, we can abstract away what type of function is being called,
    # and use a uniform interface for task execution.
    standard_analysis_function: framework_task.AnalysisBound = functools.partial(
        analysis_function,
        source_index_identifiers=source_index_identifiers
    )

    return steer_task_execution(
        task_settings=task_settings,
        task_metadata=task_metadata,
        iter_arrays=iter_arrays,
        output_settings=output_settings,
        analysis_function=standard_analysis_function,
        validation_mode=validation_mode,
    )