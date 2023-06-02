"""Task (and analysis) related functionality

Overall concept:
    task: Corresponds to one unit (eg. file), which will be handled by an app
    analysis: Corresponds to analysis of one chunk (eg. one file, or one chunk of a file). Doesn't care about I/O, etc.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import concurrent.futures
import functools
import importlib
import logging
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import attrs
import awkward as ak
import hist
import uproot

from mammoth.framework import sources
from mammoth.framework.io import output_utils

logger = logging.getLogger(__name__)

####################
# Task functionality
####################

Metadata = dict[str, Any]


def description_from_metadata(metadata: dict[str, Any]) -> str:
    return ", ".join([f"{k}={v}" for k, v in metadata.items()])


def description_and_output_metadata(task_metadata: Metadata) -> tuple[str, dict[str, Any]]:
    """Create a str description and output metadata from the task metadata."""
    description = description_from_metadata(metadata=task_metadata)
    output_metadata = {
        # Useful to have the summary as a string
        "description": description,
        # but also useful to have programmatic access
        "parameters": task_metadata,
    }
    return description, output_metadata


@attrs.frozen(kw_only=True)
class Settings:
    """Settings for the task.

    Attributes:
        production_identifier: Unique production identifier for the task.
        collision_system: Collision system
        chunk_size: Chunk size for the task.
    """
    production_identifier: str
    collision_system: str
    chunk_size: sources.T_ChunkSize


@attrs.frozen(kw_only=True)
class PrimaryOutput:
    type: str = attrs.field(validator=[attrs.validators.in_(["hists", "skim"])])
    reference_name: str
    name: str = attrs.field(default="")

    @classmethod
    def from_config(cls, config: dict[str, str]) -> PrimaryOutput:
        return cls(
            type=config["type"],
            reference_name=config["reference_name"],
            name=config.get("name", ""),
        )


@attrs.frozen(kw_only=True)
class OutputSettings:
    """Task output settings.

    Note:
        Default: None for many of the parameters means that it will be determined by whether output was provided.
        If it's available, then it will be written and/or returned.

    Attributes:
        output_filename: Output filename.
        primary_output: Primary output, which should be used as a proxy for whether there are existing outputs.
        return_skim: Whether to return the skim. Default: None.
        write_chunk_skim: Whether to write the skim for each chunk. Default: None.
        return_merged_hists: Whether to return the merged histograms. Default: None.
        write_chunk_hists: Whether to write the histograms for each chunk. Default: None.
        write_merged_hists: Whether to write the merged histograms. Default: None.
    """
    output_filename: Path
    primary_output: PrimaryOutput
    return_skim: bool | None = attrs.field(default=None)
    write_chunk_skim: bool | None = attrs.field(default=None)
    return_merged_hists: bool | None = attrs.field(default=None)
    write_chunk_hists: bool | None = attrs.field(default=None)
    write_merged_hists: bool | None = attrs.field(default=None)

    @classmethod
    def from_config(cls, output_filename: Path, config: dict[str, Any]) -> OutputSettings:
        return cls(
            output_filename=output_filename,
            primary_output=PrimaryOutput.from_config(config=config["primary_output"]),
            return_skim=config.get("return_skim", None),
            write_chunk_skim=config.get("write_chunk_skim", None),
            return_merged_hists=config.get("return_merged_hists", None),
            write_chunk_hists=config.get("write_chunk_hists", None),
            write_merged_hists=config.get("write_merged_hists", None),
        )

    def with_new_output_filename(self, new_output_filename: Path) -> OutputSettings:
        return type(self)(
            output_filename=new_output_filename,
            primary_output=self.primary_output,
            return_skim=self.return_skim,
            write_chunk_skim=self.write_chunk_skim,
            return_merged_hists=self.return_merged_hists,
            write_chunk_hists=self.write_chunk_hists,
            write_merged_hists=self.write_merged_hists,
        )


@attrs.frozen
class Output:
    """Task output wrapper.

    Attributes:
        production_identifier: Unique production identifier for the task.
        collision_system: Collision system
        success: Whether the task was successful.
        message: Message about the task execution.
        hists: Histograms returned by this task. Default: {}.
        results: Any additional results returned by this task. eg. skimmed arrays. Default: {}.
        metadata: Any additional metadata returned by this task. Collision system could go
            into this metadata, but it's fairly useful for identification, etc, so we call
            it out explicitly. Default: {}.
    """
    production_identifier: str
    collision_system: str
    success: bool
    message: str
    hists: dict[str, hist.Hist] = attrs.field(factory=dict, kw_only=True, repr=lambda value: str(list(value.keys())))
    results: dict[str, Any] = attrs.field(factory=dict, kw_only=True)
    metadata: dict[str, Any] = attrs.field(factory=dict, kw_only=True)

    def print(self) -> None:
        """Print the message to the logger.

        Note:
            This is implemented as a convenience function since it otherwise
            won't evaluate newlines in the message. We don't implement __str__
            or __repr__ since those require an additional function call (ie. explicit print).
            This is all just for convenience.
        """
        logger.info(f"collision_system={self.collision_system}, success={self.success}, identifier={self.production_identifier}")
        # NOTE: Evaluate the message separately to ensure that newlines are evaluated.
        logger.info(self.message)


#############
# I/O sources
#############


def check_for_task_output(
    output_options: OutputSettings,
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
                res = output_utils.check_for_task_root_skim_output_file(
                    output_filename=output_utils.task_output_path_skim(output_filename=output_options.output_filename, skim_name=output_options.primary_output.name),
                    reference_tree_name=output_options.primary_output.reference_name,
                )
            else:
                res = output_utils.check_for_task_parquet_skim_output_file(
                    output_filename=output_utils.task_output_path_skim(output_filename=output_options.output_filename, skim_name=output_options.primary_output.name),
                    reference_array_name=output_options.primary_output.reference_name,
                )
        if output_options.primary_output.type == "hists":
            res = output_utils.check_for_task_hist_output_file(
                output_filename=output_utils.task_output_path_hist(output_filename=output_options.output_filename),
                reference_hist_name=output_options.primary_output.reference_name,
            )

    return res


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

class SetupSource(Protocol):
    def __call__(
            self,
            *,
            task_settings: Settings,
            task_metadata: Metadata,
            output_options: OutputSettings,
            **kwargs: Any,
        ) -> tuple[Iterator[ak.Array]]:
            ...

class SetupEmbeddingSource(Protocol):
    def __call__(
            self,
            *,
            task_settings: Settings,
            task_metadata: Metadata,
            output_options: OutputSettings,
            **kwargs: Any,
        ) -> tuple[dict[str, int], Iterator[ak.Array]]:
        ...


##############################
# Chunk analysis functionality
##############################


class Analysis(Protocol):
    def __call__(
            self,
            *,
            arrays: ak.Array,
            validation_mode: bool = False,
            return_skim: bool = False
        ) -> AnalysisOutput:
        ...

class EmbeddingAnalysis(Protocol):
    def __call__(
            self,
            *,
            source_index_identifiers: dict[str, int],
            arrays: ak.Array,
            validation_mode: bool = False,
            return_skim: bool = False,
        ) -> AnalysisOutput:
        ...
class CustomizeAnalysisMetadata(Protocol):
    """Customize metadata based on the analysis arguments.

    """
    def __call__(
            self,
            task_settings: Settings,
            **analysis_arguments: dict[str, Any],
        ) -> Metadata:
        ...

    @property
    def __name__(self) -> str:
        ...

def NoOpCustomizeAnalysisMetadata(
    task_settings: Settings,
    **analysis_arguments: dict[str, Any],
) -> Metadata:
    return {}

@attrs.frozen(kw_only=True)
class AnalysisOutput:
    hists: dict[str, hist.Hist] = attrs.field(factory=dict)
    skim: ak.Array | dict[str, ak.Array] = attrs.field(factory=dict)

    def _write_hists(self, output_filename: Path) -> None:
        output_hist_filename = output_utils.task_output_path_hist(output_filename=output_filename)
        output_hist_filename.parent.mkdir(parents=True, exist_ok=True)
        with uproot.recreate(output_hist_filename) as f:
            output_utils.write_hists_to_file(hists=self.hists, f=f)

    def _write_skim(self, output_filename: Path, skim: dict[str, ak.Array]) -> None:
        for skim_name, skim_array in skim.items():
            # Enables the possibility of writing a single standard file (just wrap in a dict with an empty string as key).
            if skim_name != "":
                _skim_output_filename = output_utils.task_output_path_skim(output_filename=output_filename, skim_name=skim_name)
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

    def write(self, output_filename: Path, write_hists: bool | None, write_skim: bool | None) -> None:
        # If not specified, fall back to the default, which is to write the analysis outputs if they're provided
        if write_hists is None:
            write_hists = bool(self.hists)
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


#############
# Python apps
#############


def _module_and_name_from_func(func: Callable[..., Any]) -> tuple[str, str]:
    """Get the module and name from a function.

    Used to import them in the app, which is required by parsl.
    """
    return (func.__module__, func.__name__)


def python_app_embed_MC_into_thermal_model(
    *,
    analysis: Callable[..., AnalysisOutput],
    analysis_metadata: CustomizeAnalysisMetadata | None = None,
) -> Callable[..., concurrent.futures.Future[Output]]:
    # Delay this import since we don't want to load all job utils functionality
    # when simply loading the module. However, it doesn't to make sense to split this up otherwise, so
    # we live with the delayed import.
    from parsl import File

    from mammoth import job_utils

    # We need to reimport in the parsl app, so we grab the parameters here, and then put them into the closure.
    # Analysis function
    analysis_function_module_import_path, analysis_function_function_name = _module_and_name_from_func(func=analysis)
    # Task metadata
    analysis_metadata_module_import_path = ""
    analysis_metadata_function_name = ""
    if analysis_metadata is not None:
        analysis_metadata_module_import_path, analysis_metadata_function_name = _module_and_name_from_func(func=analysis_metadata)

    @job_utils.python_app
    @functools.wraps(analysis)
    def my_app(
        *,
        production_identifier: str,
        collision_system: str,
        chunk_size: sources.T_ChunkSize,
        thermal_model_parameters: sources.ThermalModelParameters,
        analysis_arguments: dict[str, Any],
        job_framework: job_utils.JobFramework,  # noqa: ARG001
        inputs: list[File] = [],
        outputs: list[File] = [],
    ) -> Output:
        # Standard imports in the app
        import traceback
        from pathlib import Path

        from mammoth.framework import task as framework_task

        # Get the analysis
        module_containing_analysis_function = importlib.import_module(analysis_function_module_import_path)
        analysis_function = getattr(module_containing_analysis_function, analysis_function_function_name)
        # And metadata customization
        # We handle this more carefully because it may not always be specified
        if analysis_metadata_module_import_path:
            module_containing_analysis_metadata_function = importlib.import_module(analysis_metadata_module_import_path)
            metadata_function: CustomizeAnalysisMetadata = getattr(module_containing_analysis_metadata_function, analysis_metadata_function_name)
        else:
            metadata_function = NoOpCustomizeAnalysisMetadata

        try:
            result = steer_embed_task(
                # General task settings
                task_settings=Settings(
                    production_identifier=production_identifier,
                    collision_system=collision_system,
                    chunk_size=chunk_size,
                ),
                # Inputs
                setup_input_source=functools.partial(
                    setup_source_for_embedding_thermal_model,
                    signal_input=[Path(_input_file.filepath) for _input_file in inputs],
                    thermal_model_parameters=thermal_model_parameters,
                ),
                output_options=OutputSettings(
                    ...
                ),
                # Analysis
                analysis_function=functools.partial(
                    analysis_function,
                    **analysis_arguments,
                ),
                analysis_metadata_function=functools.partial(
                    metadata_function,
                    **analysis_arguments,
                ),
                # Pass it separately to ensure that it's accounted for explicitly.
                validation_mode=analysis_arguments.get("validation_mode", False),
            )
        except Exception:
            result = framework_task.Output(
                production_identifier=production_identifier,
                collision_system=collision_system,
                success=False,
                message=f"failure during execution of task with: \n{traceback.format_exc()}",
            )
        return result

    return my_app