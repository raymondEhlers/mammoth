"""Task related functionality

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import concurrent.futures
import functools
import importlib
import logging
from typing import Any, Callable

import attrs
import hist

from mammoth.framework import sources

logger = logging.getLogger(__name__)


Metadata = dict[str, Any]

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

@attrs.frozen
class Output:
    """Task output.

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


def python_app_embed_MC_into_thermal_model(
    *,
    analysis: Callable[..., Output],
    analysis_metadata: Callable[..., Metadata] | None = None,
) -> Callable[..., concurrent.futures.Future[Output]]:
    # Delay this import since we don't want to load all job utils functionality
    # when simply loading the module. However, it doesn't to make sense to split this up otherwise, so
    # we live with the delayed import.
    from parsl import File

    from mammoth import job_utils

    # We need to reimport in the parsl app, so we grab the parameters here, and then put them into the closure.
    # Analysis function
    analysis_function_module_import_path = analysis.__module__
    analysis_function_function_name = analysis.__name__
    # Task metadata
    analysis_metadata_module_import_path = ""
    analysis_metadata_function_name = ""
    if analysis_metadata is not None:
        analysis_metadata_module_import_path = analysis_metadata.__module__
        analysis_metadata_function_name = analysis_metadata.__name__

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
            metadata_function = getattr(module_containing_analysis_metadata_function, analysis_metadata_function_name)
        else:
            metadata_function = lambda *args, **kwargs: {}

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