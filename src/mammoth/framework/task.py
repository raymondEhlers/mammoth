"""Task related functionality

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

import concurrent.futures
import functools
import importlib
import logging
from typing import Any, Callable

import attrs
import hist

from mammoth.framework import sources

logger = logging.getLogger(__name__)


@attrs.frozen(kw_only=True)
class Settings:
    production_identifier: str
    collision_system: str
    chunk_size: sources.T_ChunkSize

@attrs.frozen
class Output:
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
    f: Callable[..., Output],
) -> Callable[..., concurrent.futures.Future[Output]]:
    # Delay this import since we don't want to load all job utils functionality
    # when simply loading the module. However, it doesn't to make sense to split this up otherwise, so
    # we live with the delayed import.
    from parsl import File

    from mammoth import job_utils

    module_import_path = f.__module__
    function_name = f.__name__

    @job_utils.python_app
    @functools.wraps(f)
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

        # Get the task
        module_containing_function = importlib.import_module(module_import_path)
        func = getattr(module_containing_function, function_name)

        try:
            result = steer_embed_task(
                # General task settings
                task_settings=Settings(
                    production_identifier=production_identifier,
                    collision_system=collision_system,
                    chunk_size=chunk_size,
                ),
                # Inputs
                signal_input=[Path(_input_file.filepath) for _input_file in inputs],
                thermal_model_parameters=thermal_model_parameters,
                # Analysis
                analysis_func=functools.partial(
                    func,
                    **analysis_arguments
                ),
                analysis_arguments=analysis_arguments,
            )
        except Exception:
            result = framework_task.Output(
                production_identifier=production_identifier,
                collision_system=collision_system,
                success=False,
                message=f"failure for {collision_system}, signal={[_f.filepath for _f in inputs]} with: \n{traceback.format_exc()}",
            )
        return result

    return my_app