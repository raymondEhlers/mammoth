"""Steering for jobs

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import copy
import logging
import secrets
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from parsl.data_provider.files import File

from mammoth import job_utils
from mammoth.framework import production, sources
from mammoth.framework import task as framework_task
from mammoth.framework.io import output_utils

logger = logging.getLogger(__name__)


class SetupTasks(Protocol):
    def __call__(
        self,
        *,
        prod: production.ProductionSettings,
        job_framework: job_utils.JobFramework,
        debug_mode: bool,
    ) -> list[Future[framework_task.Output]]:
        ...

class PreprocessArguments(Protocol):
    def __call__(
            self,
            **analysis_arguments: Any,
        ) -> dict[str, Any]:
        ...


def no_op_preprocess_arguments(
    **analysis_arguments: Any,
) -> dict[str, Any]:
    return {}


def safe_output_filename_from_relative_path(
    filename: Path,
    output_dir: Path,
    number_of_parent_directories_for_relative_output_filename: int | None = None,
) -> str:
    """Safe and identifiable name for naming output files based on the relative path.

    Converts: "2111/run_by_run/LHC17p_CENT_woSDD/282341/AnalysisResults.17p.001.root"
           -> "2111__run_by_run__LHC17p_CENT_woSDD__282341__AnalysisResults_17p_001"

    Returns:
        Filename that is safe for using as the output filename.
    """
    if number_of_parent_directories_for_relative_output_filename is not None:
        # Add one more since we usually forget the current directory will count as one
        number_of_parent_directories_for_relative_output_filename += 1
        reference_dir = filename
        # Now, walk back up the tree for the specific number of times
        for _ in range(number_of_parent_directories_for_relative_output_filename):
            reference_dir = reference_dir.parent
    else:
        # NOTE: We use the grandparent of the output dir because the input filename is going to be a different train
        #       than our output. For the case of embedding trains, we might not even share the collision system.
        #       So by going to the grandparent (ie `trains`), we end up with a shared path
        reference_dir = output_dir.parent.parent
        # `relative_to` requires that both filenames are the same type (either absolute or relative)
        # `reference_dir` is usually relative, so we may need to resolve it to ensure that the comparison will work.
        if filename.is_absolute():
            # NOTE: We can't use `resolve()` because it will resolve symlinks, which we probably don't want it to do
            #       since we usually symlink the `train` directory.
            # NOTE: `pathlib.Path.absolute()` would be perfect here, but it requires 3.11
            reference_dir = Path.cwd() / reference_dir
    return str(filename.relative_to(reference_dir).with_suffix("")).replace("/", "__").replace(".", "_")


# def setup_embed() -> list[Future[framework_task.Output]]


# Or possibly:
#run_embedding = framework_task.run_embedding(analysis_function, analysis_steering_argument_preprocessing_function, task_description_metadata_function)
# And then we can use run embedding here, with the expected signature (which we could call TaskSteering).

# Needed here:
# - Argument preprocessing: construct types, etc. Optional...
# - description output metadata. Optional...
# - Analysis function
# Will define in function:
# - Output definition from config...
# - Source definition. Can only be partial - need to construct based on the config.
#   - I think this necessarily means that we have separate embedding, thermal model, etc functions. Fair enough.

def setup_embed_MC_into_thermal_model(
    # Analysis function
    analysis_function: framework_task.Analysis,
    # Preprocess the arguments in the steering
    argument_preprocessing: PreprocessArguments | None = None,
    # Customize the task metadata
    analysis_metadata: framework_task.CustomizeAnalysisMetadata | None = None
) -> SetupTasks:
    # Validation
    # NOTE: We change the function name here to help out mypy. Something about the way that we're
    #       wrapping the function causes an issue otherwise.
    if argument_preprocessing is None:
        defined_argument_preprocessing = no_op_preprocess_arguments
    else:
        defined_argument_preprocessing = argument_preprocessing
    # Note: We'll handle analysis_metadata possibly being None in the python app

    def wrap_setup(
        prod: production.ProductionSettings,
        job_framework: job_utils.JobFramework,
        debug_mode: bool,
    ) -> list[Future[framework_task.Output]]:

        # Setup
        python_app_func = framework_task.python_app_embed_MC_into_thermal_model(
            analysis=analysis_function,
            analysis_metadata=analysis_metadata,
        )

        # Setup input and output
        # Need to handle pt hat bin productions differently than standard productions
        # since we need to keep track of the pt hat bin
        if "n_pt_hat_bins" in prod.config["metadata"]["dataset"]:
            input_files: dict[int, list[Path]] = prod.input_files_per_pt_hat()
        else:
            input_files = {-1: prod.input_files()}
            _msg = "Need pt hat production for embedding into a thermal model..."
            raise RuntimeError(_msg)
        output_dir = prod.output_dir / "skim"
        output_dir.mkdir(parents=True, exist_ok=True)

        if debug_mode:
            #input_files = {10: [Path("trains/pythia/2619/run_by_run/LHC18b8_cent_woSDD/282008/10/AnalysisResults.18b8_cent_woSDD.003.root")]}
            input_files = {10: [
                Path("trains/pythia/2640/run_by_run/LHC20g4/296415/4/AnalysisResults.20g4.007.root"),
                Path("trains/pythia/2640/run_by_run/LHC20g4/296415/4/AnalysisResults.20g4.010.root"),
            ]}
        # Setup for dataset settings
        _metadata_config = prod.config["metadata"]
        _analysis_config = prod.config["settings"]
        _output_options_config = _analysis_config.pop("output_options")
        # Sample fraction of input events (for quick analysis)
        sample_dataset_fraction =_metadata_config.get("sample_dataset_fraction", 1.0)
        if sample_dataset_fraction < 1.0:
            logger.warning(f"Sampling only a fraction of the statistics! Using {sample_dataset_fraction}")
            # Sample the input files, but require at least one entry so we have something
            # in each pt hat bin
            input_files = {
                _pt_hat_bin: [
                    secrets.choice(_input_files) for _ in range(int(np.ceil(sample_dataset_fraction * len(_input_files))))
                ]
                for _pt_hat_bin, _input_files in input_files.items()
            }
        # Thermal model parameters
        thermal_model_parameters = sources.THERMAL_MODEL_SETTINGS[
            f"{_metadata_config['dataset']['sqrt_s']}_{_analysis_config['event_activity']}"
        ]
        chunk_size = _analysis_config["chunk_size"]
        logger.info(f"Processing chunk size for {chunk_size}")

        # Analysis settings
        analysis_arguments = copy.deepcopy(_analysis_config)
        # Preprocess the arguments
        analysis_arguments.update(
            defined_argument_preprocessing(
                **analysis_arguments,
            )
        )
        # Scale factors
        # NOTE: We need to mix these in below because we expect the single pt hat scale factor per func call
        scale_factors = None
        if prod.has_scale_factors:
            scale_factors = prod.scale_factors()
        else:
            _msg = "Check the thermal model config - you need a signal dataset."
            raise ValueError(_msg)
        analysis_arguments["scale_factors"] = scale_factors
        # Cross check
        if set(scale_factors) != set(input_files) and not debug_mode:
            # NOTE: We have to skip this on debug mode because we frequently only want to run a few files
            _msg = f"Mismatch between the pt hat bins in the scale factors ({set(scale_factors)}) and the pt hat bins ({set(input_files)})"
            raise ValueError(_msg)

        results = []
        _file_counter = 0
        # Reversed because the higher pt hard bins are of more importance to get done sooner.
        for pt_hat_bin, input_filenames in reversed(input_files.items()):
            # Need to customize the analysis arguments to pass the relevant scale factor
            analysis_arguments_with_pt_hat_scale_factor = copy.deepcopy(analysis_arguments)
            analysis_arguments_with_pt_hat_scale_factor["scale_factor"] = scale_factors[pt_hat_bin]

            for input_filename in input_filenames:
                if _file_counter % 500 == 0 or debug_mode:
                    logger.info(f"Adding {input_filename} for analysis")

                # For debugging
                if debug_mode and _file_counter > 1:
                    break

                # Setup file I/O
                # Converts: "2111/run_by_run/LHC17p_CENT_woSDD/282341/AnalysisResults.17p.001.root"
                #        -> "2111__run_by_run__LHC17p_CENT_woSDD__282341__AnalysisResults_17p_001"
                output_identifier = safe_output_filename_from_relative_path(
                    filename=input_filename, output_dir=prod.output_dir,
                    number_of_parent_directories_for_relative_output_filename=_metadata_config["dataset"].get(
                        "number_of_parent_directories_for_relative_output_filename", None
                    ),
                )
                output_filename = output_dir / f"{output_identifier}.parquet"
                # And create the tasks
                results.append(
                    python_app_func(
                        # Task settings
                        production_identifier=prod.identifier,
                        collision_system=prod.collision_system,
                        chunk_size=chunk_size,
                        # I/O
                        input_source_config=_metadata_config["dataset"],
                        thermal_model_parameters=thermal_model_parameters,
                        output_options_config=_output_options_config,
                        # Arguments
                        analysis_arguments=analysis_arguments_with_pt_hat_scale_factor,
                        job_framework=job_framework,
                        inputs=[
                            File(str(input_filename)),
                        ],
                        outputs=[File(str(output_filename))],
                    )
                )

                _file_counter += 1

        return results

    return wrap_setup