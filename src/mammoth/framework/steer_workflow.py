"""Steering for jobs

Overall concept:
    job: Group of tasks which are generated together.
    task: Corresponds to one unit (eg. file), which will be handled by an app
    analysis: Corresponds to analysis of one chunk (eg. one file, or one chunk of a file). Doesn't care about I/O, etc.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import copy
import logging
import secrets
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Protocol

import IPython
import numpy as np
from pachyderm import yaml
from parsl.data_provider.files import File

from mammoth import helpers, job_utils
from mammoth.framework import production, sources
from mammoth.framework import task as framework_task

logger = logging.getLogger(__name__)


class SetupTasks(Protocol):
    """Interface for setting up tasks."""

    def __call__(
        self,
        *,
        prod: production.ProductionSettings,
        execution_settings: job_utils.ExecutionSettings,
    ) -> list[Future[framework_task.Output]]: ...


class PreprocessArguments(Protocol):
    """Interface for preprocessing arguments for a given task."""

    def __call__(
        self,
        **analysis_arguments: Any,
    ) -> dict[str, Any]: ...


def no_op_preprocess_arguments(
    **analysis_arguments: Any,  # noqa: ARG001
) -> dict[str, Any]:
    """No-op for preprocessing arguments."""
    return {}


class OutputIdentifier(Protocol):
    """Interface for determining the output identifier for a given task."""

    def __call__(
        self,
        **analysis_arguments: Any,
    ) -> str: ...


def no_op_output_identifier(
    **analysis_arguments: Any,  # noqa: ARG001
) -> str:
    """No-op for determining the output identifier."""
    return ""


def _validate_setup_functions(
    preprocess_arguments: PreprocessArguments | None = None,
    output_identifier: OutputIdentifier | None = None,
) -> tuple[PreprocessArguments, OutputIdentifier]:
    """Standard validation for setup functions.

    Args:
        preprocess_arguments: Preprocess the arguments in the steering.
        output_identifier: Customize the output identifier.
    Returns:
        Tuple of the validated functions.
    """
    defined_preprocess_arguments = no_op_preprocess_arguments if preprocess_arguments is None else preprocess_arguments
    defined_output_identifier = no_op_output_identifier if output_identifier is None else output_identifier

    return defined_preprocess_arguments, defined_output_identifier


class SetupSteeringTask(Protocol):
    """Setup processing calculation (ie. single input collection).

    Args:
        analyze_chunk: Analysis function to be run.
        preprocess_arguments: Preprocess the arguments in the steering.
        output_identifier: Customize the output identifier.
        metadata_for_labeling: Customize the task metadata.

    Returns:
        Function that will setup the embedding of MC into data with the specified analysis function.
    """

    def __call__(
        self,
        analyze_chunk: framework_task.Analysis,
        preprocess_arguments: PreprocessArguments | None = None,
        output_identifier: OutputIdentifier | None = None,
        metadata_for_labeling: framework_task.CustomizeMetadataForLabeling | None = None,
    ) -> SetupTasks: ...


def safe_slug_from_relative_path_of_input_filename(
    filename: Path,
    production_base_dir: Path,
    cwd: Path,
    number_of_parent_directories_for_relative_output_filename: int | None = None,
) -> str:
    """Safe and identifiable name for naming output files based on the relative input path.

    Converts: "pp/2111/run_by_run/LHC17p_CENT_woSDD/282341/AnalysisResults.17p.001.root"
           -> "pp_2111__run_by_run__LHC17p_CENT_woSDD__282341__AnalysisResults_17p_001"

    NOTE:
        Per the python docs, `relative_to` is a string operation, so we shouldn't hit
        the filesystem here!

    Args:
        filename: Filename to be converted.
        production_base_dir: Production base directory. e.g. `trains/`
        cwd: Current working directory. Only used in the case of absolute paths, but better
            to have it available so we don't have to call it repeatedly here.
        number_of_parent_directories_for_relative_output_filename: Number of parent directories above
            filename to use as the reference directory for the relative path. If None, the
            production_base_dir is used (e.g. which ends up as `trains`). Default: None.
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
        reference_dir = production_base_dir
        # `relative_to` requires that both filenames are the same type (either absolute or relative)
        # `reference_dir` is usually relative, so we may need to resolve it to ensure that the comparison will work.
        if filename.is_absolute():
            # NOTE: We can't use `resolve()` because it will resolve symlinks, which we probably don't want it to do
            #       since we usually symlink the `train` directory.
            # NOTE: `pathlib.Path.absolute()` would be perfect here, but it requires 3.11
            reference_dir = cwd / reference_dir
    return str(filename.relative_to(reference_dir).with_suffix("")).replace("/", "__").replace(".", "_")


def _extract_info_from_signal_file_list(
    signal_input_files_per_pt_hat: dict[int, list[Path]],
) -> tuple[list[int], list[tuple[int, Path]]]:
    """Helper to extract the pt hat bins and flatten the input list."""

    # Since we would sample the pt hat bins, it's better to keep track of them directly.
    pt_hat_bins = list(signal_input_files_per_pt_hat)
    # Or alternatively, we sample the pythia files directly. In this case, the PYTHIA files are sampled
    # evenly, while the pt hat bins are not (we will sample the higher pt hat bins more because there
    # are more statistics).
    # To do this sampling, we need to flatten out the list of signal input files.
    # We also store the pt hat bin since we need that for the grabbing the right scale factor.
    signal_input_files_flat = [
        (_pt_hat_bin, _signal_path)
        for _pt_hat_bin, _signal_paths in signal_input_files_per_pt_hat.items()
        for _signal_path in _signal_paths
    ]

    return pt_hat_bins, signal_input_files_flat


def _determine_unconstrained_signal_input_files(
    signal_input_files_per_pt_hat: dict[int, list[Path]],
    signal_input_files_flat: list[tuple[int, Path]],
    pt_hat_bins: list[int],
    signal_input_config: dict[str, Any],
) -> tuple[int, list[Path]]:
    """Determine the signal input files for the unconstrained case.

    We refactored this out since the logic is a bit complex to be inline.
    """
    # Sample the pt hat equally, or directly sample the signal_input_files
    _sample_each_pt_hat_bin_equally = signal_input_config["sample_each_pt_hat_bin_equally"]
    _n_files_to_use_per_task = signal_input_config["n_files_to_use_per_task"]

    # Randomly select (in some manner) an input file to match up with the background input file.
    # NOTE: The signal input file will repeat if there are more background events.
    #       So far, this doesn't seem to be terribly common, but even if it was, it
    #       would be perfectly fine as long as it doesn't happen too often.
    signal_input = []
    if _sample_each_pt_hat_bin_equally:
        # Each pt hat bin will be equally likely, and then we select the file from
        # those which are available.
        # NOTE: This doesn't mean that the embedded statistics will still be the same in the end.
        #       For example, if I compare a low and high pt hat bin, there are just going to be
        #       more accepted jets in the high pt hat sample.
        pt_hat_bin = secrets.choice(pt_hat_bins)
        signal_input = [
            secrets.choice(signal_input_files_per_pt_hat[pt_hat_bin]) for _ in range(_n_files_to_use_per_task)
        ]
    else:
        # Directly sample the files. This probes the generator stats because
        # the number of files is directly proportional to the generated statistics.
        pt_hat_bin, _signal_input_filename = secrets.choice(signal_input_files_flat)
        signal_input = [_signal_input_filename]
        # Since we want to keep the same pt hat bin, use the pt hat ban to randomly select additional files
        signal_input.extend(
            [
                secrets.choice(signal_input_files_per_pt_hat[pt_hat_bin])
                # -1 since we already have a filename
                for _ in range(_n_files_to_use_per_task - 1)
            ]
        )
    return pt_hat_bin, signal_input


def _select_files_for_source(
    input_files: list[Path],
    selected_input_file: Path,
    n_files_to_use: int,
) -> list[Path]:
    """Select n files from a list of available files without replacement."""
    _input = [selected_input_file]

    _possible_additional_files = {
        secrets.choice(input_files)
        # -1 since we already have a filename
        # +5 since we'll remove any filenames if they're repeated
        # NOTE: +5 is arbitrary, but should be sufficient. We could do more, but it would be a waste of cycles.
        #       In any case, We'll double check below.
        for _ in range(n_files_to_use - 1 + 5)
    }
    # Remove the existing file, and then add to the list
    _possible_additional_files.discard(selected_input_file)
    _input.extend(list(_possible_additional_files)[: n_files_to_use - 1])

    # Validate that we didn't somehow end up with too few files
    # This really shouldn't happen outside of exceptional cases
    if len(_input) != n_files_to_use:
        _msg = (
            "You somehow don't have enough input files."
            f" Requested: {n_files_to_use}, but only have {len(_input)} files available."
            " Check your input configuration!"
        )
        raise ValueError(_msg)

    return _input


def _determine_embed_pythia_input_files(
    signal_input_files_per_pt_hat: dict[int, list[Path]],
    background_input_files: list[Path],
    background_is_constrained_source: bool,
    input_handling_config: dict[str, Any],
) -> Iterable[tuple[int, list[Path], list[Path]]]:
    """Determine the input files for embedding with pythia."""
    # Configuration setup
    signal_input_config = input_handling_config["signal_parameters"]
    background_input_config = input_handling_config["background_parameters"]

    # Some convenient quantities for working with signal inputs
    pt_hat_bins, signal_input_files_flat = _extract_info_from_signal_file_list(
        signal_input_files_per_pt_hat=signal_input_files_per_pt_hat
    )

    if background_is_constrained_source:
        for background_file in background_input_files:
            # Determine the constrained input (background)
            # Start with the file that we iterated with
            background_input = _select_files_for_source(
                input_files=background_input_files,
                selected_input_file=background_file,
                n_files_to_use=background_input_config["constrained_source"]["n_files_to_use_per_task"],
            )

            # Determine the unconstrained input (signal)
            pt_hat_bin, signal_input = _determine_unconstrained_signal_input_files(
                signal_input_files_per_pt_hat=signal_input_files_per_pt_hat,
                signal_input_files_flat=signal_input_files_flat,
                pt_hat_bins=pt_hat_bins,
                signal_input_config=signal_input_config["unconstrained_source"],
            )

            yield pt_hat_bin, signal_input, background_input
    else:
        for pt_hat_bin, signal_file in signal_input_files_flat:
            # Determine the constrained input (signal)
            # Start with the file that we iterated with
            signal_input = _select_files_for_source(
                input_files=signal_input_files_per_pt_hat[pt_hat_bin],
                selected_input_file=signal_file,
                n_files_to_use=signal_input_config["constrained_source"]["n_files_to_use_per_task"],
            )

            # Determine unconstrained source (background)
            background_input = [
                secrets.choice(background_input_files)
                for _ in range(background_input_config["unconstrained_source"]["n_files_to_use_per_task"])
            ]

            yield pt_hat_bin, signal_input, background_input


def _determine_analysis_function(
    *,
    prod: production.ProductionSettings,
    analyze_chunk_with_one_input_lvl: framework_task.Analysis | None = None,
    analyze_chunk_with_two_input_lvl: framework_task.Analysis | None = None,
    analyze_chunk_with_three_input_level: framework_task.Analysis | None = None,
) -> framework_task.Analysis:
    """Determine the analysis function to use based on the production settings.

    We bass this off the analysis input levels that are provided. We're peeking a bit ahead,
    but this is really convenient to do here, so it's worth it.

    Args:
        prod: Production settings.
        analyze_chunk_with_one_input_lvl: Analysis function to be run with one input level. Default: None
        analyze_chunk_with_two_input_lvl: Analysis function to be run with two input levels. Default: None
        analyze_chunk_with_three_input_level: Analysis function to be run with three input levels. Default: None

    Returns:
        The analysis function to use based on the production settings.
    """
    # Determine the analysis function to use
    analyze_chunk = None
    analysis_input_levels = prod.config["settings"]["analysis_input_levels"]
    match len(analysis_input_levels):
        case 1:
            analyze_chunk = analyze_chunk_with_one_input_lvl
        case 2:
            analyze_chunk = analyze_chunk_with_two_input_lvl
        case 3:
            analyze_chunk = analyze_chunk_with_three_input_level
        case _:
            _msg = f"Unexpected number of analysis input levels: {analysis_input_levels}"
            raise ValueError(_msg)

    if analyze_chunk is None:
        _msg = f"Selected analysis function is None! Provided {analysis_input_levels=}"
        raise ValueError(_msg)
    return analyze_chunk


def setup_framework_standard_workflow(  # noqa: C901
    *,
    analyze_chunk_with_one_input_lvl: framework_task.Analysis | None = None,
    analyze_chunk_with_two_input_lvl: framework_task.Analysis | None = None,
    preprocess_arguments: PreprocessArguments | None = None,
    output_identifier: OutputIdentifier | None = None,
    metadata_for_labeling: framework_task.CustomizeMetadataForLabeling | None = None,
) -> SetupTasks:
    """Setup standard workflow for the framework.

    Args:
        analyze_chunk_with_one_input_lvl: Analysis function to be run with one input level.
        analyze_chunk_with_two_input_lvl: Analysis function to be run with two input levels.
        preprocess_arguments: Preprocess the arguments in the steering.
        output_identifier: Customize the output identifier.
        metadata_for_labeling: Customize the task metadata.

    Note:
        Only one analyze function needs to be provided since we may not have valid ones for all cases.
        However, they cannot all be None!

    Returns:
        Function that will setup the standard workflow with the using the provided analysis
            functions. The correct function will be selected based on the options selected in
            the configuration file.
    """
    # Validation
    # We must have something...
    assert analyze_chunk_with_one_input_lvl is not None or analyze_chunk_with_two_input_lvl is not None
    # NOTE: We change the function name here to help out mypy. Something about the way that we're
    #       wrapping the function causes an issue otherwise.
    defined_preprocess_arguments, defined_output_identifier = _validate_setup_functions(
        preprocess_arguments=preprocess_arguments,
        output_identifier=output_identifier,
    )
    # Note: We'll handle metadata_for_labeling possibly being None in the python app

    def wrap_setup(  # noqa: C901
        prod: production.ProductionSettings,
        execution_settings: job_utils.ExecutionSettings,
    ) -> list[Future[framework_task.Output]]:
        """Execute to setup the standard workflow for the framework.

        Args:
            prod: Production settings.
            job_framework: Job framework to use.
            debug_mode: Debug mode.

        Returns:
            List of futures for the tasks to be run.
        """
        # First, we need to select the function that we'll use. This is based on the production settings
        analyze_chunk = _determine_analysis_function(
            prod=prod,
            analyze_chunk_with_one_input_lvl=analyze_chunk_with_one_input_lvl,
            analyze_chunk_with_two_input_lvl=analyze_chunk_with_two_input_lvl,
        )
        # And then we can define the app we'll execute
        python_app_func = framework_task.python_app_data(
            analysis=analyze_chunk,
            metadata_for_labeling=metadata_for_labeling,
        )

        # Setup input and output
        # Need to handle pt hat bin productions differently than standard productions
        # since we need to keep track of the pt hat bin
        if "n_pt_hat_bins" in prod.config["metadata"]["dataset"]:
            input_files: dict[int, list[Path]] = prod.input_files_per_pt_hat()
        else:
            input_files = {-1: prod.input_files()}
        output_dir = prod.output_dir / "skim"
        output_dir.mkdir(parents=True, exist_ok=True)

        # If we want to debug some particular files, we can directly set them here
        logger.info(f"{execution_settings.debug_mode=}")
        if execution_settings.debug_mode and isinstance(execution_settings.debug_mode, Mapping):
            # input_files = {10: [Path("trains/pythia/2619/run_by_run/LHC18b8_cent_woSDD/282008/10/AnalysisResults.18b8_cent_woSDD.003.root")]}
            # input_files = {-1: [Path("trains/pp/2111/run_by_run/LHC17p_CENT_woSDD/282341/AnalysisResults.17p.586.root")]}
            # input_files = {-1: [Path("trains/PbPb/645/run_by_run/LHC18r/297595/AnalysisResults.18r.551.root")]}
            # We rely on you to get this input type correct wrong since it's just a debug tool for convenience.
            logger.info(f"Using debug mode input files: {execution_settings.debug_mode}")
            input_files = execution_settings.debug_mode  # type: ignore[assignment]

        # Setup for analysis and dataset settings
        _metadata_config = prod.config["metadata"]
        _analysis_config: dict[str, Any] = copy.deepcopy(prod.config["settings"])
        _output_settings_config = _analysis_config.pop("output_settings")
        # NOTE: These are arguments which will be passed onto `load_data.setup_source_for_data_or_MC_task`.
        #       If you want to pass all of the metadata, we will need to add a kwargs, since this can vary from dataset to dataset.
        #       As of July 2023, explicitly specifying th arguments seems good enough.
        _input_options_for_load_data_setup_source = {
            "loading_data_rename_levels": _metadata_config.get("loading_data_rename_levels", {}),
            "minimize_IO_as_possible": execution_settings.minimize_IO_as_possible,
        }
        # Chunk size
        chunk_size = _analysis_config.pop("chunk_size", sources.ChunkSizeSentinel.FULL_SOURCE)
        logger.info(f"Processing chunk size for {chunk_size}")
        # Sample fraction of input events (for quick analysis)
        sample_dataset_fraction = _metadata_config.get("sample_dataset_fraction", 1.0)
        if sample_dataset_fraction < 1.0:
            logger.warning(f"Sampling only a fraction of the statistics! Using {sample_dataset_fraction}")
            # Sample the input files, but require at least one entry so we have something
            # in each pt hat bin
            input_files = {
                _pt_hat_bin: [
                    secrets.choice(_input_files)
                    for _ in range(int(np.ceil(sample_dataset_fraction * len(_input_files))))
                ]
                for _pt_hat_bin, _input_files in input_files.items()
            }

        # Analysis settings
        analysis_arguments = copy.deepcopy(_analysis_config)
        # Det level artificial tracking efficiency (pythia / pp_MC only)
        det_level_artificial_tracking_efficiency = None
        if prod.collision_system in ["pythia", "pp_MC", "PbPb_MC"]:
            # NOTE: Delayed import since we don't want to depend on this in the main framework directly
            from mammoth.framework.analysis import tracking as analysis_tracking

            # Artificial tracking efficiency (including the option for pt dependent tracking eff)
            # NOTE: This depends on period, so it's better to do it here!
            det_level_artificial_tracking_efficiency = _analysis_config.get(  # noqa: SIM910
                "det_level_artificial_tracking_efficiency", None
            )
            # Pt dependent for tracking efficiency uncertainty
            if _analysis_config.get("apply_pt_dependent_tracking_efficiency_uncertainty", False):
                # NOTE: The pt dependence is added on top of the baseline shift. e.g. .98 + pt dependence.
                det_level_artificial_tracking_efficiency = (
                    analysis_tracking.PtDependentTrackingEfficiencyParameters.from_file(
                        # NOTE: We select "anchor_period" and "0_100" here because we know we're analyzing pythia
                        period=_metadata_config["dataset"]["anchor_period"],
                        event_activity="0_100",
                        baseline_tracking_efficiency_shift=det_level_artificial_tracking_efficiency,
                    )
                )
        analysis_arguments["det_level_artificial_tracking_efficiency"] = det_level_artificial_tracking_efficiency
        # Preprocess the arguments
        # NOTE: We do it last so we can access the other arguments if needed
        analysis_arguments.update(
            defined_preprocess_arguments(
                **analysis_arguments,
            )
        )
        # NOTE: We need to customize the analysis arguments to pass the relevant scale factor,
        #       so we make a copy for clarity. We'll update it each loop.
        analysis_arguments_with_pt_hat_scale_factor = copy.deepcopy(analysis_arguments)

        # Scale factors
        scale_factors = None
        if prod.has_scale_factors:
            scale_factors = prod.scale_factors()

        # Helper for below, so we don't have to call it repeatedly
        cwd = Path.cwd()

        results = []
        _file_counter = 0
        for pt_hat_bin, input_filenames in input_files.items():
            # Setup the analysis arguments
            # (The pt hat value may not always be meaningful, but we always include it).
            analysis_arguments_with_pt_hat_scale_factor["pt_hat_bin"] = pt_hat_bin
            if scale_factors is not None:
                analysis_arguments_with_pt_hat_scale_factor["scale_factors"] = scale_factors

            for input_filename in input_filenames:
                if _file_counter % 500 == 0:
                    logger.info(f"Adding {input_filename} for analysis")

                # For debugging
                if execution_settings.debug_mode and _file_counter > 1:
                    break

                # Setup file I/O
                output_identifier = safe_slug_from_relative_path_of_input_filename(
                    filename=input_filename,
                    production_base_dir=prod.base_output_dir,
                    number_of_parent_directories_for_relative_output_filename=_metadata_config["dataset"].get(
                        "number_of_parent_directories_for_relative_output_filename", None
                    ),
                    cwd=cwd,
                )
                # Finally, add the customization
                output_identifier += defined_output_identifier(**analysis_arguments_with_pt_hat_scale_factor)

                # NOTE: The extension will be customized in the app....
                output_filename = output_dir / f"{output_identifier}.dummy_ext"

                if _file_counter == 0:
                    logger.warning(f"First loop: {output_identifier}\n{analysis_arguments_with_pt_hat_scale_factor=}")

                # if _file_counter % 100 == 0:
                #    import IPython; IPython.embed()
                # And create the tasks
                results.append(
                    python_app_func(
                        # Task settings
                        production_identifier=prod.identifier,
                        collision_system=prod.collision_system,
                        chunk_size=chunk_size,
                        # I/O
                        # Inputs
                        input_source_config=_metadata_config["dataset"],
                        input_options_for_load_data_setup_source=_input_options_for_load_data_setup_source,
                        # Outputs
                        output_settings_config=_output_settings_config,
                        # Arguments
                        analysis_arguments=analysis_arguments_with_pt_hat_scale_factor,
                        # Framework options
                        job_framework=execution_settings.job_framework,
                        inputs=[File(input_filename)],
                        outputs=[File(output_filename)],
                        file_staging_settings=execution_settings.file_staging_settings,
                    )
                )

                _file_counter += 1

        logger.info("Done with creating apps!")
        return results

    return wrap_setup


def setup_framework_embed_workflow(  # noqa: C901
    *,
    analyze_chunk_with_one_input_lvl: framework_task.Analysis | None = None,
    analyze_chunk_with_two_input_lvl: framework_task.Analysis | None = None,
    analyze_chunk_with_three_input_lvl: framework_task.Analysis | None = None,
    preprocess_arguments: PreprocessArguments | None = None,
    output_identifier: OutputIdentifier | None = None,
    metadata_for_labeling: framework_task.CustomizeMetadataForLabeling | None = None,
) -> SetupTasks:
    """Setup embed workflow for the framework.

    This is distinct enough from the standard workflow that it's worth separating out
    (eg. we need separate inputs for the background)

    Note:
        Only one analyze function needs to be provided since we may not have valid ones for all cases.
        However, they cannot all be None!

    Args:
        analyze_chunk_with_one_input_lvl: Analysis function to be run with one input level.
        analyze_chunk_with_two_input_lvl: Analysis function to be run with two input levels.
        analyze_chunk_with_three_input_lvl: Analysis function to be run with three input levels. (ie. embedding)
        preprocess_arguments: Preprocess the arguments in the steering.
        output_identifier: Customize the output identifier.
        metadata_for_labeling: Customize the task metadata.

    Returns:
        Function that will setup the standard workflow with the using the provided analysis
            functions. The correct function will be selected based on the options selected in
            the configuration file.
    """
    # Validation
    # We must have something...
    assert (
        analyze_chunk_with_one_input_lvl is not None
        or analyze_chunk_with_two_input_lvl is not None
        or analyze_chunk_with_three_input_lvl is not None
    )
    # NOTE: We change the function name here to help out mypy. Something about the way that we're
    #       wrapping the function causes an issue otherwise.
    defined_preprocess_arguments, defined_output_identifier = _validate_setup_functions(
        preprocess_arguments=preprocess_arguments,
        output_identifier=output_identifier,
    )
    # Note: We'll handle metadata_for_labeling possibly being None in the python app

    def wrap_setup_embed_MC_into_data(  # noqa: C901
        prod: production.ProductionSettings,
        execution_settings: job_utils.ExecutionSettings,
    ) -> list[Future[framework_task.Output]]:
        """Create futures to produce embed MC into data skim"""
        # First, we need to select the function that we'll use. This is based on the production settings
        analyze_chunk = _determine_analysis_function(
            prod=prod,
            analyze_chunk_with_one_input_lvl=analyze_chunk_with_one_input_lvl,
            analyze_chunk_with_two_input_lvl=analyze_chunk_with_two_input_lvl,
            analyze_chunk_with_three_input_level=analyze_chunk_with_three_input_lvl,
        )
        # And then we can define the app we'll execute
        python_app_func = framework_task.python_app_embed_MC_into_data(
            analysis=analyze_chunk,
            metadata_for_labeling=metadata_for_labeling,
        )

        # Setup input and output
        output_dir = prod.output_dir / "skim"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect input files (signal and background)
        background_input_files = prod.input_files()
        # We store the signal input files in a few different formats to enable sampling different ways.
        # We can sample pt hat bins equally by sampling the pt hat bin, and then taking a random file
        # from that bin. In this case, the pythia files _are not_ sampled equally.
        signal_input_files_per_pt_hat = prod.input_files_per_pt_hat()

        # If we want to debug some particular files, we can directly set them here
        logger.info(f"{execution_settings.debug_mode=}")
        if execution_settings.debug_mode and isinstance(execution_settings.debug_mode, Mapping):
            # background_input_files = [Path("trains/PbPb/645/run_by_run/LHC18r/297595/AnalysisResults.18r.551.root")]
            # signal_input_files_per_pt_hat = {
            #     12: [
            #         Path("trains/pythia/2640/run_by_run/LHC20g4/296690/12/AnalysisResults.20g4.008.root"),
            #         Path("trains/pythia/2640/run_by_run/LHC20g4/295819/12/AnalysisResults.20g4.009.root"),
            #         Path("trains/pythia/2640/run_by_run/LHC20g4/297479/12/AnalysisResults.20g4.009.root"),
            #     ]
            # }
            logger.info(f"Using debug mode input files: {execution_settings.debug_mode}")
            if not (
                "signal_input_files_per_pt_hat" in execution_settings.debug_mode
                and "background_input_files" in execution_settings.debug_mode
            ):
                msg = "Must specify both signal_input_files_per_pt_hat and background_input_files in debug mode!"
                raise ValueError(msg)
            # By using the intermediate values, it allows us to pass None in the dict to
            # indicate that we should use the standard files. However, this also requires
            # the user to explicitly say so to avoid doing it on accident.
            _temp_value = execution_settings.debug_mode["signal_input_files_per_pt_hat"]
            if _temp_value is not None:
                signal_input_files_per_pt_hat = _temp_value
            _temp_value = execution_settings.debug_mode["background_input_files"]
            if _temp_value is not None:
                background_input_files = _temp_value

        # Setup for dataset and input
        _metadata_config: dict[str, Any] = prod.config["metadata"]
        _input_handling_config: dict[str, Any] = _metadata_config["input_handling"]
        _background_is_constrained_source: bool = _metadata_config["input_constrained_source"].lower() != "signal"
        source_input_options = {
            "background_is_constrained_source": _background_is_constrained_source,
            # In June 2024, we switched from grabbing the statically defined collision system in the configuration
            # to actually grabbing it from the metadata of the input datasets.
            # "signal_source_collision_system": _input_handling_config["signal_parameters"]["collision_system"],
            # "background_source_collision_system": _input_handling_config["background_parameters"]["collision_system"],
            "signal_source_collision_system": _metadata_config["signal_dataset"]["collision_system"],
            "background_source_collision_system": _metadata_config["dataset"]["collision_system"],
            "post_embedding_rename_levels": _metadata_config.get("post_embedding_rename_levels", {}),
            "minimize_IO_as_possible": execution_settings.minimize_IO_as_possible,
        }
        _analysis_config: dict[str, Any] = copy.deepcopy(prod.config["settings"])
        _output_settings_config = _analysis_config.pop("output_settings")
        # Chunk size
        chunk_size = _analysis_config.pop("chunk_size", sources.ChunkSizeSentinel.FULL_SOURCE)
        logger.info(f"Processing chunk size for {chunk_size}")
        logger.info(
            f"Configuring embed pythia with {'background' if _background_is_constrained_source else 'signal'} as the constrained source."
        )
        # Sample fraction of input events (for quick analysis)
        sample_dataset_fraction = _metadata_config.get("sample_dataset_fraction", 1.0)
        rng_for_sample_dataset_fraction = np.random.default_rng()
        # NOTE: In this case (ie. embedding MC into data), we need to implement this as
        #       we loop over files because it's significantly simpler. See below.
        if sample_dataset_fraction < 1.0:
            logger.warning(f"Sampling only a fraction of the statistics! Using {sample_dataset_fraction}")

        # Analysis settings
        analysis_arguments = copy.deepcopy(_analysis_config)
        # Artificial tracking efficiency (including the option for pt dependent tracking eff)
        # NOTE: This depends on centrality and period, so it's better to do it here!
        det_level_artificial_tracking_efficiency = _analysis_config["det_level_artificial_tracking_efficiency"]
        # Pt dependent for tracking efficiency uncertainty
        if _analysis_config["apply_pt_dependent_tracking_efficiency_uncertainty"]:
            # NOTE: Delayed import since we don't want to depend on this in the main framework directly
            from mammoth.framework.analysis import tracking as analysis_tracking

            _event_activity_name_to_centrality_values = {
                "central": "0_10",
                "semi_central": "30_50",
            }
            # NOTE: The pt dependence is added on top of the baseline shift. e.g. .98 + pt dependence.
            det_level_artificial_tracking_efficiency = (
                analysis_tracking.PtDependentTrackingEfficiencyParameters.from_file(
                    period=_metadata_config["dataset"]["period"],
                    event_activity=_event_activity_name_to_centrality_values[_analysis_config["event_activity"]],
                    baseline_tracking_efficiency_shift=det_level_artificial_tracking_efficiency,
                )
            )
        analysis_arguments["det_level_artificial_tracking_efficiency"] = det_level_artificial_tracking_efficiency
        # Preprocess the arguments
        # NOTE: We do it last so we can access the other arguments if needed
        analysis_arguments.update(
            defined_preprocess_arguments(
                **analysis_arguments,
            )
        )

        # Scale factors
        scale_factors = None
        if prod.has_scale_factors:
            scale_factors = prod.scale_factors()
        else:
            _msg = "Check the embedding config - you need a signal dataset."
            raise ValueError(_msg)

        # NOTE: We need to customize the analysis arguments to pass the relevant scale factor,
        #       so we make a copy for clarity. We'll update it each loop.
        analysis_arguments_with_pt_hat_scale_factor = copy.deepcopy(analysis_arguments)

        # Cross check input
        # NOTE: Need to wait until here because need the scale factors
        # NOTE: We usually need to skip this during debug mode because we may not have all pt hat bins in the input,
        #       so it will fail trivially.
        if not execution_settings.debug_mode:
            pt_hat_bins, _ = _extract_info_from_signal_file_list(
                signal_input_files_per_pt_hat=signal_input_files_per_pt_hat
            )
            if set(scale_factors) != set(pt_hat_bins):
                _msg = f"Mismatch between the pt hat bins in the scale factors ({set(scale_factors)}) and the pt hat bins ({set(pt_hat_bins)})"
                raise ValueError(_msg)

        # Helper for below, so we don't have to call it repeatedly
        cwd = Path.cwd()

        results = []
        _embedding_file_pairs = {}
        # Keep track of output identifiers. If there is already an existing identifier, then we can try again to avoid overwriting it.
        _output_identifiers = []
        input_generator = _determine_embed_pythia_input_files(
            signal_input_files_per_pt_hat=signal_input_files_per_pt_hat,
            background_input_files=background_input_files,
            background_is_constrained_source=_background_is_constrained_source,
            input_handling_config=_input_handling_config,
        )
        for _file_counter, (pt_hat_bin, signal_input, background_input) in enumerate(input_generator):
            # Need to customize the analysis arguments to pass the relevant scale factor
            analysis_arguments_with_pt_hat_scale_factor["scale_factor"] = scale_factors[pt_hat_bin]

            if _file_counter % 500 == 0:
                logger.info(
                    f"Adding {(background_input if _background_is_constrained_source else signal_input)} for analysis"
                )

            # Sample fraction
            # NOTE: It's much easier to implement here than messing with all of the input file determination.
            #       This trades off non-uniform sampling for small samples (ie. if the fraction is very small,
            #       the input pt hat bins may not be sampled uniformly). However, the reduced complexity
            #       is worth this trade off.
            # NOTE: This isn't very efficient to call for each file, but it's good enough for this purpose
            #       since it's just in the steering.
            if sample_dataset_fraction < 1.0 and rng_for_sample_dataset_fraction.random() > sample_dataset_fraction:
                continue

            # For debugging
            if execution_settings.debug_mode and _file_counter > 1:
                break

            # Setup file I/O
            # We want to identify as: "{signal_identifier}__embedded_into__{background_identifier}"
            # Take the first signal and first background filenames as the main identifier to the path.
            # Otherwise, the filename could become indefinitely long... (apparently there are file length limits in unix...)
            output_identifier = safe_slug_from_relative_path_of_input_filename(
                filename=signal_input[0],
                production_base_dir=prod.base_output_dir,
                number_of_parent_directories_for_relative_output_filename=_metadata_config["signal_dataset"].get(
                    "number_of_parent_directories_for_relative_output_filename", None
                ),
                cwd=cwd,
            )
            output_identifier += "__embedded_into__"
            output_identifier += safe_slug_from_relative_path_of_input_filename(
                filename=background_input[0],
                production_base_dir=prod.base_output_dir,
                number_of_parent_directories_for_relative_output_filename=_metadata_config["dataset"].get(
                    "number_of_parent_directories_for_relative_output_filename", None
                ),
                cwd=cwd,
            )
            # Finally, add the customization
            output_identifier += defined_output_identifier(**analysis_arguments_with_pt_hat_scale_factor)

            # Ensure that we don't use an output identifier twice.
            # If we've already used it, we add a counter to it
            _modifiable_output_identifier = output_identifier
            _output_identifier_counter = 0
            _output_identifier_stored = False
            while not _output_identifier_stored:
                if _modifiable_output_identifier in _output_identifiers:
                    # If the identifier is in the list, try to add some counter to it.
                    _output_identifier_counter += 1
                    _modifiable_output_identifier = output_identifier + f"__{_output_identifier_counter:03}"
                else:
                    output_identifier = _modifiable_output_identifier
                    _output_identifiers.append(output_identifier)
                    _output_identifier_stored = True

            # logger.info(f"output_identifier: {output_identifier}")
            # NOTE: The extension will be customized in the app....
            output_filename = output_dir / f"{output_identifier}.dummy_ext"

            # Store the file pairs for our records
            # The output identifier contains the first signal filename, as well as the background filename.
            # We use it here rather than _just_ the background filename because we may embed into data multiple times
            _embedding_file_pairs[output_identifier] = [str(_filename) for _filename in signal_input] + [
                str(_filename) for _filename in background_input
            ]

            if _file_counter == 0:
                logger.warning(f"First loop: {output_identifier}\n{analysis_arguments_with_pt_hat_scale_factor=}")

            # And create the tasks
            results.append(
                python_app_func(
                    # Task settings
                    production_identifier=prod.identifier,
                    collision_system=prod.collision_system,
                    chunk_size=chunk_size,
                    # I/O
                    # Inputs
                    signal_input_source_config=_metadata_config["signal_dataset"],
                    n_signal_input_files=len(signal_input),
                    background_input_source_config=_metadata_config["dataset"],
                    input_options_for_load_data_setup_source=source_input_options,
                    # Outputs
                    output_settings_config=_output_settings_config,
                    # Arguments
                    analysis_arguments=analysis_arguments_with_pt_hat_scale_factor,
                    # Framework options
                    job_framework=execution_settings.job_framework,
                    inputs=[
                        *[File(str(_filename)) for _filename in signal_input],
                        *[File(str(_filename)) for _filename in background_input],
                    ],
                    outputs=[File(str(output_filename))],
                    file_staging_settings=execution_settings.file_staging_settings,
                )
            )

        # And write the file pairs, again for our records
        y = yaml.yaml()
        embedding_file_pairs_filename = prod.output_dir / "embedding_file_pairs.yaml"
        _existing_embedding_file_pairs = {}
        if embedding_file_pairs_filename.exists():
            with embedding_file_pairs_filename.open() as f:
                _existing_embedding_file_pairs = y.load(f)
        # Add back in the existing file pairs if we've read them
        if _existing_embedding_file_pairs:
            _embedding_file_pairs.update(_existing_embedding_file_pairs)
        # And then (re)write the file pairs
        with embedding_file_pairs_filename.open("w") as f:
            y.dump(_embedding_file_pairs, f)

        logger.info("Done with creating apps!")
        return results

    def wrap_setup_embed_MC_into_thermal_model(
        prod: production.ProductionSettings,
        execution_settings: job_utils.ExecutionSettings,
    ) -> list[Future[framework_task.Output]]:
        # First, we need to select the function that we'll use. This is based on the production settings
        analyze_chunk = _determine_analysis_function(
            prod=prod,
            analyze_chunk_with_one_input_lvl=analyze_chunk_with_one_input_lvl,
            analyze_chunk_with_two_input_lvl=analyze_chunk_with_two_input_lvl,
            analyze_chunk_with_three_input_level=analyze_chunk_with_three_input_lvl,
        )
        # And then we can define the app we'll execute
        python_app_func = framework_task.python_app_embed_MC_into_thermal_model(
            analysis=analyze_chunk,
            metadata_for_labeling=metadata_for_labeling,
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

        # If we want to debug some particular files, we can directly set them here
        logger.info(f"{execution_settings.debug_mode=}")
        if execution_settings.debug_mode and isinstance(execution_settings.debug_mode, Mapping):
            # input_files = {10: [Path("trains/pythia/2619/run_by_run/LHC18b8_cent_woSDD/282008/10/AnalysisResults.18b8_cent_woSDD.003.root")]}
            # input_files = {10: [
            #    Path("trains/pythia/2640/run_by_run/LHC20g4/296415/4/AnalysisResults.20g4.007.root"),
            #    Path("trains/pythia/2640/run_by_run/LHC20g4/296415/4/AnalysisResults.20g4.010.root"),
            # ]}
            # We rely on you to get this input type correct wrong since it's just a debug tool for convenience.
            logger.info(f"Using debug mode input files: {execution_settings.debug_mode}")
            input_files = execution_settings.debug_mode  # type: ignore[assignment]

        # Setup for dataset settings (and grab analysis config for Output settings)
        _metadata_config = prod.config["metadata"]
        _analysis_config = copy.deepcopy(prod.config["settings"])
        _output_settings_config = _analysis_config.pop("output_settings")
        _input_options_for_load_data_setup_source = {
            "signal_source_collision_system": _metadata_config["dataset"]["collision_system"],
            "post_embedding_rename_levels": _metadata_config.get("post_embedding_rename_levels", {}),
            "minimize_IO_as_possible": execution_settings.minimize_IO_as_possible,
        }
        # Thermal model parameters
        thermal_model_parameters = sources.THERMAL_MODEL_SETTINGS[
            f"{_metadata_config['dataset']['sqrt_s']}_{_analysis_config['event_activity']}"
        ]
        # Chunk size
        chunk_size = _analysis_config.pop("chunk_size")
        logger.info(f"Processing chunk size for {chunk_size}")
        # Sample fraction of input events (for quick analysis)
        sample_dataset_fraction = _metadata_config.get("sample_dataset_fraction", 1.0)
        if sample_dataset_fraction < 1.0:
            logger.warning(f"Sampling only a fraction of the statistics! Using {sample_dataset_fraction}")
            # Sample the input files, but require at least one entry so we have something
            # in each pt hat bin
            input_files = {
                _pt_hat_bin: [
                    secrets.choice(_input_files)
                    for _ in range(int(np.ceil(sample_dataset_fraction * len(_input_files))))
                ]
                for _pt_hat_bin, _input_files in input_files.items()
            }

        # Analysis settings
        analysis_arguments = copy.deepcopy(_analysis_config)
        # Preprocess the arguments
        # NOTE: We do it last so we can access the other arguments if needed
        analysis_arguments.update(
            defined_preprocess_arguments(
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

        # NOTE: We need to customize the analysis arguments to pass the relevant scale factor,
        #       so we make a copy for clarity. We'll update it each loop.
        analysis_arguments_with_pt_hat_scale_factor = copy.deepcopy(analysis_arguments)

        # Cross check
        # NOTE: Need to wait until here because need the scale factors
        # NOTE: We usually need to skip this during debug mode because we may not have all pt hat bins in the input,
        #       so it will fail trivially.
        if set(scale_factors) != set(input_files) and not execution_settings.debug_mode:
            _msg = f"Mismatch between the pt hat bins in the scale factors ({set(scale_factors)}) and the pt hat bins ({set(input_files)})"
            raise ValueError(_msg)

        # Helper for below, so we don't have to call it repeatedly
        cwd = Path.cwd()

        results = []
        _file_counter = 0
        # Reversed because the higher pt hard bins are of more importance to get done sooner.
        for pt_hat_bin, input_filenames in reversed(input_files.items()):
            analysis_arguments_with_pt_hat_scale_factor["scale_factor"] = scale_factors[pt_hat_bin]

            for input_filename in input_filenames:
                if _file_counter % 500 == 0 or execution_settings.debug_mode:
                    logger.info(f"Adding {input_filename} for analysis")

                # For debugging
                if execution_settings.debug_mode and _file_counter > 1:
                    break

                # Setup file I/O
                output_identifier = safe_slug_from_relative_path_of_input_filename(
                    filename=input_filename,
                    production_base_dir=prod.base_output_dir,
                    number_of_parent_directories_for_relative_output_filename=_metadata_config["dataset"].get(
                        "number_of_parent_directories_for_relative_output_filename", None
                    ),
                    cwd=cwd,
                )
                # Finally, add the customization
                output_identifier += defined_output_identifier(**analysis_arguments_with_pt_hat_scale_factor)

                # NOTE: The extension will be customized in the app....
                output_filename = output_dir / f"{output_identifier}.dummy_ext"

                if _file_counter == 0:
                    logger.warning(f"First loop: {output_identifier}\n{analysis_arguments_with_pt_hat_scale_factor=}")

                # And create the tasks
                results.append(
                    python_app_func(
                        # Task settings
                        production_identifier=prod.identifier,
                        collision_system=prod.collision_system,
                        chunk_size=chunk_size,
                        # I/O
                        # Inputs
                        thermal_model_parameters=thermal_model_parameters,
                        input_source_config=_metadata_config["dataset"],
                        input_options_for_load_data_setup_source=_input_options_for_load_data_setup_source,
                        # Output
                        output_settings_config=_output_settings_config,
                        # Arguments
                        analysis_arguments=analysis_arguments_with_pt_hat_scale_factor,
                        job_framework=execution_settings.job_framework,
                        inputs=[
                            File(str(input_filename)),
                        ],
                        outputs=[File(str(output_filename))],
                        file_staging_settings=execution_settings.file_staging_settings,
                    )
                )

                _file_counter += 1

        logger.info("Done with creating apps!")
        return results

    # Select the correct setup function based on the collision system
    def wrap_setup(
        prod: production.ProductionSettings,
        execution_settings: job_utils.ExecutionSettings,
    ) -> list[Future[framework_task.Output]]:
        """Simple wrapper to forward on setup of the embed workflow for the framework.

        We do this so we don't have to specify which embedding function to use in the returned
        setup function. Instead, we determine it dynamically here.

        Args:
            prod: Production settings.
            job_framework: Job framework to use.
            debug_mode: Debug mode.
        Returns:
            List of futures for the tasks to be run.
        """
        # Usually we want general embedding of MC into data, so handle embedding into the thermal model as the special case
        if prod.collision_system == "embed_thermal_model":
            return wrap_setup_embed_MC_into_thermal_model(prod=prod, execution_settings=execution_settings)
        return wrap_setup_embed_MC_into_data(prod=prod, execution_settings=execution_settings)

    return wrap_setup


def setup_framework_default_workflows(
    *,
    analyze_chunk_with_one_input_lvl: framework_task.Analysis | None = None,
    analyze_chunk_with_two_input_lvl: framework_task.Analysis | None = None,
    analyze_chunk_with_three_input_lvl: framework_task.Analysis | None = None,
    preprocess_arguments: PreprocessArguments | None = None,
    output_identifier: OutputIdentifier | None = None,
    metadata_for_labeling: framework_task.CustomizeMetadataForLabeling | None = None,
) -> tuple[SetupTasks, SetupTasks]:
    """Create setup functions for the standard and embed workflows in the analysis framework.

    Note:
        Only one analyze function needs to be provided since we may not have valid ones for all cases.
        However, they cannot all be None!

    Note:
        If you need to customize how this is setup, you can always define conventions yourself!
        This is just for convenience (hence the "default" in the name).

    Args:
        analyze_chunk_with_one_input_lvl: Analysis function to be run with one input level.
        analyze_chunk_with_two_input_lvl: Analysis function to be run with two input levels.
        analyze_chunk_with_three_input_lvl: Analysis function to be run with three input levels. (ie. embedding)
        preprocess_arguments: Preprocess the arguments in the steering.
        output_identifier: Customize the output identifier.
        metadata_for_labeling: Customize the task metadata.

    Returns:
        Function that will setup the standard workflow with the using the provided analysis
            functions. The correct function will be selected based on the options selected in
            the configuration file.
    """
    standard = setup_framework_standard_workflow(
        analyze_chunk_with_one_input_lvl=analyze_chunk_with_one_input_lvl,
        analyze_chunk_with_two_input_lvl=analyze_chunk_with_two_input_lvl,
        preprocess_arguments=preprocess_arguments,
        output_identifier=output_identifier,
        metadata_for_labeling=metadata_for_labeling,
    )
    embed = setup_framework_embed_workflow(
        analyze_chunk_with_one_input_lvl=analyze_chunk_with_one_input_lvl,
        analyze_chunk_with_two_input_lvl=analyze_chunk_with_two_input_lvl,
        analyze_chunk_with_three_input_lvl=analyze_chunk_with_three_input_lvl,
        preprocess_arguments=preprocess_arguments,
        output_identifier=output_identifier,
        metadata_for_labeling=metadata_for_labeling,
    )
    return standard, embed


def setup_job_framework(
    job_framework: job_utils.JobFramework,
    productions: list[production.ProductionSettings],
    task_config: job_utils.TaskConfig,
    facility: job_utils.FACILITIES,
    walltime: str,
    target_n_tasks_to_run_simultaneously: int,
    log_level: int,
    conda_environment_name: str | None = None,
    tasks_requiring_root: list[str] | None = None,
    tasks_requiring_aliphysics: list[str] | None = None,
    override_minimize_IO_as_possible: bool | None = None,
    debug_mode: bool | dict[str | int, Any] = False,
) -> (
    tuple[job_utils.parsl.DataFlowKernel, job_utils.parsl.Config, job_utils.ExecutionSettings]
    | tuple[job_utils.dask.distributed.Client, job_utils.dask.distributed.SpecCluster, job_utils.ExecutionSettings]
):
    # Validation
    if tasks_requiring_root is None:
        tasks_requiring_root = []
    if tasks_requiring_aliphysics is None:
        tasks_requiring_aliphysics = []

    # Delayed import since it's a main framework module
    from mammoth.alice import job_utils as alice_job_utils

    # First, need to figure out if we need additional environments such as ROOT
    _additional_worker_init_script = alice_job_utils.determine_additional_worker_init(
        productions=productions,
        conda_environment_name=conda_environment_name,
        tasks_requiring_root=tasks_requiring_root,
        tasks_requiring_aliphysics=tasks_requiring_aliphysics,
    )
    return job_utils.setup_job_framework(
        job_framework=job_framework,
        task_config=task_config,
        facility=facility,
        walltime=walltime,
        target_n_tasks_to_run_simultaneously=target_n_tasks_to_run_simultaneously,
        log_level=log_level,
        additional_worker_init_script=_additional_worker_init_script,
        override_minimize_IO_as_possible=override_minimize_IO_as_possible,
        debug_mode=debug_mode,
    )


def process_futures(
    productions: Sequence[production.ProductionSettings],
    all_results: Sequence[Future[framework_task.Output]],
    job_framework: job_utils.JobFramework,
    # delete_outputs_in_futures: bool = True,
) -> None:
    # Setup
    # Delayed import to reduce dependence
    from mammoth.framework.io import output_utils

    # Process the futures, showing processing progress
    # Since it returns the results, we can actually use this to accumulate results.
    if job_framework == job_utils.JobFramework.dask_delayed:
        gen_results: Iterable[Any] = job_utils.dask.distributed.as_completed(all_results)  # type: ignore[no-untyped-call]
    elif job_framework == job_utils.JobFramework.immediate_execution_debug:
        gen_results = all_results
    else:
        gen_results = job_utils.provide_results_as_completed(all_results, running_with_parsl=True)

    # In order to support writing histograms from multiple systems, we need to index the output histograms
    # by the collision system + centrality.
    output_hists: dict[tuple[str, str], dict[Any, Any]] = {
        (_p.identifier, _p.collision_system): {} for _p in productions
    }
    with helpers.progress_bar() as progress:
        track_results = progress.add_task(total=len(all_results), description="Processing results...")
        for result in gen_results:
            logger.info(f"result: {result.summary_str()}")
            if result.success and result.hists:
                k = result.production_identifier
                logger.info(f"Found result for key {k}")
                output_hists[(k, result.collision_system)] = output_utils.merge_results(output_hists[k], result.hists)
            logger.info(f"output_hists: {output_hists}")
            progress.update(track_results, advance=1)

    # Save hists to uproot (if needed)
    for (production_identifier, collision_system), hists in output_hists.items():
        if hists:
            import uproot

            file_label = production_identifier
            if file_label:
                file_label = f"_{file_label}"

            output_hist_filename = Path("output") / collision_system / f"task_output_{file_label}.root"
            output_hist_filename.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Writing output_hists to {output_hist_filename} for system {production_identifier=}, {collision_system=}"
            )
            with uproot.recreate(output_hist_filename) as f:
                output_utils.write_hists_to_file(hists=hists, f=f)

    # By embedded here, we can inspect results, etc in the meantime.
    # NOTE: This may be commented out sometimes when I have long running processes and will
    #       probably forget to close it.
    IPython.start_ipython(user_ns={**locals(), **globals()})  # type: ignore[no-untyped-call]

    # In case we close IPython early, wait for all apps to complete
    # Also allows for a summary at the end.
    # By taking only the first two, it just tells use the status and a quick message.
    # Otherwise, we can overwhelm with trying to print large objects
    res = []
    for r in all_results:
        try:
            if job_framework != job_utils.JobFramework.immediate_execution_debug:
                _res = r.result().success
            else:
                _res = r.success  # type: ignore[attr-defined]
            res.append(_res)
        except FileNotFoundError:
            # Apparently if the file isn't found, it will show up here, so best if we catch it
            # and note that it was unsuccessful
            res.append(False)

    logger.info(res)
