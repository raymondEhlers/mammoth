"""Run mammoth EEC skimming and analysis tasks

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
import secrets
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Iterable, MutableMapping, Sequence

import attrs
import IPython
import numpy as np
from parsl.data_provider.files import File

from mammoth import helpers, job_utils
from mammoth.alice import job_utils as alice_job_utils
from mammoth.alice import steer_scale_factors
from mammoth.eec import analysis_alice
from mammoth.framework import production, sources, steer_job
from mammoth.framework import task as framework_task
from mammoth.framework.io import output_utils
from mammoth.job_utils import python_app

logger = logging.getLogger(__name__)


@attrs.frozen()
class EECProductionSpecialization:
    def customize_identifier(self, analysis_settings: MutableMapping[str, Any]) -> str:
        # NOTE: Only handle things here that need special treatment!
        # Trigger pt ranges
        name = "_trigger_pt_ranges_"
        trigger_pt_ranges: dict[str, tuple[float, float]] = analysis_settings.pop("trigger_pt_ranges")
        for trigger_name, trigger_pt_tuple in trigger_pt_ranges.items():
            #name += f"_{trigger_name}_{'_'.join(map(str, trigger_pt_tuple))}"
            name += f"_{trigger_name}_{trigger_pt_tuple[0]:g}_{trigger_pt_tuple[1]:g}"
        name += "_"
        # Min track pt
        name += "_min_track_pt"
        for level, value in analysis_settings.pop("min_track_pt").items():
            name += f"_{level}_{value:g}"
        return name

    def tasks_to_execute(self, collision_system: str) -> list[str]:
        _tasks = []

        # Skim task
        _base_name = "calculate_{label}_skim"
        _label_map = {
            "pp": "data",
            "pythia": "data",
            "pp_MC": "data",
            "PbPb": "data",
            "embedPythia": "embed_pythia",
            "embed_pythia": "embed_pythia",
            "embed_thermal_model": "embed_thermal_model",
        }
        _tasks.append(_base_name.format(label=_label_map[collision_system]))
        return _tasks


@python_app
def _run_embed_thermal_model_skim(
    collision_system: str,
    trigger_pt_ranges: dict[str, tuple[float, float]],
    min_track_pt: dict[str, float],
    momentum_weight_exponent: int | float,
    det_level_artificial_tracking_efficiency: float,
    thermal_model_parameters: sources.ThermalModelParameters,
    scale_factor: float,
    chunk_size: int,
    output_trigger_skim: bool,
    return_hists: bool,
    production_identifier: str,
    job_framework: job_utils.JobFramework,  # noqa: ARG001
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> framework_task.Output:
    import traceback
    from pathlib import Path

    from mammoth.eec import analysis_using_track_skim
    from mammoth.framework import task as framework_task

    try:
        result = analysis_using_track_skim.eec_embed_thermal_model_analysis(
            production_identifier=production_identifier,
            collision_system=collision_system,
            signal_input=[Path(_input_file.filepath) for _input_file in inputs],
            trigger_pt_ranges=trigger_pt_ranges,
            min_track_pt=min_track_pt,
            momentum_weight_exponent=momentum_weight_exponent,
            det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
            thermal_model_parameters=thermal_model_parameters,
            scale_factor=scale_factor,
            chunk_size=chunk_size,
            output_trigger_skim=output_trigger_skim,
            return_hists=return_hists,
            output_filename=Path(outputs[0].filepath),
        )
    except Exception:
        result = framework_task.Output(
            production_identifier,
            collision_system,
            False,
            f"failure for {collision_system}, signal={[_f.filepath for _f in inputs]} with: \n{traceback.format_exc()}",
        )
    return result



def setup_calculate_embed_thermal_model_skim(
    prod: production.ProductionSettings,
    job_framework: job_utils.JobFramework,
    debug_mode: bool,
) -> list[Future[framework_task.Output]]:
    """Create futures to produce hardest kt embedded pythia skim"""
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

    # Setup for analysis and dataset settings
    _metadata_config = prod.config["metadata"]
    _analysis_config = prod.config["settings"]
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
    # Scale factors
    scale_factors = None
    if prod.has_scale_factors:
        scale_factors = prod.scale_factors()
    else:
        _msg = "Check the thermal model config - you need a signal dataset."
        raise ValueError(_msg)

    # Cross check
    if set(scale_factors) != set(input_files) and not debug_mode:
        # NOTE: We have to skip this on debug mode because we frequently only want to run a few files
        _msg = f"Mismatch between the pt hat bins in the scale factors ({set(scale_factors)}) and the pt hat bins ({set(input_files)})"
        raise ValueError(_msg)

    results = []
    _file_counter = 0
    # Reversed because the higher pt hard bins are of more importance to get done sooner.
    for pt_hat_bin, input_filenames in reversed(input_files.items()):
        for input_filename in input_filenames:
            if _file_counter % 500 == 0 or debug_mode:
                logger.info(f"Adding {input_filename} for analysis")

            # For debugging
            if debug_mode and _file_counter > 1:
                break

            # Setup file I/O
            # Converts: "2111/run_by_run/LHC17p_CENT_woSDD/282341/AnalysisResults.17p.001.root"
            #        -> "2111__run_by_run__LHC17p_CENT_woSDD__282341__AnalysisResults_17p_001"
            output_identifier = steer_job.safe_output_filename_from_relative_path(
                filename=input_filename, output_dir=prod.output_dir,
                number_of_parent_directories_for_relative_output_filename=_metadata_config["dataset"].get(
                    "number_of_parent_directories_for_relative_output_filename", None
                ),
            )
            output_filename = output_dir / f"{output_identifier}.parquet"
            # And create the tasks
            results.append(
                _run_embed_thermal_model_skim(
                    collision_system=prod.collision_system,
                    trigger_pt_ranges=_analysis_config["trigger_pt_ranges"],
                    min_track_pt=_analysis_config["min_track_pt"],
                    momentum_weight_exponent=_analysis_config["momentum_weight_exponent"],
                    det_level_artificial_tracking_efficiency=_analysis_config[
                        "det_level_artificial_tracking_efficiency"
                    ],
                    thermal_model_parameters=thermal_model_parameters,
                    scale_factor=scale_factors[pt_hat_bin],
                    chunk_size=chunk_size,
                    output_trigger_skim=_analysis_config["output_trigger_skim"],
                    # This can become quite expensive in terms of memory since we have to keep copies in memory.
                    # So by default, we will write them to file, but won't return them.
                    return_hists=_analysis_config.get("return_hists_during_execution", False),
                    production_identifier=prod.identifier,
                    job_framework=job_framework,
                    inputs=[
                        File(str(input_filename)),
                    ],
                    outputs=[File(str(output_filename))],
                )
            )

            _file_counter += 1

    return results

# Define the steering apps

#setup_data_skim = steer_job.setup_data_calculation(
#    analysis_function=analysis_alice.analysis_data,
#    analysis_metadata=analysis_alice.customize_analysis_metadata,
#)

setup_embed_MC_into_thermal_model_skim = steer_job.setup_embed_MC_into_data_calculation(
    analysis_function=analysis_alice.analysis_embedding,
    analysis_metadata=analysis_alice.customize_analysis_metadata,
)

setup_embed_MC_into_thermal_model_skim = steer_job.setup_embed_MC_into_thermal_model_calculation(
    analysis_function=analysis_alice.analysis_embedding,
    analysis_metadata=analysis_alice.customize_analysis_metadata,
)

def setup_job_framework(
    job_framework: job_utils.JobFramework,
    productions: Sequence[production.ProductionSettings],
    task_config: job_utils.TaskConfig,
    facility: job_utils.FACILITIES,
    walltime: str,
    target_n_tasks_to_run_simultaneously: int,
    log_level: int,
    conda_environment_name: str | None = None,
) -> tuple[job_utils.parsl.DataFlowKernel, job_utils.parsl.Config] | tuple[job_utils.dask.distributed.Client, job_utils.dask.distributed.SpecCluster]:
    # First, need to figure out if we need additional environments such as ROOT
    _additional_worker_init_script = alice_job_utils.determine_additional_worker_init(
        productions=productions,
        conda_environment_name=conda_environment_name,
        tasks_requiring_root=[],
        tasks_requiring_aliphysics=[],
    )
    return job_utils.setup_job_framework(
        job_framework=job_framework,
        task_config=task_config,
        facility=facility,
        walltime=walltime,
        target_n_tasks_to_run_simultaneously=target_n_tasks_to_run_simultaneously,
        log_level=log_level,
        additional_worker_init_script=_additional_worker_init_script,
    )


def define_productions() -> list[production.ProductionSettings]:
    # We want to provide the opportunity to run multiple productions at once.
    # We'll do so by defining each production below and then iterating over them below
    productions = []

    # Create and store production information
    _here = Path(__file__).parent
    config_filename = Path(_here.parent / "alice" / "config" / "track_skim_config.yaml")
    productions.extend(
        [
            # Debug
            # production.ProductionSettings.read_config(
            #     collision_system="embed_pythia", number=3,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            # Production
            # production.ProductionSettings.read_config(
            #     collision_system="embed_pythia", number=65,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            # production.ProductionSettings.read_config(
            #     collision_system="embed_pythia", number=66,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            # Debug
            # production.ProductionSettings.read_config(
            #     collision_system="PbPb", number=3,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            # Production
            production.ProductionSettings.read_config(
                collision_system="embed_thermal_model", number=3,
                specialization=EECProductionSpecialization(),
                track_skim_config_filename=config_filename,
            ),
        ]
    )

    # Write out the production settings
    for production_settings in productions:
        production_settings.store_production_parameters()

    return productions


def _hours_in_walltime(walltime: str) -> int:
    return int(walltime.split(":")[0])


def setup_and_submit_tasks(
    productions: Sequence[production.ProductionSettings],
    task_config: job_utils.TaskConfig,
    job_framework: job_utils.JobFramework,
    debug_mode: bool,
    job_executor: job_utils.parsl.DataFlowKernel | job_utils.dask.distributed.Client,
) -> list[Future[Any]]:
    all_results: list[Future[framework_task.Output]] = []
    for prod in productions:
        tasks_to_execute = prod.tasks_to_execute
        logger.info(f"Tasks to execute: {tasks_to_execute} for production \"{prod.collision_system}\" #{prod.formatted_number}")

        # Setup tasks
        system_results = []
        if "extract_scale_factors" in tasks_to_execute:
            # NOTE: This will block on the result since it needs to be done before anything can proceed
            system_results.extend(
                steer_scale_factors.steer_extract_scale_factors(
                    prod=prod,
                    job_framework=job_framework,
                )
            )
        #if "calculate_data_skim" in tasks_to_execute:
        #    system_results.extend(
        #        setup_calculate_data_skim(
        #            prod=prod,
        #            job_framework=job_framework,
        #            debug_mode=debug_mode,
        #        )
        #    )
        if "calculate_embed_thermal_model_skim" in tasks_to_execute:
            system_results.extend(
                setup_calculate_embed_thermal_model_skim(
                    prod=prod,
                    job_framework=job_framework,
                    debug_mode=debug_mode,
                )
            )
        #if "calculate_embed_pythia_skim" in tasks_to_execute:
        #    system_results.extend(
        #        setup_calculate_embed_pythia_skim(
        #            prod=prod,
        #            job_framework=job_framework,
        #            debug_mode=debug_mode,
        #        )
        #    )

        all_results.extend(system_results)
        logger.info(f"Accumulated {len(system_results)} futures for {prod.collision_system}")

    logger.info(f"Accumulated {len(all_results)} total futures")

    if job_framework == job_utils.JobFramework.dask_delayed:
        assert isinstance(job_executor, job_utils.dask.distributed.Client)
        all_results = job_executor.compute(  # type: ignore[no-untyped-call]
            all_results,
            # Distributed assumes functions are pure, but usually mine are not (ie. they create files)
            pure=False,
            resources={"n_cores": task_config.n_cores_per_task}
        )

    return all_results


def process_futures(
    productions: Sequence[production.ProductionSettings],
    all_results: Sequence[Future[framework_task.Output]],
    job_framework: job_utils.JobFramework,
) -> None:
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
    output_hists: dict[str, dict[Any, Any]] = {_p.identifier: {} for _p in productions}
    with helpers.progress_bar() as progress:
        track_results = progress.add_task(total=len(all_results), description="Processing results...")
        for result in gen_results:
            logger.info(f"result: {result.production_identifier}")
            if result.success and result.hists:
                k = result.collision_system
                logger.info(f"Found result for key {k}")
                output_hists[k] = output_utils.merge_results(output_hists[k], result.hists)
            logger.info(f"output_hists: {output_hists}")
            progress.update(track_results, advance=1)

    # Save hists to uproot (if needed)
    for system, hists in output_hists.items():
        if hists:
            import uproot

            split_system_name = system.split("_")
            # Either "pp" or "PbPb"
            collision_system = split_system_name[0]
            # Additional label for centrality when appropriate
            # NOTE: If the list is of length 1, it will be empty
            file_label = "_".join(split_system_name[1:])
            if file_label:
                file_label = f"_{file_label}"

            output_hist_filename = Path("output") / collision_system / f"hardest_kt_{file_label}.root"
            output_hist_filename.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing output_hists to {output_hist_filename} for system {system}")
            with uproot.recreate(output_hist_filename) as f:
                output_utils.write_hists_to_file(hists=hists, f=f)

    # By embedded here, we can inspect results, etc in the meantime.
    # NOTE: This may be commented out sometimes when I have long running processes and wil
    #       probably forget to close it.
    IPython.start_ipython(user_ns={**locals(), **globals()})  # type: ignore[no-untyped-call]

    # In case we close IPython early, wait for all apps to complete
    # Also allows for a summary at the end.
    # By taking only the first two, it just tells use the status and a quick message.
    # Otherwise, we can overwhelm with trying to print large objects
    res = [r.result().success for r in all_results]
    logger.info(res)


def run(job_framework: job_utils.JobFramework) -> list[Future[Any]]:
    # Job execution parameters
    productions = define_productions()
    task_name = "EEC_mammoth"

    # Job execution configuration
    conda_environment_name = ""
    task_config = job_utils.TaskConfig(name=task_name, n_cores_per_task=1)
    # target_n_tasks_to_run_simultaneously = 120
    # target_n_tasks_to_run_simultaneously = 110
    target_n_tasks_to_run_simultaneously = 60
    log_level = logging.INFO
    walltime = "24:00:00"
    debug_mode = True
    if debug_mode:
        # Usually, we want to run in the short queue
        target_n_tasks_to_run_simultaneously = 2
        walltime = "1:59:00"
    #facility: job_utils.FACILITIES = "ORNL_b587_long" if _hours_in_walltime(walltime) >= 2 else "ORNL_b587_short"
    facility: job_utils.FACILITIES = "rehlers_mbp_m1pro"

    # Keep the job executor just to keep it alive
    job_executor, _job_framework_config = setup_job_framework(
        job_framework=job_framework,
        productions=productions,
        task_config=task_config,
        facility=facility,
        walltime=walltime,
        target_n_tasks_to_run_simultaneously=target_n_tasks_to_run_simultaneously,
        log_level=log_level,
        conda_environment_name=conda_environment_name,
    )
    all_results = setup_and_submit_tasks(
        productions=productions,
        task_config=task_config,
        job_framework=job_framework,
        debug_mode=debug_mode,
        job_executor=job_executor
    )

    process_futures(productions=productions, all_results=all_results, job_framework=job_framework)

    return all_results


if __name__ == "__main__":
    run(job_framework=job_utils.JobFramework.immediate_execution_debug)
