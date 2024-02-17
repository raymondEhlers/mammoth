"""Run mammoth EEC skimming and analysis tasks

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from mammoth import job_utils
from mammoth.alice import groomed_substructure_analysis, groomed_substructure_steering, steer_scale_factors
from mammoth.framework import production, steer_job
from mammoth.framework import task as framework_task
from mammoth.framework.steer_job import process_futures, setup_job_framework

logger = logging.getLogger(__name__)


# Define the steering apps
setup_data_skim = steer_job.setup_data_calculation(
    analysis_function=groomed_substructure_analysis.analysis_data,
    argument_preprocessing=groomed_substructure_steering.argument_preprocessing,
    analysis_metadata=groomed_substructure_analysis.customize_analysis_metadata,
    analysis_output_identifier=groomed_substructure_steering.analysis_output_identifier,
)

setup_MC_skim = steer_job.setup_data_calculation(
    analysis_function=groomed_substructure_analysis.analysis_MC,
    argument_preprocessing=groomed_substructure_steering.argument_preprocessing,
    analysis_metadata=groomed_substructure_analysis.customize_analysis_metadata,
    analysis_output_identifier=groomed_substructure_steering.analysis_output_identifier,
)

setup_embed_MC_into_data_skim = steer_job.setup_embed_MC_into_data_calculation(
    analysis_function=groomed_substructure_analysis.analysis_embedding,
    argument_preprocessing=groomed_substructure_steering.argument_preprocessing,
    analysis_metadata=groomed_substructure_analysis.customize_analysis_metadata,
    analysis_output_identifier=groomed_substructure_steering.analysis_output_identifier,
)

setup_embed_MC_into_thermal_model_skim = steer_job.setup_embed_MC_into_thermal_model_calculation(
    analysis_function=groomed_substructure_analysis.analysis_embedding,
    argument_preprocessing=groomed_substructure_steering.argument_preprocessing,
    analysis_metadata=groomed_substructure_analysis.customize_analysis_metadata,
    analysis_output_identifier=groomed_substructure_steering.analysis_output_identifier,
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
            # Production
            # production.ProductionSettings.read_config(
            #    collision_system="pp_MC", number=69,
            #    specialization=groomed_substructure_steering.ProductionSpecialization(),
            #    track_skim_config_filename=config_filename,
            # ),
            # production.ProductionSettings.read_config(
            #    collision_system="pp_MC", number=70,
            #    specialization=groomed_substructure_steering.ProductionSpecialization(),
            #    track_skim_config_filename=config_filename,
            # ),
            # production.ProductionSettings.read_config(
            #    collision_system="pp_MC", number=71,
            #    specialization=groomed_substructure_steering.ProductionSpecialization(),
            #    track_skim_config_filename=config_filename,
            # ),
            # production.ProductionSettings.read_config(
            #    collision_system="pp_MC", number=72,
            #    specialization=groomed_substructure_steering.ProductionSpecialization(),
            #    track_skim_config_filename=config_filename,
            # ),
            production.ProductionSettings.read_config(
                collision_system="pp_MC",
                number=6,
                specialization=groomed_substructure_steering.ProductionSpecialization(),
                track_skim_config_filename=config_filename,
            ),
        ]
    )

    # Write out the production settings
    for production_settings in productions:
        production_settings.store_production_parameters()

    return productions


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
        logger.info(
            f'Tasks to execute: {tasks_to_execute} for production "{prod.collision_system}" #{prod.formatted_number}'
        )

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
        if "calculate_data_skim" in tasks_to_execute:
            system_results.extend(
                setup_data_skim(
                    prod=prod,
                    job_framework=job_framework,
                    debug_mode=debug_mode,
                )
            )
        if "calculate_pp_MC_skim" in tasks_to_execute:
            system_results.extend(
                setup_MC_skim(
                    prod=prod,
                    job_framework=job_framework,
                    debug_mode=debug_mode,
                )
            )
        if "calculate_embed_pythia_skim" in tasks_to_execute:
            system_results.extend(
                setup_embed_MC_into_data_skim(
                    prod=prod,
                    job_framework=job_framework,
                    debug_mode=debug_mode,
                )
            )
        if "calculate_embed_thermal_model_skim" in tasks_to_execute:
            system_results.extend(
                setup_embed_MC_into_thermal_model_skim(
                    prod=prod,
                    job_framework=job_framework,
                    debug_mode=debug_mode,
                )
            )

        all_results.extend(system_results)
        logger.info(f"Accumulated {len(system_results)} futures for {prod.collision_system}")

    logger.info(f"Accumulated {len(all_results)} total futures")

    if job_framework == job_utils.JobFramework.dask_delayed:
        assert isinstance(job_executor, job_utils.dask.distributed.Client)
        all_results = job_executor.compute(  # type: ignore[no-untyped-call]
            all_results,
            # Distributed assumes functions are pure, but usually mine are not (ie. they create files)
            pure=False,
            resources={"n_cores": task_config.n_cores_per_task},
        )

    return all_results


def run(job_framework: job_utils.JobFramework) -> list[Future[Any]]:
    # Job execution parameters
    productions = define_productions()
    task_name = "time_reclustering_mammoth"

    # Job execution configuration
    conda_environment_name = ""
    task_config = job_utils.TaskConfig(name=task_name, n_cores_per_task=1, memory_per_task=4)
    target_n_tasks_to_run_simultaneously = 120
    # target_n_tasks_to_run_simultaneously = 110
    # target_n_tasks_to_run_simultaneously = 60
    log_level = logging.INFO
    walltime = "24:00:00"
    debug_mode = True
    if debug_mode:
        # Usually, we want to run in the short queue
        target_n_tasks_to_run_simultaneously = 2
        walltime = "1:59:00"
    # facility: job_utils.FACILITIES = "ORNL_b587_long" if job_utils.hours_in_walltime(walltime) >= 2 else "ORNL_b587_short"
    facility: job_utils.FACILITIES = "hiccup_std" if job_utils.hours_in_walltime(walltime) >= 2 else "hiccup_quick"
    # facility: job_utils.FACILITIES = "rehlers_mbp_m1pro"

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
        job_executor=job_executor,
    )

    process_futures(productions=productions, all_results=all_results, job_framework=job_framework)

    return all_results


if __name__ == "__main__":
    # run(job_framework=job_utils.JobFramework.immediate_execution_debug)
    run(job_framework=job_utils.JobFramework.parsl)
