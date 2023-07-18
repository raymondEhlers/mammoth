"""Run mammoth EEC skimming and analysis tasks

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Iterable, Sequence

import IPython

from mammoth import helpers, job_utils
from mammoth.alice import steer_scale_factors
from mammoth.framework import production, steer_job
from mammoth.framework import task as framework_task
from mammoth.framework.io import output_utils
from mammoth.framework.steer_job import setup_job_framework

# This will be moved around at some point, but it's a start
from mammoth.hardest_kt import produce_flat_skim_from_track_skim
from mammoth.time_reclustering import analysis_alice

logger = logging.getLogger(__name__)


# Define the steering apps
setup_data_skim = steer_job.setup_data_calculation(
    analysis_function=analysis_alice.analysis_data,
    argument_preprocessing=produce_flat_skim_from_track_skim.argument_preprocessing,
    analysis_metadata=analysis_alice.customize_analysis_metadata,
    analysis_output_identifier=produce_flat_skim_from_track_skim.analysis_output_identifier,
)

setup_MC_skim = steer_job.setup_data_calculation(
    analysis_function=analysis_alice.analysis_MC,
    argument_preprocessing=produce_flat_skim_from_track_skim.argument_preprocessing,
    analysis_metadata=analysis_alice.customize_analysis_metadata,
    analysis_output_identifier=produce_flat_skim_from_track_skim.analysis_output_identifier,
)

setup_embed_MC_into_data_skim = steer_job.setup_embed_MC_into_data_calculation(
    analysis_function=analysis_alice.analysis_embedding,
    argument_preprocessing=produce_flat_skim_from_track_skim.argument_preprocessing,
    analysis_metadata=analysis_alice.customize_analysis_metadata,
    analysis_output_identifier=produce_flat_skim_from_track_skim.analysis_output_identifier,
)

setup_embed_MC_into_thermal_model_skim = steer_job.setup_embed_MC_into_thermal_model_calculation(
    analysis_function=analysis_alice.analysis_embedding,
    argument_preprocessing=produce_flat_skim_from_track_skim.argument_preprocessing,
    analysis_metadata=analysis_alice.customize_analysis_metadata,
    analysis_output_identifier=produce_flat_skim_from_track_skim.analysis_output_identifier,
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
            production.ProductionSettings.read_config(
                collision_system="PbPb", number=4,
                specialization=produce_flat_skim_from_track_skim.HardestKtProductionSpecialization(),
                track_skim_config_filename=config_filename,
            ),
            # Production
            #production.ProductionSettings.read_config(
            #    collision_system="embed_thermal_model", number=3,
            #    specialization=produce_flat_skim_from_track_skim.HardestKtProductionSpecialization(),
            #    track_skim_config_filename=config_filename,
            #),
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
        if "calculate_data_skim" in tasks_to_execute:
            system_results.extend(
                setup_data_skim(
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
            resources={"n_cores": task_config.n_cores_per_task}
        )

    return all_results


def process_futures(
    productions: Sequence[production.ProductionSettings],
    all_results: Sequence[Future[framework_task.Output]],
    job_framework: job_utils.JobFramework,
    #delete_outputs_in_futures: bool = True,
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
    #facility: job_utils.FACILITIES = "ORNL_b587_long" if job_utils.hours_in_walltime(walltime) >= 2 else "ORNL_b587_short"
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
