"""Run mammoth EEC skimming and analysis tasks

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import attrs

from mammoth import job_utils
from mammoth.alice import steer_scale_factors
from mammoth.eec import analyze_chunk
from mammoth.framework import production, steer_workflow
from mammoth.framework import task as framework_task
from mammoth.framework.steer_workflow import process_futures, setup_job_framework

logger = logging.getLogger(__name__)


@attrs.frozen()
class EECProductionSpecialization:
    def customize_identifier(self, analysis_settings: MutableMapping[str, Any]) -> str:
        """Customize the production identifier."""
        name = ""
        # NOTE: Only handle things here that need special treatment!
        trigger_parameters = analyze_chunk.TriggerParameters.from_config(analysis_settings.pop("trigger_parameters"))
        name += f"__{trigger_parameters.label}"
        for trigger_name, trigger_range_tuple in trigger_parameters.classes.items():
            name += f"_{trigger_name}_{trigger_range_tuple[0]:g}_{trigger_range_tuple[1]:g}"
        name += "_"
        # Min track pt
        name += "_min_track_pt"
        for level, value in analysis_settings.pop("min_track_pt").items():
            name += f"_{level}_{value:g}"
        # Don't include the output_settings. Since we don't want to display them at all,
        # we just pop them and ignore the return value.
        analysis_settings.pop("output_settings")
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
            "PbPb_MC": "data",
            "embedPythia": "embed_pythia",
            "embed_pythia": "embed_pythia",
            "embed_thermal_model": "embed_thermal_model",
        }
        _tasks.append(_base_name.format(label=_label_map[collision_system]))
        return _tasks


# Define the steering apps
setup_standard_workflow, setup_embed_workflow = steer_workflow.setup_framework_default_workflows(
    analyze_chunk_with_one_input_lvl=analyze_chunk.analyze_chunk_one_input_level,
    analyze_chunk_with_two_input_lvl=analyze_chunk.analyze_chunk_two_input_level,
    analyze_chunk_with_three_input_lvl=analyze_chunk.analyze_chunk_three_input_level,
    preprocess_arguments=analyze_chunk.preprocess_arguments,
    output_identifier=analyze_chunk.output_identifier,
    metadata_for_labeling=analyze_chunk.customize_analysis_metadata,
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
                collision_system="pp_MC",
                number=75,
                specialization=EECProductionSpecialization(),
                track_skim_config_filename=config_filename,
                base_output_dir=Path("/rstorage/rehlers/trains"),
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
    execution_settings: job_utils.ExecutionSettings,
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
                    job_framework=execution_settings.job_framework,
                )
            )
        standard_workflows = ["calculate_data_skim", "calculate_pp_MC_skim"]
        for wf in standard_workflows:
            if wf in tasks_to_execute:
                system_results.extend(
                    setup_standard_workflow(
                        prod=prod,
                        execution_settings=execution_settings,
                    )
                )
        embed_workflows = ["calculate_embed_thermal_model_skim", "calculate_embed_pythia_skim"]
        for wf in embed_workflows:
            if wf in tasks_to_execute:
                system_results.extend(
                    setup_embed_workflow(
                        prod=prod,
                        execution_settings=execution_settings,
                    )
                )

        all_results.extend(system_results)
        logger.info(f"Accumulated {len(system_results)} futures for {prod.collision_system}")

    logger.info(f"Accumulated {len(all_results)} total futures")

    if execution_settings.job_framework == job_utils.JobFramework.dask_delayed:
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
    task_name = "EEC_mammoth"

    # Job execution configuration
    conda_environment_name = ""
    task_config = job_utils.TaskConfig(name=task_name, n_cores_per_task=1)
    # target_n_tasks_to_run_simultaneously = 120
    target_n_tasks_to_run_simultaneously = 110
    # target_n_tasks_to_run_simultaneously = 60
    log_level = logging.INFO
    walltime = "24:00:00"
    override_minimize_IO_as_possible = None
    debug_mode = False
    if debug_mode:
        # Usually, we want to run in the short queue
        target_n_tasks_to_run_simultaneously = 2
        walltime = "1:59:00"
    facility: job_utils.FACILITIES = (
        "hiccup_staging_std" if job_utils.hours_in_walltime(walltime) >= 2 else "hiccup_staging_quick"
    )
    # facility: job_utils.FACILITIES = "hiccup_std" if job_utils.hours_in_walltime(walltime) >= 2 else "hiccup_quick"
    # facility: job_utils.FACILITIES = "rehlers_mbp_m1pro"

    # Keep the job executor just to keep it alive
    job_executor, _job_framework_config, execution_settings = setup_job_framework(
        job_framework=job_framework,
        productions=productions,
        task_config=task_config,
        facility=facility,
        walltime=walltime,
        target_n_tasks_to_run_simultaneously=target_n_tasks_to_run_simultaneously,
        log_level=log_level,
        conda_environment_name=conda_environment_name,
        override_minimize_IO_as_possible=override_minimize_IO_as_possible,
        debug_mode=debug_mode,
    )
    all_results = setup_and_submit_tasks(
        productions=productions,
        task_config=task_config,
        execution_settings=execution_settings,
        job_executor=job_executor,
    )

    process_futures(productions=productions, all_results=all_results, job_framework=job_framework)

    return all_results


if __name__ == "__main__":
    # run(job_framework=job_utils.JobFramework.immediate_execution_debug)
    run(job_framework=job_utils.JobFramework.parsl)
