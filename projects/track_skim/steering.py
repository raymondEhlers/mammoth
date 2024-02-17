# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %load_ext rich

# ### Cross check environment

# +
from __future__ import annotations

import logging
import os

print(os.getenv("CONDA_BUILD_SYSROOT"))
print(os.getenv("ROOUNFOLD_ROOT"))
print(os.getenv("LD_LIBRARY_PATH"))
print(os.getenv("PATH"))
# -

# # Running jobs

# ## General setup

# +
from importlib import reload
from pathlib import Path

from mammoth import job_utils
from mammoth.hardest_kt import produce_flat_skim_from_track_skim

# -

# ## Reload

# reload(job_utils)
reload(produce_flat_skim_from_track_skim)


# ## Tasks

# ### Running


def define_productions() -> list[produce_flat_skim_from_track_skim.production.ProductionSettings]:
    # We want to provide the opportunity to run multiple productions at once.
    # We'll do so by defining each production below and then iterating over them below
    productions = []

    # Create and store production information
    _here = Path(produce_flat_skim_from_track_skim.__file__).parent
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
            # produce_flat_skim_from_track_skim.production.ProductionSettings.read_config(
            #     collision_system="pp",
            #     number=60,
            #     specialization=produce_flat_skim_from_track_skim.HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            produce_flat_skim_from_track_skim.production.ProductionSettings.read_config(
                collision_system="pythia",
                number=4,
                specialization=produce_flat_skim_from_track_skim.HardestKtProductionSpecialization(),
                track_skim_config_filename=config_filename,
            ),
        ]
    )

    # Write out the production settings
    for production_settings in productions:
        production_settings.store_production_parameters()

    return productions


# +
# Job execution parameters
productions = define_productions()

# Base settings
job_framework = job_utils.JobFramework.dask_delayed
# job_framework = job_utils.JobFramework.immediate_execution_debug
facility: job_utils.FACILITIES = "rehlers_mbp_m1pro"
# conda_environment_name = "substructure_c_24_06"
conda_environment_name = ""

# Job execution configuration
task_name = "hardest_kt_mammoth"
task_config = job_utils.TaskConfig(name=task_name, n_cores_per_task=1)
if task_config.n_cores_per_task > 1:
    facility = "rehlers_mbp_m1pro_multi_core"
# Formerly n_cores_to_allocate, this new variable is == n_cores_to_allocate if n_cores_per_task == 1
target_n_tasks_to_run_simultaneously = 8
log_level = logging.INFO
walltime = "24:00:00"
debug_mode = False

if debug_mode:
    # Usually, we want to run in the short queue
    target_n_tasks_to_run_simultaneously = 4
    walltime = "1:59:00"

# Keep the job executor just to keep it alive
job_executor, job_cluster = produce_flat_skim_from_track_skim.setup_job_framework(
    job_framework=job_framework,
    productions=productions,
    task_config=task_config,
    facility=facility,
    walltime=walltime,
    target_n_tasks_to_run_simultaneously=target_n_tasks_to_run_simultaneously,
    log_level=log_level,
    conda_environment_name=conda_environment_name,
)
# -

# ## Status

job_executor

# ## Submit jobs

futures = produce_flat_skim_from_track_skim.setup_and_submit_tasks(
    productions=productions,
    task_config=task_config,
    job_framework=job_framework,
    debug_mode=debug_mode,
    job_executor=job_executor if job_framework == job_utils.JobFramework.dask_delayed else None,  # type: ignore[arg-type]
)
# futures

futures

futures[0].result()

all(f.result() for f in futures[2:])

import numpy as np

np.count_nonzero([r.result()[0] == False for r in futures]) / len(futures)  # noqa: E712

# ## Cleanup

job_executor.close()
job_cluster.close()
