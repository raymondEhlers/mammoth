"""Run multiple configurations on a single node.

This is more efficient than requesting individual jobs.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import os.path
import shutil
import subprocess
import time
import timeit
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass
class Config:
    sqrts: int
    trigger: str

    def __str__(self) -> str:
        return f"{self.sqrts}_{self.trigger}_trigger"


def run_job_with_config(
    scratch_dir: Path, base_config_dir: Path, model: str, config_name: str, node_type: str, index: int | None = None
) -> subprocess.Popen:  # type: ignore[type-arg]
    # Setup the config path
    config_path = (base_config_dir / model / config_name).with_suffix(".xml")
    # We need to separate this into a new directory so we don't overwrite it.
    new_config_path = config_path.parent / node_type
    if index is not None:
        new_config_path = new_config_path / str(index)
    new_config_path = (new_config_path / config_name).with_suffix(".xml")
    # Create our new directory...
    if not new_config_path.parent.exists():
        new_config_path.parent.mkdir(parents=True, exist_ok=True)
    # ... and copy the config to the new location...
    shutil.copy(config_path, new_config_path)
    # ... and then update our config path to the new file.
    config_path = new_config_path

    # Setup stdout
    stdout = (config_path.parent / config_name).with_suffix(".log")
    if not stdout.parent.exists():
        stdout.parent.mkdir(parents=True, exist_ok=True)
    # Need to use Popen to ensure that we're not blocking.
    res = subprocess.Popen(
        ["time", "singularity", "run", "--cleanenv", scratch_dir / "jetscape-stat-estimate_latest.sif", config_path],
        # ["sleep", "3"],
        # stdout=(base_config_dir / model / node_type / config_name).with_suffix(".log")
        stdout=open(stdout, "w"),  # noqa: SIM115,PTH123
        stderr=subprocess.STDOUT,
    )

    return res  # noqa: RET504
    # subprocess.run(
    #    #["echo", "singularity", "shell", "--cleanenv", scratch_dir / "jetscape-stat-estimate_latest.sif", (base_config_dir / model / config_name).with_suffix(".xml")],
    #    ["sleep", "10"],
    #    #stdout=(base_config_dir / model / node_type / config_name).with_suffix(".log")
    #    stderr=subprocess.STDOUT,
    # )


def setup_all_configs() -> dict[str, list[Config]]:
    base_configs = [
        Config(sqrts=5020, trigger="single_hard"),
        Config(sqrts=5020, trigger="neutral"),
        Config(sqrts=2760, trigger="single_hard"),
        Config(sqrts=2760, trigger="neutral"),
        Config(sqrts=200, trigger="single_hard"),
        Config(sqrts=200, trigger="neutral"),
    ]
    configs = {
        "lbt": list(base_configs),
        "matter": list(base_configs),
        "martini": list(base_configs),
        "matter_lbt": list(base_configs),
        "matter_martini": list(base_configs),
    }
    return configs  # noqa: RET504


def monitor_processes(processes: Sequence[subprocess.Popen]) -> None:  # type: ignore[type-arg]
    # Convert to psutil Process so we can monitor them.
    process_list = [psutil.Process(p.pid) for p in processes]
    # Will return will all processes are completed.
    gone, alive = psutil.wait_procs(process_list)


if __name__ == "__main__":
    # Settings
    n_cores = 48
    node_type = f"skylake_{n_cores}"
    # scratch_dir = Path("scratch_dir")
    scratch_dir = Path(os.path.expandvars("$SCRATCH"))

    available_configs = setup_all_configs()

    start_time = timeit.default_timer()
    processes = []
    i = 0
    for model in available_configs:
        for config in available_configs[model]:
            print(f"Starting {i}")
            processes.append(
                run_job_with_config(
                    scratch_dir=scratch_dir,
                    base_config_dir=scratch_dir / "jetscape-an" / "config" / "jetscape" / "STAT",
                    # base_config_dir=Path("..") / "config" / "jetscape" / "STAT",
                    model=model,
                    config_name=str(config),
                    node_type=node_type,
                    index=i,
                )
            )
            print(f"Started {i}")
            # Try to avoid going too crazy with the I/O.
            time.sleep(10)
            i += 1

    print(processes)

    monitor_processes(processes=processes)
    elapsed = timeit.default_timer() - start_time
    print(f"Done! Elapsed time since starting first process: {elapsed}")
