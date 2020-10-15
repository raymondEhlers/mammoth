#!/usr/bin/env python3

""" Run multiple configurations on a single node.

This is more efficienct than requesting individual jobs.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import os.path
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import psutil

@dataclass
class Config:
    sqrts: int
    trigger: str

    def __str__(self):
        return f"{self.sqrts}_{self.trigger}"


def run_job_with_config(scratch_dir: Path, base_config_dir: Path, model: str, config_name: str, node_type: str, index: int) -> None:
    stdout = (base_config_dir / model / node_type / f"{config_name}_{index}").with_suffix(".log")
    if not stdout.parent.exists():
        stdout.parent.mkdir(parents=True, exist_ok=True)
    # Need to use Popen to ensure that we're not blocking.
    res = subprocess.Popen(
        ["singularity", "run", "--cleanenv", scratch_dir / "jetscape-stat-estimate_latest.sif", (base_config_dir / model / config_name).with_suffix(".xml")],
        #["sleep", "3"],
        #stdout=(base_config_dir / model / node_type / config_name).with_suffix(".log")
        stdout=open(stdout, "w"),
        stderr=subprocess.STDOUT,
    )
    return res
    #subprocess.run(
    #    #["echo", "singularity", "shell", "--cleanenv", scratch_dir / "jetscape-stat-estimate_latest.sif", (base_config_dir / model / config_name).with_suffix(".xml")],
    #    ["sleep", "10"],
    #    #stdout=(base_config_dir / model / node_type / config_name).with_suffix(".log")
    #    stderr=subprocess.STDOUT,
    #)


def setup_all_configs() -> Dict[str, List[Config]]:
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
    return configs

def monitor_processes(processes: Sequence[subprocess.Popen]) -> None:
    # Convert to psutil Process so we can monitor them.
    process_list = [psutil.Process(p.pid) for p in processes]
    # Will return will all processes are completed.
    gone, alive = psutil.wait_procs(process_list)


if __name__ == "__main__":
    # Settings
    n_cores = 6
    node_type = f"KNL_{n_cores}"
    scratch_dir = Path("scratch_dir")
    #scratch_dir = Path(os.path.expandvars("$SCRATCH"))

    available_configs = setup_all_configs()

    processes = []
    for i in range(10):
        print(f"Starting {i}")
        processes.append(run_job_with_config(
            scratch_dir=scratch_dir,
            base_config_dir=scratch_dir / "jetscape-an" / "config" / "jetscape" / "STAT",
            model="lbt",
            config_name="5020_single_hard_trigger",
            node_type=node_type,
            index=i,
        ))
        print(f"Finished {i}")
        # Try to avoid going too crazy with the I/O.
        time.sleep(10)

    print(processes)

    monitor_processes(processes=processes)
