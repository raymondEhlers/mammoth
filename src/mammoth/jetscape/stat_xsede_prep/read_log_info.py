""" Read log info to extract scaling time information.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""
from __future__ import annotations

import os.path
import pprint
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    sqrts: int
    trigger: str

    def __str__(self) -> str:
        return f"{self.sqrts}_{self.trigger}_trigger"


def read_log_for_runtime(filename: Path) -> float:
    """

    Returns:
        Runtime in seconds.
    """

    with filename.open() as f:
        # The "Real time" output by JETSCAPE should be on the fourth to last line.
        # It's slightly shorter than the full system time, but easier to parse,
        # and should be good enough.
        timing_line = f.readlines()[-4]
        real_time = float(timing_line.split()[2])

    return real_time  # noqa: RET504


def read_logs(base_path: Path, model: str, tag: str, config: Config) -> list[float]:
    p = base_path / model / tag

    times = []
    for d in sorted(p.glob("*")):
        times.append(read_log_for_runtime(filename=(d / str(config)).with_suffix(".log")))

    return times


def run(
    base_path: Path, models: Sequence[str], tags: Sequence[str], config: Config
) -> dict[str, dict[str, list[float]]]:
    output: dict[str, dict[str, list[float]]] = {}
    for tag in tags:
        output[tag] = {}
        for model in models:
            output[tag][model] = read_logs(base_path, model=model, tag=tag, config=config)
    return output


if __name__ == "__main__":
    # scratch_dir = Path("scratch_dir")
    scratch_dir = Path(os.path.expandvars("$SCRATCH"))
    res = run(
        base_path=scratch_dir / "jetscape-an" / "config" / "jetscape" / "STAT",
        models=["matter_lbt"],
        tags=[
            "skylake_6_seed_1_1",
            "skylake_36_seed_1_1",
            "skylake_42_seed_1_1",
            "skylake_48_seed_1_1",
        ],
        config=Config(
            sqrts=5020,
            trigger="single_hard",
        ),
    )

    pprint.pprint(res)  # noqa: T203
