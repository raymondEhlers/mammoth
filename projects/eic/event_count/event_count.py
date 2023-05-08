#!/usr/bin/env python3

"""One off script to count the number of events in a set of root trees.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path

import uproot

from mammoth import helpers

logger = logging.getLogger(__name__)


def run(p: Path) -> None:
    root_files = list(p.glob("*.root"))

    logger.info(f"Found {len(root_files)} files")

    n_events = 0
    for filename in root_files:
        with uproot.open(filename) as f:
            logger.info(f"{filename}: {f['event_tree'].num_entries}")
            n_events += f["event_tree"].num_entries

    logger.info(f"For {p}, there are {n_events} events")


if __name__ == "__main__":
    helpers.setup_logging()
    run(p=Path("/alf/data/rehlers/eic/official_prod/prop.4/prop.4.0/HFandJets/pythia8/ep-10x100-q2-100/eval_00002"))
