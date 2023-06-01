"""Functionality related to conventions for running analyses.

DEPRECATED!

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import uproot


def description_from_parameters(parameters: Mapping[str, Any]) -> str:
    return ", ".join([f"{k}={v}" for k, v in parameters.items()])


def check_for_root_skim_output_file(output_filename: Path, description: str) -> tuple[bool, str]:
    # Try to bail out as early to avoid reprocessing if possible.
    # First, check for the empty filename
    empty_filename = output_filename.with_suffix(".empty")
    if empty_filename.exists():
        # It will be empty, so there's nothing to check. Just return
        return (True, f"Done - found empty file indicating that there are no jets to analyze for {description}")

    # Next, check the contents of the output file
    if output_filename.exists():
        try:
            with uproot.open(output_filename) as f:
                # If the tree exists, can be read, and has more than 0 entries, we should be good
                if f["tree"].num_entries > 0:
                    # Return immediately to indicate that we're done.
                    return (True, f"already processed for {description}")
        except Exception:
            # If it fails for some reason, give up - we want to try again
            pass

    return (False, "")
