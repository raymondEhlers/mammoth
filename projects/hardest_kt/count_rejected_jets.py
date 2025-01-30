"""Count the number of rejected jets

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import uproot

from mammoth import helpers

logger = logging.getLogger(__name__)


def count(hist_path: Path) -> dict[str, np.float64]:
    # Grab the hist
    with uproot.open(hist_path) as f:
        hist = f["data_jet_n_accepted"].to_hist()

    # Initialize the dictionary to store the results
    results = {}

    # Get the bin values. The names are convenient, so we'll retrieve those too
    # (couldn't find a way to them in an array-like way)
    bin_values = hist.values()
    names = [hist.axes[0].bin(i) for i in range(hist.axes[0].size)]
    names_and_values = list(zip(names, bin_values, strict=True))

    # Iterate over the bin values and compare each bin value as a fraction to the previous bin
    for (lower_bin_name, lower_bin), (upper_bin_name, upper_bin) in itertools.pairwise(names_and_values):
        # Handle division by zero
        fraction = upper_bin / lower_bin if lower_bin != 0 else np.inf
        # Return percentages - it's more convenient.
        results[f"{upper_bin_name} / {lower_bin_name}"] = fraction * 100

    return results


if __name__ == "__main__":
    # Setup
    helpers.setup_logging()
    logger.info("Percentage of jets accepted after each step")

    base_path = Path("trains") / "PbPb" / "0073"
    # hadd
    logger.warning("hadd ")
    result_hadd = count(hist_path=base_path / "skim_hists_merged_hadd.root")
    logger.info(f"{result_hadd=}")

    # shadd (cross check)
    logger.warning("shadd")
    result_shadd = count(hist_path=base_path / "skim_hists_merged_hadd.root")
    logger.info(f"{result_shadd=}")

    # And complete the cross check for completeness
    assert result_hadd == result_shadd
    logger.info("Shadd and hadd match :-)")
