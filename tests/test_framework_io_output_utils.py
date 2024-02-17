"""Tests for the framework.io.output_utils module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np
import pytest
import uproot
from pachyderm import binned_data

# from mammoth.framework.io import output_utils

logger = logging.getLogger(__name__)


def _consistency_check(h_root: Any, h_uproot: Any) -> bool:
    """Check hists for consistency."""
    r = binned_data.BinnedData.from_existing_data(h_root)
    u = binned_data.BinnedData.from_existing_data(h_uproot)
    logger.warning("Yup")
    return r == u


def _generate_profile_with_root(ROOT: Any) -> Any:
    # Create the TProfile
    # NOTE: We need the random string to avoid clobbering existing hists
    tag = uuid.uuid4()
    profile_root = ROOT.TProfile(f"test_{tag}", f"test_{tag}", 10, -0.5, 9.5)
    for i in range(5):
        profile_root.Fill(i, i + 0.5, 1 + (i * 0.1))
    return profile_root


def test_hist_profile_round_trip() -> None:
    """Test round trip of TProfile through hist.

    This isn't guaranteed by uproot, so we need to confirm it all works.
    """
    # TODO: Conditionally create the TProfile. We should normally save it...
    ROOT = pytest.importorskip("ROOT")
    # Create the TProfile
    profile_root = _generate_profile_with_root(ROOT)

    # Convert to uproot
    profile_uproot = uproot.from_pyroot(profile_root)

    # Check the values for consistency
    assert _consistency_check(profile_root, profile_uproot)


@pytest.mark.xfail(reason="Hist doesn't support adding TProfile properly yet (Jan 2024).")
def test_hist_merged_profile_round_trip() -> None:
    """Test round trip of TProfile through hist.

    This isn't guaranteed by uproot, so we need to confirm it all works.
    """
    # TODO: Conditionally create the TProfile. We should normally save it...
    ROOT = pytest.importorskip("ROOT")
    # Create the TProfile
    profile_root = _generate_profile_with_root(ROOT)

    # Convert to uproot
    profile_uproot = uproot.from_pyroot(profile_root)

    # Check the values for consistency
    assert _consistency_check(profile_root, profile_uproot)

    # Add the profiles within their respective types
    # NOTE: We recreate the profiles to avoid memory + ownership issues. There's something subtle here,
    #       but I'm not sure what it is. This is easiest. (Maybe just because they don't implement __add__ properly?)
    merged_profile_root = _generate_profile_with_root(ROOT)
    merged_profile_root.Add(_generate_profile_with_root(ROOT))
    # Next, the uproot derived hists
    # NOTE: We of course want to convert to hist because we want to test this round trip. Uproot tests the standard one.
    merged_profile_uproot = profile_uproot.to_hist() + profile_uproot.to_hist()
    # Need to convert uproot NaNs to 0s to compare with root
    with uproot.recreate("test_temp.root") as f:
        f["test"] = merged_profile_uproot
    merged_profile_uproot = binned_data.BinnedData.from_existing_data(merged_profile_uproot)
    logger.info(f"merged_profile_uproot.values: {merged_profile_uproot.values}")
    logger.info(f"merged_profile_uproot.variances: {merged_profile_uproot.variances}")
    merged_profile_uproot.values[np.isnan(merged_profile_uproot.values)] = 0
    merged_profile_uproot.variances[np.isnan(merged_profile_uproot.variances)] = 0
    assert _consistency_check(
        merged_profile_root,
        merged_profile_uproot,
    )

    # with tempfile.NamedTemporaryFile() as output_file:
    #    with ROOT.TFile.Open(output_file.name, "RECREATE") as f:
    #        prof.Write()

    #    output_file.seek(0)


# TODO: To be implemented...
# def test_shadd() -> None:
#    """Test shadd, comparing to a known result from hadd."""
#    ...
