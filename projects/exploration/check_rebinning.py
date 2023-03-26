"""Check rebinning with a final result

Basically, the goal here is to confirm that we do the same thing as ROOT!

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

import logging
from typing import Any

import numpy as np
from pachyderm import binned_data

from mammoth import helpers

logger = logging.getLogger(__name__)

TH1D = Any

def root_example() -> tuple[TH1D, TH1D]:
    # Delay import to avoid explicit dependenceT
    import ROOT

    binning = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
    h_ROOT = ROOT.TH1D("test", "test", len(binning) - 1, binning)
    h_ROOT.Sumw2()
    binning_broader = np.array([0.25, 0.5, 1, 1.5, 2, 3, 6], dtype=np.float64)
    h_ROOT_broader = ROOT.TH1D("test_broader", "test_broader", len(binning_broader) - 1, binning_broader)
    h_ROOT_broader.Sumw2()

    # Mock up data
    values_filled = []
    for i, bin_center in enumerate((binning[1:] - binning[:-1]) / 2 + binning[:-1], start=1):
        logger.info(f"{i=}, {bin_center=}")
        for _j in range(10-i):
            values_filled.append((i, 10-i))
            h_ROOT.Fill(bin_center, 10-i)
            h_ROOT_broader.Fill(bin_center, 10-i)

    logger.info(f"{values_filled=}")

    h_temp = binned_data.BinnedData.from_existing_data(h_ROOT)
    logger.info(f"{h_temp.values=}, {h_temp.variances=}")
    h_temp_broader = binned_data.BinnedData.from_existing_data(h_ROOT_broader)
    logger.info(f"{h_temp_broader.values=}, {h_temp_broader.variances=}")
    logger.info("=== Done creating hists")

    return h_ROOT, h_ROOT_broader


def test_rebin(h_ROOT: TH1D, h_ROOT_broader: TH1D) -> bool:
    h_temp = binned_data.BinnedData.from_existing_data(h_ROOT)
    h_temp_broader = binned_data.BinnedData.from_existing_data(h_ROOT_broader)
    binning_broader = h_temp_broader.axes[0].bin_edges

    # ROOT
    h_ROOT_rebinned = h_ROOT.Rebin(len(binning_broader)-1, "test_rebinned", binning_broader)
    h_temp_rebinned = binned_data.BinnedData.from_existing_data(h_ROOT_rebinned)
    logger.info(f"{h_temp_rebinned.values=}, {h_temp_rebinned.variances=}")
    h_temp_already_rebinned = binned_data.BinnedData.from_existing_data(h_ROOT_broader)

    # binned_data
    h_binned_data_rebin = h_temp[::binning_broader]  # type: ignore[misc]
    logger.info(f"{h_binned_data_rebin.axes[0].bin_edges=}")
    logger.info(f"{h_binned_data_rebin.values=}, {h_binned_data_rebin.variances=}")

    np.testing.assert_allclose(h_temp_rebinned.values, h_binned_data_rebin.values)
    np.testing.assert_allclose(h_temp_rebinned.variances, h_binned_data_rebin.variances)
    assert h_temp_rebinned == h_binned_data_rebin
    np.testing.assert_allclose(h_temp_already_rebinned.values, h_binned_data_rebin.values)
    np.testing.assert_allclose(h_temp_already_rebinned.variances, h_binned_data_rebin.variances)
    assert h_temp_already_rebinned == h_binned_data_rebin

    return True

def test_steer_rebinning() -> None:

    h_ROOT, h_ROOT_broader = root_example()
    # First, without bin width scaling
    logger.info("===== Testing before scaling")
    test_rebin(h_ROOT=h_ROOT, h_ROOT_broader=h_ROOT_broader)
    # Next, try to do bin width scaling on the ROOT hists, and then rebin
    h_ROOT_scale = h_ROOT.Clone("h_ROOT_scale")
    h_ROOT_scale.Scale(1, "width")
    h_ROOT_broader_scale = h_ROOT_broader.Clone("h_ROOT_broader_scale")
    h_ROOT_broader_scale.Scale(1, "width")
    logger.info("===== Testing after scaling")
    try:
        test_rebin(h_ROOT=h_ROOT_scale, h_ROOT_broader=h_ROOT_broader_scale)
    except AssertionError:
        logger.info("Failed as expected!")
    # Finally, take the bin width scaled, unscale it, rebin and then compare to the h_ROOT_broader_scale (ie. right binning and then scaled)
    # (I would do this with the ROOT hist too, but I don't know how to undo bin width scaling unless I do it by hand, which sounds annoying)
    h_binned_data_handled_properly = binned_data.BinnedData.from_existing_data(h_ROOT_scale)
    # Undo scaling
    h_binned_data_handled_properly *= h_binned_data_handled_properly.axes[0].bin_widths
    # Rebin
    h_binned_data_handled_properly = h_binned_data_handled_properly[::binned_data.BinnedData.from_existing_data(h_ROOT_broader_scale).axes[0].bin_edges]
    # Scale by bin width
    h_binned_data_handled_properly /= h_binned_data_handled_properly.axes[0].bin_widths
    # Finally, compare the scale + rebin with the rebin + scale
    # And we'll pass it back as a ROOT hist.
    logger.info("==== Testing after handled properly")
    test_rebin(h_ROOT=h_binned_data_handled_properly.to_ROOT(), h_ROOT_broader=h_ROOT_broader_scale)


if __name__ == "__main__":
    helpers.setup_logging()
    test_steer_rebinning()