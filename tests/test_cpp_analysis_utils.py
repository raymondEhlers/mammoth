
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from mammoth_cpp._ext import smooth_array

def test_smooth_array(caplog: Any) -> None:
    n_times = 1
    arr = np.array([1,2,7,4,5], dtype=np.float64)
    #arr = np.array([1,3,1,1,1], dtype=np.float64)
    res = smooth_array(arr, n_times=n_times)

    # Cross check that the smoothing did something
    assert not np.allclose(arr, res)

    # Expected result for n_times = 1
    expected_result = np.array([1, 2.25, 3.75, 4.75, 5], dtype=np.float64)
    np.testing.assert_allclose(res, expected_result)


def smooth_array_ROOT(array: npt.NDArray[np.float64], n_times: int = 1) -> npt.NDArray[np.float64]:
    """ Smooths an array using ROOT TH1::Smooth() smoothing algorithm.

    Args:
        array: The array to smooth.

    Returns:
        The smoothed array.
    """
    ROOT = pytest.importorskip("ROOT")
    h = ROOT.TH1D("h", "h", len(array), 0, len(array))
    for i, v in enumerate(array):
        h.SetBinContent(i+1, v)

    out = np.zeros_like(array)
    h.Smooth(n_times)
    for i in range(len(array)):
        out[i] = h.GetBinContent(i+1)

    return out


@pytest.mark.parametrize("n_times", list(range(1, 4)), ids=lambda x: f"n_times={x}")
def test_ROOT_comparison(caplog: Any, n_times: int) -> None:
    """ Compare the smoothing algorithm to ROOT's TH1::Smooth() method."""
    ROOT = pytest.importorskip("ROOT")

    arr = np.array([1,2,7,4,5], dtype=np.float64)
    res = smooth_array(arr, n_times=n_times)
    res_ROOT = smooth_array_ROOT(arr, n_times=n_times)

    np.testing.assert_allclose(res, res_ROOT)
