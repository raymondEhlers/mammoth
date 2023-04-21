""" Tests for array helpers

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, UCB/LBL
"""

from __future__ import annotations

import logging
from typing import Any

import awkward as ak
import pytest

from mammoth.framework.analysis import array_helpers as analysis_array_helpers


@pytest.mark.parametrize("random_seed", [None, 123456])
def test_random_choice_jagged(caplog: Any, random_seed: int | None) -> None:
    # Setup
    caplog.set_level(logging.INFO)

    array = ak.Array([[1,2,3],[1,2],[],[2,3,4]])
    mask = analysis_array_helpers.random_choice_jagged(arrays=array, random_seed=random_seed)

    if random_seed is None:
        # We don't know the exact values since we didn't set the seed, but we can check the shape
        assert ak.count(array[mask], axis=-1).to_list() == [1, 1, 0, 1]
    else:
        assert array[mask].to_list() == [[2], [1], [], [4]]
