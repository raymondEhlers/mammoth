"""Tests for utility functions.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""
from __future__ import annotations

import awkward as ak
import pytest

from mammoth.framework import utils


@pytest.mark.parametrize(
    "columns",
    [
        [0, 1],
    ], ids = ["first and second"])
def test_group_by(columns: list[int]) -> None:
    # fmt: off
    array = ak.Array([
        [   2.,    1.,    2.,    0.],
        [   4.,    2.,    4.,    0.],
        [   2.,    1.,  100.,    0.],
        [   3.,    1.,    2.,    0.],
        [   3.,    3.,    6.,    0.],
        [   2.,    2.,  100.,    0.],
        [   4.,    2.,    2.,    0.],
        [   3.,    1.,    4.,    0.],
        [   4.,    2.,    6.,    0.],
        [   5.,    2.,    6.,    0.],
    ])
    group_by = utils.group_by(array=array, by=columns)

    #desired_result = ak.Array([
    #    [   2.,   1.,   2.,   0.],
    #    [   3.,   1.,   2.,   0.],
    #    [   4.,   1.,   2.,   0.],
    #    [   2.,   2., 100.,   0.],
    #    [   3.,   2.,   4.,   0.],
    #    [   4.,   2.,   4.,   0.],
    #    [   2.,   3., 100.,   0.],
    #    [   3.,   3.,   6.,   0.],
    #    [   4.,   3.,   6.,   0.],
    #])

    #desired_result = ak.Array([
    #    [   2.,   1.,   2.,   0.],
    #    [   2.,   1., 100.,   0.],
    #    [   2.,   2., 100.,   0.],
    #    [   3.,   1.,   2.,   0.],
    #    [   3.,   1.,   4.,   0.],
    #    [   3.,   3.,   6.,   0.],
    #    [   4.,   2.,   2.,   0.],
    #    [   4.,   2.,   4.,   0.],
    #    [   4.,   2.,   6.,   0.],
    #])

    desired_result_group_by_first = ak.Array([
        [
            [   2.,   1.,   2.,   0.],
            [   2.,   1., 100.,   0.],
            [   2.,   2., 100.,   0.],
        ],
        [
            [   3.,   1.,   2.,   0.],
            [   3.,   3.,   6.,   0.],
            [   3.,   1.,   4.,   0.],
        ],
        [
            [   4.,   2.,   4.,   0.],
            [   4.,   2.,   2.,   0.],
            [   4.,   2.,   6.,   0.],
        ],
        [
            [   5.,    2.,    6.,    0.]
        ],
    ])

    desired_result_group_by_first_and_second = ak.Array([
        [
            [   2.,   1.,   2.,   0.],
            [   2.,   1., 100.,   0.],
        ],
        [
            [   2.,   2., 100.,   0.],
        ],
        [
            [   3.,   1.,   2.,   0.],
            [   3.,   1.,   4.,   0.],
        ],
        [
            [   3.,   3.,   6.,   0.],
        ],
        [
            [   4.,   2.,   4.,   0.],
            [   4.,   2.,   2.,   0.],
            [   4.,   2.,   6.,   0.],
        ],
        [
            [   5.,    2.,    6.,    0.]
        ]
    ])
    # fmt: on

    comparison = desired_result_group_by_first
    if len(columns) == 2:
        comparison = desired_result_group_by_first_and_second

    assert ak.all(
        ak.flatten(group_by == comparison, axis=None)
    )

