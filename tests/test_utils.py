"""Tests for utility functions.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from typing import Sequence

import pytest

import awkward as ak

from mammoth.framework import utils

@pytest.mark.parametrize(
    "columns",
    [
        #0,
        #[0],
        [0, 1],
    #], ids = ["first, int", "first, sequence", "first and second"])
    ], ids = ["first and second"])
def test_group_by(columns: Sequence[int]) -> None:
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

    comparison = desired_result_group_by_first
    if not isinstance(columns, int) and len(columns) == 2:
        comparison = desired_result_group_by_first_and_second

    assert ak.all(
        ak.flatten(group_by == comparison, axis=None)
    )

