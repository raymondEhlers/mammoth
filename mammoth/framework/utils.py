""" Helpers and utilities.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import collections.abc
from typing import Optional, Sequence, Union

import attr

import awkward as ak
import numpy as np


@attr.s(frozen=True)
class Range:
    min: Optional[float] = attr.ib()
    max: Optional[float] = attr.ib()


def _groupby_lexsort(array: ak.Array, columns: Sequence[Union[str, int]]) -> ak.Array:
    """Sort for groupby."""
    sort = np.lexsort(tuple(np.asarray(array[:, col]) for col in reversed(columns)))
    return array[sort]


def group_by(array: ak.Array, by: Sequence[Union[str, int]]) -> ak.Array:
    """ Group by for awkward arrays.

    Args:
        array: Array to be grouped. Must be convertable to numpy arrays.
        by: Names or indices of columns to group by. The first column is the primary index for sorting,
            second is secondary, etc.
    Returns:
        Array grouped by the columns.
    """
    # Validation
    if not isinstance(by, collections.abc.Iterable):
        by = [by]

    # First, sort
    sorted = _groupby_lexsort(array=array, columns=by)

    # And then construct the array
    return ak.unflatten(
        sorted, ak.run_lengths(sorted[:, by[-1]])
    )

