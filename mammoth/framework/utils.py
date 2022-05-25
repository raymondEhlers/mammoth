""" Helpers and utilities.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import collections.abc
from pathlib import Path
from typing import List, Optional, Sequence, Union

import attr

import awkward as ak
import numpy as np


@attr.frozen
class Range:
    min: Optional[float]
    max: Optional[float]


def expand_wildcards_in_filenames(paths: Sequence[Path]) -> List[Path]:
    return_paths: List[Path] = []
    for path in paths:
        p = str(path)
        if "*" in p:
            # Glob all associated filenames.
            # NOTE: This assumes that the paths are relative to the execution directory. But that's
            #       almost always the case.
            return_paths.extend(list(Path(".").glob(p)))
        else:
            return_paths.append(path)

    # Sort in the expected order (just according to alphabetical, which should handle numbers
    # fine as long as they have leading 0s (ie. 03 instead of 3)).
    return_paths = sorted(return_paths, key=lambda p: str(p))
    return return_paths


def ensure_and_expand_paths(paths: Sequence[Union[str, Path]]) -> List[Path]:
    return expand_wildcards_in_filenames([Path(p) for p in paths])


def _lexsort_for_groupby(array: ak.Array, columns: Sequence[Union[str, int]]) -> ak.Array:
    """Sort for groupby."""
    sort = np.lexsort(tuple(np.asarray(array[:, col]) for col in reversed(columns)))  # type: ignore
    return array[sort]


def group_by(array: ak.Array, by: Sequence[Union[str, int]]) -> ak.Array:
    """Group by for awkward arrays.

    Args:
        array: Array to be grouped. Must be convertible to numpy arrays.
        by: Names or indices of columns to group by. The first column is the primary index for sorting,
            second is secondary, etc.
    Returns:
        Array grouped by the columns.
    """
    # Validation
    if not (isinstance(by, collections.abc.Sequence) and not isinstance(by, str)):
        by = [by]

    # First, sort
    # See: https://stackoverflow.com/a/64053838/12907985
    sorted_array = _lexsort_for_groupby(array=array, columns=by)

    # Now, we need to combine the run lengths from the different columns. We need to split
    # every time any of them change.
    run_lengths = [ak.run_lengths(sorted_array[:, k]) for k in by]
    # We can match them up more easily by using the starting index of each run.
    run_starts = [np.cumsum(np.asarray(l)) for l in run_lengths]  # noqa: E741
    # Combine all of the indices together into one array. Note that this isn't unique.
    combined = np.concatenate(run_starts)  # type: ignore
    # NOTE: Unique can be done more efficient than using the naive implementation.
    #       See: https://stackoverflow.com/a/12427633/12907985
    #       However, it's not worth messing around with at the moment (May 2022).
    combined = np.unique(combined)  # type: ignore

    run_length = np.zeros(len(combined), dtype=np.int64)
    run_length[0] = combined[0]
    # run_length[1:] = combined[1:] - combined[:-1]
    run_length[1:] = np.diff(combined)  # type: ignore

    # And then construct the array
    return ak.unflatten(
        # Somehow, these seem to be equivalent, even though they shouldn't be...
        # sorted_array, ak.run_lengths(sorted_array[:, by[-1]])
        sorted_array,
        run_length,
    )
