"""Helpers for working with arrays

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from typing import Any

import awkward as ak
import numba as nb
import numpy as np
import numpy.typing as npt

from mammoth_cpp._ext import smooth_array, smooth_array_f  # noqa: F401

logger = logging.getLogger(__name__)


@nb.njit  # type: ignore[misc]
def _random_choice_jagged(local_index: ak.Array, n_counts: int, random_seed: int | None) -> npt.NDArray[np.bool_]:
    # Validation
    if random_seed is not None:
        # NOTE: This needs to be kept using the old interface because it needs to compile with numba,
        #       and numba doesn't yet support the new np rng interface
        np.random.seed(random_seed)  # noqa: NPY002
    # Setup
    mask = np.zeros(n_counts, dtype=np.bool_)

    i = 0
    for indices in local_index:
        if len(indices):
            selected = np.random.choice(np.asarray(indices))  # noqa: NPY002
            mask[i + selected] = True
            i += len(indices)

    return mask


def random_choice_jagged(arrays: ak.Array, random_seed: int | None = None) -> ak.Array:
    """Generate a mask which randomly select one element from each jagged entry.

    Args:
        arrays: The array from which to select one element per jagged entry. (i.e. if event structured,
            then pick one element per event.)
        random_seed: The random seed to use for the random selection. Default: None, which uses the default
            RNG seed.

    Returns:
        A mask which selects one random element per jagged entry.
    """
    mask = _random_choice_jagged(local_index=ak.local_index(arrays), n_counts=ak.count(arrays), random_seed=random_seed)
    return ak.unflatten(mask, ak.num(arrays))


def shape_like(flat_array: npt.NDArray[np.number[npt.NBitBase]], array_with_desired_structure: ak.Array) -> ak.Array:
    """Convert a flat array to the same structure as another array.

    Code from https://github.com/scikit-hep/awkward/discussions/3101#discussioncomment-9332008,
    which I asked for help in how best to generically convert a flat array into an array with
    a particular structure. For a further explanation of how this works, see the `ak.transform`
    documentation - it's pretty non-trivial.

    Args:
        flat_array: The flat array to convert to the structure of the other array.
        array_with_desired_structure: The array which has the desired structure.
    Returns:
        The flat array converted to have the same structure as the array with the desired structure.
    """
    flat_array_as_layout = ak.to_layout(flat_array)
    # flat_array needs to be flat
    assert flat_array_as_layout.is_numpy

    def transformation(layout: ak.contents.content.Content, **kwargs: Any) -> ak.contents.content.Content | None:  # noqa: ARG001
        # this function only makes sense for non-branching layouts
        assert not layout.is_record and not layout.is_union  # noqa: PT018

        if layout.is_numpy:
            # they need to have the same number of numerical values
            assert len(layout) == len(flat_array)
            return flat_array_as_layout
        return None

    return ak.transform(transformation, array_with_desired_structure)
