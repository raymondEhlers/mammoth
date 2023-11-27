"""Helpers for working with arrays

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging

import awkward as ak
import numba as nb
import numpy as np
import numpy.typing as npt

from mammoth_cpp._ext import smooth_array  # noqa: F401

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
    mask = _random_choice_jagged(local_index=ak.local_index(arrays), n_counts=ak.count(arrays), random_seed=random_seed)
    return ak.unflatten(mask, ak.num(arrays))
