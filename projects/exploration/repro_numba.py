""" Reproduce awkward 1.5.1 issue with numba

Reported in https://github.com/scikit-hep/awkward-1.0/issues/1158

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import pickle

import awkward as ak
import numba as nb


@nb.njit  # type: ignore
def repro(
    generator_like_jet_constituents: ak.Array,
) -> None:
    for i, generator_like_constituents in enumerate(generator_like_jet_constituents):
        s = 0
        for generator_like_constituent in generator_like_constituents:
            s += generator_like_constituent.pt

def reproducer() -> None:
    # NOTE: Need to write with to_buffers to maintain the right structure to reproduce this issue
    with open("repro.pkl", "rb") as f:
        a = ak.from_buffers(*pickle.load(f))
    repro(generator_like_jet_constituents=ak.packed(a.constituents))

if __name__ == "__main__":
    reproducer()