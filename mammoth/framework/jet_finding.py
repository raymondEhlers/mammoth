""" Jet finding

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from typing import Any

import awkward as ak
import numpy as np
import vector

#import fastjet_wrapper as fj_wrapper

fj_wrapper: Any = object()

vector.register_awkward()

def find_jets(particles: ak.Array, jet_R: float, algorithm: fj_wrapper.JetAlgorithm = fj_wrapper.JetAlgorithm.anti_kt) -> ak.Array:
    counts = ak.num(particles, axis=1)
    flattened_particles = ak.flatten(particles)
    sum_counts = np.cumsum(np.asarray(counts))

    for lower, upper in zip(sum_counts[:-1], sum_counts[1:]):
        # Skip if there are somehow no particles.
        if lower == upper:
            continue

        res = fj_wrapper.find_jets(
            px=flattened_particles[lower:upper].px,
            py=flattened_particles[lower:upper].py,
            pz=flattened_particles[lower:upper].pz,
            E=flattened_particles[lower:upper].E,
            jet_R=jet_R,
            algorithm=algorithm,
        )

        if len(res) == 3:
            # NOTE: Here, constituent_indices is the _subtracted_ constituent indices.
            jetsArray, constituent_indices, (subtracted_constituents, subtracted_to_unsubtracted_indices) = res
        else:
            jetsArray, constituent_indices = res

        # Make an output based on this information...
        ...
        jets = ak.Array(
            ak.layout.ListOffsetArray64(
                ak.layout.Index64(np.array([0, len(array)])),
                ak.layout.RecordArray(
                    [
                        ak.layout.NumpyArray(jetsArray[0]),
                        ak.layout.NumpyArray(jetsArray[1]),
                        ak.layout.NumpyArray(jetsArray[2]),
                        ak.layout.NumpyArray(jetsArray[3]),
                    ],
                    ["px", "py", "pz", "E"],
                ),
            )
        )

    return jets

