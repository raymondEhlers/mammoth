""" Jet finding interface.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from typing import Dict, List, Optional, Tuple

import awkward as ak
import numpy as np
import vector

import mammoth._ext
from mammoth._ext import ConstituentSubtractionSettings

vector.register_awkward()

def find_jets(particles: ak.Array, jet_R: float,
              algorithm: str = "anti-kt",
              eta_range: Tuple[float, float] = (-0.9, 0.9),
              min_jet_pt: float = 1.0,
              background_subtraction: bool = False,
              constituent_subtraction: Optional[ConstituentSubtractionSettings] = None,
              ) -> ak.Array:
    # Validation
    # Without this, we may have argument mismatches.
    min_jet_pt = float(min_jet_pt)

    # Keep track of the event transitions.
    counts = ak.num(particles, axis=1)
    # To use for indexing, we need to keep track of the cumulative sum. That way, we can
    # slice using these indices.
    sum_counts = np.cumsum(np.asarray(counts))
    # However, to use as slices, we need one more entry than the number of events. We
    # account for this by inserting 0 at the beginning since the first indices starts at 0.
    sum_counts = np.insert(sum_counts, 0, 0)

    # Validate that there is at least one particle per event
    if np.any(sum_counts[1:] == sum_counts[:-1]):
        raise ValueError("There are some events with zero particles, which is going to mess up the alignment. Check the input!")

    # Now, deal with the particles themselves.
    # This will flatten the awkward array contents while keeping the record names.
    flattened_particles = ak.flatten(particles, axis=1)
    # We only want vector to calculate the components once (the input components may not
    # actually be px, py, pz, and E), so calculate them now, and view them as numpy arrays
    # so we can pass them directly into our function.
    # NOTE: To avoid argument mismatches when calling to the c++, we view as float64 (which
    #       will be converted to a double). As of July 2021, tests seem to indicate that it's =
    #       not making the float32 -> float conversion properly.
    px = np.asarray(flattened_particles.px, dtype=np.float64)
    py = np.asarray(flattened_particles.py, dtype=np.float64)
    pz = np.asarray(flattened_particles.pz, dtype=np.float64)
    E = np.asarray(flattened_particles.E, dtype=np.float64)

    #offsets = []
    # Keep track of the jet four vector components. Although this will have to be converted later,
    # it seems that this is good enough enough to start.
    # NOTE: If this gets too slow, we can do the jet finding over multiple events in c++ like what
    #       is done in the new fj bindings. I skip this for now because my existing code seems to 
    #       be good enough.
    jets: Dict[str, List[np.ndarray]] = {
        "px": [],
        "py": [],
        "pz": [],
        "E": [],
    }
    constituent_indices = []
    #subtracted_constituents = []
    #subtracted_to_unsubtracted_indices = []
    for lower, upper in zip(sum_counts[:-1], sum_counts[1:]):
        # Run the actual jet finding.
        res = mammoth._ext.find_jets(
            px=px[lower:upper],
            py=py[lower:upper],
            pz=pz[lower:upper],
            E=E[lower:upper],
            jet_R=jet_R,
            jet_algorithm=algorithm,
            eta_range=eta_range,
            min_jet_pt=min_jet_pt,
            background_subtraction=background_subtraction,
            constituent_subtraction=constituent_subtraction,
        )

        temp_jets = res.jets
        jets["px"].append(temp_jets[0])
        jets["py"].append(temp_jets[1])
        jets["pz"].append(temp_jets[2])
        jets["E"].append(temp_jets[3])
        constituent_indices.append(res.constituent_indices)
        #if len(res) == 3:
        #    # NOTE: Here, constituent_indices is the _subtracted_ constituent indices.
        #    subtracted_constituents.append(res[2][0])
        #    subtracted_to_unsubtracted_indices.append(res[2][1])
        #    #jetsArray, constituent_indices, (subtracted_constituents, subtracted_to_unsubtracted_indices) = res

    # Determine constituent indices
    output_constituent_indices = ak.Array(constituent_indices)
    # Following the example
    constituents_shape = ak.num(output_constituent_indices, axis=1)
    # Duplicate constituents so we can project along them
    duplicate_mask = ak.unflatten(np.zeros(np.sum(constituents_shape), np.int64), constituents_shape)
    duplicated_particles = particles[:, np.newaxis][duplicate_mask]
    output_constituents = duplicated_particles[output_constituent_indices]

    #outputs_to_inputs = self.constituent_index(min_pt)
    #shape = ak.num(outputs_to_inputs)
    #total = np.sum(shape)
    #duplicate = ak.unflatten(np.zeros(total, np.int64), shape)
    #prepared = self.data[:, np.newaxis][duplicate]
    #return prepared[outputs_to_inputs]
    #output_constituents = ak.Array(
    #    ak.layout.ListOffsetArray64(
    #        ak.layout.Index64(output_constituent_indices.layout),
    #        particles.layout,
    #        #ak.layout.RecordArray(
    #        #    [
    #        #        ak.layout.NumpyArray(array.px),
    #        #        ak.layout.NumpyArray(array.py),
    #        #        ak.layout.NumpyArray(array.pz),
    #        #        ak.layout.NumpyArray(array.E),
    #        #    ],
    #        #    ["px", "py", "pz", "E"],
    #        #),
    #    )
    #)

    # Make an output based on this information...
    output_jets = ak.zip({
        "px": jets["px"],
        "py": jets["py"],
        "pz": jets["pz"],
        "E": jets["E"],
        "constituents": output_constituents,
    }, with_name="Momentum4D", depth_limit=2)
    #outputJets = ak.Array([output.jets for output in outputs], with_name="Momentum4D")
    #jets = ak.Array(
    #    ak.layout.ListOffsetArray64(
    #        ak.layout.Index64(np.array([0, len(array)])),
    #        ak.layout.RecordArray(
    #            [
    #                ak.layout.NumpyArray(jetsArray[0]),
    #                ak.layout.NumpyArray(jetsArray[1]),
    #                ak.layout.NumpyArray(jetsArray[2]),
    #                ak.layout.NumpyArray(jetsArray[3]),
    #            ],
    #            ["px", "py", "pz", "E"],
    #        ),
    #    )
    #)

    return output_jets

