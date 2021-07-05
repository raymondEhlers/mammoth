""" Jet finding interface.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from typing import Dict, List

import awkward as ak
import numpy as np
import vector

import mammoth._ext

vector.register_awkward()

def find_jets(particles: ak.Array, jet_R: float,
              algorithm: str = "anti-kt") -> ak.Array:
    #counts = ak.num(particles, axis=1)
    #flattened_particles = ak.flatten(particles)
    #sum_counts = np.cumsum(np.asarray(counts))

    #jets_offsets = []
    #offsets = []
    jets: Dict[str, List[np.ndarray]] = {
        "px": [],
        "py": [],
        "pz": [],
        "E": [],
    }
    constituent_indices = []
    #subtracted_constituents = []
    #subtracted_to_unsubtracted_indices = []
    #for lower, upper in zip(sum_counts[:-1], sum_counts[1:]):
    for event_particles in particles:
        # Skip if there are somehow no particles.
        #if lower == upper:
        #    continue

        res = mammoth._ext.find_jets(
            #px=flattened_particles[lower:upper].px,
            #py=flattened_particles[lower:upper].py,
            #pz=flattened_particles[lower:upper].pz,
            #E=flattened_particles[lower:upper].E,
            px=np.array(event_particles.px, dtype=np.float64),
            py=np.array(event_particles.py, dtype=np.float64),
            pz=np.array(event_particles.pz, dtype=np.float64),
            E=np.array(event_particles.E, dtype=np.float64),
            jet_R=jet_R,
            jet_algorithm=algorithm,
            eta_range=(-0.9, 0.9),
            min_jet_pt=1.0,
        )

        # Determine the offsets for jets immediately
        #outputs.append(res)
        #jets_offsets.append(len(res[0]))
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

    # Make an output based on this information...
    output_jets = ak.Array({
        "px": jets["px"],
        "py": jets["py"],
        "pz": jets["pz"],
        "E": jets["E"],
    }, with_name="Momentum4D")
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

