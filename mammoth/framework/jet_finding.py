""" Jet finding interface.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from typing import Dict, List, Optional, Tuple

import awkward as ak
import numpy as np
import numpy.typing as npt
import vector

import mammoth._ext
from mammoth._ext import ConstituentSubtractionSettings

vector.register_awkward()

def _expand_array_for_applying_constituent_indices(array_to_expand: ak.Array, array_with_constituent_indices: ak.Array) -> ak.Array:
    # We need to duplicate the array_to_expand so we can project along it with constituent indices.
    # Basically, we'll duplicate the particles as many times as there are constituents by selecting
    # the first constituent that is promoted by np.newaxis (namely, the constituent that is there).
    # It's not the most intuitive operation, but it seems to work.
    constituents_shape = ak.num(array_with_constituent_indices, axis=1)
    duplicate_mask = ak.unflatten(
        np.zeros(np.sum(constituents_shape), np.int64),
        constituents_shape
    )
    duplicated_elements = array_to_expand[:, np.newaxis][duplicate_mask]
    # Once we have the duplicated input particles, we can finally retrieve the output constituents.
    return duplicated_elements[array_with_constituent_indices]


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

    # Keep track of the jet four vector components. Although this will have to be converted later,
    # it seems that this is good enough enough to start.
    # NOTE: If this gets too slow, we can do the jet finding over multiple events in c++ like what
    #       is done in the new fj bindings. I skip this for now because my existing code seems to
    #       be good enough.
    jets: Dict[str, List[npt.NDArray[np.float32 | np.float64]]] = {
        "px": [],
        "py": [],
        "pz": [],
        "E": [],
    }
    constituent_indices = []
    subtracted_constituents: Dict[str, List[npt.NDArray[np.float32 | np.float64]]] = {
        "px": [],
        "py": [],
        "pz": [],
        "E": [],
    }
    subtracted_to_unsubtracted_indices = []
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
        subtracted_info = res.subtracted_info
        if subtracted_info:
            subtracted_constituents["px"].append(subtracted_info[0][0])
            subtracted_constituents["py"].append(subtracted_info[0][1])
            subtracted_constituents["pz"].append(subtracted_info[0][2])
            subtracted_constituents["E"].append(subtracted_info[0][3])
            subtracted_to_unsubtracted_indices.append(subtracted_info[1])

        # Temp to end early
        if lower > 1000:
            break

    # To create the output, we first start with the constituents.
    # If we have subtracted constituents, we need to handle them very carefully.
    if subtracted_to_unsubtracted_indices:
        # If we have subtracted constituents, the indices that were returned reference
        # the subtracted constituents.
        particles_for_constistuents = ak.Array(subtracted_constituents)
    else:
        particles_for_constistuents = particles

    # Determine constituents from constituent indices
    # NOTE: This follows the example in the scikit-hep fastjet bindings.
    # First, we convert the indices into an awkward array to make the next operations simpler.
    # NOTE: This requires a copy.
    output_constituent_indices = ak.Array(constituent_indices)

    output_constituents = _expand_array_for_applying_constituent_indices(
        array_to_expand=particles_for_constistuents,
        array_with_constituent_indices=output_constituent_indices,
    )
    ## We need to duplicate the constituents so we can project along them
    ## Basically, we'll duplicate the particles as many times as there are constituents by selecting
    ## the first constituent that is promoted by np.newaxis (namely, the constituent that is there).
    ## It's not the most intuitive operation, but it seems to work.
    #constituents_shape = ak.num(output_constituent_indices, axis=1)
    #duplicate_mask = ak.unflatten(
    #    np.zeros(np.sum(constituents_shape), np.int64),
    #    constituents_shape
    #)
    #duplicated_particles = particles_for_constistuents[:, np.newaxis][duplicate_mask]
    ## Once we have the duplicated input particles, we can finally retrieve the output constituents.
    #output_constituents = duplicated_particles[output_constituent_indices]
    # Add their constituent indices to the array so we can keep track of where they came from
    # NOTE: I'm not entirely convinced that this is necessary...
    output_constituents = ak.zip(
        {
            **dict(zip(ak.fields(output_constituents), ak.unzip(output_constituents))),
            "index": output_constituent_indices,
        },
    )

    # Now, handle the subtracted constituents if they exist.
    # NOTE: We don't need to play the same game for the subtracted constituents because they are
    #       returned as four vectors.
    #       However, we add the subtracted-to-unsubtracted index map
    if subtracted_to_unsubtracted_indices:
        # TODO: Merge with up above.
        #output_constituents = ak.zip(
        #    {
        #        **dict(zip(ak.fields(output_constituents), ak.unzip(output_constituents))),
        #        "unsubtracted_constituent_indices": ak.Array(subtracted_to_unsubtracted_indices),
        #    },
        #)
        expanded_subtracted_to_unsbtracted_indices = _expand_array_for_applying_constituent_indices(
            array_to_expand=ak.Array(subtracted_to_unsubtracted_indices),
            array_with_constituent_indices=output_constituents["index"],
        )
        output_constituents = ak.zip(
            {
                **dict(zip(ak.fields(output_constituents), ak.unzip(output_constituents))),
                "unsubtracted_index": expanded_subtracted_to_unsbtracted_indices,
            },
        )

        # We have subtracted constituents, so we need to convert them
        #output_subtracted_constituents = ak.Array(subtracted_constituents)
        #output_subtracted_constituents = ak.zip(
        #    {
        #        **dict(zip(ak.fields(output_subtracted_constituents), ak.unzip(output_subtracted_constituents))),
        #        # NOTE: We also need to convert the subtracted-to-unsubtracted index map so that
        #        #       it will zip properly.
        #        "indices": subtracted_to_unsubtracted_indices,
        #    },
        #)

    # Make an output based on this information...
    output_kwargs = {
        "px": jets["px"],
        "py": jets["py"],
        "pz": jets["pz"],
        "E": jets["E"],
        "constituents": output_constituents,
    }
    # Add subtracted constituents to the output when appropriate.
    #if subtracted_to_unsubtracted_indices:
    #    output_kwargs["subtracted_constituents"] = output_subtracted_constituents

    import IPython; IPython.embed()

    # Finally, construct the output
    try:
        output_jets = ak.zip(
            output_kwargs,
            with_name="Momentum4D",
            # Limit of 2 is based on: 1 for events + 1 for jets
            depth_limit=2,
        )
    except Exception as e:
        print(e)
        import IPython; IPython.embed()
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

