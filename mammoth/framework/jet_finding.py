""" Jet finding functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from typing import Dict, List, Optional, Tuple

import awkward as ak
import numba as nb
import numpy as np
import numpy.typing as npt
import vector

import mammoth._ext
from mammoth._ext import AreaSettings, ConstituentSubtractionSettings

logger = logging.getLogger(__name__)

vector.register_awkward()

AREA_PP = AreaSettings("active_area", 0.01)
AREA_AA = AreaSettings("active_area", 0.005)
AREA_SUBSTRUCTURE = AreaSettings("passive_area", 0.05)

@nb.njit
def _jet_matching_geometrical_impl(jets_first: ak.Array, jets_second: ak.Array, n_jets_first: int, max_matching_distance: float) -> npt.NDArray[np.int64]:
    """Implementation of geometrical jet matching.

    Args:
        jets_first: The first jet collection, which we iterate over first. Indices are stored according
            to this collection, pointing to the `jets_second` collection.
        jets_second: The second jet collection, which we iterate over first. The stored indices
            point to entries in this collection.
        n_jets_first: Total number of jets in the first collection (summed over all events).
        max_matching_distance: Maximum matching distance.

    Returns:
        Indices matching from jets in jets_first to jets in jets_second. Note that it has no event structure,
            which must be applied separately.
    """
    # Setup for storing the indices
    matching_indices = -1 * np.ones(n_jets_first, dtype=np.int64)

    # We need global indices, so we keep track of them externally.
    jet_index_base = 0
    jet_index_tag = 0
    # Implement simple geometrical matching.
    for event_base, event_tag in zip(jets_first, jets_second):
        for jet_base in event_base:
            j = jet_index_tag
            closest_tag_jet_distance = 9999
            closest_tag_jet = -1
            for jet_tag in event_tag:
                delta_R = jet_base.deltaR(jet_tag)
                if delta_R < closest_tag_jet_distance and delta_R < max_matching_distance:
                    closest_tag_jet = j
                    closest_tag_jet_distance = delta_R
                j += 1

            # print(f"Storing closest_tag_jet: {closest_tag_jet} with distance {closest_tag_jet_distance}")
            matching_indices[jet_index_base] = closest_tag_jet
            jet_index_base += 1

        # print(f"Right before increment jet_index_tag: {jet_index_tag}")
        jet_index_tag += len(event_tag)

    return matching_indices


@nb.njit
def _jet_matching(jets_base: ak.Array, jets_tag: ak.Array, max_matching_distance: float) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Main jet matching implementation in numba.

    """
    # First, setup for the base jets
    counts_base = np.zeros(len(jets_base), dtype=np.int64)
    for i, events_base in enumerate(jets_base):
        counts_base[i] = len(events_base)
    n_jets_base = np.sum(counts_base)
    # We need the index where each event starts, which is known as the starts in awkward.
    # Since we want to start at 0, we define a new array, and then store the cumulative sum
    # in the rest of the array.
    starts_base = np.zeros(len(jets_base) + 1, dtype=np.int64)
    starts_base[1:] = np.cumsum(counts_base)
    # Now, the same setup for the tag jets
    counts_tag = np.zeros(len(jets_tag), dtype=np.int64)
    for i, events_tag in enumerate(jets_tag):
        counts_tag[i] = len(events_tag)
    n_jets_tag = np.sum(counts_tag)
    # We need the index where each event starts, which is known as the starts in awkward.
    # Since we want to start at 0, we define a new array, and then store the cumulative sum
    # in the rest of the array.
    starts_tag = np.zeros(len(jets_tag) + 1, dtype=np.int64)
    starts_tag[1:] = np.cumsum(counts_tag)

    # Perform the actual matching
    event_base_matching_indices = _jet_matching_geometrical_impl(
        jets_first=jets_base, jets_second=jets_tag, n_jets_first=n_jets_base, max_matching_distance=max_matching_distance
    )
    event_tag_matching_indices = _jet_matching_geometrical_impl(
        jets_first=jets_tag, jets_second=jets_base, n_jets_first=n_jets_tag, max_matching_distance=max_matching_distance
    )

    # Now, we'll check for true matches, which are pairs where the base jet is the
    # closest to the tag jet and vice versa
    # NOTE: The numpy arrays are not event-by-event, but the jets are, so we need
    #       to keep careful track of the offsets. Awkward arrays would be nice here,
    #       but we would have to pass around a couple of array builders, so this was
    #       a little simpler.
    matching_output_base = -1 * np.ones(n_jets_base, dtype=np.int64)
    matching_output_tag = -1 * np.ones(n_jets_tag, dtype=np.int64)
    # i is the global index for base jets. We only need to keep track of this to index
    # the numpy matching arays for above. The tag jet indices will come from the base
    # jets (we just have to keep track of the base and tag offsets).
    i = 0
    # We don't care about the last offset (it's just the total number of jets), so we skip it
    # here in order to match with the length of the base counts.
    # NOTE: I guess this kind of reimplements broadcast_arrays. But okay, it seems to be fine.
    for event_offset_base, event_offset_tag, count_base in zip(starts_base[:-1], starts_tag[:-1], counts_base):
        # print(f"{i=}")
        # We don't care about this counter, we just need to iterate this many times.
        for _ in range(0, count_base):
            # print(f"{local_counter=}")
            # print(f"{event_base_matching_indices[i]=}")
            if event_base_matching_indices[i] > -1 and event_base_matching_indices[i] > -1 and i == event_tag_matching_indices[event_base_matching_indices[i]]:
                # We found a true match! Store the indices.
                # print(f"Found match! {i=}, {event_offset_base=}, {event_offset_tag=}, {event_base_matching_indices[i]=}, {event_tag_matching_indices[event_base_matching_indices[i]]=}")
                # NOTE: We need to correct the indices for the array offsets. This ensures
                #       that we will have meaningful indices when we add back the event
                #       structure after this function.
                matching_output_base[i] = event_base_matching_indices[i] - event_offset_tag
                matching_output_tag[event_base_matching_indices[i]] = i - event_offset_base
            i += 1

    return matching_output_base, matching_output_tag


def jet_matching(jets_base: ak.Array, jets_tag: ak.Array, max_matching_distance: float) -> ak.Array:
    # TODO: Fully Wrap the results in and out with ak.zip
    base_to_tag_matching_np, tag_to_base_matching_np = _jet_matching(jets_base=jets_base, jets_tag=jets_tag, max_matching_distance=max_matching_distance)

    # Add back event structure.
    base_to_tag_matching = ak.unflatten(base_to_tag_matching_np, ak.num(jets_base, axis=1))
    tag_to_base_matching = ak.unflatten(tag_to_base_matching_np, ak.num(jets_tag, axis=1))

    logger.debug(f"base_to_tag_matching_np: {base_to_tag_matching_np}, tag_to_base_matching_np: {tag_to_base_matching_np}")
    logger.debug(f"base_to_tag_matching: {base_to_tag_matching}, tag_to_base_matching: {tag_to_base_matching}")

    return base_to_tag_matching, tag_to_base_matching


def _expand_array_for_applying_constituent_indices(array_to_expand: ak.Array, constituent_indices: ak.Array) -> ak.Array:
    """Duplicate array for applying constituent indices.

    We end up with doubly-jagged constituent indices, but singly-jagged arrays (`array_to_expand`).
    To be able to apply the doubly-jagged constituent indices to the singly-jagged arrays, we need
    to expand the dimension of the single-jagged array. Practically, this is done by duplicating the
    singly-jagged array entries, and then applying the constituent indices to that newly expanded
    array to extract the values.

    Args:
        array_to_expand: Singly-jagged array where we want to apply the constituent indices.
        constituent_indices: Doubly-jagged constituent indices which are to index the
            `array_to_expand`.
    Returns:
        constituent indices applied to the array to expand (e.g. constituents from input particles).
    """
    # Further details on the strategy:
    #
    # We need to duplicate the array_to_expand so we can apply the constituent indices to it.
    # Basically, we'll duplicate each entry in each event in array_to_expand as many times as
    # there are constituents by selecting the first element generated by a new axis (basically,
    # just making the 1D new axis into a variable length entry).
    # It's not the most intuitive operation, but it seems to work.
    #
    # NOTE: I _think_ awkward is not making a _ton_ of copies, but just indexing. If so, this
    #       should be reasonably efficient.
    constituents_shape = ak.num(constituent_indices, axis=1)
    duplication_mask = ak.unflatten(
        np.zeros(np.sum(constituents_shape), np.int64),
        constituents_shape
    )
    duplicated_elements = array_to_expand[:, np.newaxis][duplication_mask]
    # Once we have the duplicated array elements, we can finally retrieve the elements which
    # are indexed by the constituent indices.
    return duplicated_elements[constituent_indices]


def find_jets(particles: ak.Array, jet_R: float,
              algorithm: str = "anti-kt",
              area_settings: Optional[AreaSettings] = None,
              eta_range: Tuple[float, float] = (-0.9, 0.9),
              min_jet_pt: float = 1.0,
              background_subtraction: bool = False,
              constituent_subtraction: Optional[ConstituentSubtractionSettings] = None,
              ) -> ak.Array:
    """Main jet finding interface.

    """
    # Validation
    # Without this, we may have argument mismatches.
    min_jet_pt = float(min_jet_pt)
    if area_settings is None:
        area_settings = AREA_AA

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
            area_settings=area_settings,
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

    # Then, we perform the manipulations necessary to get all of the dimensions to match up.
    # Namely, we have to get a singly-jagged array to broadcast with a doubly-jagged array
    # of constituent indices.
    output_constituents = _expand_array_for_applying_constituent_indices(
        array_to_expand=particles_for_constistuents,
        constituent_indices=output_constituent_indices,
    )
    # Add their constituent indices to the array so we can keep track of where they came from
    # NOTE: I'm not entirely convinced that this info is necessary...
    output_constituents = ak.zip(
        {
            **dict(zip(ak.fields(output_constituents), ak.unzip(output_constituents))),
            "index": output_constituent_indices,
        },
    )

    # Now, handle the subtracted constituents if they exist.
    if subtracted_to_unsubtracted_indices:
        # We have to play the same expand array game as above for the subtracted-to-unsubtracted
        # constituents mapping.
        expanded_subtracted_to_unsbtracted_indices = _expand_array_for_applying_constituent_indices(
            array_to_expand=ak.Array(subtracted_to_unsubtracted_indices),
            constituent_indices=output_constituents["index"],
        )
        # And then include that in the output.
        output_constituents = ak.zip(
            {
                **dict(zip(ak.fields(output_constituents), ak.unzip(output_constituents))),
                "unsubtracted_index": expanded_subtracted_to_unsbtracted_indices,
            },
        )

    # Finally, construct the output
    output_jets = ak.zip(
        {
            "px": jets["px"],
            "py": jets["py"],
            "pz": jets["pz"],
            "E": jets["E"],
            "constituents": output_constituents,
        },
        with_name="Momentum4D",
        # Limit of 2 is based on: 1 for events + 1 for jets
        depth_limit=2,
    )

    return output_jets

