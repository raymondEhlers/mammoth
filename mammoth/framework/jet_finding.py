""" Jet finding functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from typing import Any, Dict, Final, List, Optional, Tuple, Union

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
AREA_AA = AreaSettings("active_area_explicit_ghosts", 0.005)
AREA_SUBSTRUCTURE = AreaSettings("passive_area", 0.05)

DISTANCE_DELTA: Final[float] = 0.01


@nb.njit  # type: ignore
def _shared_momentum_fraction_for_flat_array_implementation(
    generator_like_jet_pts: ak.Array,
    generator_like_jet_constituents: ak.Array,
    generator_like_jet_constituent_indices: ak.Array,
    measured_like_jet_constituents: ak.Array,
    measured_like_jet_constituent_indices: ak.Array,
    match_using_distance: bool = False
) -> npt.NDArray[np.float32]:
    """ Implementation of the shared momentum fraction

    Why passed the indices separately? Because when awkward has a momentum field, it doesn't seem to pass
    the other fields along. So we workaround it by passing it separately so we can use it now, at the cost
    of some extra bookkeeping.
    """
    # Setup
    delta = DISTANCE_DELTA
    shared_momentum_fraction = np.zeros(len(generator_like_jet_constituents), dtype=np.float32)

    for i, (generator_like_jet_pt, generator_like_constituents, generator_like_constituent_indices, measured_like_constituents, measured_like_constituent_indices) in \
        enumerate(zip(generator_like_jet_pts,
                      generator_like_jet_constituents, generator_like_jet_constituent_indices,
                      measured_like_jet_constituents, measured_like_jet_constituent_indices)):
        sum_pt = 0
        for generator_like_constituent, generator_like_constituent_index in zip(generator_like_constituents, generator_like_constituent_indices):
            #print(f"generator: index: {generator_like_constituent.index}, pt: {generator_like_constituent.pt}")
            for measured_like_constituent, measured_like_constituent_index in zip(measured_like_constituents, measured_like_constituent_indices):
                #print(f"measured: index: {measured_like_constituent.index}, pt: {measured_like_constituent.pt}")
                if match_using_distance:
                    if np.abs(measured_like_constituent.eta - generator_like_constituent.eta) > delta:
                        continue
                    if np.abs(measured_like_constituent.phi - generator_like_constituent.phi) > delta:
                        continue
                else:
                    #if generator_like_constituent.index != measured_like_constituent.index:
                    #if generator_like_constituent["index"] != measured_like_constituent["index"]:
                    if generator_like_constituent_index != measured_like_constituent_index:
                        continue

                sum_pt += generator_like_constituent.pt
                #print(f"Right after sum_pt: {sum_pt}")
                # We've matched once - no need to match again.
                # Otherwise, the run the risk of summing a generator-like constituent pt twice.
                break

        shared_momentum_fraction[i] = sum_pt / generator_like_jet_pt
    return shared_momentum_fraction


def shared_momentum_fraction_for_flat_array(
    generator_like_jet_pts: ak.Array,
    generator_like_jet_constituents: ak.Array,
    measured_like_jet_constituents: ak.Array,
    match_using_distance: bool = False
) -> npt.NDArray[np.float32]:
    """Calculate shared momentum fraction for a flat jet array

    Should be used _after_ jet matching, so that we only have a flat jet array.

    NOTE:
        The index matching is based on jets coming from the same source and therefore having the same index.
        Which is to say, this function won't work for part <-> det because they are matched by _label_, not _index_.

    Note:
        Calculate the momentum fraction as a scalar sum of constituent pt.

    Args:
        generator_like_jet_pts: Generator-like jet pt.
        generator_like_jet_constituents: Generator-like jet constituents.
        measured_like_jet_constituents: Measured-like jet constituents.
        match_using_distance: If True, match using distance instead of index. Default: False.
    Return:
        Fraction of generator-like jet momentum contained in the measured-like jet.
    """
    # Validation
    if len(generator_like_jet_constituents) != len(measured_like_jet_constituents):
        raise ValueError(
            f"Number of jets mismatch: generator: {len(generator_like_jet_constituents)} measured: {len(measured_like_jet_constituents)}"
        )

    return _shared_momentum_fraction_for_flat_array_implementation(  # type: ignore
        generator_like_jet_pts=generator_like_jet_pts,
        generator_like_jet_constituents=generator_like_jet_constituents,
        generator_like_jet_constituent_indices=generator_like_jet_constituents.index,
        measured_like_jet_constituents=measured_like_jet_constituents,
        measured_like_jet_constituent_indices=measured_like_jet_constituents.index,
        match_using_distance=match_using_distance,
    )


@nb.njit  # type: ignore
def _jet_matching_geometrical_impl(
    jets_first: ak.Array, jets_second: ak.Array, n_jets_first: int, max_matching_distance: float
) -> npt.NDArray[np.int64]:
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


@nb.njit  # type: ignore
def _jet_matching(
    jets_base: ak.Array, jets_tag: ak.Array, max_matching_distance: float
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Main geometrical jet matching implementation in numba.

    Args:
        jets_base: Base jet collection to match, in an event structure.
        jets_tag: Tag jet collection to match, in an event structure.
        max_matching_distance: Maximum matching distance.
    Returns:
        base -> tag matching indices, tag -> base matching indices
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
        jets_first=jets_base,
        jets_second=jets_tag,
        n_jets_first=n_jets_base,
        max_matching_distance=max_matching_distance,
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
    # the numpy matching arrays from above. The tag jet indices will come from the base
    # jets (we just have to keep track of the base and tag offsets).
    i = 0
    # We don't care about the last offset (it's just the total number of jets), so we skip it
    # here in order to match with the length of the base counts.
    # NOTE: I guess this kind of re-implements broadcast_arrays. But okay, it seems to be fine.
    for event_offset_base, event_offset_tag, count_base in zip(starts_base[:-1], starts_tag[:-1], counts_base):
        # print(f"{i=}")
        # We don't care about this counter, we just need to iterate this many times.
        for _ in range(0, count_base):
            # print(f"{local_counter=}")
            # print(f"{event_base_matching_indices[i]=}")
            if (
                event_base_matching_indices[i] > -1
                and event_base_matching_indices[i] > -1
                and i == event_tag_matching_indices[event_base_matching_indices[i]]
            ):
                # We found a true match! Store the indices.
                # print(f"Found match! {i=}, {event_offset_base=}, {event_offset_tag=}, {event_base_matching_indices[i]=}, {event_tag_matching_indices[event_base_matching_indices[i]]=}")
                # NOTE: We need to correct the indices for the array offsets. This ensures
                #       that we will have meaningful indices when we add back the event
                #       structure after this function.
                matching_output_base[i] = event_base_matching_indices[i] - event_offset_tag
                matching_output_tag[event_base_matching_indices[i]] = i - event_offset_base
            i += 1

    return matching_output_base, matching_output_tag


def jet_matching_geometrical(jets_base: ak.Array, jets_tag: ak.Array, max_matching_distance: float) -> ak.Array:
    """ Main interface for geometrical jet matching

    Matches are required to be bijective (ie. base <-> tag).

    Args:
        jets_base: Base jet collection to match, in an event structure.
        jets_tag: Tag jet collection to match, in an event structure.
        max_matching_distance: Maximum matching distance.
    Returns:
        (base -> tag matching indices, tag -> base matching indices)
    """
    base_to_tag_matching_np, tag_to_base_matching_np = _jet_matching(
        jets_base=jets_base, jets_tag=jets_tag, max_matching_distance=max_matching_distance
    )

    # Add back event structure.
    base_to_tag_matching = ak.unflatten(base_to_tag_matching_np, ak.num(jets_base, axis=1))
    tag_to_base_matching = ak.unflatten(tag_to_base_matching_np, ak.num(jets_tag, axis=1))

    # TODO: Probably best to remove these printouts when done debugging...
    logger.debug(
        f"base_to_tag_matching_np: {base_to_tag_matching_np}, tag_to_base_matching_np: {tag_to_base_matching_np}"
    )
    logger.debug(f"base_to_tag_matching: {base_to_tag_matching}, tag_to_base_matching: {tag_to_base_matching}")

    return base_to_tag_matching, tag_to_base_matching


def _apply_constituent_indices_to_expanded_array(
    array_to_expand: ak.Array, constituent_indices: ak.Array
) -> ak.Array:
    """Expand array by duplicating entries and then apply the constituent indices to select from that array.

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
        constituent indices applied to the array which was expanded (e.g. constituents from input particles).
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
    duplication_mask = ak.unflatten(np.zeros(np.sum(constituents_shape), np.int64), constituents_shape)
    duplicated_elements = array_to_expand[:, np.newaxis][duplication_mask]
    # Once we have the duplicated array elements, we can finally retrieve the elements which
    # are indexed by the constituent indices.
    return duplicated_elements[constituent_indices]


def area_percentage(percentage: float, jet_R: float) -> float:
    """Calculate jet R area percentage (for cuts)."""
    # Validation
    if percentage < 1:
        raise ValueError(f"Did you pass a fraction? Passed {percentage}. Check it!")
    return percentage / 100.0 * np.pi * jet_R * jet_R


def find_jets(
    particles: ak.Array,
    jet_R: float,
    algorithm: str = "anti-kt",
    area_settings: Optional[AreaSettings] = None,
    eta_range: Tuple[float, float] = (-0.9, 0.9),
    min_jet_pt: float = 1.0,
    background_particles: Optional[ak.Array] = None,
    background_subtraction: bool = False,
    constituent_subtraction: Optional[ConstituentSubtractionSettings] = None,
) -> ak.Array:
    """Main jet finding interface."""
    # Validation
    # Without this, we may have argument mismatches.
    min_jet_pt = float(min_jet_pt)
    if area_settings is None:
        area_settings = AREA_AA
    if background_subtraction and constituent_subtraction:
        raise ValueError("Enabled background subtraction and constituent subtraction together. This isn't compatible. Please disable one.")

    # Keep track of the event transitions.
    counts = ak.num(particles, axis=1)
    # To use for indexing, we need to keep track of the cumulative sum. That way, we can
    # slice using these indices.
    sum_counts = np.cumsum(np.asarray(counts))
    # However, to use as slices, we need one more entry than the number of events. We
    # account for this by inserting 0 at the beginning since the first indices starts at 0.
    sum_counts = np.insert(sum_counts, 0, 0)  # type: ignore

    # Validate that there is at least one particle per event
    event_with_no_particles = sum_counts[1:] == sum_counts[:-1]
    if np.any(event_with_no_particles):
        raise ValueError(
            f"There are some events with zero particles, which is going to mess up the alignment. Check the input! 0s are at {np.where(event_with_no_particles)}"
        )

    # Now, deal with the particles themselves.
    # This will flatten the awkward array contents while keeping the record names.
    flattened_particles = ak.flatten(particles, axis=1)
    # We only want vector to calculate the components once (the input components may not
    # actually be px, py, pz, and E), so calculate them now, and view them as numpy arrays
    # so we can pass them directly into our function.
    # NOTE: To avoid argument mismatches when calling to the c++, we view as float64 (which
    #       will be converted to a double). As of July 2021, tests seem to indicate that it's
    #       not making the float32 -> float conversion properly.
    px = np.asarray(flattened_particles.px, dtype=np.float64)
    py = np.asarray(flattened_particles.py, dtype=np.float64)
    pz = np.asarray(flattened_particles.pz, dtype=np.float64)
    E = np.asarray(flattened_particles.E, dtype=np.float64)

    # Now, onto the background particles. If background particles were passed, we want to do the
    # same thing as the input particles
    if background_particles is not None:
        background_counts = ak.num(background_particles, axis=1)
        # To use for indexing, we need to keep track of the cumulative sum. That way, we can
        # slice using these indices.
        background_sum_counts = np.cumsum(np.asarray(background_counts))
        # However, to use as slices, we need one more entry than the number of events. We
        # account for this by inserting 0 at the beginning since the first indices starts at 0.
        background_sum_counts = np.insert(background_sum_counts, 0, 0)  # type: ignore

        # Validate that there is at least one particle per event
        event_with_no_particles = background_sum_counts[1:] == background_sum_counts[:-1]
        if np.any(event_with_no_particles):
            raise ValueError(
                f"There are some background events with zero particles, which is going to mess up the alignment. Check the input! 0s are at {np.where(event_with_no_particles)}"
            )

        # Now, deal with the particles themselves.
        # This will flatten the awkward array contents while keeping the record names.
        flattened_background_particles = ak.flatten(background_particles, axis=1)
        # We only want vector to calculate the components once (the input components may not
        # actually be px, py, pz, and E), so calculate them now, and view them as numpy arrays
        # so we can pass them directly into our function.
        # NOTE: To avoid argument mismatches when calling to the c++, we view as float64 (which
        #       will be converted to a double). As of July 2021, tests seem to indicate that it's
        #       not making the float32 -> float conversion properly.
        background_px = np.asarray(flattened_background_particles.px, dtype=np.float64)
        background_py = np.asarray(flattened_background_particles.py, dtype=np.float64)
        background_pz = np.asarray(flattened_background_particles.pz, dtype=np.float64)
        background_E = np.asarray(flattened_background_particles.E, dtype=np.float64)
    else:
        # If we don't have any particles, we just want to pass empty arrays. This will be interpreted
        # to use the signal events for the background estimator.
        # For this to work, we need to at least fake some values.
        # In particular, slices of 0 (ie. [0:0]) will return empty arrays, even if the arrays themselves
        # are already empty. So we make the counts 0 for all events
        background_sum_counts = np.zeros(len(sum_counts), dtype=np.int64)
        # And then create the empty arrays.
        background_px = np.zeros(0, dtype=np.float64)
        background_py = np.zeros(0, dtype=np.float64)
        background_pz = np.zeros(0, dtype=np.float64)
        background_E = np.zeros(0, dtype=np.float64)

    # Validate that the number of background events match the number of signal events
    if len(sum_counts) != len(background_sum_counts):
        raise ValueError(
            f"Mismatched between number of events for signal and background. Signal: {len(sum_counts) -1}, background: {len(background_sum_counts) - 1}"
        )

    # Keep track of the jet four vector components. Although this will have to be converted later,
    # it seems that this is good enough enough to start.
    # NOTE: If this gets too slow, we can do the jet finding over multiple events in c++ like what
    #       is done in the new fj bindings. I skip this for now because my existing code seems to
    #       be good enough.
    jets: Dict[str, List[npt.NDArray[Union[np.float32, np.float64]]]] = {
        "px": [],
        "py": [],
        "pz": [],
        "E": [],
        "area": [],
    }
    constituent_indices = []
    subtracted_constituents: Dict[str, List[npt.NDArray[Union[np.float32, np.float64]]]] = {
        "px": [],
        "py": [],
        "pz": [],
        "E": [],
    }
    subtracted_to_unsubtracted_indices = []
    for lower, upper, background_lower, background_upper in zip(sum_counts[:-1], sum_counts[1:], background_sum_counts[:-1], background_sum_counts[1:]):
        # Run the actual jet finding.
        res = mammoth._ext.find_jets(
            px=px[lower:upper],
            py=py[lower:upper],
            pz=pz[lower:upper],
            E=E[lower:upper],
            background_px=background_px[background_lower:background_upper],
            background_py=background_py[background_lower:background_upper],
            background_pz=background_pz[background_lower:background_upper],
            background_E=background_E[background_lower:background_upper],
            jet_R=jet_R,
            jet_algorithm=algorithm,
            area_settings=area_settings,
            eta_range=eta_range,
            min_jet_pt=min_jet_pt,
            background_subtraction=background_subtraction,
            constituent_subtraction=constituent_subtraction,
        )

        # Store the results temporarily so we can perform all of the jet finding immediately.
        # We'll format them for returning below.
        temp_jets = res.jets
        # Unpack and store the jet four vector (it doesn't appear to be possible to append via tuple unpacking...)
        jets["px"].append(temp_jets[0])
        jets["py"].append(temp_jets[1])
        jets["pz"].append(temp_jets[2])
        jets["E"].append(temp_jets[3])
        # Next, store additional jet properties
        jets["area"].append(res.jets_area)
        # Next, associate the indices of the constituents that are associated with each jet
        constituent_indices.append(res.constituent_indices)

        # Finally, we'll handle the subtracted constituents if relevant
        subtracted_info = res.subtracted_info
        if subtracted_info:
            # Unpack and store the substracted constituent four vector
            subtracted_constituents["px"].append(subtracted_info[0][0])
            subtracted_constituents["py"].append(subtracted_info[0][1])
            subtracted_constituents["pz"].append(subtracted_info[0][2])
            subtracted_constituents["E"].append(subtracted_info[0][3])
            # Plus the association for each subtracted constituent index into the unsubtracted constituent.
            # NOTE: These are the indices assigned via the user_index.
            subtracted_to_unsubtracted_indices.append(subtracted_info[1])

    # To create the output, we start with the constituents.
    # First, we convert the fastjet user_index indices that we use for book keeping during jet finding
    # into an awkward array to make the next operations simpler.
    # NOTE: Remember, these indices are just for internal mapping during jet finding! They're totally
    #       separate from the `index` field that we include in all particles arrays to identify the source
    #       of jet constituents (ie. in terms of the input particle collection that they're from).
    # NOTE: This requires a copy, but that's fine - there's nothing to be done about it, so it's just the cost.
    _constituent_indices_awkward = ak.Array(constituent_indices)

    # If we have subtracted constituents, we need to handle them very carefully.
    if subtracted_to_unsubtracted_indices:
        # In the case of subtracted constituents, the indices that were returned reference the subtracted
        # constituents. Consequently, we need to build our jet constituents using the subtracted constituent
        # four vectors that were returned from the jet finding.
        # NOTE: Constituents are expected to have an `index` field to identify their input source.
        #       However, the `subtracted_constituents` assigned here to `_particles_for_constituents`
        #       don't contain an `index` field yet because we only returned the four vectors from the
        #       jet finding. We need to add it in below!
        _particles_for_constituents = ak.Array(subtracted_constituents)

        # Now, to add in the indices. Since the subtracted-to-unsubtracted mapping is actually just a
        # list of unsubtracted indices (where the location of unsubtracted index corresponds to the
        # subtracted particles), we can directly apply this "mapping" to the unsubtracted particles
        # `index` (after converting it to awkward)
        _subtracted_to_unsubtracted_indices_awkward = ak.Array(subtracted_to_unsubtracted_indices)
        _subtracted_indices = particles["index"][_subtracted_to_unsubtracted_indices_awkward]
        # Then, we just need to zip it in to the particles for constituents, and it will be brought
        # along when the constituents are associated with the jets.
        _particles_for_constituents = ak.zip(
            {
                **dict(zip(ak.fields(_particles_for_constituents), ak.unzip(_particles_for_constituents))),
                "index": _subtracted_indices,
            },
            with_name="Momentum4D",
        )
    else:
        # Since `index` is already included in particles, there's nothing else we need to do here.
        _particles_for_constituents = particles

    # Now, determine constituents from constituent indices
    # To do this, we perform the manipulations necessary to get all of the dimensions to match up.
    # Namely, we have to get a singly-jagged array (_particles_for_constituents) to broadcast with a
    # doubly-jagged array (_constituent_indices_awkward).
    # NOTE: This follows the example in the scikit-hep fastjet bindings.
    output_constituents = _apply_constituent_indices_to_expanded_array(
        array_to_expand=_particles_for_constituents,
        constituent_indices=_constituent_indices_awkward,
    )

    """
    NOTE: We don't need the constituent indices themselves since we've already mapped the constituents
          to the jets. Those constituents can identify their source via `index`. If we later decide that
          we need them, it's as simple as the zipping everything together. Example:

    ```python
    output_constituents = ak.zip(
        {
            **dict(zip(ak.fields(output_constituents), ak.unzip(output_constituents))),
            "name_of_field_for_constituent_indices": _constituent_indices_awkward,
        },
        with_name="Momentum4D",
    )
    ```
    """

    # Finally, construct the output
    output_jets = ak.zip(
        {
            "px": jets["px"],
            "py": jets["py"],
            "pz": jets["pz"],
            "E": jets["E"],
            "area": jets["area"],
            "constituents": output_constituents,
        },
        with_name="Momentum4D",
        # Limit of 2 is based on: 1 for events + 1 for jets
        depth_limit=2,
    )

    return output_jets

def _splittings_output() -> Dict[str, List[Any]]:
    return {
        "kt": [],
        "delta_R": [],
        "z": [],
        "parent_index": [],
    }


def _subjets_output() -> Dict[str, List[Any]]:
    return {
        "splitting_node_index": [],
        "part_of_iterative_splitting": [],
        "constituent_indices": [],
    }


def recluster_jets(jets: ak.Array) -> ak.Array:
    # Validation. There must be jets
    if len(jets) == 0:
        raise ValueError("No jets present for reclustering!")

    # To iterate over the constituents in an efficient manner, we need to flatten them and
    # their four-momenta. To make this more manageable, we want to determine the constituent
    # start and stop indices, and then build those up into doubly-jagged arrays and iterate
    # over those. Those doubly-jagged arrays then provide the slice indices for selecting
    # the flattened constituents.
    # First, we need to get the starts and stops. We'll do this via:
    # number -> offsets -> starts and stops.
    # NOTE: axis=2 is really the key to making this work nicely. It provides the number of
    #       constituents, and crucially, they're in the right jagged form.
    num_constituents = ak.num(jets.constituents, axis=2)
    # Convert to offsets
    offsets_constituents = np.cumsum(np.asarray(ak.flatten(num_constituents, axis=1)))
    offsets_constituents = np.insert(offsets_constituents, 0, 0)  # type: ignore
    # Then into starts and stops
    starts_constituents = ak.unflatten(
        offsets_constituents[:-1],
        ak.num(num_constituents, axis=1)
    )
    stops_constituents = ak.unflatten(
        offsets_constituents[1:],
        ak.num(num_constituents, axis=1)
    )

    # Now, setup the constituents themselves.
    # It would be nice to flatten them and then assign the components like for standard jet
    # finding, but since we have to flatten with `None` to deals with the multiple levels,
    # it also flattens over the records.
    # NOTE: `axis="records"` isn't yet available (July 2021).
    # We only want vector to calculate the components once (the input components may not
    # actually be px, py, pz, and E), so calculate them now, and view them as numpy arrays
    # so we can pass them directly into our function.
    # NOTE: To avoid argument mismatches when calling to the c++, we view as float64 (which
    #       will be converted to a double). As of July 2021, tests seem to suggest that it's =
    #       not making the float32 -> float conversion properly.
    px = np.asarray(ak.flatten(jets.constituents.px, axis=None), dtype=np.float64)
    py = np.asarray(ak.flatten(jets.constituents.py, axis=None), dtype=np.float64)
    pz = np.asarray(ak.flatten(jets.constituents.pz, axis=None), dtype=np.float64)
    E = np.asarray(ak.flatten(jets.constituents.E, axis=None), dtype=np.float64)

    # import IPython; IPython.embed()

    event_splittings = _splittings_output()
    event_subjets = _subjets_output()
    for starts, stops in zip(starts_constituents, stops_constituents):
        jets_splittings = _splittings_output()
        jets_subjets = _subjets_output()
        for lower, upper in zip(starts, stops):
            res = mammoth._ext.recluster_jet(
                px=px[lower:upper],
                py=py[lower:upper],
                pz=pz[lower:upper],
                E=E[lower:upper],
                jet_R=1.0,
                #area_settings=area_settings,
                #eta_range=eta_range,
            )
            _temp_splittings = res.splittings()
            _temp_subjets = res.subjets()
            jets_splittings["kt"].append(_temp_splittings.kt)
            jets_splittings["delta_R"].append(_temp_splittings.delta_R)
            jets_splittings["z"].append(_temp_splittings.z)
            jets_splittings["parent_index"].append(_temp_splittings.parent_index)
            jets_subjets["splitting_node_index"].append(_temp_subjets.splitting_node_index)
            jets_subjets["part_of_iterative_splitting"].append(_temp_subjets.part_of_iterative_splitting)
            jets_subjets["constituent_indices"].append(_temp_subjets.constituent_indices)

        # Now, move to the overall output objects.
        # NOTE: We want to fill this even if we didn't perform any reclustering to ensure that
        #       we keep the right shape.
        for k in event_splittings:
            event_splittings[k].append(jets_splittings[k])
        for k in event_subjets:
            event_subjets[k].append(jets_subjets[k])

    return ak.zip(
        {
            "jet_splittings": ak.zip(
                event_splittings
            ),
            "subjets": ak.zip(
                event_subjets,
                # zip over events, jets, and subjets (but can't go all the way due to the constituent indices)
                depth_limit=3,
            )
        },
        depth_limit=2
    )

