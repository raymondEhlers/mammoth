""" Jet finding functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations  # noqa: I001

import functools
import logging
from typing import Any, Final

import awkward as ak
import numba as nb
import numpy as np
import numpy.typing as npt
import vector

import mammoth_cpp._ext
from mammoth_cpp._ext import (  # noqa: F401
    AreaSettings,
    NegativeEnergyRecombiner,
    JetFindingSettings,
    JetMedianBackgroundEstimator,
    GridMedianBackgroundEstimator,
    BackgroundSubtractionType,
    RhoSubtractor,
    ConstituentSubtractor,
    BackgroundSubtraction,
    DEFAULT_RAPIDITY_MAX,
)

logger = logging.getLogger(__name__)

vector.register_awkward()

"""Default area settings for pp."""
AreaPP = functools.partial(
    AreaSettings,
    area_type="active_area",
    ghost_area=0.01,
)
"""Default area settings for AA."""
AreaAA = functools.partial(
    AreaSettings,
    area_type="active_area_explicit_ghosts",
    ghost_area=0.005,
)

"""Default jet finding settings for jet median background determination."""
JetMedianJetFindingSettings = functools.partial(
    JetFindingSettings,
    R=0.2,
    algorithm="kt",
    recombination_scheme="E_scheme",
    strategy="Best",
    pt_range=(0.0, 10000.0),
    eta_range=(-0.9 + 0.2, 0.9 - 0.2),
    area_settings=AreaAA(),
)

"""Default area settings for reclustering / substructure."""
AreaSubstructure = functools.partial(
    AreaSettings,
    area_type="passive_area",
    ghost_area=0.05,
)
"""Default jet finding settings for reclustering"""
ReclusteringJetFindingSettings = functools.partial(
    JetFindingSettings,
    R=1.0,
    algorithm="CA",
    recombination_scheme="E_scheme",
    strategy="Best",
    pt_range=(0.0, 10000.0),
    eta_range=(-5.0, 5.0),
)

DISTANCE_DELTA: Final[float] = 0.01
VALIDATION_MODE_RANDOM_SEED: Final[list[int]] = [12345, 67890]


def pt_range(pt_min: float = 0.0, pt_max: float = 10000.0) -> tuple[float, float]:
    """Helper to create the pt range, including common default values.

    Args:
        pt_min: Min jet pt range.
        pt_max: Max jet pt range.
    Returns:
        tuple of pt range according to the settings.
    """
    # Wrap in floats to ensure that we don't have any type mismatches in pybind11
    return float(pt_min), float(pt_max)


def eta_range(
    jet_R: float, fiducial_acceptance: bool, eta_min: float = -0.9, eta_max: float = 0.9
) -> tuple[float, float]:
    """Helper to create the eta range, including common default values.

    Args:
        jet_R: Jet R. Used for fiducial acceptance. Could be optional, but I think it's better to just have a consistent interface.
        fiducial_acceptance: If True, we want a fiducial acceptance.
        eta_min: Min jet eta range.
        eta_max: Max jet eta range.
    Returns:
        tuple of eta range according to the settings.
    """
    if fiducial_acceptance:
        eta_min, eta_max = eta_min + jet_R, eta_max - jet_R
    # Wrap in floats to ensure that we don't have any type mismatches in pybind11
    return float(eta_min), float(eta_max)


@nb.njit  # type: ignore[misc]
def _shared_momentum_fraction_for_flat_array_implementation(
    generator_like_jet_pts: ak.Array,
    generator_like_jet_constituents: ak.Array,
    generator_like_jet_constituent_identifiers: ak.Array,
    measured_like_jet_constituents: ak.Array,
    measured_like_jet_constituent_identifiers: ak.Array,
    match_using_distance: bool = False,
    max_matching_distance: float = DISTANCE_DELTA,
) -> npt.NDArray[np.float32]:
    """Implementation of the shared momentum fraction

    Why passed the identifiers separately? Because when awkward has a momentum field, it doesn't seem to pass
    the other fields along. So we workaround it by passing it separately so we can use it now, at the cost
    of some extra bookkeeping.
    """
    # Setup
    shared_momentum_fraction = np.zeros(len(generator_like_jet_constituents), dtype=np.float32)

    for i, (
        generator_like_jet_pt,
        generator_like_constituents,
        generator_like_constituent_identifiers,
        measured_like_constituents,
        measured_like_constituent_identifiers,
    ) in enumerate(
        zip(
            generator_like_jet_pts,
            generator_like_jet_constituents,
            generator_like_jet_constituent_identifiers,
            measured_like_jet_constituents,
            measured_like_jet_constituent_identifiers,
        )
    ):
        sum_pt = 0
        for generator_like_constituent, generator_like_constituent_identifier in zip(
            generator_like_constituents, generator_like_constituent_identifiers
        ):
            # print(f"generator: identifier: {generator_like_constituent.identifier}, pt: {generator_like_constituent.pt}")
            for measured_like_constituent, measured_like_constituent_identifier in zip(
                measured_like_constituents, measured_like_constituent_identifiers
            ):
                # print(f"measured: identifier: {measured_like_constituent.identifier}, pt: {measured_like_constituent.pt}")
                if match_using_distance:
                    if np.abs(measured_like_constituent.eta - generator_like_constituent.eta) > max_matching_distance:
                        continue
                    if np.abs(measured_like_constituent.phi - generator_like_constituent.phi) > max_matching_distance:
                        continue
                else:
                    # if generator_like_constituent.identifier != measured_like_constituent.identifier:
                    # if generator_like_constituent["identifier"] != measured_like_constituent["identifier"]:
                    if generator_like_constituent_identifier != measured_like_constituent_identifier:
                        continue

                sum_pt += generator_like_constituent.pt
                # print(f"Right after sum_pt: {sum_pt}")
                # We've matched once - no need to match again.
                # Otherwise, the run the risk of summing a generator-like constituent pt twice.
                break

        shared_momentum_fraction[i] = sum_pt / generator_like_jet_pt
    return shared_momentum_fraction


def shared_momentum_fraction_for_flat_array(
    generator_like_jet_pts: ak.Array,
    generator_like_jet_constituents: ak.Array,
    measured_like_jet_constituents: ak.Array,
    match_using_distance: bool = False,
    max_matching_distance: float = DISTANCE_DELTA,
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
        max_matching_distance: Maximum matching distance if matching using distance. Default: DISTANCE_DELTA (0.01).
    Return:
        Fraction of generator-like jet momentum contained in the measured-like jet.
    """
    # Validation
    if len(generator_like_jet_constituents) != len(measured_like_jet_constituents):
        _msg = f"Number of jets mismatch: generator: {len(generator_like_jet_constituents)} measured: {len(measured_like_jet_constituents)}"
        raise ValueError(_msg)

    return _shared_momentum_fraction_for_flat_array_implementation(  # type: ignore[no-any-return]
        generator_like_jet_pts=generator_like_jet_pts,
        generator_like_jet_constituents=generator_like_jet_constituents,
        generator_like_jet_constituent_identifiers=generator_like_jet_constituents.identifier,
        measured_like_jet_constituents=measured_like_jet_constituents,
        measured_like_jet_constituent_identifiers=measured_like_jet_constituents.identifier,
        match_using_distance=match_using_distance,
        max_matching_distance=max_matching_distance,
    )


@nb.njit  # type: ignore[misc]
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


@nb.njit  # type: ignore[misc]
def _jet_matching(
    jets_base: ak.Array, jets_tag: ak.Array, max_matching_distance: float
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
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
    """Main interface for geometrical jet matching

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

    # These messages were useful for debugging, but we're quite noisy. We don't need them anymore, so they're
    # commented out, but they're left here in case they're needed in the future
    # logger.debug(
    #    f"base_to_tag_matching_np: {base_to_tag_matching_np}, tag_to_base_matching_np: {tag_to_base_matching_np}"
    # )
    # logger.debug(f"base_to_tag_matching: {base_to_tag_matching}, tag_to_base_matching: {tag_to_base_matching}")

    return base_to_tag_matching, tag_to_base_matching


@nb.njit  # type: ignore[misc]
def _calculate_unsubtracted_constituent_max_pt(
    input_arrays: ak.Array,
    input_arrays_source_indices: ak.Array,
    input_constituents_source_indices: ak.Array,
    builder: ak.ArrayBuilder,
) -> ak.ArrayBuilder:
    """Implementation of calculating the unsubtracted constituent max pt

    Since matching all of the source_indices, etc is a pain, we just implement via numba, where it appears to
    be fast enough.

    Note:
        It would be nice to pass in directly via the whole particle or jet type, but it seems that
        numba narrows to just the vector object, so we have to pass source_indices separately.

    Args:
        input_arrays: Particle arrays
        input_arrays_source_indices: Particle array source_indices
        input_constituents_source_indices: Constituent source_indices from subtracted jet collection.
    Returns:
        ArrayBuilder containing the unsubtracted max pt for each jet.
    """
    for particles_in_event, particles_source_indices_in_event, jets_constituents_source_indices_in_event in zip(
        input_arrays, input_arrays_source_indices, input_constituents_source_indices
    ):
        builder.begin_list()
        for constituents_source_indices in jets_constituents_source_indices_in_event:
            unsubtracted_constituent_pt = []
            for constituent_identifier in constituents_source_indices:
                for particle, particle_identifier in zip(particles_in_event, particles_source_indices_in_event):
                    if constituent_identifier == particle_identifier:
                        unsubtracted_constituent_pt.append(particle.pt)
            builder.append(max(unsubtracted_constituent_pt))
        builder.end_list()

    return builder


def calculate_unsubtracted_constituent_max_pt(arrays: ak.Array, constituents: ak.Array) -> ak.Array:
    """Calculate the unsubtracted constituent max pt

    This function is used for calculating the unsubtracted constituent max pt when performing constituent
    subtraction.

    Args:
        arrays: Array for a particular particle collection (eg. hybrid). These are the particles
            used for jet finding.
        constituents: Jet constituents for a particular jet collection (eg. hybrid).
            These must be constituent subtracted.
    Returns:
        unsubtracted constituent max pt, in the same shape as the jet collection.
    """
    return _calculate_unsubtracted_constituent_max_pt(
        input_arrays=arrays,
        input_arrays_source_indices=arrays.source_index,
        input_constituents_source_indices=constituents.source_index,
        builder=ak.ArrayBuilder(),
    ).snapshot()


def _apply_constituent_indices_to_expanded_array(array_to_expand: ak.Array, constituent_indices: ak.Array) -> ak.Array:
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
        _msg = f"Did you pass a fraction? Passed {percentage}. Check it!"
        raise ValueError(_msg)
    return percentage / 100.0 * np.pi * jet_R * jet_R


def _indices_for_event_boundaries(array: ak.Array) -> npt.NDArray[np.int64]:
    """Determine indies of event boundaries.

    Args:
        array: Event-based array of particles.
    Returns:
        Cumulative sum of indices of event boundaries (eg. [0, 3, 7, 11, ..] for 3, 4, 4
            particles in 3 total events)
    """
    counts: npt.NDArray[np.int64] = ak.num(array, axis=1)
    # To use for indexing, we need to keep track of the cumulative sum. That way, we can
    # slice using these indices.
    sum_counts = np.cumsum(np.asarray(counts))
    # However, to use as slices, we need one more entry than the number of events. We
    # account for this by inserting 0 at the beginning since the first indices starts at 0.
    sum_counts = np.insert(sum_counts, 0, 0)
    return sum_counts  # noqa: RET504


#@nb.njit  # type: ignore[misc]
def _find_constituent_indices_via_user_index(
    user_indices: ak.Array, constituents_user_index: ak.Array, number_of_constituents: int
) -> ak.Array:
    output = np.ones(number_of_constituents, dtype=np.int64) * -1
    output_counter = 0
    for event_user_index, event_constituents_user_index in zip(user_indices, constituents_user_index):
        for jet_constituents_user_index in event_constituents_user_index:
            for jet_constituent_index in jet_constituents_user_index:
                #for constituent_index in jet_constituents_user_index:
                #print(f"{jet_constituent_index=}, {event_user_index=}")
                for i_original_constituent, user_index in enumerate(event_user_index):
                    if jet_constituent_index == user_index:
                        #print(f"Found match for {jet_constituent_index} at original index {i_original_constituent}")
                        output[output_counter] = i_original_constituent
                        output_counter += 1
                        break
                else:
                    _msg = "Could not find match " + str(jet_constituent_index)
                    print(_msg)  # noqa: T201
                    # NOTE: Can't pass the message directly with numba since it would have to be a compile time constant (as of Mar 2023).
                    #       As an alternative, we print the message, and then we raise the exception. As long as we don't catch it, it
                    #       achieves basically the same thing.
                    raise ValueError

    # Make sure we've found a match everywhere.
    #assert np.all(output_counter != -1)
    return output


def find_constituent_indices_via_user_index(
    user_indices: ak.Array,
    constituents_user_index: ak.Array
) -> ak.Array:
    res = _find_constituent_indices_via_user_index(
        user_indices=user_indices,
        constituents_user_index=constituents_user_index,
        number_of_constituents=ak.count(constituents_user_index),
    )

    first_step = ak.unflatten(res, ak.count(user_indices, axis=1))
    return ak.unflatten(
        first_step,
        ak.flatten(ak.count(constituents_user_index, axis=1)),
        axis=1
    )

def _find_unsubtracted_constituent_index_from_subtracted_index_via_user_index(
    user_indices: ak.Array,
    subtracted_index_to_unsubtracted_user_index: ak.Array,
    number_of_subtracted_constituents: int,
) -> ak.Array:
    output = np.ones(number_of_subtracted_constituents, dtype=np.int64) * -1
    output_counter = 0

    for event_user_index, event_subtracted_index_to_unsubtracted_user_index in zip(user_indices, subtracted_index_to_unsubtracted_user_index):
        for unsubtracted_user_index in event_subtracted_index_to_unsubtracted_user_index:
            for i_original_constituent, user_index in enumerate(event_user_index):
                if unsubtracted_user_index == user_index:
                    #print(f"Found match for {jet_constituent_index} at original index {i_original_constituent}")
                    output[output_counter] = i_original_constituent
                    output_counter += 1
                    break
            else:
                _msg = "Could not find match " + str(unsubtracted_user_index)
                print(_msg)  # noqa: T201
                # NOTE: Can't pass the message directly with numba since it would have to be a compile time constant (as of Mar 2023).
                #       As an alternative, we print the message, and then we raise the exception. As long as we don't catch it, it
                #       achieves basically the same thing.
                raise ValueError

    return output


def find_unsubtracted_constituent_index_from_subtracted_index_via_user_index(
    user_indices: ak.Array,
    subtracted_index_to_unsubtracted_user_index: ak.Array,
) -> ak.Array:
    res = _find_unsubtracted_constituent_index_from_subtracted_index_via_user_index(
        user_indices=user_indices,
        subtracted_index_to_unsubtracted_user_index=subtracted_index_to_unsubtracted_user_index,
        number_of_subtracted_constituents=ak.count(subtracted_index_to_unsubtracted_user_index),
    )

    return ak.unflatten(res, ak.count(subtracted_index_to_unsubtracted_user_index, axis=1))


def calculate_user_index_with_encoded_sign_info(
    particles: ak.Array,
    mask_to_encode_with_negative: ak.Array,
) -> ak.Array:
    """Calculate the user index and encode information into the sign"""
    # Validation
    # If the 0th particle was to be encoded negative, we could miss this since it won't store the sign.
    # In practice, I don't think this is terribly likely, but we can warn the user if this would happen.
    if ak.any(
        ak.local_index(particles.px, axis=-1)[mask_to_encode_with_negative] == 0
    ):
        # TODO: Test, but with and without this!
        _msg = "Particles requested to be encoded contain index of 0. We will miss this encoded info for this index. This is probably wrong, but you need to think through how to fix this!"
        raise ValueError(_msg)

    # Use px as a proxy - any particle property field would be fine
    user_index = ak.local_index(particles.px, axis=-1)
    user_index[mask_to_encode_with_negative] = -1 * user_index[mask_to_encode_with_negative]
    return user_index


def _handle_subtracted_constituents(
    particles: ak.Array,
    user_index: npt.NDArray[np.int64] | None,
    constituents_user_index_awkward: ak.Array,
    subtracted_constituents: dict[str, list[npt.NDArray[np.float32 | np.float64]]],
    subtracted_index_to_unsubtracted_user_index: list[npt.NDArray[np.int64]],
) -> tuple[ak.Array, ak.Array]:
    # In the case of subtracted constituents, the indices that were returned reference the subtracted
    # constituents. Consequently, we need to build our jet constituents using the subtracted constituent
    # four vectors that were returned from the jet finding.
    # NOTE: Constituents are expected to have a `source_index` field to identify their input source,
    #       and an `identifier` field to specify their relationship between collections.
    #       However, the `subtracted_constituents` assigned here to `_particles_for_constituents`
    #       only contain the kinematic fields because we only returned the four vectors from the
    #       jet finding. We need to add them in below!
    _particles_for_constituents = ak.Array(subtracted_constituents)

    # Now, to figure out the proper mapping between the original input particles and the subtracted constituents.
    _subtracted_index_to_unsubtracted_user_index_awkward = ak.Array(subtracted_index_to_unsubtracted_user_index)
    # We need to match the value stored with each subtracted index with the index stored with each unsubtracted.
    # We need to handle the case of the user_index carefully!
    if user_index is not None:
        # First, deal with the subtracted index -> unsubtracted user_index mapping.
        # We need to convert it into subtracted index -> unsubtracted index by finding the index where
        # the user_index matches
        _subtracted_index_to_unsubtracted_index_awkward = find_unsubtracted_constituent_index_from_subtracted_index_via_user_index(
            user_indices=particles.user_index,
            subtracted_index_to_unsubtracted_user_index=_subtracted_index_to_unsubtracted_user_index_awkward,
        )

        # Next, we need to deal with the subtracted index -> user_index map.
        # Here, we don't allow the user to directly set this value. Instead, we make the indexing
        # continuous (up to ghost particles, which are filtered out). When the user provides the
        # user_index, it can encode the sign of the unsubtracted user_index, so we take abs here.
        _constituent_indices_awkward = np.abs(constituents_user_index_awkward)
    else:
        # Since the user didn't provide the user_index, the subtracted indices map directly to unsubtracted indices
        _subtracted_index_to_unsubtracted_index_awkward = _subtracted_index_to_unsubtracted_user_index_awkward

        # Here, since we didn't provide the user_index, we don't need to do anything else - just assign
        # for consistent variable naming
        _constituent_indices_awkward = constituents_user_index_awkward

    # Now, map the subtracted-to-unsubtracted. In this case, since the subtracted-to-unsubtracted mapping is
    # actually just a list of unsubtracted indices (where the location of unsubtracted index corresponds to
    # the subtracted particles), we can directly apply this "mapping" to the unsubtracted particles `index`
    # (after converting it to awkward)

    # As an interlude, we want to pass on all fields which are associated with the particles.
    # However, we need to filter out all kinematic variables (for which we already have the subtracted values).
    # This is kind of a dumb way to do it, but it works, so good enough.
    _kinematic_fields_to_skip_for_applying_subtracted_indices_mask = [
        "px", "py", "pz", "E",
        "pt", "eta", "phi", "m",
        "x", "y", "z", "t",
    ]
    _additional_fields_for_subtracted_constituents_names = list(set(ak.fields(particles)) - set(_kinematic_fields_to_skip_for_applying_subtracted_indices_mask))
    # Now, we can finally grab the additional fields
    _additional_fields_for_subtracted_constituents = particles[_additional_fields_for_subtracted_constituents_names][
        _subtracted_index_to_unsubtracted_index_awkward
    ]

    # Then, we just need to zip it in to the particles for constituents, and it will be brought
    # along when the constituents are associated with the jets.
    _particles_for_constituents = ak.zip(
        {
            **dict(
                zip(
                    ak.fields(_particles_for_constituents),
                    ak.unzip(_particles_for_constituents)
                )
            ),
            **dict(
                zip(
                    ak.fields(_additional_fields_for_subtracted_constituents),
                    ak.unzip(_additional_fields_for_subtracted_constituents),
                )
            ),
        },
        with_name="Momentum4D",
    )

    return _particles_for_constituents, _constituent_indices_awkward


def find_jets(
    particles: ak.Array,
    jet_finding_settings: JetFindingSettings,
    background_particles: ak.Array | None = None,
    background_subtraction: BackgroundSubtraction | None = None,
) -> ak.Array:
    # Validation
    if background_subtraction is None:
        background_subtraction = BackgroundSubtraction(type=BackgroundSubtractionType.disabled)
    if jet_finding_settings.recombiner is not None and \
        isinstance(jet_finding_settings.recombiner, NegativeEnergyRecombiner) and "user_index" not in ak.fields(particles):  # type: ignore[redundant-expr]
        _msg = "The Negative Energy Recombiner requires you to encode the relevant info into the user index and pass them to the jet finder."
        raise ValueError(_msg)

    logger.info(f"Jet finding settings: {jet_finding_settings}")
    logger.info(f"Background subtraction settings: {background_subtraction}")

    # Keep track of the event transitions.
    sum_counts = _indices_for_event_boundaries(particles)

    # Validate that there is at least one particle per event
    # NOTE: This can be avoided by requiring that there are particles in each level for each event.
    #       However, this is left as a user preprocessing step to avoid surprising users!
    event_with_no_particles = sum_counts[1:] == sum_counts[:-1]
    if np.any(event_with_no_particles):
        _msg = f"There are some events with zero particles, which is going to mess up the alignment. Check the input! 0s are at {np.where(event_with_no_particles)}"
        raise ValueError(_msg)

    # Now, deal with the particles themselves.
    # This will flatten the awkward array contents while keeping the record names.
    flattened_particles = ak.flatten(particles, axis=1)
    # We only want vector to calculate the components once (the input components may not
    # actually be px, py, pz, and E), so calculate them now, and view them as numpy arrays
    # so we can pass them directly into our function.
    # NOTE: To avoid argument mismatches when calling to the c++, we view as float64 (which
    #       will be converted to a double). As of July 2021, tests seem to indicate that it's
    #       not making the float32 -> float conversion properly.
    px: npt.NDArray[np.float64] = np.asarray(flattened_particles.px, dtype=np.float64)
    py: npt.NDArray[np.float64] = np.asarray(flattened_particles.py, dtype=np.float64)
    pz: npt.NDArray[np.float64] = np.asarray(flattened_particles.pz, dtype=np.float64)
    E: npt.NDArray[np.float64] = np.asarray(flattened_particles.E, dtype=np.float64)
    # Provide this value only if it's available in the array
    user_index = None
    if "user_index" in ak.fields(flattened_particles):
        user_index = np.asarray(flattened_particles.user_index, dtype=np.int64)

    # Now, onto the background particles. If background particles were passed, we want to do the
    # same thing as the input particles
    if background_particles is not None:
        background_sum_counts = _indices_for_event_boundaries(background_particles)

        # Validate that there is at least one particle per event
        event_with_no_particles = background_sum_counts[1:] == background_sum_counts[:-1]
        if np.any(event_with_no_particles):
            _msg = f"There are some background events with zero particles, which is going to mess up the alignment. Check the input! 0s are at {np.where(event_with_no_particles)}"
            raise ValueError(_msg)

        # Now, deal with the particles themselves.
        # This will flatten the awkward array contents while keeping the record names.
        flattened_background_particles = ak.flatten(background_particles, axis=1)
        # We only want vector to calculate the components once (the input components may not
        # actually be px, py, pz, and E), so calculate them now, and view them as numpy arrays
        # so we can pass them directly into our function.
        # NOTE: To avoid argument mismatches when calling to the c++, we view as float64 (which
        #       will be converted to a double). As of July 2021, tests seem to indicate that it's
        #       not making the float32 -> float conversion properly.
        background_px: npt.NDArray[np.float64] = np.asarray(flattened_background_particles.px, dtype=np.float64)
        background_py: npt.NDArray[np.float64] = np.asarray(flattened_background_particles.py, dtype=np.float64)
        background_pz: npt.NDArray[np.float64] = np.asarray(flattened_background_particles.pz, dtype=np.float64)
        background_E: npt.NDArray[np.float64] = np.asarray(flattened_background_particles.E, dtype=np.float64)
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
        _msg = f"Mismatched between number of events for signal and background. Signal: {len(sum_counts) -1}, background: {len(background_sum_counts) - 1}"
        raise ValueError(_msg)

    # Keep track of the jet four vector components. Although this will have to be converted later,
    # it seems that this is good enough enough to start.
    # NOTE: If this gets too slow, we can do the jet finding over multiple events in c++ like what
    #       is done in the new fj bindings. I skip this for now because my existing code seems to
    #       be good enough.
    jets: dict[str, list[np.float32 | np.float64 | npt.NDArray[np.float32 | np.float64]]] = {
        "px": [],
        "py": [],
        "pz": [],
        "E": [],
        "area": [],
        # NOTE: We can't call it "rho" because that will interfere with the four vector with a length "rho".
        "rho_value": [],
    }
    # Maps from index of the constituent to the user_index of the input particles.
    # If a user_index is not provided by the user, then the user_index that is returned here will
    # simply be an index. However, if a user_index is provided by the user, then we need to map
    # from the values here (ie. the user_index) to the index of the particle.
    constituents_user_index = []
    subtracted_constituents: dict[str, list[npt.NDArray[np.float32 | np.float64]]] = {
        "px": [],
        "py": [],
        "pz": [],
        "E": [],
    }
    # Map the index of the subtracted constituent to the user_index of the unsubtracted (ie. input) particles
    # If a user_index is not provided by the user, then the user_index that is returned here will
    # simply be an index. However, if a user_index is provided by the user, then we need to map
    # from the values here (ie. the unsubtracted user_index) to the index of the unsubtracted particle.
    subtracted_index_to_unsubtracted_user_index = []
    for lower, upper, background_lower, background_upper in zip(
        sum_counts[:-1], sum_counts[1:], background_sum_counts[:-1], background_sum_counts[1:]
    ):
        # Run the actual jet finding.
        res = mammoth_cpp._ext.find_jets(
            px=px[lower:upper],
            py=py[lower:upper],
            pz=pz[lower:upper],
            E=E[lower:upper],
            jet_finding_settings=jet_finding_settings,
            background_px=background_px[background_lower:background_upper],
            background_py=background_py[background_lower:background_upper],
            background_pz=background_pz[background_lower:background_upper],
            background_E=background_E[background_lower:background_upper],
            background_subtraction=background_subtraction,
            user_index=user_index[lower:upper] if user_index is not None else None,
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
        jets["rho_value"].append(res.rho_value)
        # Next, associate the indices of the constituents that are associated with each jet
        constituents_user_index.append(res.constituents_user_index)

        # Finally, we'll handle the subtracted constituents if relevant
        subtracted_info = res.subtracted_info
        if subtracted_info:
            # Unpack and store the subtracted constituent four vector
            subtracted_constituents["px"].append(subtracted_info[0][0])
            subtracted_constituents["py"].append(subtracted_info[0][1])
            subtracted_constituents["pz"].append(subtracted_info[0][2])
            subtracted_constituents["E"].append(subtracted_info[0][3])
            # Plus the association for each subtracted constituent index into the unsubtracted constituent.
            # NOTE: These are the indices assigned via the user_index.
            subtracted_index_to_unsubtracted_user_index.append(subtracted_info[1])

    # To create the output, we start with the constituents.
    # First, we convert the fastjet user_index indices that we use for book keeping during jet finding
    # into an awkward array to make the next operations simpler.
    # NOTE: Remember, these indices are just for internal book keeping during jet finding! (Although if the user_index
    #       is passed, then they utilize this information). They're totally separate from the `source_index` and `identifier`
    #       fields that we include in all particles arrays to (source_index:) identify the source of jet constituents
    #       (ie. in terms of the input particle collection that they're from) and (identifier): identify relationships
    #       between particles, respectively.
    # NOTE: This requires a copy, but that's fine - there's nothing to be done about it, so it's just the cost.
    _constituents_user_index_awkward = ak.Array(constituents_user_index)

    # If we have subtracted constituents, we need to handle them very carefully.
    if subtracted_index_to_unsubtracted_user_index:
        # This is quite tricky, so we handle it in a dedicated function
        _particles_for_constituents, _constituent_indices_awkward = _handle_subtracted_constituents(
            particles=particles,
            user_index=user_index,
            constituents_user_index_awkward=_constituents_user_index_awkward,
            subtracted_constituents=subtracted_constituents,
            subtracted_index_to_unsubtracted_user_index=subtracted_index_to_unsubtracted_user_index,
        )
    else:
        if user_index is not None:
            # If we passed the user_index, then the constituent_indices which are returned (which are just the user_index
            # from fastjet) won't actually be indices of particles (ie. it may be a label, rather than an index we can use
            # in the array to find the right constituents). So here, we match up the user_index that was passed with the
            # user_index that was returned, allowing us to map the returned user_index to proper indices.
            _constituent_indices_awkward = find_constituent_indices_via_user_index(
                user_indices=particles.user_index,
                constituents_user_index=_constituents_user_index_awkward,
            )
        else:
            # Nothing needs to be done here. Just assign for the next step
            _constituent_indices_awkward = _constituents_user_index_awkward

        # Since additional fields are already included in particles, there's nothing else we need
        # to do here.
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
          to the jets. Those constituents can identify their source via `identifier`. If we later decide that
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

    # Add additional columns that only apply for subtracted constituents
    additional_jet_level_fields = {}
    # Only include if we actually calculated the area
    if jet_finding_settings.area_settings:
        additional_jet_level_fields["area"] = jets["area"]
    if subtracted_index_to_unsubtracted_user_index:
        # Here, we add the unsubtracted constituent max pt
        additional_jet_level_fields["unsubtracted_leading_track_pt"] = calculate_unsubtracted_constituent_max_pt(
            arrays=particles,
            constituents=output_constituents,
        )

    # Finally, construct the output
    output_jets = ak.zip(
        {
            "px": jets["px"],
            "py": jets["py"],
            "pz": jets["pz"],
            "E": jets["E"],
            "rho_value": jets["rho_value"],
            "constituents": output_constituents,
            **additional_jet_level_fields,
        },
        with_name="Momentum4D",
        # Limit of 2 is based on: 1 for events + 1 for jets
        depth_limit=2,
    )

    return output_jets  # noqa: RET504


def _splittings_output() -> dict[str, list[Any]]:
    return {
        "kt": [],
        "delta_R": [],
        "z": [],
        "parent_index": [],
    }


def _subjets_output() -> dict[str, list[Any]]:
    return {
        "splitting_node_index": [],
        "part_of_iterative_splitting": [],
        "constituent_indices": [],
    }


def recluster_jets(
    jets: ak.Array,
    jet_finding_settings: JetFindingSettings,
    store_recursive_splittings: bool,
) -> ak.Array:
    # Validation. There must be jets
    if len(jets) == 0:
        _msg = "No jets present for reclustering!"
        raise ValueError(_msg)

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
    offsets_constituents = np.insert(offsets_constituents, 0, 0)
    # Then into starts and stops
    # NOTE: wrapping ak.num with np.asarray is needed for awkward 2.0.5 due to a minor bug in
    #       how the arguments are handled. See: https://github.com/scikit-hep/awkward/issues/2071
    starts_constituents = ak.unflatten(offsets_constituents[:-1], np.asarray(ak.num(num_constituents, axis=1)))
    stops_constituents = ak.unflatten(offsets_constituents[1:], np.asarray(ak.num(num_constituents, axis=1)))

    # Now, setup the constituents themselves.
    # It would be nice to flatten them and then assign the components like for standard jet
    # finding, but since we have to flatten with `None` to deals with the multiple levels,
    # it also flattens over the records.
    # NOTE: `axis="records"` isn't yet available (July 2021).
    # We only want vector to calculate the components once (the input components may not
    # actually be px, py, pz, and E), so calculate them now, and view them as numpy arrays
    # so we can pass them directly into our function.
    # NOTE: To avoid argument mismatches when calling to the c++, we view as float64 (which
    #       will be converted to a double). As of July 2021, tests seem to suggest that it's
    #       not making the float32 -> float conversion properly.
    px = np.asarray(ak.flatten(jets.constituents.px, axis=None), dtype=np.float64)
    py = np.asarray(ak.flatten(jets.constituents.py, axis=None), dtype=np.float64)
    pz = np.asarray(ak.flatten(jets.constituents.pz, axis=None), dtype=np.float64)
    E = np.asarray(ak.flatten(jets.constituents.E, axis=None), dtype=np.float64)

    event_splittings = _splittings_output()
    event_subjets = _subjets_output()
    for starts, stops in zip(starts_constituents, stops_constituents):
        jets_splittings = _splittings_output()
        jets_subjets = _subjets_output()
        for lower, upper in zip(starts, stops):
            res = mammoth_cpp._ext.recluster_jet(
                px=px[lower:upper],
                py=py[lower:upper],
                pz=pz[lower:upper],
                E=E[lower:upper],
                jet_finding_settings=jet_finding_settings,
                store_recursive_splittings=store_recursive_splittings,
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
            "jet_splittings": ak.zip(event_splittings),
            "subjets": ak.zip(
                event_subjets,
                # zip over events, jets, and subjets (but can't go all the way due to the constituent indices)
                depth_limit=3,
            ),
        },
        depth_limit=2,
    )
