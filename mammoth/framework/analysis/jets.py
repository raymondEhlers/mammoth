""" Jet related analysis tools.

These all build on various aspects of the framework, but are at a higher level than the basic framework
functionality itself.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL
"""

import logging
from typing import Mapping, Tuple

import awkward as ak
import numpy as np

from mammoth.framework import jet_finding


logger = logging.getLogger(__name__)


def jet_matching_MC(
    jets: ak.Array,
    part_level_det_level_max_matching_distance: float = 1.0,
) -> ak.Array:
    """Geometrical jet matching for MC

    Note:
        This throws out jets if there is no match between detector level and particle level jets.
        This shouldn't be surprising, but sometimes I overlook this point, so just making it explicit!

    Note:
        The default matching distance is larger than the matching distance that I would have
        expected to use (eg. we usually use 0.3 = in embedding), but this is apparently what
        we use in pythia. So just go with it.

    Args:
        jets: Array containing the jets to match
        part_level_det_level_max_matching distance: Maximum matching distance
            between part and det level. Default: 1.0
    Returns:
        jets array containing only the matched jets
    """
    # First, actually perform the geometrical matching
    jets["part_level", "matching"], jets["det_level", "matching"] = jet_finding.jet_matching_geometrical(
        jets_base=jets["part_level"],
        jets_tag=jets["det_level"],
        max_matching_distance=part_level_det_level_max_matching_distance,
    )

    # Now, use matching info
    # First, require that there are jets in an event. If there are jets, further require that there
    # is a valid match (further below).
    # NOTE: These can't be combined into one mask because they operate at different levels: events and jets
    logger.info("Using matching info")
    jets_present_mask = (ak.num(jets["part_level"], axis=1) > 0) & (ak.num(jets["det_level"], axis=1) > 0)
    jets = jets[jets_present_mask]
    logger.warning(
        f"post jets present mask n accepted: {np.count_nonzero(np.asarray(ak.flatten(jets['det_level'].px, axis=None)))}"
    )

    # Now, onto the individual jet collections
    # We want to require valid matched jet indices. The strategy here is to lead via the detector
    # level jets. The procedure is as follows:
    #
    # 1. Identify all of the detector level jets with valid matches.
    # 2. Apply that mask to the detector level jets.
    # 3. Use the matching indices from the detector level jets to index the particle level jets.
    # 4. We should be done and ready to flatten. Note that the matching indices will refer to
    #    the original arrays, not the masked ones. In principle, this should be updated, but
    #    I'm unsure if they'll be used again, so we wait to update them until it's clear that
    #    it's required.
    #
    # The other benefit to this approach is that it should reorder the particle level matches
    # to be the same shape as the detector level jets, so they are already paired together.
    #
    # NOTE: This method doesn't check that particle level jets match to the detector level jets.
    #       However, it doesn't need to do so because the geometrical matching implementation
    #       is bijective, so it already must be the case.
    det_level_matched_jets_mask = jets["det_level"]["matching"] > -1
    jets["det_level"] = jets["det_level"][det_level_matched_jets_mask]
    jets["part_level"] = jets["part_level"][jets["det_level", "matching"]]
    logger.warning(
        f"post requiring valid matches n accepted: {np.count_nonzero(np.asarray(ak.flatten(jets['det_level'].px, axis=None)))}"
    )

    return jets


def jet_matching_embedding(
    jets: ak.Array,
    det_level_hybrid_max_matching_distance: float = 0.3,
    part_level_det_level_max_matching_distance: float = 0.3,
) -> ak.Array:
    """Jet matching for embedding.

    Note:
        This throws out jets if there is no match between detector level and particle level jets.
        This shouldn't be surprising, but sometimes I overlook this point, so just making it explicit!

    Args:
        jets: Array containing the jets to match
        det_level_hybrid_max_matching_distance: Maximum matching distance
            between det level and hybrid. Default: 0.3
        part_level_det_level_max_matching distance: Maximum matching distance
            between part and det level. Default: 0.3
    Returns:
        jets array containing only the matched jets
    """

    # NOTE: This is a bit of a departure from the AliPhysics constituent subtraction method.
    #       Namely, in AliPhysics, we doing an additional matching step between hybrid sub
    #       and hybrid unsubtracted (and then matching hybrid unsubtracted to det level, etc).
    #       However, since we have the the additional jet constituent indexing info, we can
    #       figure out the matching directly between hybrid sub and det level. So we don't include
    #       the intermediate matching.
    jets["det_level", "matching"], jets["hybrid", "matching"] = jet_finding.jet_matching_geometrical(
        jets_base=jets["det_level"],
        jets_tag=jets["hybrid"],
        max_matching_distance=det_level_hybrid_max_matching_distance,
    )
    jets["part_level", "matching"], jets["det_level", "matching"] = jet_finding.jet_matching_geometrical(
        jets_base=jets["part_level"],
        jets_tag=jets["det_level"],
        max_matching_distance=part_level_det_level_max_matching_distance,
    )

    # Now, use matching info
    # First, require that there are jets in an event. If there are jets, and require that there
    # is a valid match.
    # NOTE: These can't be combined into one mask because they operate at different levels: events and jets
    logger.info("Using matching info")
    jets_present_mask = (
        (ak.num(jets["part_level"], axis=1) > 0)
        & (ak.num(jets["det_level"], axis=1) > 0)
        & (ak.num(jets["hybrid"], axis=1) > 0)
    )
    jets = jets[jets_present_mask]

    # Now, onto the individual jet collections
    # We want to require valid matched jet indices. The strategy here is to lead via the detector
    # level jets. The procedure is as follows:
    #
    # 1. Identify all of the detector level jets with valid matches.
    # 2. Apply that mask to the detector level jets.
    # 3. Use the matching indices from the detector level jets to index the particle level jets.
    # 4. We should be done and ready to flatten. Note that the matching indices will refer to
    #    the original arrays, not the masked ones. In principle, this should be updated, but
    #    I'm unsure if they'll be used again, so we wait to update them until it's clear that
    #    it's required.
    #
    # The other benefit to this approach is that it should reorder the particle level matches
    # to be the same shape as the detector level jets (same for hybrid, etc), so they are already
    # all paired together.
    #
    # NOTE: This method doesn't check that particle level jets match to the detector level jets.
    #       However, it doesn't need to do so because the matching is bijective, so it already
    #       must be the case.
    hybrid_to_det_level_valid_matches = jets["hybrid", "matching"] > -1
    det_to_part_level_valid_matches = jets["det_level", "matching"] > -1
    hybrid_to_det_level_including_det_to_part_level_valid_matches = det_to_part_level_valid_matches[
        jets["hybrid", "matching"][hybrid_to_det_level_valid_matches]
    ]
    # First, restrict the hybrid level, requiring hybrid to det_level valid matches and
    # det_level to part_level valid matches.
    jets["hybrid"] = jets["hybrid"][hybrid_to_det_level_valid_matches][
        hybrid_to_det_level_including_det_to_part_level_valid_matches
    ]
    # Next, restrict the det_level. Since we've restricted the hybrid to only valid matches, we should be able
    # to directly apply the masking indices.
    jets["det_level"] = jets["det_level"][jets["hybrid", "matching"]]
    # Same reasoning here.
    jets["part_level"] = jets["part_level"][jets["det_level", "matching"]]

    # After all of these gymnastics, we may not have jets at all levels, so require there to a jet of each type.
    # In principle, we've done this twice now, but logically this seems to be clearest.
    jets_present_mask = (
        (ak.num(jets["part_level"], axis=1) > 0)
        & (ak.num(jets["det_level"], axis=1) > 0)
        & (ak.num(jets["hybrid"], axis=1) > 0)
    )
    jets = jets[jets_present_mask]

    return jets


def hybrid_background_particles_only_mask(
    arrays: ak.Array,
    source_index_identifiers: Mapping[str, int],
) -> ak.Array:
    """Select only background particles from the hybrid particle collection.

    Args:
        arrays: Input particle level arrays
        source_index_identifiers: Index offset map for each source. Provided by the embedding.
    Returns:
        Mask selected only the background particles.
    """
    # NOTE: The most general approach would be some divisor argument to select the signal source indexed
    #       particles, but since the background has the higher source index, we can just select particles
    #       with an index smaller than that offset.
    background_only_particles_mask = ~(arrays["hybrid", "index"] < source_index_identifiers["background"])
    return background_only_particles_mask


def hybrid_level_particles_mask_for_jet_finding(
    arrays: ak.Array,
    det_level_artificial_tracking_efficiency: float,
    source_index_identifiers: Mapping[str, int],
    validation_mode: bool,
) -> Tuple[ak.Array, ak.Array]:
    """Select only background particles from the hybrid particle collection.

    Args:
        arrays: Input particle level arrays
        source_index_identifiers: Index offset map for each source. Provided by the embedding.
    Returns:
        Mask to apply to the hybrid level particles during jet finding, mask selecting only the background particles.
    """
    # We need the bacgkround only particles mask for the tracking inefficiency,
    # so we calculate it first
    background_particles_only_mask = hybrid_background_particles_only_mask(
        arrays=arrays, source_index_identifiers=source_index_identifiers
    )

    # Create an artificial tracking efficiency for detector level particles
    # To apply this, we want to select all background tracks + the subset of det level particles to keep
    # First, start with an all True mask
    hybrid_level_mask = (arrays["hybrid"].pt >= 0)
    if det_level_artificial_tracking_efficiency < 1.0:
        if validation_mode:
            raise ValueError("Cannot apply artificial tracking efficiency during validation mode. The randomness will surely break the validation.")

        # Here, we focus in on the detector level particles.
        # We want to select only them, determine whether they will be rejected, and then assign back
        # to the full hybrid mask. However, since awkward arrays are immutable, we need to do all of
        # this in numpy, and then unflatten.
        # First, we determine the total number of det_level particles to determine how many random
        # numbers to generate (plus, the info needed to unflatten later)
        _n_det_level_particles_per_event = ak.num(arrays["hybrid"][~background_particles_only_mask], axis=1)
        _total_n_det_level_particles = ak.sum(_n_det_level_particles_per_event)

        # Next, drop particles if their random values that are higher than the tracking efficiency
        _rng = np.random.default_rng()
        random_values = _rng.uniform(low=0.0, high=1.0, size=_total_n_det_level_particles)
        _drop_particles_mask = random_values > det_level_artificial_tracking_efficiency
        # NOTE: The check above will assign `True` when the random value is higher than the tracking efficiency.
        #       However, since we to remove those particles and keep ones below, we need to invert this selection.
        _det_level_particles_to_keep_mask = ~_drop_particles_mask

        # Now, we need to integrate it into the hybrid_level_mask
        # First, flatten that hybrid list into a numpy array, as well as the mask that selects det_level particles only
        _hybrid_level_mask_np = ak.to_numpy(ak.flatten(hybrid_level_mask))
        _det_level_particles_mask_np = ak.to_numpy(ak.flatten(~background_particles_only_mask))
        # Then, we can use the mask selecting det level particles only to assign whether to keep each det level
        # particle due to the tracking efficiency. Since the mask does the assignment
        _hybrid_level_mask_np[_det_level_particles_mask_np] = _det_level_particles_to_keep_mask

        # Unflatten so we can apply the mask to the existing particles
        hybrid_level_mask = ak.unflatten(_hybrid_level_mask_np, ak.num(arrays["hybrid"]))

        # Cross check that it worked.
        # If the entire hybrid mask is True, then it means that no particles were removed.
        # NOTE: I don't have this as an assert because if there aren't _that_ many particles and the efficiency
        #       is high, I suppose it's possible that this fails, and I don't want to kill jobs for that reason.
        if ak.all(hybrid_level_mask == True):  # noqa: E712
            logger.warning(
                "No particles were removed in the artificial tracking efficiency."
                " This is possible, but not super likely. Please check your settings!"
            )

    return hybrid_level_mask, background_particles_only_mask
