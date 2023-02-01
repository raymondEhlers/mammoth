""" Jet related analysis tools.

These all build on various aspects of the framework, but are at a higher level than the basic framework
functionality itself.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import attr
import awkward as ak
import numpy as np
import numpy.typing as npt

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


@attr.define(frozen=True)
class PtDependentTrackingEfficiencyParameters:
    bin_edges: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    baseline_tracking_efficiency_shift: float = 1.0

    @classmethod
    def from_file(cls, period: str | Sequence[str], event_activity: str, baseline_tracking_efficiency_shift: float) -> PtDependentTrackingEfficiencyParameters:
        # Validation
        if isinstance(period, str):
            period = [period]

        # Load yaml file
        from pachyderm import yaml
        y = yaml.yaml()
        _here = Path(__file__).parent
        config_filename = Path(_here.parent / "alice" / "config" / "track_efficiency_pt_dependence.yaml")
        with open(config_filename, "r") as f:
            config = y.load(f)

        # Grab values from file
        bin_edges = np.array(config["pt_binning"], dtype=np.float64)
        possible_values = []
        # Iterate over the periods to average possible contributions
        for _period in period:
            possible_values.append(
                np.array(config[_period][event_activity], dtype=np.float64)
            )
        values = np.mean(possible_values, axis=0)

        # Validate values vs pt bin
        assert len(values) + 1 == len(bin_edges), f"Bin edges don't match up to values. {len(bin_edges)=}, {len(values)=}"

        return cls(
            bin_edges=bin_edges,
            values=values,
            baseline_tracking_efficiency_shift=baseline_tracking_efficiency_shift,
        )

    def calculate_tracking_efficiency(
        self,
        pt_values: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        _indices_for_pt_dependent_values = np.searchsorted(
            self.bin_edges, pt_values, side="right",
        )
        # NOTE: We want pt values in the first bin to get mapped to 0th entry in the tracking efficiency,
        #       so we subtract one from each index
        _indices_for_pt_dependent_values -= 1
        # And determine the values
        _pt_dependent_tracking_efficiency: npt.NDArray[np.float64] = self.values[_indices_for_pt_dependent_values]
        # Apply additional baseline tracking efficiency degradation for high multiplicity environment
        # NOTE: We take 1.0 - value because it's defined as eg. 0.97, so to add it on the pt dependent values,
        # we have to determine how much _more_ to add on.
        _pt_dependent_tracking_efficiency = (
            _pt_dependent_tracking_efficiency -
            (1.0 - self.baseline_tracking_efficiency_shift)
        )

        return _pt_dependent_tracking_efficiency


def hybrid_level_particles_mask_for_jet_finding(
    arrays: ak.Array,
    det_level_artificial_tracking_efficiency: float | PtDependentTrackingEfficiencyParameters,
    source_index_identifiers: Mapping[str, int],
    validation_mode: bool,
) -> Tuple[ak.Array, ak.Array]:
    """Calculate mask to separate hybrid and det level tracks, potentially apply an additional tracking efficiency uncertainty

    Args:
        arrays: Input particle level arrays
        det_level_artificial_tracking_efficiency: Artificial tracking efficiency to apply only to detector level particles.
        source_index_identifiers: Index offset map for each source. Provided by the embedding.
        validation_mode: If True, we're running in validation mode.
    Returns:
        Mask to apply to the hybrid level particles during jet finding, mask selecting only the background particles.
    """
    # We need the background only particles mask for the tracking inefficiency,
    # so we calculate it first
    background_particles_only_mask = hybrid_background_particles_only_mask(
        arrays=arrays, source_index_identifiers=source_index_identifiers
    )

    # Create an artificial tracking efficiency for detector level particles
    # To apply this, we want to select all background tracks + the subset of det level particles to keep
    # First, start with an all True mask
    hybrid_level_mask = (arrays["hybrid"].pt >= 0)
    if isinstance(det_level_artificial_tracking_efficiency, PtDependentTrackingEfficiencyParameters) or det_level_artificial_tracking_efficiency < 1.0:
        if validation_mode:
            raise ValueError(
                "Cannot apply artificial tracking efficiency during validation mode. The randomness will surely break the validation."
            )

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
        if isinstance(det_level_artificial_tracking_efficiency, PtDependentTrackingEfficiencyParameters):
            _pt_dependent_tracking_efficiency = det_level_artificial_tracking_efficiency.calculate_tracking_efficiency(
                # NOTE: We need to flatten to be able to use searchsorted
                pt_values=ak.flatten(arrays["hybrid"][~background_particles_only_mask].pt),
            )

            _drop_particles_mask = random_values > _pt_dependent_tracking_efficiency
        else:
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


def det_level_particles_mask_for_jet_finding(
    arrays: ak.Array,
    det_level_artificial_tracking_efficiency: float | PtDependentTrackingEfficiencyParameters,
    validation_mode: bool,
) -> ak.Array:
    """Calculate mask to apply an additional tracking efficiency uncertainty to detector level

    Similar to (and code based on) `hybrid_level_particles_mask_for_jet_finding`, but simpler since we don't
    have to apply the selection to a subset of particles in a overall array.

    Args:
        arrays: Input particle level arrays
        det_level_artificial_tracking_efficiency: Artificial tracking efficiency to apply only to detector level particles.
        validation_mode: If True, we're running in validation mode.
    Returns:
        Mask to apply to the det level particles during jet finding.
    """
    det_level_mask = (arrays["det_level"].pt >= 0)
    if isinstance(det_level_artificial_tracking_efficiency, PtDependentTrackingEfficiencyParameters) or det_level_artificial_tracking_efficiency < 1.0:
        if validation_mode:
            raise ValueError(
                "Cannot apply artificial tracking efficiency during validation mode. The randomness will surely break the validation."
            )

        # Here, we focus in on the detector level particles.
        # First, we determine the total number of det_level particles to determine how many random
        # numbers to generate (plus, the info needed to unflatten later)
        _n_det_level_particles_per_event = ak.num(arrays["det_level"], axis=1)
        _total_n_det_level_particles = ak.sum(_n_det_level_particles_per_event)

        # Next, drop particles if their random values that are higher than the tracking efficiency
        _rng = np.random.default_rng()
        random_values = _rng.uniform(low=0.0, high=1.0, size=_total_n_det_level_particles)
        if isinstance(det_level_artificial_tracking_efficiency, PtDependentTrackingEfficiencyParameters):
            _pt_dependent_tracking_efficiency = det_level_artificial_tracking_efficiency.calculate_tracking_efficiency(
                # NOTE: We need to flatten to be able to use searchsorted
                pt_values=ak.flatten(arrays["det_level"].pt),
            )

            _drop_particles_mask = random_values > _pt_dependent_tracking_efficiency
        else:
            _drop_particles_mask = random_values > det_level_artificial_tracking_efficiency
        # NOTE: The check above will assign `True` when the random value is higher than the tracking efficiency.
        #       However, since we to remove those particles and keep ones below, we need to invert this selection.
        det_level_mask = ~_drop_particles_mask

        # Cross check that it worked.
        # If the entire hybrid mask is True, then it means that no particles were removed.
        # NOTE: I don't have this as an assert because if there aren't _that_ many particles and the efficiency
        #       is high, I suppose it's possible that this fails, and I don't want to kill jobs for that reason.
        if ak.all(det_level_mask == True):  # noqa: E712
            logger.warning(
                "No particles were removed in the artificial tracking efficiency."
                " This is possible, but not super likely. Please check your settings!"
            )

    return det_level_mask
