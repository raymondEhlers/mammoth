""" Analysis level functionality related to tracking.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import attrs
import awkward as ak
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@attrs.define(frozen=True)
class PtDependentTrackingEfficiencyParameters:
    bin_edges: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    baseline_tracking_efficiency_shift: float = 1.0

    @classmethod
    def from_file(
        cls, period: str | list[str], event_activity: str, baseline_tracking_efficiency_shift: float
    ) -> PtDependentTrackingEfficiencyParameters:
        # Validation
        if isinstance(period, str):
            period = [period]

        # Load yaml file
        from pachyderm import yaml

        y = yaml.yaml()
        _here = Path(__file__).parent
        config_filename = Path(_here.parent.parent / "alice" / "config" / "track_efficiency_pt_dependence.yaml")
        with config_filename.open() as f:
            config = y.load(f)

        # Grab values from file
        bin_edges = np.array(config["pt_binning"], dtype=np.float64)
        possible_values = []
        # Iterate over the periods to average possible contributions
        for _period in period:
            possible_values.append(np.array(config[_period][event_activity], dtype=np.float64))
        values = np.mean(possible_values, axis=0)

        # Validate values vs pt bin
        assert len(values) + 1 == len(
            bin_edges
        ), f"Bin edges don't match up to values. {len(bin_edges)=}, {len(values)=}"

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
            self.bin_edges,
            pt_values,
            side="right",
        )
        # NOTE: We want pt values in the first bin to get mapped to 0th entry in the tracking efficiency,
        #       so we subtract one from each index
        # NOTE: Apparently the type is passed through such that the indices are an awkward array.
        #       Due to this, we couldn't use `-=` and instead had to do the explicit assignment.
        #       It's all fine - just something to watch out for!
        _indices_for_pt_dependent_values = _indices_for_pt_dependent_values - 1
        # And determine the values
        _pt_dependent_tracking_efficiency: npt.NDArray[np.float64] = self.values[_indices_for_pt_dependent_values]
        # Apply additional baseline tracking efficiency degradation for high multiplicity environment
        # NOTE: We take 1.0 - value because it's defined as eg. 0.97, so to add it on the pt dependent values,
        # we have to determine how much _more_ to add on.
        _pt_dependent_tracking_efficiency = _pt_dependent_tracking_efficiency - (
            1.0 - self.baseline_tracking_efficiency_shift
        )

        return _pt_dependent_tracking_efficiency  # noqa: RET504


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
    background_only_particles_mask = ~(arrays["hybrid", "source_index"] < source_index_identifiers["background"])
    return background_only_particles_mask  # noqa: RET504


def hybrid_level_particles_mask_for_jet_finding(
    arrays: ak.Array,
    det_level_artificial_tracking_efficiency: float | PtDependentTrackingEfficiencyParameters,
    source_index_identifiers: Mapping[str, int],
    validation_mode: bool,
) -> tuple[ak.Array, ak.Array]:
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
    hybrid_level_mask = (arrays["hybrid"].pt >= 0)  # fmt: skip
    if (
        isinstance(det_level_artificial_tracking_efficiency, PtDependentTrackingEfficiencyParameters)
        or det_level_artificial_tracking_efficiency < 1.0
    ):
        if validation_mode:
            _message = "Cannot apply artificial tracking efficiency during validation mode. The randomness will surely break the validation."
            raise ValueError(_message)

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
    det_level_mask = (arrays["det_level"].pt >= 0)  # fmt: skip
    if (
        isinstance(det_level_artificial_tracking_efficiency, PtDependentTrackingEfficiencyParameters)
        or det_level_artificial_tracking_efficiency < 1.0
    ):
        if validation_mode:
            _message = "Cannot apply artificial tracking efficiency during validation mode. The randomness will surely break the validation."
            raise ValueError(_message)

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
                # NOTE: We need to flatten to be able to use searchsorted. This will also leave the
                #       result in the right shape to compare against the randomly generated values.
                pt_values=ak.flatten(arrays["det_level"].pt),
            )

            _drop_particles_mask = random_values > _pt_dependent_tracking_efficiency
        else:
            _drop_particles_mask = random_values > det_level_artificial_tracking_efficiency
        # NOTE: The check above will assign `True` when the random value is higher than the tracking efficiency.
        #       However, since we to remove those particles and keep ones below, we need to invert this selection.
        _det_level_particles_to_keep_np = ~_drop_particles_mask

        # Unflatten so we can apply the mask to the existing particles
        det_level_mask = ak.unflatten(_det_level_particles_to_keep_np, ak.num(arrays["det_level"]))

        # Cross check that it worked.
        # If the entire det level mask is True, then it means that no particles were removed.
        # NOTE: I don't have this as an assert because if there aren't _that_ many particles and the efficiency
        #       is high, I suppose it's possible that this fails, and I don't want to kill jobs for that reason.
        if ak.all(det_level_mask == True):  # noqa: E712
            logger.warning(
                "No particles were removed in the artificial tracking efficiency."
                " This is possible, but not super likely. Please check your settings!"
            )

    return det_level_mask
