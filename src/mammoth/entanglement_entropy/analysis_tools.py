"""Analysis tools for calculating entanglement entropy

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging

import attrs
import hist
import numpy as np
import numpy.typing as npt
from pachyderm import binned_data

logger = logging.getLogger(__name__)


def vectorized_entropy(
    dist: npt.NDArray[np.floating], sum_axes: int | tuple[int, ...], skip_sum: bool = False
) -> npt.NDArray[np.floating]:
    """Calculate Shannon entropy for each jet_pt bin (vectorized).

    Args:
        dist: numpy array with jet_pt as the first dimension
        sum_axes: int or tuple of axes to sum over for normalization and entropy
        skip_sum: Skip the final sum that's used to calculate the entropy, and instead
            return the full array. Useful for debugging. Default: False.

    Returns:
        entropy: 1D array of entropies for each jet_pt bin (in nats)
    """
    # Normalize along the specified axes (keepdims for proper broadcasting)
    norm = np.sum(dist, axis=sum_axes, keepdims=True)

    # NOTE: Need to protect against 0s. In the case of 0s, we store 0.
    prob_dist = np.divide(dist, norm, out=np.zeros_like(dist), where=norm > 0)

    # logger.info(f"Norm: {norm}")
    logger.debug(f"prob sum: {np.sum(prob_dist, axis=sum_axes)}")

    # NOTE: Need to protect against 0s to avoid log(0)
    safe_log = np.log(prob_dist, out=np.zeros_like(prob_dist), where=prob_dist > 0)

    # Calculate entropy: -sum(p * log(p))
    if skip_sum:
        return -1 * prob_dist * safe_log  # type: ignore[no-any-return]

    entropy: npt.NDArray[np.floating] = -np.sum(prob_dist * safe_log, axis=sum_axes)
    return entropy.squeeze()


@attrs.define(frozen=True)
class Result:
    lead: npt.NDArray[np.floating]
    sublead: npt.NDArray[np.floating]
    joint: npt.NDArray[np.floating]


def calculate_entropy(
    input_hists: dict[str, hist.Hist[hist.storage.Weight]],
    lead_jet_pt_range: tuple[float, float] | None = None,
    sublead_jet_pt_range: tuple[float, float] | None = None,
    skip_entropy_sum: bool = False,
    rebin_factor_for_joint_hist_to_apply_correction: int = 4,
) -> tuple[Result, Result]:
    """Calculate entropy for the lead, sublead, and joint.

    Args:
        input_hists: Input histograms containing "data_n_constituents_jet_pt", the 4D
            histogram of (pt_1, pt_2, n_1, n_2).
        lead_jet_pt_range: Lead jet pt range. Default: None - no selection.
        sublead_jet_pt_range: Sublead jet pt range. Default: None - no selection.
        skip_entropy_sum: Skip the final sum that's used to calculate the entropy, and instead
            return the full array. Useful for debugging. Default: False.
        rebin_Factor_for_joint_hist_to_apply_correction: Factor to rebin the joint distribution by
            to facilitate correction of the mutual information. Default: 4. (e.g. assuming n^const_bins = 28).
    Returns:
        Calculation of entropy in a Result wrapper, n_jets in a Result wrapper (for corrections)
    """
    if lead_jet_pt_range is None:
        lead_jet_slice = slice(None, None)
    else:
        lead_jet_slice = slice(lead_jet_pt_range[0] * 1.0j, lead_jet_pt_range[1] * 1.0j)
    # NOTE: For the sublead, we always want to sum...
    if sublead_jet_pt_range is None:
        sublead_jet_slice = slice(None, None)
        sublead_jet_slice_sum = slice(None, None, sum)
    else:
        sublead_jet_slice = slice(sublead_jet_pt_range[0] * 1.0j, sublead_jet_pt_range[1] * 1.0j)
        sublead_jet_slice_sum = slice(sublead_jet_pt_range[0] * 1.0j, sublead_jet_pt_range[1] * 1.0j, sum)

    input_hist = input_hists["data_n_constituents_jet_pt"]
    # This formulation ignores the subleading pt axis entirely
    # Axes: (lead jet pt, lead n_const, sublead n_const)
    # NOTE: This is where we rebin the joint distribution. This is critical, since this will impact
    #       how the correction is applied!
    h_joint = binned_data.BinnedData.from_existing_data(
        input_hist[
            lead_jet_slice,
            sublead_jet_slice_sum,
            :: hist.rebin(rebin_factor_for_joint_hist_to_apply_correction),  # type: ignore[misc]
            :: hist.rebin(rebin_factor_for_joint_hist_to_apply_correction),  # type: ignore[misc]
        ]
    )
    # Axes: (lead jet pt, lead n_const)
    h_lead = binned_data.BinnedData.from_existing_data(input_hist[lead_jet_slice, sublead_jet_slice_sum, :, sum])  # type: ignore[index]
    # Axes: (lead jet pt, sublead n_const)
    h_sublead = binned_data.BinnedData.from_existing_data(input_hist[lead_jet_slice, sublead_jet_slice_sum, sum, :])  # type: ignore[index]

    entropy_lead = vectorized_entropy(h_lead.values, sum_axes=1, skip_sum=skip_entropy_sum)
    # NOTE: This is the same axis as for the lead because we've summed over the lead above
    entropy_sublead = vectorized_entropy(h_sublead.values, sum_axes=1, skip_sum=skip_entropy_sum)
    entropy_joint = vectorized_entropy(h_joint.values, sum_axes=(1, 2), skip_sum=skip_entropy_sum)

    ##############################
    # Including the sublead jet pt
    ##############################
    # This formulation attempts to maintain the subleading jet pt axis.
    # NOTE: This requires special care w.r.t the normalization, so as of 14 April, we don't use this.
    # Axes: (lead jet pt, sublead jet pt, lead n_const, sublead n_const)
    # h_joint = binned_data.BinnedData.from_existing_data(input_hist)
    # Axes: (lead jet pt, sublead jet pt, lead n_const)
    # h_lead = binned_data.BinnedData.from_existing_data(input_hist[:, :, :, sum])
    # Axes: (lead jet pt, sublead jet pt, sublead n_const)
    # h_sublead = binned_data.BinnedData.from_existing_data(input_hist[:, sum, sum, :])

    # entropy_lead = vectorized_entropy(h_lead.values, sum_axes=2)
    # # NOTE: This is the same axis as for the lead because we've summed over the lead above
    # entropy_sublead = vectorized_entropy(h_sublead.values, sum_axes=2)
    # entropy_joint = vectorized_entropy(h_joint.values, sum_axes=(2, 3))

    # Also need the number of jets to calculate the correction factor.
    n_jets = Result(
        lead=binned_data.BinnedData.from_existing_data(
            input_hist[lead_jet_slice, sublead_jet_slice_sum, sum, sum]  # type: ignore[index]
        ).values,
        # NOTE: In practice, these are the same because we need them as a function of jet pt
        sublead=binned_data.BinnedData.from_existing_data(
            input_hist[lead_jet_slice, sublead_jet_slice_sum, sum, sum]  # type: ignore[index]
        ).values,
        # It needs to be as a function of lead jet pt to
        joint=np.sum(
            binned_data.BinnedData.from_existing_data(input_hist[lead_jet_slice, sublead_jet_slice, sum, sum]).values,  # type: ignore[index]
            axis=1,
        ),
    )

    # Returned values are the entropy values as a function of (lead jet pt, sublead jet pt)
    return Result(lead=entropy_lead, sublead=entropy_sublead, joint=entropy_joint), n_jets


def calculate_mutual_information(
    input_hists: dict[str, hist.Hist[hist.storage.Weight]], entropy: Result | None
) -> npt.NDArray[np.floating]:
    if entropy is None:
        entropy = calculate_entropy(input_hists=input_hists)[0]

    mutual_information: npt.NDArray[np.floating] = entropy.lead + entropy.sublead - entropy.joint
    return mutual_information
