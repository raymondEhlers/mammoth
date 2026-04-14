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


def vectorized_entropy(dist: npt.NDArray[np.floating], sum_axes: int | tuple[int, ...]) -> npt.NDArray[np.floating]:
    """Calculate Shannon entropy for each jet_pt bin (vectorized).

    Args:
        dist: numpy array with jet_pt as the first dimension
        sum_axes: int or tuple of axes to sum over for normalization and entropy

    Returns:
        entropy: 1D array of entropies for each jet_pt bin (in nats)
    """
    # Normalize along the specified axes (keepdims for proper broadcasting)
    norm = np.sum(dist, axis=sum_axes, keepdims=True)

    # NOTE: Need to protect against 0s. In the case of 0s, we store 0.
    prob_dist = np.divide(dist, norm, out=np.zeros_like(dist), where=norm > 0)

    # NOTE: Need to protect against 0s to avoid log(0)
    safe_log = np.log(prob_dist, out=np.zeros_like(prob_dist), where=prob_dist > 0)

    # Calculate entropy: -sum(p * log(p))
    entropy: npt.NDArray[np.floating] = -np.sum(prob_dist * safe_log, axis=sum_axes)

    return entropy.squeeze()


@attrs.define(frozen=True)
class Result:
    lead: npt.NDArray[np.floating]
    sublead: npt.NDArray[np.floating]
    joint: npt.NDArray[np.floating]


def calculate_entropy(input_hists: dict[str, hist.Hist[hist.storage.Weight]]) -> Result:
    """Calculate entropy for the lead, sublead, and joint.

    Args:
        input_hists: Input histograms containing "data_n_constituents_jet_pt", the 4D
            histogram of (pt_1, pt_2, n_1, n_2).
    Returns:
        Calculation of entropy in a Result wrapper.
    """
    input_hist = input_hists["data_n_constituents_jet_pt"]
    # This formulation ignores the subleading pt axis entirely
    # Axes: (lead jet pt, lead n_const, sublead n_const)
    h_joint = binned_data.BinnedData.from_existing_data(input_hist[:, sum, :, :])  # type: ignore[index]
    # Axes: (lead jet pt, lead n_const)
    h_lead = binned_data.BinnedData.from_existing_data(input_hist[:, sum, :, sum])  # type: ignore[index]
    # Axes: (lead jet pt, sublead n_const)
    h_sublead = binned_data.BinnedData.from_existing_data(input_hist[:, sum, sum, :])  # type: ignore[index]

    entropy_lead = vectorized_entropy(h_lead.values, sum_axes=1)
    # NOTE: This is the same axis as for the lead because we've summed over the lead above
    entropy_sublead = vectorized_entropy(h_sublead.values, sum_axes=1)
    entropy_joint = vectorized_entropy(h_joint.values, sum_axes=(1, 2))

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

    # Returned values are the entropy values as a function of (lead jet pt, sublead jet pt)
    return Result(lead=entropy_lead, sublead=entropy_sublead, joint=entropy_joint)


def calculate_mutual_information(input_hists: dict[str, hist.Hist[hist.storage.Weight]]) -> npt.NDArray[np.floating]:
    entropy = calculate_entropy(input_hists=input_hists)

    mutual_information: npt.NDArray[np.floating] = entropy.lead + entropy.sublead - entropy.joint
    return mutual_information
