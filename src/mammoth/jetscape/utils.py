"""Jetscape related utilities

..codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""
from __future__ import annotations

import awkward as ak
import numba as nb


@nb.njit  # type: ignore[misc]
def subtract_holes_from_jet_pt(jets: ak.Array, particles_holes: ak.Array, jet_R: float, builder: ak.ArrayBuilder) -> ak.ArrayBuilder:
    """Subtract holes from the jet pt

    Args:
        jets: Jets found (grouped by events).
        particles_holes: Holes included (grouped by events).
        jet_R: Jet R, for which holes should be included.
        builder: Awkward array builder to be used to build the output info.

    Returns:
        ArrayBuilder with the output (note that it needs to have the snapshot applied).
    """
    for jets_in_event, holes_in_event in zip(jets, particles_holes):
        builder.begin_list()
        for jet in jets_in_event:
            jet_pt = jet.pt
            for hole in holes_in_event:
                if jet.deltaR(hole) < jet_R:
                    jet_pt -= hole.pt
            builder.append(jet_pt)
        builder.end_list()

    return builder

