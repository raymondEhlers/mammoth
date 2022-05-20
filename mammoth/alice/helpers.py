""" Helpers and utilities for ALICE analyses

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

import logging
import functools
import operator
from typing import Final, List, Optional, Sequence

import awkward as ak
import numpy as np

from mammoth.framework import particle_ID


logger = logging.getLogger(__name__)


_DEFAULT_CHARGED_HADRON_PIDs: Final[List[int]] = [11, 13, 211, 321, 2212, 3222, 3112, 3312, 3334]

def standard_event_selection(arrays: ak.Array) -> ak.Array:
    """ALICE standard event selection

    Includes selections on:

    - removed events that are marked as rejected
    - removing events with a reconstructed z vertex larger than 10 cm

    NOTE:
        These selections are only applied if the columns are actually present in the array.

    Args:
        arrays: The array to apply the event selection to. If there are multiple classes (eg. embedding before
            combining), then you can pass the selected array and assign the result.
    Returns:
        The array with the event selection applied.
    """
    # Event selection
    logger.debug(f"pre  event sel n events: {len(arrays)}")
    event_level_mask = np.ones(len(arrays)) > 0
    if "is_ev_rej" in ak.fields(arrays):
        event_level_mask = event_level_mask & (arrays["is_ev_rej"] == 0)
    if "z_vtx_reco" in ak.fields(arrays):
        event_level_mask = event_level_mask & (np.abs(arrays["z_vtx_reco"]) < 10)
    arrays = arrays[event_level_mask]
    logger.debug(f"post event sel n events: {len(arrays)}")

    return arrays

def standard_track_selection(arrays: ak.Array,
                             require_at_least_one_particle_in_each_collection_per_event: bool,
                             selected_particle_column_name: str = "",
                             columns_to_explicitly_select_charged_particles: Optional[Sequence[str]] = None,
                             charged_hadron_PIDs: Optional[Sequence[int]] = None,
                             ) -> ak.Array:
    """ALICE standard track selection

    Includes selections on:

    - minimum pt of 150 MeV for all particle collections
    - requiring at least one particle in each collection per event (if requested).

    This applies to "part_level", "det_level", and "hybrid" collections, unless we specify
    a particular single collection to select on.

    NOTE:
        We don't need to explicitly select charged particles here because when creating
        the ALICE track skims, we explicitly select charged particles there.
        However, this would be required for separate MC (eg. jetscape, herwig, etc) productions.

    NOTE:
        Since the HF Tree and TrackSkim tasks use track containers, the min pt is usually applied
        by default.  However, we apply it explicitly to be certain.

    """
    # Validation
    particle_columns = []
    if not selected_particle_column_name:
        # If no particle column is selected, then automatically detect the columns to use.
        if "part_level" in arrays:
            particle_columns.append("part_level")
        if "det_level" in arrays:
            particle_columns.append("det_level")
        if "hybrid" in arrays:
            particle_columns.append("hybrid")

        # Double check that we got something
        if not particle_columns:
            raise ValueError(f"No particle columns found in the array. Double check your inputs. columns: {ak.fields(arrays)}")
    else:
        particle_columns = [selected_particle_column_name]
    # Charged particle selection
    # If nothing is passed, we don't want to explicitly select charged particles.
    # For most ALICE skims, this selection has already been done, so we don't need to do anything here.
    if columns_to_explicitly_select_charged_particles is None:
        columns_to_explicitly_select_charged_particles = []
    #
    if charged_hadron_PIDs is None:
        charged_hadron_PIDs = _DEFAULT_CHARGED_HADRON_PIDs

    # Track cuts
    logger.debug("Track level cuts")
    for column_name in particle_columns:
        # Uniformly apply min pt cut of 150 MeV
        particle_mask = arrays[column_name].pt >= 0.150
        # Optionally apply selection of only charged particles if requested
        if column_name in columns_to_explicitly_select_charged_particles:
            if not "particle_ID" in ak.fields(arrays["column_name"]):
                raise ValueError(
                    f"Cannot select charged particles for {column_name} because the particle_ID column is not present."
                )
            charged_particles_mask = particle_ID.build_PID_selection_mask(
                arrays[column_name], absolute_pids=charged_hadron_PIDs
            )
            particle_mask = particle_mask & charged_particles_mask
        # Actually apply the mask
        arrays[column_name] = arrays[column_name][particle_mask]

    # Finally, if requested, require that we have at least one particle in each particle column for each event
    if require_at_least_one_particle_in_each_collection_per_event:
        logger.debug(f"pre  requiring a particle in every event n events: {len(arrays)}")
        # NOTE: We have to do it in a separate mask from above because the above is masked as the particle level,
        #       but here we need to mask at the event level. (If you try to mask at the particle, you'll
        #       end up with empty events)
        # NOTE: Remember that the lengths of particle collections need to match up, so be careful with the mask!
        masks = [
            ak.num(arrays[column_name], axis=1) > 0 for column_name in particle_columns
        ]
        # We need to do a bitwise hand of the masks
        event_has_particles_mask = functools.reduce(operator.and_, masks)

        arrays = arrays[event_has_particles_mask]
        logger.debug(f"post requiring a particle in every event n events: {len(arrays)}")

    return arrays