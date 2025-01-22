"""Helpers and utilities for ALICE analyses

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import enum
import functools
import logging
import operator
from collections.abc import Mapping, Sequence
from typing import Final

import awkward as ak
import hist
import numpy as np

from mammoth.framework import jet_finding, particle_ID

logger = logging.getLogger(__name__)


# (e-, mu-, pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-)
_DEFAULT_CHARGED_HADRON_PIDs: Final[list[int]] = [11, 13, 211, 321, 2212, 3222, 3112, 3312, 3334]


def standard_event_selection(arrays: ak.Array, return_mask: bool = False) -> ak.Array:
    """ALICE standard event selection

    Includes selections on:

    - removed events that are marked as rejected
    - removing events with a reconstructed z vertex larger than 10 cm

    NOTE:
        These selections are only applied if the columns are actually present in the array.

    Args:
        arrays: The array to apply the event selection to. If there are multiple classes (eg. embedding before
            combining), then you can pass the selected array and assign the result.
        return_mask: If True, return the mask rather than applying it to the array.
    Returns:
        The array with the event selection applied, or the mask if return_mask is True.
    """
    # Event selection
    logger.debug(f"pre  event sel n events: {len(arrays)}")
    event_level_mask = np.ones(len(arrays)) > 0
    if "is_ev_rej" in ak.fields(arrays):
        event_level_mask = event_level_mask & (arrays["is_ev_rej"] == 0)
    if "z_vtx_reco" in ak.fields(arrays):
        event_level_mask = event_level_mask & (np.abs(arrays["z_vtx_reco"]) < 10)

    # Return early with just the mask if requested.
    if return_mask:
        return event_level_mask

    arrays = arrays[event_level_mask]
    logger.debug(f"post event sel n events: {len(arrays)}")

    return arrays


def _determine_particle_column_names(arrays: ak.Array, selected_particle_column_name: str) -> list[str]:
    """Determine particle column names

    Args:
        arrays: Input array.
        selected_particle_column_name: Name of the selected particle column. If it's empty, then we automatically
            extract the column names from the array based on the expected names.
    Returns:
        List of particle column names.
    """
    particle_columns = []
    if not selected_particle_column_name:
        # If no particle column is selected, then automatically detect the columns to use.
        possible_columns = ["part_level", "det_level", "hybrid_level"]
        for possible_col in possible_columns:
            if possible_col in ak.fields(arrays):
                particle_columns.append(possible_col)

        # Double check that we got something
        if not particle_columns:
            _msg = f"No particle columns found in the array. Double check your inputs. columns: {ak.fields(arrays)}"
            raise ValueError(_msg)
    else:
        particle_columns = [selected_particle_column_name]
    return particle_columns


def standard_track_selection(
    arrays: ak.Array,
    require_at_least_one_particle_in_each_collection_per_event: bool,
    selected_particle_column_name: str = "",
    columns_to_explicitly_select_charged_particles: Sequence[str] | None = None,
    charged_hadron_PIDs: Sequence[int] | None = None,
    columns_to_skip_min_pt_requirement: Sequence[str] | None = None,
) -> ak.Array:
    """ALICE standard track selection

    Includes selections on:

    - minimum pt of 150 MeV for all particle collections
    - requiring at least one particle in each collection per event (if requested).

    This applies to "part_level", "det_level", and "hybrid_level" collections, unless we specify
    a particular single collection to select on.

    NOTE:
        We don't need to explicitly select charged particles here because when creating
        the ALICE track skims, we explicitly select charged particles there.
        However, this would be required for separate MC (eg. jetscape, herwig, etc) productions.

    NOTE:
        Since the HF Tree and TrackSkim tasks use track containers, the min pt is usually applied
        by default. However, we apply it explicitly to be certain.

    Args:
        arrays: Input particle level arrays
        require_at_least_one_particle_in_each_collection_per_event: If True, require at least one
            particle in each collection per event.
        selected_particle_column_name: If specified, only apply the track selection to this column. Default: all available columns.
        columns_to_explicitly_select_charged_particles: If specified, apply the charged particle selection
            on these columns. This requires the particle ID to be available.
        charged_hadron_PIDs: Charged hadron PIDs to select. If not specified, use the default ALICE selection.
        columns_to_skip_min_pt_requirement: List of columns that should skip the minimum pt requirement.
            This is used if we need to do some fancier selection externally (e.g. for generators). Default: None.
            (e.g. apply the min pt to all columns)
    Returns:
        The array with the track selection applied.
    """
    # Validation
    if columns_to_skip_min_pt_requirement is None:
        columns_to_skip_min_pt_requirement = []
    particle_columns = _determine_particle_column_names(
        arrays=arrays, selected_particle_column_name=selected_particle_column_name
    )
    # Charged particle selection
    # If nothing is passed, we don't want to explicitly select charged particles.
    # For most ALICE skims, this selection has already been done, so we don't need to do anything here.
    if columns_to_explicitly_select_charged_particles is None:
        columns_to_explicitly_select_charged_particles = []
    # If not specified, use the default ALICE charged hadron PID
    if charged_hadron_PIDs is None:
        charged_hadron_PIDs = _DEFAULT_CHARGED_HADRON_PIDs

    # Track cuts
    logger.debug("Track level cuts")
    for column_name in particle_columns:
        # Uniformly apply min pt cut of 150 MeV
        min_pt_value = 0.150
        if column_name in columns_to_skip_min_pt_requirement:
            logger.debug(f'Skipping the min pt requirement for "{column_name}"')
            min_pt_value = 0.0
        particle_mask = arrays[column_name].pt >= min_pt_value
        # Optionally apply selection of only charged particles if requested
        if column_name in columns_to_explicitly_select_charged_particles:
            if "particle_ID" not in ak.fields(arrays[column_name]):
                _msg = (
                    f"Cannot select charged particles for {column_name} because the particle_ID column is not present."
                )
                raise ValueError(_msg)
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
        masks = [ak.num(arrays[column_name], axis=1) > 0 for column_name in particle_columns]
        # We need to do a bitwise and of the masks
        event_has_particles_mask = functools.reduce(operator.and_, masks)

        arrays = arrays[event_has_particles_mask]
        logger.debug(f"post requiring a particle in every event n events: {len(arrays)}")

    return arrays


class JetRejectionReason(enum.StrEnum):  # type: ignore[name-defined,misc]
    n_initial = "n_initial"
    n_accepted = "n_accepted"
    constituents_max_pt = "constituents_max_pt"
    minimum_area = "minimum_area"
    substructure_n_constituents = "substructure_n_constituents"


def create_jet_selection_QA_hists(particle_columns: list[str]) -> dict[str, hist.Hist]:
    """Jet selection QA hists."""
    hists = {}
    for level in particle_columns:
        # Acceptance reason
        hists[f"{level}_jet_n_accepted"] = hist.Hist(
            hist.axis.StrCategory([str(v) for v in list(JetRejectionReason)], growth=True),
            label="Jet acceptance",
            storage=hist.storage.Weight(),
        )

    return hists


def fill_jet_qa_reason(
    reason: str,
    hists: dict[str, hist.Hist],
    masks: ak.Array,
    column_name: str,
) -> None:
    """Fill Jet QA reason hists.

    hists are modified in place.
    """
    # Jets which pass, cumulatively
    hists["f{column_name}_{reason}"].fill(
        reason, np.count_nonzero(np.asarray(ak.flatten(masks[column_name], axis=None)))
    )


def standard_jet_selection(
    jets: ak.Array,
    jet_R: float,
    collision_system: str,
    substructure_constituent_requirements: bool,
    selected_particle_column_name: str = "",
    max_constituent_pt_values: Mapping[str, float] | None = None,
) -> tuple[ak.Array, dict[str, hist.Hist]]:
    """Standard ALICE jet selection

    Includes selections on:

    - Remove detector level jets with constituents with pt > 100 GeV for det and hybrid level (1000 GeV for part level). This is configurable.
    - Require jet area greater than 60% of jet_R
    - If requested, remove jets with insufficient constituents for substructure. Regardless of the request,
      this will only be performed in some datasets (pp or det level).

    Args:
        jets: Jets array
        jet_R: Jet resolution parameter
        collision_system: Collision system
        substructure_constituent_requirements: If True, require certain jets to have sufficient constituents
            for non-trivial substructure. Implements the requirement on pp data or at det level.
        selected_particle_column_name: If specified, only apply the track selection to this column. Default: all available columns.
        max_constituent_pt_values: Max constituent pt values to apply to the jet collections. Default:
            100 GeV for det level and hybrid, 1000 GeV for part level.

    Returns:
        Jets array with the jet selection applied, QA hists
    """
    # Validation
    particle_columns = _determine_particle_column_names(
        arrays=jets, selected_particle_column_name=selected_particle_column_name
    )
    if max_constituent_pt_values is not None:
        _max_constituent_pt_values = dict(max_constituent_pt_values)
    else:
        # We only need to specify the cuts that aren't equal to 100 GeV, which is the default.
        _max_constituent_pt_values = {
            "part_level": 1000.0,
        }

    # Setup
    hists = create_jet_selection_QA_hists(particle_columns=particle_columns)

    # Start with all true mask
    masks = {
        # NOTE: Since these arrays could be jagged, we want to compare to a value which we can be
        #       confident that it will always give positive, leading to an all true mask.
        # NOTE: If there are no jets in any events, it won't lead to a True mask (which would be
        #       inconsistent because it shouldn't be selecting anything), but rather keeps the event
        #       structure with an array that would flatten to a zero length list.
        column_name: ak.ones_like(jets[column_name].px) > 0
        for column_name in particle_columns
    }

    # Apply jet level cuts.
    for column_name in masks:  # noqa: PLC0206
        # Cross check - if there are no entries at all, then this masking won't do anything,
        # and there's no point in continuing
        if len(ak.flatten(jets[column_name].pt)) == 0:
            # Fix up the case where there are no entries so that they have the right type!
            masks[column_name] = ak.values_astype(masks[column_name], bool, including_unknown=True)
            logger.info(
                f"There are no jets available for {column_name}, so skipping masking since it's not meaningful and can cause problems"
            )
            continue

        # Record initial number of jets.
        fill_jet_qa_reason(JetRejectionReason.n_initial, hists, masks, column_name)

        # **************
        # Remove detector level jets with constituents with pt > 100 GeV
        # Those tracks are almost certainly fake at detector level.
        # NOTE: For part level, we set it to 1000 GeV as a convention because it doesn't share these
        #       detector effects. It should be quite rare that it has an effect, but it's included for consistency.
        # NOTE: We apply this _after_ jet finding because applying it before jet finding would bias the z distribution.
        # **************
        masks[column_name] = (masks[column_name]) & (
            ~ak.any(jets[column_name].constituents.pt > _max_constituent_pt_values.get(column_name, 100), axis=-1)
        )
        logger.info(
            f"{column_name}: max track constituent max accepted: {np.count_nonzero(np.asarray(ak.flatten(masks[column_name] == True, axis=None)))}"  # noqa: E712
        )
        # Jets which pass, cumulatively
        fill_jet_qa_reason(JetRejectionReason.constituents_max_pt, hists, masks, column_name)
        # **************
        # Apply area cut
        # Requires at least 60% of possible area.
        # **************
        min_area = jet_finding.area_percentage(60, jet_R)
        masks[column_name] = (masks[column_name]) & (jets[column_name, "area"] > min_area)
        logger.info(
            f"{column_name}: add area cut n accepted: {np.count_nonzero(np.asarray(ak.flatten(masks[column_name] == True, axis=None)))}"  # noqa: E712
        )
        # Jets which pass, cumulatively
        fill_jet_qa_reason(JetRejectionReason.minimum_area, hists, masks, column_name)

        # *************
        # Require more than one constituent at detector level (or in data) if we're not in PbPb.
        # This basically requires there to be non-trivial substructure in these systems (pp and pythia).
        # Matches a cut in AliAnalysisTaskJetDynamicalGrooming
        # We generically associate it with substructure, so we describe the switch for it as:
        # `substructure_constituent_requirements`
        # *************
        if (  # noqa: SIM102
            substructure_constituent_requirements
            and collision_system not in ["PbPb"]
            and "embed" not in collision_system
        ):
            # We only want to apply this to det_level or data, so skip both "part_level" and "hybrid_level"
            if column_name not in ["part_level", "hybrid_level"]:
                masks[column_name] = (masks[column_name]) & (ak.num(jets[column_name, "constituents"], axis=2) > 1)
                logger.info(
                    f"{column_name}: require more than one constituent n accepted: {np.count_nonzero(np.asarray(ak.flatten(masks[column_name] == True, axis=None)))}"  # noqa: E712
                )
        # Jets which pass, cumulatively
        # NOTE: We put it here since in the case that the cut is disabled, we want to be clear that nothing was selected here!
        fill_jet_qa_reason(JetRejectionReason.substructure_n_constituents, hists, masks, column_name)

    # Actually apply the masks
    for column_name, mask in masks.items():
        jets[column_name] = jets[column_name][mask]

    return jets, hists
