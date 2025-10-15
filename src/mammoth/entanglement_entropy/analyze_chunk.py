"""Run ALICE analysis for entropy correlators for pp, PbPb, MC, and embedding

Here, we have a working definition of the analysis functions:
- `analyze_chunk_one_input_level`: Run the analysis for a single input level. This includes data, one MC column, or one embedding column
- `analyze_chunk_two_input_level`: Run the analysis for two input levels. This includes two MC columns. In principle,
    it could also include embedding, but it's not tested as such.
- `analyze_chunk_three_input_levels`: Run the analysis for three input levels (ie. embedding).

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

try:
    # Python 3.11+ (need the mypy ignore since it's targeted at 3.10)
    from enum import StrEnum  # type: ignore[attr-defined]
except ImportError:
    # Backport for Python 3.10
    from strenum import StrEnum  # pyright: ignore[reportMissingImports]
import logging
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import attrs
import awkward as ak
import hist
import numpy as np
import vector

from mammoth import helpers
from mammoth.alice import helpers as alice_helpers
from mammoth.framework import jet_finding, load_data
from mammoth.framework import task as framework_task
from mammoth.framework.analysis import generator_settings
from mammoth.framework.analysis import tracking as analysis_tracking
from mammoth.framework.io import output_utils, track_skim

logger = logging.getLogger(__name__)
vector.register_awkward()


@attrs.define
class TriggerParameters:
    classes: dict[str, Any] = attrs.field(factory=dict)
    parameters: dict[str, Any] = attrs.field(factory=dict)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TriggerParameters:
        classes = dict(config["classes"])
        parameters = dict(config["parameters"])

        return cls(
            classes=classes,
            parameters=parameters,
        )

    @property
    def label(self) -> str:
        return "dijet_trigger_pt"


def preprocess_arguments(**analysis_arguments: Any) -> dict[str, Any]:
    trigger_parameters = TriggerParameters.from_config(analysis_arguments["trigger_parameters"])
    return {
        "trigger_parameters": trigger_parameters,
    }


def output_identifier(**analysis_arguments: Any) -> str:
    identifier = ""
    trigger_parameters: TriggerParameters = analysis_arguments.pop("trigger_parameters")
    identifier += f"__{trigger_parameters.label}"
    for trigger_name, trigger_range_tuple in trigger_parameters.classes.items():
        identifier += f"_{trigger_name}_{trigger_range_tuple[0]:g}_{trigger_range_tuple[1]:g}"
    return identifier


def customize_analysis_metadata(
    task_settings: framework_task.Settings,  # noqa: ARG001
    **analysis_arguments: Any,  # noqa: ARG001
) -> framework_task.Metadata:
    """Customize the analysis metadata for the analysis.

    Nothing special required here as of June 2023.
    """
    return {}


def _setup_base_hists(levels: list[str], trigger_parameters: TriggerParameters) -> dict[str, hist.Hist]:  # noqa: ARG001
    """Setup the histograms for the embedding analysis."""
    hists = {}

    # Spectra
    for level in levels:
        # Inclusive spectra
        hists[f"{level}_inclusive_jet_spectra"] = hist.Hist(
            hist.axis.Regular(200, 0, 100, label="inclusive_jet_pt"), storage=hist.storage.Weight()
        )
        # Leading, subleading spectra
        hists[f"{level}_leading_jet_spectra"] = hist.Hist(
            hist.axis.Regular(151, -0.5, 150.5, label="leading_jet_pt"), storage=hist.storage.Weight()
        )
        hists[f"{level}_subleading_jet_spectra"] = hist.Hist(
            hist.axis.Regular(151, -0.5, 150.5, label="subleading_jet_pt"), storage=hist.storage.Weight()
        )

    return hists


class DijetRejectionReason(StrEnum):  # type: ignore[misc]
    n_initial = "n_initial"
    not_enough_jets = "not_enough_jets"
    pt_selection = "jet_pt_selection"
    delta_phi_selection = "delta_phi_selection"
    n_accepted = "n_accepted"


def create_dijet_selection_QA_hists(particle_columns: list[str]) -> dict[str, hist.Hist]:
    """Dijet selection QA hists."""
    hists = {}
    for level in particle_columns:
        # Acceptance reason
        hists[f"{level}_dijet_n_accepted"] = hist.Hist(
            hist.axis.StrCategory([str(v) for v in list(DijetRejectionReason)], growth=True, name="Dijet N Accepted"),
            label="Dijet acceptance",
            storage=hist.storage.Weight(),
        )

    return hists


def fill_dijet_QA_reason(
    reason: str,
    hists: dict[str, hist.Hist],
    masks: ak.Array,
    column_name: str,
) -> None:
    """Fill dijet QA reason hists.

    Args:
        reason: Acceptance / rejection reason.
        hists: Dict containing QA hists.
        masks: Selection masks.
        column_name: Name of column / level which we are recording for.

    Returns:
        None. The histogram is modified in place.
    """
    # Since we use the general mask, this will store the cumulative number of jets
    # (i.e. it depends on the selection order). However, this is probably good enough
    # for these purposes.
    hists[f"{column_name}_dijet_n_accepted"].fill(
        reason, weight=np.count_nonzero(np.asarray(ak.flatten(masks[column_name], axis=None)))
    )


def dijet_selection(
    jets: ak.Array,
    jet_R: float,  # noqa: ARG001
    collision_system: str,  # noqa: ARG001
    trigger_parameters: TriggerParameters,
    particle_columns: list[str],
    max_delta_phi_from_axis: float,
    require_exclusive_leading_dijets: bool = False,
) -> tuple[ak.Array, dict[str, hist.Hist]]:
    """Di-jet selections for EBC analysis

    Requirements:
    - There are at least two jets within our pt selections.
    - The leading two jets are within a delta_phi window.

    Args:
        jets: Jets array
        jet_R: Jet resolution parameter
        collision_system: Collision system
        trigger_parameter: Trigger parameters.
        level_names: Names of the level (i.e. particle) columns to apply these selections.
        max_delta_phi_from_axis: Max delta_phi from the dijet axis.
        require_exclusive_leading_dijets: If True, only the two leading dijets will be
            returned. If False, all inclusive jets that pass out inclusive conditions
            which are in an event passing our dijet conditions are returned. Default: False.
    Returns:
        Jets array with the jet selection applied, QA hists
    """
    # Setup
    hists = create_dijet_selection_QA_hists(particle_columns=particle_columns)
    leading_jet_pt_range, subleading_jet_pt_range = (
        trigger_parameters.classes["leading"],
        trigger_parameters.classes["subleading"],
    )

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
        fill_dijet_QA_reason(DijetRejectionReason.n_initial, hists, masks, column_name)

        # **************
        # Must have at least two jets in the first place
        # **************
        masks[column_name] = (masks[column_name]) & (ak.num(jets[column_name], axis=1) >= 2)
        fill_dijet_QA_reason(DijetRejectionReason.not_enough_jets, hists, masks, column_name)

        # **************
        # Apply pt selections based on the two leading jets
        # **************
        # First, select the two leading jets in an event
        # NOTE: Ascending is important because we're clipping everything but the two leading for many checks.
        sorted_by_pt_indices = ak.argsort(jets[column_name].pt, axis=1, ascending=False)
        leading_two_jets_indices = sorted_by_pt_indices[:, :2]
        # And retrieve their pt. To allow the projection into the leading and subleading, we'll fill a sentinel value (-1000)
        # when we don't have sufficient jets available. Given that value, it will evaluate to false
        leading_two_jets_pt = ak.fill_none(
            ak.pad_none(jets[column_name][leading_two_jets_indices].pt, 2, axis=-1), -1000
        )
        masks[column_name] = (masks[column_name]) & (
            (leading_two_jets_pt[:, 0] > leading_jet_pt_range[0])
            & (leading_two_jets_pt[:, 1] > subleading_jet_pt_range[0])
        )
        logger.info(
            f"{column_name}: dijet pt selection: {np.count_nonzero(np.asarray(ak.flatten(masks[column_name] == True, axis=None)))}"  # noqa: E712
        )
        fill_dijet_QA_reason(DijetRejectionReason.pt_selection, hists, masks, column_name)

        # **************
        # Require a jet with |delta_phi| <= c from the recoil axis
        # **************
        # First, we need the recoil axis, which is the recoil from the leading jet.
        recoil_axis = -1 * jets[column_name][sorted_by_pt_indices][:, :1]
        # This is much easier to calculate if it's in a regular array, so we'll pad None.
        # NOTE: By using pad_none, delta_phi will work despite not having entries for each event.
        recoil_axis = ak.to_regular(ak.pad_none(recoil_axis, 1, axis=1))
        # Also need to keep track of the index of the leading jet for constructing the delta_phi mask later.
        leading_jet_index = ak.to_regular(ak.pad_none(sorted_by_pt_indices[:, :1], 1, axis=1))
        # NOTE: We only want to select on phi!
        #       But sort by the leading pt to ease making the comparison below (at least as of Oct 2025)
        delta_phi_from_recoil_axis = recoil_axis.deltaphi(jets[column_name])
        # Keep jets that are within max_delta_phi of the recoil axis.
        # NOTE: We calculate the recoil for every jet, including the leading jet.
        #       Obviously we don't want to mask out the leading jet, so specifically assign True
        #       when we get to that index.
        delta_phi_mask = ak.where(
            ak.local_index(delta_phi_from_recoil_axis, axis=1) == leading_jet_index,
            True,
            np.abs(delta_phi_from_recoil_axis) <= max_delta_phi_from_axis,
        )
        masks[column_name] = masks[column_name] & delta_phi_mask
        logger.info(
            f"{column_name}: dijet eta selection: {np.count_nonzero(np.asarray(ak.flatten(masks[column_name] == True, axis=None)))}"  # noqa: E712
        )
        fill_dijet_QA_reason(DijetRejectionReason.delta_phi_selection, hists, masks, column_name)

        # All done - make sure we get a final count (this may be trivial in some cases,
        # but I'd rather be clear).
        fill_dijet_QA_reason(DijetRejectionReason.n_accepted, hists, masks, column_name)

    # Actually apply the masks
    for column_name, mask in masks.items():
        jets[column_name] = jets[column_name][mask]

    # Now that we've applied the mask, we can take the exclusive leading dijets if we want.
    # NOTE: It's better to apply it here rather than above because it allows us to take a sub-sub-leading
    #       jet which still passes the pt cut and the eta cut. If we take just the leading and subleading,
    #       and the subleading doesn't pass the eta cut, then we lose the event.
    # NOTE: As of Oct 2025, by default we don't want exclusive dijets. Instead, we apply the dijet
    #       conditions to the event, and if it passes, we'll take **ALL** jets in that event which
    #       pass our inclusive jet cuts. Since this is called after our inclusive jet cuts, we don't
    #       need to apply any stricter selection on jets. However, we also add an option for exclusive
    #       leading dijets if so inclined.
    if require_exclusive_leading_dijets:
        for column_name in particle_columns:
            # Keep the leading two jets that survived the selections!
            sorted_by_pt_indices = ak.argsort(jets[column_name].pt, axis=1, ascending=False)
            jets[column_name] = jets[column_name][sorted_by_pt_indices][:, :2]
            # NOTE: We don't yet require that there are only two jets because this would remove events,
            #       which will mess up the existing `jets` awkward array. We'll need to flatten later
            #       to handle that correctly.
            # jets[column_name] = jets[column_name][ak.num(jets[column_name], axis=1) == 2]

    return jets, hists


def _setup_one_input_level_hists(level_names: list[str], trigger_parameters: TriggerParameters) -> dict[str, hist.Hist]:
    return _setup_base_hists(
        levels=level_names,
        trigger_parameters=trigger_parameters,
    )


def analyze_chunk_one_input_level(
    *,
    collision_system: str,
    arrays: ak.Array,
    input_metadata: dict[str, Any],
    # Analysis arguments
    trigger_parameters: TriggerParameters,
    min_track_pt: dict[str, float],  # noqa: ARG001
    max_delta_phi_from_axis: float,
    level_name: str = "data",
    background_subtraction: Mapping[str, Any] | None = None,
    generator_analysis_arguments: framework_task.GeneratorAnalysisArguments | None = None,
    # Injected analysis arguments (when appropriate)
    pt_hat_bin: int = -1,  # noqa: ARG001
    scale_factors: dict[int, float] | None = None,
    # Default analysis arguments
    validation_mode: bool = False,
    return_skim: bool = False,  # noqa: ARG001
    # NOTE: kwargs are required because we pass the config as the analysis arguments,
    #       and it contains additional values.
    **kwargs: Any,  # noqa: ARG001
) -> framework_task.AnalysisOutput:
    """Run the analysis for one input level.

    This includes data, or analyzing a single column of pp_MC, PbPb_MC, or embedding.
    """
    # Validation
    if background_subtraction is None:
        background_subtraction = {}
    # If we don't need a scale factor, it's more convenient to just set it to 1.0
    if scale_factors is None:
        scale_factors = {-1: 1.0}

    # Setup
    logger.info("Start analyzing")
    hists = _setup_one_input_level_hists(level_names=[level_name], trigger_parameters=trigger_parameters)
    gen_settings = generator_settings.create_generator_settings_if_valid(
        input_source_config=input_metadata["signal_source_config"],
        generator_analysis_arguments=generator_analysis_arguments,
    )
    track_selection_additional_kwargs = {}
    if gen_settings:
        # Generator is valid - time to customize
        arrays, _res = generator_settings.configure_generator_options_before_starting_analysis(
            generator=gen_settings,
            collision_system=collision_system,
            arrays=arrays,
            levels=[level_name],
        )
        track_selection_additional_kwargs.update(_res)

    # Event selection
    # This would apply to the signal events, because this is what we propagate from the embedding transform
    arrays = alice_helpers.standard_event_selection(arrays=arrays)

    # Track cuts
    arrays = alice_helpers.standard_track_selection(
        arrays=arrays,
        require_at_least_one_particle_in_each_collection_per_event=False,
        selected_particle_column_name=level_name,
        **track_selection_additional_kwargs,
    )

    # Jet finding
    logger.info("Find jets")
    # First, setup what is needed for validation mode if enabled
    area_kwargs = {}
    if validation_mode:
        area_kwargs["random_seed"] = jet_finding.VALIDATION_MODE_RANDOM_SEED

    area_settings = jet_finding.AreaPP(**area_kwargs)  # type: ignore[arg-type]
    additional_kwargs: dict[str, Any] = {}
    if collision_system in ["PbPb", "embedPythia", "embed_pythia", "embed_thermal_model"]:
        area_settings = jet_finding.AreaAA(**area_kwargs)  # type: ignore[arg-type]
        additional_kwargs["background_subtraction"] = jet_finding.BackgroundSubtraction(
            type=jet_finding.BackgroundSubtractionType.event_wise_constituent_subtraction,
            estimator=jet_finding.JetMedianBackgroundEstimator(
                jet_finding_settings=jet_finding.JetMedianJetFindingSettings(
                    area_settings=jet_finding.AreaAA(**area_kwargs)  # type: ignore[arg-type]
                )
            ),
            subtractor=jet_finding.ConstituentSubtractor(
                r_max=background_subtraction["r_max"],
            ),
        )

    # Plug-in point before jet finding for configuring the generator options
    # We need to define some basic kwargs here so we can override them later
    levels = [level_name]
    jet_finding_settings_additional_kwargs: dict[str, dict[str, Any]] = {level: {} for level in levels}
    # NOTE: This has to be defined here with the all of relevant arguments since we may want
    #       e.g. override the "particles" argument.
    jet_finding_kwargs = {
        level: {
            "particles": arrays[level],
            **additional_kwargs,
        }
        for level in levels
    }
    if gen_settings:
        arrays, jet_finding_kwargs_return, jet_finding_settings_kwargs_return = (
            generator_settings.configure_generator_options_before_jet_finding(
                generator=gen_settings,
                collision_system=collision_system,
                arrays=arrays,
                levels=levels,
            )
        )
        # Update the relevant jet finding settings
        for level in levels:
            jet_finding_kwargs[level].update(jet_finding_kwargs_return[level])
            jet_finding_settings_additional_kwargs[level].update(jet_finding_settings_kwargs_return[level])

    # Additional parameters
    jet_R = trigger_parameters.parameters["jet_R"]
    # Take the lowest value of any of the trigger classes, but at least 1 GeV
    min_jet_pt = min([r[0] for r in trigger_parameters.classes.values()] + [1.0])
    # And run the jet finding
    logger.warning(f"For particle column '{level_name}', additional_kwargs: {additional_kwargs}")
    jets = ak.zip(
        {
            level_name: jet_finding.find_jets(
                **jet_finding_kwargs[level_name],
                jet_finding_settings=jet_finding.JetFindingSettings(
                    R=jet_R,
                    algorithm="anti-kt",
                    pt_range=jet_finding.pt_range(pt_min=min_jet_pt),
                    eta_range=jet_finding.eta_range(jet_R=jet_R, fiducial_acceptance=True),
                    area_settings=area_settings,
                    **jet_finding_settings_additional_kwargs[level_name],
                ),
            ),
        },
        depth_limit=1,
    )
    logger.warning(f"Found n jets: {np.count_nonzero(np.asarray(ak.flatten(jets[level_name].px, axis=None)))}")
    # Now, apply the jet selections
    jets, _qa_hists = alice_helpers.standard_jet_selection(
        jets=jets,
        jet_R=jet_R,
        collision_system=collision_system,
        substructure_constituent_requirements=False,
        selected_particle_column_name=level_name,
    )
    hists.update(_qa_hists)

    # And now the dijet selections
    logger.info("Applying dijet selections")
    jets, _qa_hists = dijet_selection(
        jets=jets,
        jet_R=jet_R,
        collision_system=collision_system,
        trigger_parameters=trigger_parameters,
        particle_columns=[level_name],
        max_delta_phi_from_axis=max_delta_phi_from_axis,
        require_exclusive_leading_dijets=True,
    )

    return framework_task.AnalysisOutput(
        hists=hists,
        skim=jets,
    )


def _setup_two_input_level_hists(
    level_names: list[str],
    trigger_parameters: TriggerParameters,
) -> dict[str, hist.Hist]:
    return _setup_base_hists(
        levels=level_names,
        trigger_parameters=trigger_parameters,
    )


def analyze_chunk_two_input_level(
    *,
    collision_system: str,  # noqa: ARG001
    arrays: ak.Array,  # noqa: ARG001
    input_metadata: dict[str, Any],  # noqa: ARG001
    # Analysis arguments
    trigger_parameters: TriggerParameters,  # noqa: ARG001
    correlator_type: str,  # noqa: ARG001
    min_track_pt: dict[str, float],  # noqa: ARG001
    max_delta_phi_from_axis: float,  # noqa: ARG001
    momentum_weight_exponent: int | float,  # noqa: ARG001
    combinatorics_chunk_size: int,  # noqa: ARG001
    det_level_artificial_tracking_efficiency: float | analysis_tracking.PtDependentTrackingEfficiencyParameters,  # noqa: ARG001
    # Injected analysis arguments
    pt_hat_bin: int,  # noqa: ARG001
    scale_factors: dict[int, float],  # noqa: ARG001
    level_names: list[str] | None = None,
    # Default analysis arguments
    validation_mode: bool = False,  # noqa: ARG001
    return_skim: bool = False,  # noqa: ARG001
    # NOTE: kwargs are required because we pass the config as the analysis arguments,
    #       and it contains additional values.
    **kwargs: Any,  # noqa: ARG001
) -> framework_task.AnalysisOutput:
    """Run the analysis for the two input levels."""
    # Validation
    if level_names is None:
        level_names = ["part_level", "det_level"]

    #     # Setup
    #     hists = _setup_two_input_level_hists(level_names=level_names, trigger_parameters=trigger_parameters)
    #     trigger_skim_output: dict[str, ak.Array] = {}
    #
    #     # Event selection
    #     # This would apply to the signal events, because this is what we propagate from the embedding transform
    #     arrays = alice_helpers.standard_event_selection(arrays=arrays)
    #
    #     # Track cuts
    #     arrays = alice_helpers.standard_track_selection(
    #         arrays=arrays, require_at_least_one_particle_in_each_collection_per_event=True
    #     )
    #
    #     # Calculate the relevant masks for det level particles to potentially apply an
    #     # artificial tracking inefficiency
    #     det_level_mask = analysis_tracking.det_level_particles_mask_for_jet_finding(
    #         arrays=arrays,
    #         det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
    #         validation_mode=validation_mode,
    #     )
    #     # Require that events have at least one particle after any possible masking.
    #     # If not, the entire array will be thrown out during jet finding, so better to
    #     # remove them now and be able to analyze the rest of the array. We're not missing
    #     # anything meaningful by doing this because we can't analyze such a case anyway
    #     # (and it might only be useful for an efficiency of losing events due to tracking,
    #     # which should be exceptionally rare).
    #     _events_with_at_least_one_particle = (ak.num(arrays["part_level"]) > 0) & (
    #         ak.num(arrays["det_level"][det_level_mask]) > 0
    #     )
    #     arrays = arrays[_events_with_at_least_one_particle]
    #     # NOTE: We need to apply it to the det level mask as well because we
    #     #       may be dropping some events, which then needs to be reflected in
    #     det_level_mask = det_level_mask[_events_with_at_least_one_particle]
    #     arrays["det_level"] = arrays["det_level"][det_level_mask]
    #
    #     # Find trigger(s)
    #     logger.debug("Finding trigger(s)")
    #     triggers_dict: dict[str, dict[str, ak.Array]] = {}
    #     event_selection_mask: dict[str, dict[str, ak.Array]] = {}
    #     for level in level_names:
    #         triggers_dict[level] = {}
    #         event_selection_mask[level] = {}
    #         triggers_dict[level], event_selection_mask[level] = _find_triggers(
    #             level=level,
    #             arrays=arrays,
    #             trigger_parameters=trigger_parameters,
    #             scale_factor=scale_factors[pt_hat_bin],
    #             validation_mode=validation_mode,
    #             hists=hists,
    #         )
    #
    #     # Calculate combinatorics
    #     for level in level_names:
    #         for trigger_name, _ in trigger_parameters.classes.items():
    #             res = calculate_correlators(
    #                 level=level,
    #                 trigger_name=trigger_name,
    #                 triggers=triggers_dict[level][trigger_name],
    #                 arrays=arrays[level],
    #                 event_selection_mask=event_selection_mask[level][trigger_name],
    #                 correlator_type=correlator_type,
    #                 min_track_pt=min_track_pt[level],
    #                 max_delta_phi_from_axis=max_delta_phi_from_axis,
    #                 momentum_weight_exponent=momentum_weight_exponent,
    #                 combinatorics_chunk_size=combinatorics_chunk_size,
    #                 scale_factor=scale_factors[pt_hat_bin],
    #                 hists=hists,
    #                 return_skim=return_skim,
    #             )
    #             if res:
    #                 trigger_skim_output.update(res)
    #
    return framework_task.AnalysisOutput(
        # hists=hists,
        # skim=trigger_skim_output,
        hists={},
        skim={},
    )


def _setup_three_input_level_hists(
    level_names: list[str], trigger_parameters: TriggerParameters
) -> dict[str, hist.Hist]:
    return _setup_base_hists(levels=level_names, trigger_parameters=trigger_parameters)


def analyze_chunk_three_input_level(
    *,
    collision_system: str,  # noqa: ARG001
    source_index_identifiers: dict[str, int],  # noqa: ARG001
    arrays: ak.Array,  # noqa: ARG001
    input_metadata: dict[str, Any],  # noqa: ARG001
    # Analysis arguments
    trigger_parameters: TriggerParameters,
    correlator_type: str,  # noqa: ARG001
    min_track_pt: dict[str, float],  # noqa: ARG001
    max_delta_phi_from_axis: float,  # noqa: ARG001
    momentum_weight_exponent: int | float,  # noqa: ARG001
    combinatorics_chunk_size: int,  # noqa: ARG001
    scale_factor: float,  # noqa: ARG001
    det_level_artificial_tracking_efficiency: float | analysis_tracking.PtDependentTrackingEfficiencyParameters,  # noqa: ARG001
    # Default analysis arguments
    validation_mode: bool = False,  # noqa: ARG001
    return_skim: bool = False,  # noqa: ARG001
    # NOTE: kwargs are required because we pass the config as the analysis arguments,
    #       and it contains additional values.
    **kwargs: Any,  # noqa: ARG001
) -> framework_task.AnalysisOutput:
    """Run the analysis for the three input levels.

    ie. This means analysis for embedding
    """
    # Setup
    level_names = ["part_level", "det_level", "hybrid_level"]
    hists = _setup_three_input_level_hists(level_names=level_names, trigger_parameters=trigger_parameters)

    #     # Event selection
    #     # This would apply to the signal events, because this is what we propagate from the embedding transform
    #     arrays = alice_helpers.standard_event_selection(arrays=arrays)
    #
    #     # Track cuts
    #     arrays = alice_helpers.standard_track_selection(
    #         arrays=arrays, require_at_least_one_particle_in_each_collection_per_event=True
    #     )
    #
    #     # Apply artificial tracking efficiency
    #     # 1. We may need to mask the hybrid level particles to apply an artificial tracking inefficiency
    #     # 2. We usually calculate rho only using the PbPb particles (ie. not including the embedded det_level),
    #     #    so we need to select only them.
    #     # NOTE: Technically, we aren't doing jet finding here, but the convention is still the same.
    #     #       I suppose this would call for a refactor, but I don't want to deal with that right now.
    #     hybrid_level_mask, _ = analysis_tracking.hybrid_level_particles_mask_for_jet_finding(
    #         arrays=arrays,
    #         det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
    #         source_index_identifiers=source_index_identifiers,
    #         validation_mode=validation_mode,
    #     )
    #     arrays["hybrid_level"] = arrays["hybrid_level"][hybrid_level_mask]
    #
    #     # Find trigger(s)
    #     logger.debug("Finding trigger(s)")
    #     triggers_dict: dict[str, dict[str, ak.Array]] = {}
    #     event_selection_mask: dict[str, dict[str, ak.Array]] = {}
    #     for level in level_names:
    #         triggers_dict[level], event_selection_mask[level] = _find_triggers(
    #             level=level,
    #             arrays=arrays,
    #             trigger_parameters=trigger_parameters,
    #             scale_factor=scale_factor,
    #             validation_mode=validation_mode,
    #             hists=hists,
    #         )
    #
    #     # Calculate combinatorics
    #     for level in level_names:
    #         for trigger_name, _ in trigger_parameters.classes.items():
    #             res = calculate_correlators(
    #                 level=level,
    #                 trigger_name=trigger_name,
    #                 triggers=triggers_dict[level][trigger_name],
    #                 arrays=arrays[level],
    #                 event_selection_mask=event_selection_mask[level][trigger_name],
    #                 correlator_type=correlator_type,
    #                 min_track_pt=min_track_pt[level],
    #                 max_delta_phi_from_axis=max_delta_phi_from_axis,
    #                 momentum_weight_exponent=momentum_weight_exponent,
    #                 combinatorics_chunk_size=combinatorics_chunk_size,
    #                 background_index_identifier=source_index_identifiers["background"],
    #                 scale_factor=scale_factor,
    #                 hists=hists,
    #                 return_skim=return_skim,
    #             )
    #             if res:
    #                 trigger_skim_output.update(res)
    #
    #     # IPython.embed()  # type: ignore[no-untyped-call]
    #
    return framework_task.AnalysisOutput(
        hists=hists,
        skim={},
    )


def run_some_standalone_tests() -> None:
    # Some tests:
    #######
    # Data:
    ######
    # pp: needs min_jet_pt = 5 to have any jets
    # collision_system = "pp"
    # pythia: Can test both "part_level" and "det_level" in the rename map.
    # collision_system = "pythia"
    # PbPb:
    # collision_system = "PbPb"
    # for collision_system in ["pp", "pythia", "PbPb"]:
    # for collision_system in ["pp"]:
    #    logger.info(f'Analyzing "{collision_system}"')
    #    jets = analysis_data(
    #        collision_system=collision_system,
    #        arrays=load_data.data(
    #            data_input=Path(
    #                f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults_track_skim.parquet"
    #            ),
    #            data_source=partial(track_skim.FileSource, collision_system=collision_system),
    #            collision_system=collision_system,
    #            rename_levels={"data": "data"} if collision_system != "pythia" else {"data": "det_level"},
    #        ),
    #        jet_R=0.4,
    #        min_jet_pt={"data": 5.0 if collision_system == "pp" else 20.0},
    #    )
    #
    #    # import IPython; IPython.embed()
    ######
    # MC
    ######
    # jets = analysis_MC(
    #     arrays=load_MC(filename=Path("/software/rehlers/dev/mammoth/projects/framework/pythia/AnalysisResults.parquet")),
    #     jet_R=0.4,
    #     min_jet_pt={
    #         "det_level": 20,
    #     },
    # )
    ###########
    # Embedding
    ###########
    # jets = analysis_embedding(
    #     *load_embedding(
    #         signal_filename=Path("/software/rehlers/dev/mammoth/projects/framework/pythia/AnalysisResults.parquet"),
    #         background_filename=Path("/software/rehlers/dev/mammoth/projects/framework/PbPb/AnalysisResults.parquet"),
    #     ),
    #     jet_R=0.4,
    #     min_jet_pt={
    #         "hybrid_level": 20,
    #         "det_level": 1,
    #         "part_level": 1,
    #     },
    # )
    ###############
    # Thermal model
    ###############
    # jets = analysis_embedding(
    #     *load_embed_thermal_model(
    #         signal_filename=Path("/software/rehlers/dev/substructure/trains/pythia/641/run_by_run/LHC20g4/295612/1/AnalysisResults.20g4.001.root"),
    #         thermal_model_parameters=sources.THERMAL_MODEL_SETTINGS["central"],
    #     ),
    #     jet_R=0.2,
    #     min_jet_pt={
    #         "hybrid_level": 20,
    #         #"det_level": 1,
    #         #"part_level": 1,
    #     },
    #     r_max=0.25,
    # )
    ...


def minimal_test() -> None:
    helpers.setup_logging(level=logging.INFO)

    # run_some_standalone_tests()

    ###########################
    # Explicitly for testing...
    ###########################
    # collision_system = "PbPb"
    # logger.info(f'Analyzing "{collision_system}"')
    # jets = analysis_one_input_level(
    #    collision_system=collision_system,
    #    arrays=load_data.data(
    #        data_input=Path(
    #            "trains/PbPb/645/run_by_run/LHC18q/296270/AnalysisResults.18q.580.root"
    #        ),
    #        data_source=partial(track_skim.FileSource, collision_system=collision_system),
    #        collision_system=collision_system,
    #        rename_levels={"data": "data"} if collision_system != "pythia" else {"data": "det_level"},
    #    ),
    #    jet_R=0.2,
    #    min_jet_pt={"data": 20.0 if collision_system == "pp" else 20.0},
    # )

    # source_index_identifiers, iter_arrays = load_data.embedding(
    #    #signal_input=[Path("trains/pythia/2640/run_by_run/LHC20g4/296191/1/AnalysisResults.20g4.008.root")],
    #    # NOTE: This isn't anchored, but it's convenient for testing...
    #    signal_input=[Path("trains/pythia/2619/run_by_run/LHC18b8_fast/282125/14/AnalysisResults.18b8_fast.008.root")],
    #    signal_source=partial(track_skim.FileSource, collision_system="pythia"),
    #    background_input=[
    #        Path("trains/PbPb/645/run_by_run/LHC18r/296799/AnalysisResults.18r.179.root"),
    #        Path("trains/PbPb/645/run_by_run/LHC18r/296894/AnalysisResults.18r.337.root"),
    #    ],
    #    background_source=partial(track_skim.FileSource, collision_system="PbPb"),
    #    background_is_constrained_source=False,
    #    chunk_size=2500,
    # )

    # from mammoth.framework import sources
    # source_index_identifiers, iter_arrays = load_data.embedding_thermal_model(
    #    # NOTE: This isn't anchored, but it's convenient for testing...
    #    # signal_input=[Path("trains/pythia/2619/run_by_run/LHC18b8_fast/282125/14/AnalysisResults.18b8_fast.008.root")],
    #    signal_input=[Path("trains/pythia/2640/run_by_run/LHC20g4/296415/4/AnalysisResults.20g4.011.root")],
    #    signal_source=partial(track_skim.FileSource, collision_system="pp_MC"),
    #    thermal_model_parameters=sources.THERMAL_MODEL_SETTINGS["5020_central"],
    #    # chunk_size=2500,
    #    chunk_size=1000,
    # )

    ## NOTE: Just for quick testing
    # merged_hists: dict[str, hist.Hist] = {}
    ## END NOTE
    # for i_chunk, arrays in enumerate(iter_arrays):
    #    logger.info(f"Processing chunk: {i_chunk}")
    #    analysis_output = analyze_chunk_three_input_level(
    #        collision_system="embed_thermal_model",
    #        source_index_identifiers=source_index_identifiers,
    #        arrays=arrays,
    #        input_metadata={},
    #        trigger_parameters=TriggerParameters(
    #            type="hadron",
    #            kinematic_label="pt",
    #            classes={
    #                "reference": (5, 7),
    #                "signal": (20, 50),
    #            },
    #            parameters={},
    #        ),
    #        correlator_type="two_particle",
    #        min_track_pt={
    #            "part_level": 1.0,
    #            "det_level": 1.0,
    #            "hybrid_level": 1.0,
    #        },
    #        momentum_weight_exponent=1,
    #        combinatorics_chunk_size=500,
    #        scale_factor=1,
    #        det_level_artificial_tracking_efficiency=0.99,
    #        use_jet_trigger=False,
    #        return_skim=False,
    #    )
    #    merged_hists = output_utils.merge_results(merged_hists, analysis_output.hists)

    # iter_arrays = load_data.data(
    #    data_source=partial(track_skim.FileSource, collision_system="pp"),
    #    collision_system="pp",
    #    rename_levels={"data": "data"},
    #    chunk_size=1000,
    # )
    iter_arrays = load_data.data(
        # data_input=[Path("trains/pythia/2640/run_by_run/LHC20g4/296415/4/AnalysisResults.20g4.011.root")],
        data_input=[Path("trains/pythia/2640/run_by_run/LHC20g4/296244/20/AnalysisResults.20g4.011.root")],
        data_source=partial(track_skim.FileSource, collision_system="pp_MC"),
        collision_system="pp_MC",
        rename_levels={"data": "part_level"},
        chunk_size=1000,
    )
    # NOTE: Just for quick testing
    merged_hists: dict[str, hist.Hist] = {}
    for i_chunk, arrays in enumerate(iter_arrays):
        logger.info(f"Processing chunk: {i_chunk}")
        analysis_output = analyze_chunk_one_input_level(
            collision_system="pp_MC",
            arrays=arrays,
            input_metadata={"signal_source_config": {}},
            trigger_parameters=TriggerParameters(
                classes={
                    "leading": [20, 140],
                    "subleading": [10, 140],
                },
                parameters={"jet_R": 0.4},
            ),
            min_track_pt={
                "data": 1.0,
            },
            max_delta_phi_from_axis=np.pi / 4,
            scale_factor=1,
            # det_level_artificial_tracking_efficiency=0.99,
            use_jet_trigger=False,
            return_skim=False,
        )
        merged_hists = output_utils.merge_results(merged_hists, analysis_output.hists)

    import IPython

    IPython.embed()  # type: ignore[no-untyped-call]

    # Just for quick testing!
    import uproot

    with uproot.recreate("test_eec.root") as f:
        output_utils.write_hists_to_file(hists=merged_hists, f=f)


if __name__ == "__main__":
    minimal_test()
