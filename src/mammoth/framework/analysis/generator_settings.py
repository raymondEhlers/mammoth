"""Generator settings and configuration for analysis.

The concept here is that we define some settings related to how the generator was run
and how we should analyze the output. Once this configuration is defined, we have some
plugin points where we can configure how analysis is handled. As of May 2024, the points
include:

- Immediately before starting the analysis (i.e. right before event and track selection)
- Immediately after the standard track selection (i.e. for e.g. charged track selection)
- Immediately before starting jet finding (i.e. for configuring jet finding settings, background
  subtraction, etc).

As of May 2024, the convention is just to define the customizations here. If it gets out
of hand, then we can refactor elsewhere, but I don't see a point in adding further complexity
until it's truly necessary

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any, Protocol

import attrs
import awkward as ak
import numpy as np

from mammoth.framework.task import GeneratorAnalysisArguments

logger = logging.getLogger(__name__)


@attrs.define
class GeneratorSettings:
    """Settings used for a generator.

    To be clear, these are not used to configure a generator to generate events.
    Instead, this just encodes the parameters that were used for generating events
    and then how those events should be analyzed (ie. recoil subtraction, etc)

    Attributes:
        name: The name of the generator.
        identifier: The identifier for the generator. Just for convenience.
        parameters: Intrinsic parameters used for the generator. e.g. whether it has recoils.
        analysis_arguments: Arguments used for analysis of the generators. e.g. how to handle
            recoils in the analysis.
    """

    name: str
    identifier: str
    parameters: dict[str, Any] = attrs.field(factory=dict)
    analysis_arguments: dict[str, Any] = attrs.field(factory=dict)

    @classmethod
    def from_config(
        cls, config: dict[str, Any], generator_analysis_arguments: GeneratorAnalysisArguments | None
    ) -> GeneratorSettings:
        if generator_analysis_arguments is None:
            generator_analysis_arguments = {}
        return cls(
            name=config["name"],
            # There is an opportunity to customize it, but we take the standard name if it's not specified.
            # It's just more convenient since I often don't want anything other than the name.
            identifier=config.get("identifier", config["name"]),
            parameters=config.get("parameters", {}),
            analysis_arguments=generator_analysis_arguments,
        )


class ConfigureGeneratorOptions(Protocol):
    """The signature that must be implemented for configuring generator options.

    We don't specify the return type because it will depend on the step. This isn't used
    so widely - it's mostly just for documentation purposes.
    """

    def __call__(
        self,
        generator: GeneratorSettings,
        collision_system: str,
        arrays: ak.Array,
        levels: list[str],
    ) -> Any: ...


def create_generator_settings_if_valid(
    input_source_config: dict[str, Any], generator_analysis_arguments: GeneratorAnalysisArguments | None
) -> GeneratorSettings | None:
    """Create the generator config if it is valid.

    This is the main entry point for starting to use the generator settings.
    If we're analyzing data instead of MC, this will be a no-op, returning None.

    Args:
        input_source_config: The input source configuration.
        generator_analysis_arguments: Generator specific arguments related to the analysis.
    Returns:
        The generator configuration if it is valid, otherwise None.
    """
    # NOTE: If there's one input source, this config corresponds to it.
    #       However, if there are multiple input sources, this is the signal input source.
    generator_config: dict[str, Any] | None = input_source_config.get("generator")

    if generator_config is None:
        # This is data...
        return None
    # Help out mypy...
    assert generator_config is not None
    return GeneratorSettings.from_config(
        config=generator_config, generator_analysis_arguments=generator_analysis_arguments
    )


def configure_generator_options_before_starting_analysis(
    generator: GeneratorSettings,
    collision_system: str,
    arrays: ak.Array,
    levels: list[str],
) -> tuple[ak.Array, dict[str, Any]]:
    """Configure generator specific before starting the analysis.

    Args:
        generator: The generator settings.
        collision_system: The collision system.
        arrays: The arrays to be analyzed.
        levels: The levels in the arrays to be analyzed.

    Returns:
        (arrays, track_selection_kwargs): arrays (possible modified), additional arguments to be passed to the track selection.
    """
    # Determine what to do (if anything) for each generator
    track_selection_kwargs: dict[str, Any] = {}

    # We want to support this option for each generator, so we define it generally.
    # Anything down below can customize it further
    # NOTE: We'll default to selecting charged particles since I default to charged analysis
    #       and I'm far more likely to forget to enable it than to need to disable it.
    if generator.analysis_arguments.get("selected_charged_particles", True):
        track_selection_kwargs["columns_to_explicitly_select_charged_particles"] = list(levels)
    if generator.analysis_arguments.get("custom_charged_hadron_PIDs"):
        track_selection_kwargs["charged_hadron_PIDs"] = generator.analysis_arguments.get("custom_charged_hadron_PIDs")

    # Customize for each generator
    match generator.name:
        case "jewel":
            # NOTE: The JEWEL production at LBL doesn't have any e.g. eta cuts. I don't apply them here
            #       because the jet finding will implicitly cut them. However, it's important to keep them
            #       in mind if we're using the tracks for something else.
            # Requirements:
            # - Uniformly apply min pt cut of 150 MeV, except for the recoils. We keep all of them
            #   since a cut of them seems unphysical (because they're just a model detail).
            # - An optional eta range requirement. Default: 2.0 (just to cut down on the needed processing time)
            if collision_system in ["PbPb_MC", "embed_pythia"]:
                eta_range = generator.analysis_arguments.get("eta_range", 2.0)
                for column_name in levels:
                    # Selection is >= 150 MeV or a recoil
                    # NOTE: If there are no recoils, then the selection of identifier == 3 won't change anything
                    particle_mask = (arrays[column_name].pt >= 0.150) | (arrays[column_name].identifier == 3)
                    if eta_range:
                        particle_mask = particle_mask & (np.abs(arrays[column_name].eta) < eta_range)

                    # Apply the mask
                    arrays[column_name] = arrays[column_name][particle_mask]

                    # And note to skip the pt selection in the standard track selection
                    if "columns_to_skip_min_pt_requirement" not in track_selection_kwargs:
                        track_selection_kwargs["columns_to_skip_min_pt_requirement"] = []
                    track_selection_kwargs["columns_to_skip_min_pt_requirement"].append(column_name)
        case "jetscape":
            ...
        case "pythia":
            ...
        case "herwig":
            ...
        case _:
            _msg = f"Unrecognized generator: {generator}"
            raise ValueError(_msg)

    logger.info(
        f"Generator customization analysis arguments: {generator.analysis_arguments=} -> {track_selection_kwargs=}"
    )

    return arrays, track_selection_kwargs


def _negative_energy_recombiner_for_generator(
    recombiner_name: str,
    background_mask_function: Callable[[ak.Array], ak.Array],
    arrays: ak.Array,
    levels: list[str],
) -> tuple[ak.Array, dict[str, dict[str, ak.Any]]]:
    """Configure the negative energy recombiner for a generator.

    Note:
        arrays is necessary since we add in the user_index, while the

    Args:
        recombiner_name: The name of the recombiner to use.
        background_mask_function: A function which returns a mask for the background particles.
            This function will provide the array at a given level, and it must return a mask.
        arrays: The arrays to be used for determining the background particles.
        levels: The particle levels in the analysis.

    Returns:
        arrays, jet_finding_settings_kwargs: The arrays and the jet finding settings kwargs.
            The arrays are modified to include the `user_index`, while the jet finding settings include
            the necessary settings for the negative energy recombiner.
    """
    # Setup
    from mammoth.framework import jet_finding

    available_recombiners = {
        "negative_energy_recombiner": functools.partial(jet_finding.NegativeEnergyRecombiner, identifier_index=-123456),
    }
    # Validation
    if recombiner_name not in available_recombiners:
        msg = f"Unrecognized recombiner: {recombiner_name}"
        raise ValueError(msg)
    if not all("identifier" in ak.fields(arrays[level]) for level in levels):
        msg = "The identifier field is required for the NegativeEnergyRecombiner"
        raise ValueError(msg)

    # We loop over levels because we need to handle each one separately (or not, depending on our intentions).
    jet_finding_settings_kwargs: dict[str, dict[str, Any]] = {level: {} for level in levels}
    for level in levels:
        # There doesn't seem to be any need to customized the identifier index,
        # so we'll just use our usual value
        jet_finding_settings_kwargs[level]["recombiner"] = available_recombiners[recombiner_name]()

        # Next, need to encode the user_index.
        # Here, the convention will be that we encode the user_index as negative if it is a recoil particle.
        recoil_particles_mask = arrays[level]["identifier"] == background_mask_function(arrays[level])
        arrays[level, "user_index"] = jet_finding.calculate_user_index_with_encoded_sign_info(
            particles=arrays[level],
            mask_to_encode_with_negative=recoil_particles_mask,
        )

    return arrays, jet_finding_settings_kwargs


def _setup_before_jet_finding(levels: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    jet_finding_kwargs: dict[str, dict[str, Any]] = {level: {} for level in levels}
    jet_finding_settings_kwargs: dict[str, dict[str, Any]] = {level: {} for level in levels}
    return jet_finding_kwargs, jet_finding_settings_kwargs


def _handle_jewel_options_before_jet_finding(  # noqa: C901
    generator: GeneratorSettings,
    collision_system: str,
    arrays: ak.Array,
    levels: list[str],
) -> tuple[ak.Array, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Handle JEWEL options before jet finding.

    NOTE: Status code == 3 denotes the thermal particles, per https://arxiv.org/abs/2207.14814
            We use the "identifier" field to propagate the MC particle status.

    Example generator configuration:
    ```yaml
    generator:
        name: "jewel"
        identifier: "jewel_no_recoils"
        parameters:
            recoils: false
    ```

    Available background treatment options:
    - "negative_energy_recombiner"
    - "constituent_subtraction"
    - "4mon_subtraction"
    - None

    Args:
        generator: The generator settings.
        collision_system: The collision system.
        arrays: The arrays to be analyzed.
        levels: The levels in the arrays to be analyzed.
    Returns:
        arrays, jet_finding_kwargs, jet_finding_settings_kwargs: arrays (possibly modified), the arguments
            to be passed to the jet finding and jet finding settings.
    """
    from mammoth.framework import jet_finding

    jet_finding_kwargs, jet_finding_settings_kwargs = _setup_before_jet_finding(levels=levels)

    # Setup
    recoil_particle_identifier = 3
    jewel_background_treatment_options: dict[str | None, dict[str, Any]] = {
        "negative_energy_recombiner": {"recombiner": "negative_energy_recombiner", "background_subtraction": None},
        "constituent_subtraction": {
            "recombiner": None,
            "background_subtraction": "constituent_subtraction",
        },
        "4mon_subtraction": {
            "recombiner": None,
            "background_subtraction": "4mon_subtraction",
        },
        None: {"recombiner": None, "background_subtraction": None},
    }
    background_treatment = generator.analysis_arguments.get("background_treatment")
    background_treatment_options = jewel_background_treatment_options[background_treatment]

    # Validation and cross checks
    # Our treatment will vary based on the availability of recoils
    has_recoils = generator.parameters.get("recoils")
    # Recoils are not a valid argument in pp_MC
    if collision_system in ["pp_MC", "pythia"] and has_recoils:
        msg = "Recoils are not a valid argument in pp_MC"
        raise ValueError(msg)
    if background_treatment is not None and not has_recoils:
        msg = "Background treatment is only valid with recoils"
        raise ValueError(msg)
    if background_treatment_options["recombiner"] and background_treatment_options["background_subtraction"]:
        msg = "Cannot have both a recombiner and background subtraction"
        raise ValueError(msg)

    # Now, handle the background options
    if background_treatment is None:
        # Nothing to be done
        ...
    elif background_treatment_options["recombiner"]:
        # Case of using a special recombiner for combining pseudo jets in the jet finding
        # Validation
        if not has_recoils:
            msg = "Recoils are required for the NegativeEnergyRecombiner to be meaningful"
            raise ValueError(msg)

        def _recoil_particles_mask_func(array: ak.Array) -> ak.Array:
            return array["identifier"] == recoil_particle_identifier

        # Configure the negative energy recombiner
        arrays, return_args = _negative_energy_recombiner_for_generator(
            recombiner_name=background_treatment_options["recombiner"],
            background_mask_function=_recoil_particles_mask_func,
            arrays=arrays,
            levels=levels,
        )
        # And propagate the settings
        for level in levels:
            jet_finding_kwargs[level].update(return_args[level])

    elif background_treatment_options["background_subtraction"]:
        if background_treatment_options["background_subtraction"] == "constituent_subtraction":
            # Use event-wise constituent subtraction
            for level in levels:
                # Need to define the subtractor
                jet_finding_kwargs[level]["background_subtraction"] = jet_finding.BackgroundSubtraction(
                    type=jet_finding.BackgroundSubtractionType.event_wise_constituent_subtraction,
                    estimator=jet_finding.JetMedianBackgroundEstimator(
                        jet_finding_settings=jet_finding.JetMedianJetFindingSettings(area_settings=jet_finding.AreaAA())
                    ),
                    subtractor=jet_finding.ConstituentSubtractor(
                        # Value is selected based on their paper, right after Eq 4.2 on page 7
                        r_max=0.5,
                    ),
                )

                recoil_particles_mask = arrays[level]["identifier"] == recoil_particle_identifier
                # Select the signal particles. We want to exclude the recoils from those that are subtracted
                jet_finding_kwargs[level]["particles"] = arrays[level][~recoil_particles_mask]
                # And then pick out the recoil particles, passing them as the explicit background subtraction
                jet_finding_kwargs[level]["background_particles"] = arrays[level][recoil_particles_mask]
        elif background_treatment_options["background_subtraction"] == "4mon_subtraction":
            # NOTE: The "dummy" particle may already be filtered out in the track selection.
            #       This should be checked if we implement this option.
            msg = "4mon subtraction is not yet implemented"
            # NOTE: Need to filter both the signal and background particles
            raise NotImplementedError(msg)

    return arrays, jet_finding_kwargs, jet_finding_settings_kwargs


def _handle_jetscape_options_before_jet_finding(
    generator: GeneratorSettings,  # noqa: ARG001
    collision_system: str,
    arrays: ak.Array,
    levels: list[str],
) -> tuple[ak.Array, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Handle JETSCAPE options before jet finding.

    We should use the negative energy recombiner.

    To do so, we need to:
    - Create the negative energy recombiner
    - Encode the user_index with the hole (status < 0) vs primary particle information (>= 0)

    To do so, we can use the same function as usual.

    Remember that user_info also needs to be passed to the reclustering.
    Fortunately, the encoding that we do for the standard jet finding will pass along
    to the reclustering as well, so we shouldn't need to do anything else!

    Args:
        generator: The generator settings.
        collision_system: The collision system.
        arrays: The arrays to be analyzed.
        levels: The levels in the arrays to be analyzed.
    Returns:
        arrays, jet_finding_kwargs, jet_finding_settings_kwargs: arrays (possibly modified), the arguments
            to be passed to the jet finding and jet finding settings.
    """
    # Setup
    jet_finding_kwargs, jet_finding_settings_kwargs = _setup_before_jet_finding(levels=levels)

    if collision_system in ["PbPb_MC"]:
        # Case of using a special recombiner for combining pseudo jets in the jet finding
        def _holes_mask_func(array: ak.Array) -> ak.Array:
            return array["identifier"] < 0

        # Configure the negative energy recombiner
        arrays, return_args = _negative_energy_recombiner_for_generator(
            recombiner_name="negative_energy_recombiner",
            background_mask_function=_holes_mask_func,
            arrays=arrays,
            levels=levels,
        )
        # And propagate the settings
        for level in levels:
            jet_finding_kwargs[level].update(return_args[level])

    return arrays, jet_finding_kwargs, jet_finding_settings_kwargs


def configure_generator_options_before_jet_finding(
    generator: GeneratorSettings,
    collision_system: str,
    arrays: ak.Array,
    levels: list[str],
) -> tuple[ak.Array, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Configure generator specific before jet finding.

    Args:
        generator: The generator settings.
        collision_system: The collision system.
        arrays: The arrays to be analyzed.
        levels: The levels in the arrays to be analyzed.
    Returns:
        arrays, jet_finding_kwargs, jet_finding_settings_kwargs: arrays (possibly modified), the arguments
            to be passed to the jet finding and jet finding settings.
    """
    # Setup
    jet_finding_kwargs, jet_finding_settings_kwargs = _setup_before_jet_finding(levels=levels)

    # Determine what to do for each generator
    match generator.name:
        case "jewel":
            return _handle_jewel_options_before_jet_finding(
                generator=generator,
                collision_system=collision_system,
                arrays=arrays,
                levels=levels,
            )
        case "jetscape":
            return _handle_jetscape_options_before_jet_finding(
                generator=generator,
                collision_system=collision_system,
                arrays=arrays,
                levels=levels,
            )
        case "pythia":
            # No special treatment
            ...
        case "herwig":
            # No special treatment
            ...
        case _:
            _msg = f"Unrecognized generator: {generator}"
            raise ValueError(_msg)

    return arrays, jet_finding_kwargs, jet_finding_settings_kwargs
