"""Steering for groomed substructure analyses.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""
from __future__ import annotations

import enum
from collections.abc import MutableMapping
from typing import Any

import attrs


def _safe_str_reclustering_settings(reclustering_settings: MutableMapping[str, Any]) -> str:
    """Convert reclustering settings to a string, but safely handle missing keys."""
    algorithm = reclustering_settings.get("algorithm", "")
    additional_algorithm_parameter = reclustering_settings.get("additional_algorithm_parameter", "")
    if isinstance(additional_algorithm_parameter, float):
        additional_algorithm_parameter = f"{additional_algorithm_parameter}"
        # And then replace any possible decimal point with an "p"
        additional_algorithm_parameter = additional_algorithm_parameter.replace(".", "p")
    return f"{algorithm}_{additional_algorithm_parameter}"


class SplittingsSelection(enum.Enum):
    recursive = 0
    iterative = 1

    def __str__(self) -> str:
        return f"{self.name}_splittings"


def argument_preprocessing(**analysis_arguments: Any) -> dict[str, Any]:
    splittings_selection = SplittingsSelection[analysis_arguments["splittings_selection"]]
    return {
        "iterative_splittings": splittings_selection == SplittingsSelection.iterative,
    }


def analysis_output_identifier(**analysis_arguments: Any) -> str:
    output_identifier = ""
    # Selection of grooming methods
    selected_grooming_methods = analysis_arguments.get("selected_grooming_methods", [])
    if selected_grooming_methods:
        output_identifier += f"__grooming_methods_{'_'.join(selected_grooming_methods)}"
    # Selection of splittings
    splittings_selection = SplittingsSelection[analysis_arguments["splittings_selection"]]
    output_identifier += f"__{splittings_selection!s}"
    # Reclustering settings
    reclustering_settings = analysis_arguments.pop("reclustering_settings", {})
    if reclustering_settings:
        output_identifier += f"__reclustering_settings_{_safe_str_reclustering_settings(reclustering_settings)}"
    return f"{output_identifier}"


@attrs.frozen()
class ProductionSpecialization:
    def customize_identifier(self, analysis_settings: MutableMapping[str, Any]) -> str:
        """Customize the identifier used for this production.

        NOTE:
            This is _not_ the same as the output identifier, which is used for output filenames.

        Args:
            analysis_settings: Settings for the analysis.
        Returns:
            Customization of the identifier for the production.
        """
        name = ""
        # Selection of grooming methods
        selected_grooming_methods = analysis_settings.pop("selected_grooming_methods", [])
        if selected_grooming_methods:
            name += f"__grooming_methods_{'_'.join(selected_grooming_methods)}"
        # Selection of splittings
        splittings_selection_value = SplittingsSelection[analysis_settings.pop("splittings_selection")]
        name += f"__{splittings_selection_value!s}"
        # Reclustering settings
        reclustering_settings = analysis_settings.pop("reclustering_settings", {})
        if reclustering_settings:
            name += f"__reclustering_settings_{_safe_str_reclustering_settings(reclustering_settings)}"
        # Always pop the output settings, but don't include it. It's just not needed
        analysis_settings.pop("output_settings")
        # Also the prefix conversion, since it's just not important for the identifier
        analysis_settings.pop("convert_data_format_prefixes")
        # Additional "_" at the end to add full offsets for the customized identifier properties
        return f"{name}_"

    def tasks_to_execute(self, collision_system: str) -> list[str]:
        _tasks = []

        # Skim task
        _base_name = "calculate_{label}_skim"
        _label_map = {
            "pp": "data",
            "pythia": "pp_MC",
            "pp_MC": "pp_MC",
            "PbPb": "data",
            "PbPb_MC": "PbPb_MC",
            "embedPythia": "embed_pythia",
            "embed_pythia": "embed_pythia",
            "embed_thermal_model": "embed_thermal_model",
        }
        _tasks.append(_base_name.format(label=_label_map[collision_system]))
        return _tasks
