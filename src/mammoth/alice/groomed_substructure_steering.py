"""Steering for groomed substructure analyses.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

import enum
from collections.abc import MutableMapping
from typing import Any

import attrs


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
    splittings_selection = SplittingsSelection[analysis_arguments["splittings_selection"]]
    return f"{splittings_selection!s}"


@attrs.frozen()
class ProductionSpecialization:
    def customize_identifier(self, analysis_settings: MutableMapping[str, Any]) -> str:
        name = ""
        # Selection of splittings
        splittings_selection_value = SplittingsSelection[analysis_settings.pop("splittings_selection")]
        name += f"_{splittings_selection_value!s}"
        return name

    def tasks_to_execute(self, collision_system: str) -> list[str]:
        _tasks = []

        # Skim task
        _base_name = "calculate_{label}_skim"
        _label_map = {
            "pp": "data",
            "pythia": "pp_MC",
            "pp_MC": "pp_MC",
            "PbPb": "data",
            "embedPythia": "embed_pythia",
            "embed_pythia": "embed_pythia",
            "embed_thermal_model": "embed_thermal_model",
        }
        _tasks.append(_base_name.format(label=_label_map[collision_system]))
        return _tasks
