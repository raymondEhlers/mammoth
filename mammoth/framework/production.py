"""Track and define parameters which determine a production.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

import datetime
import functools
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

import attrs
import pachyderm.yaml

from mammoth.framework import utils
from mammoth.framework.analysis import objects as analysis_objects

logger = logging.getLogger(__name__)


def _git_hash_from_module(module: Any) -> str:
    """Retrieve the git hash from a particular module

    Adapted from: https://stackoverflow.com/a/21901260/12907985

    Note:
        This assumes it is stored in a git repository, and it doesn't check.

    Args:
        module: Module to retrieve the git hist for.
    Returns:
        The git hash associated with the module.
    """
    return (
        subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(module.__file__).parent.parent, capture_output=True)
        .stdout.decode("ascii")
        .strip()
    )


def _installed_python_software() -> List[str]:
    """Extract all installed python software via `pip freeze`

    Adapted from: https://stackoverflow.com/a/58013217/12907985

    NOTE:
        This doesn't really work as expected with poetry - it just points to local packages.
        However, we also have poetry.lock, so as long as we have the has of the repo, we'll
        know the versions of all of the software.

    Args:
        None
    Returns:
        List of str, which each entry specifying a package + version.
    """
    import sys

    return (
        subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True)
        .stdout.decode("ascii")
        .strip("\n")
        .split("\n")
    )


def _describe_production_software(
    production_config: Mapping[str, Any], modules_to_record: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    # Validation
    if modules_to_record is None:
        # By default, We want to store the git hash of:
        # - pachyderm
        # - mammoth
        # - jet_substructure
        modules_to_record = ["pachyderm", "mammoth", "jet_substructure"]

    output: Dict[str, Any] = {}
    output["software"] = {}

    # To determine the location, we do something kind of lazy and import the file to determine the
    # location of the git repo
    output["software"]["hashes"] = {}
    import importlib

    for module_name in modules_to_record:
        try:
            _m = importlib.import_module(module_name)
            output["software"]["hashes"][module_name] = _git_hash_from_module(_m)
        except ImportError:
            logger.info(
                f"Skipping recording module {module_name} in the production details because it's not available."
            )

    # We also want a full pip freeze. We'll store each package as an entry in a list
    output["software"]["packages"] = _installed_python_software()

    return output


def _read_full_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Read full YAML configuration file.

    Args:
        config_path: Path to the configuration file. Default: "config/new_config.yaml".
    Returns:
        Full YAML configuration. Requires some interpretation.
    """
    if config_path is None:
        config_path = Path("config/track_skim_config.yaml")

    # Here, we use ruamel.yaml directly with the "safe" type because the roundtrip
    # types that we usually use don't play so nicely when we rewrite a subset of the data
    # (eg. anchors don't seem to resolve correctly because we only rewrite the subset, there
    # are some stray comments that we don't really want to keep, etc)
    import ruamel.yaml

    y = ruamel.yaml.YAML(typ="safe")
    with open(config_path, "r") as f:
        full_config: Dict[str, Any] = y.load(f)

    return full_config


class ProductionSpecialization(Protocol):
    def customize_identifier(self, analysis_settings: Mapping[str, Any]) -> str:
        ...

    def tasks_to_execute(self, collision_system: str) -> List[str]:
        ...


_possible_collision_systems = [
    "pp",
    "pythia",
    "PbPb",
    "embedPythia",
    "embed_pythia",
    "embed_thermal_model",
]
_collision_systems_with_scale_factors = ["pythia", "embedPythia", "embed_pythia", "embed_thermal_model"]


def _validate_collision_system(instance: "ProductionSettings", attribute: attrs.Attribute[str], value: str) -> None:
    if value not in _possible_collision_systems:
        raise ValueError(f"Invalid collisions system. Provided: {value}")


@attrs.frozen(slots=False)
class ProductionSettings:
    collision_system: str = attrs.field(validator=_validate_collision_system)
    number: int
    config: Dict[str, Any]
    specialization: ProductionSpecialization
    _manual_analysis_parameter_keys: List[str] = attrs.field(default=["jet_R", "min_jet_pt", "background_subtraction"])
    _base_output_dir: Path = attrs.field(default=Path("trains"))

    @functools.cached_property
    def formatted_number(self) -> str:
        # Put some leading 0s for consistency in sorting, etc
        return f"{self.number:04}"

    @functools.cached_property
    def identifier(self) -> str:
        name = ""
        # First, handle the case of possible embedding
        signal_dataset = self.config["metadata"].get("signal_dataset")
        if signal_dataset:
            name += f"{self.config['metadata']['signal_dataset']['name']}_embedded_into"
        # Then, the production name
        name = f"{self.config['metadata']['dataset']['name']}"
        # The label
        extra_label = self.config.get("label")
        if extra_label:
            name += f"_{extra_label}"
        # New section: the analysis parameters
        # First, we want to denote a new section with an extra "__"
        name += "__"
        _analysis_settings = self.config["settings"]
        # We want particular handling for some analysis settings, so we do those by hand.
        # The rest are included automatically
        # Jet R
        jet_R_value = _analysis_settings["jet_R"]
        name += f"_jet_R{round(jet_R_value * 100):03}"
        # Min jet pt
        name += "_min_jet_pt"
        for k, v in _analysis_settings["min_jet_pt"].items():
            name += f"_{k}_{round(v)}"
        # Background subtraction
        name += "_background_subtraction"
        for k, v in _analysis_settings["background_subtraction"].items():
            name += f"_{k}_{str(v)}"
        # Allow for customization
        name += self.specialization.customize_identifier(analysis_settings=_analysis_settings)
        # And then all the rest
        for k, v in _analysis_settings.items():
            if k in self._manual_analysis_parameter_keys:
                continue
            name += f"_{k}_{str(v)}"
        # And finally, the production details
        # First, we want to denote a new section with an extra "__"
        name += "__"
        # The production number itself
        name += f"_production_{self.number}"
        # The date for good measure
        name += f"_{datetime.datetime.utcnow().strftime('%Y_%m_%d')}"
        return name

    def input_files(self) -> List[Path]:
        n_pt_hat_bins = self.config["metadata"]["dataset"].get("n_pt_hat_bins")
        if n_pt_hat_bins is not None:
            # Handle pt hat binned production
            _files_per_pt_hat = self.input_files_per_pt_hat()
            _files = []
            for _files_in_single_pt_hat in _files_per_pt_hat.values():
                _files.extend(_files_in_single_pt_hat)
            return _files

        # Otherwise, we just can blindly expand
        return utils.ensure_and_expand_paths(self.config["metadata"]["dataset"]["files"])

    @property
    def has_scale_factors(self) -> bool:
        return (
            "signal_dataset" in self.config["metadata"] or "n_pt_hat_bins" in self.config["metadata"]["dataset"]
        ) and (self.collision_system in _collision_systems_with_scale_factors)

    def input_files_per_pt_hat(self) -> Dict[int, List[Path]]:
        if self.has_scale_factors:
            raise ValueError(
                f"Asking for input files per pt hat doesn't make sense for collision system {self.collision_system}"
            )

        # Will be signal_dataset if embedded, but otherwise will be the standard "dataset" key
        dataset_key = "signal_dataset" if "signal_dataset" in self.config["metadata"] else "dataset"

        # +1 due to pt hat bins being 1-indexed
        _files = {}
        for pt_hat_bin in range(1, self.config["metadata"][dataset_key]["n_pt_hat_bins"] + 1):
            _files[pt_hat_bin] = utils.ensure_and_expand_paths(
                [Path(s.format(pt_hat_bin=pt_hat_bin)) for s in self.config["metadata"][dataset_key]["files"]]
            )

        return _files

    @property
    def scale_factors_filename(self) -> Path:
        # Validation
        if not self.has_scale_factors:
            raise ValueError(f"Invalid collision system for extracting scale factors: {self.collision_system}")

        dataset_key = "signal_dataset" if "signal_dataset" in self.config["metadata"] else "dataset"
        # Need to go up twice to get back to the "trains" directory because the collision system
        # stored in the production config may not be the same as the dataset that we're actually
        # extracting the scale factors from (ie. embedPythia != pythia)
        return (
            self.output_dir.parent.parent
            # NOTE: The values from the config are wrapped in a string to help out mypy.
            #       Otherwise, it can't determine the type for some reason...
            / str(self.config["metadata"][dataset_key]["collision_system"])
            / str(self.config["metadata"][dataset_key]["name"])
            / "scale_factors.yaml"
        )

    def scale_factors(self) -> Dict[int, float]:
        # Validation
        if self.collision_system not in _collision_systems_with_scale_factors:
            raise ValueError(f"Invalid collision system for extracting scale factors: {self.collision_system}")

        scale_factors = analysis_objects.read_extracted_scale_factors(self.scale_factors_filename)
        return scale_factors

    @functools.cached_property
    def output_dir(self) -> Path:
        output_dir = self._base_output_dir / self.collision_system / self.formatted_number
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @functools.cached_property
    def tasks_to_execute(self) -> List[str]:
        # Could in principle be multiple tasks.
        _tasks = []

        # Need scale factors
        if self.collision_system in ["pythia"] or "embed" in self.collision_system:
            _tasks.append("extract_scale_factors")

        _tasks.extend(self.specialization.tasks_to_execute(collision_system=self.collision_system))
        return _tasks

    def store_production_parameters(self) -> None:
        output: Dict[str, Any] = {}
        output["identifier"] = self.identifier
        output["date"] = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        output["config"] = dict(self.config)
        output["input_filenames"] = [str(p) for p in self.input_files()]
        if "signal_dataset" in self.config["metadata"]:
            output["signal_filenames"] = [
                str(_filename) for filenames in self.input_files_per_pt_hat().values() for _filename in filenames
            ]
        # Add description of the software
        output.update(_describe_production_software(production_config=self.config))

        # If we've already run this production before, we don't want to overwrite the existing production.yaml
        # Instead, we want to add a new production file with the new parameters (which should be the same as before,
        # except for the production date).
        # In order to avoid overwriting, we try adding an additional index to the filename.
        # 100 is arbitrarily selected, but I see it as highly unlikely that we would have 100 productions...
        for _additional_production_number in range(0, 100):
            _production_filename = self.output_dir / "production.yaml"
            # No need for an index for the first file.
            if _additional_production_number > 0:
                _production_filename = _production_filename.parent / f"production_{_additional_production_number}.yaml"

            if _production_filename.exists():
                # Don't overwrite the production file
                continue
            else:
                y = pachyderm.yaml.yaml()
                with open(_production_filename, "w") as f:
                    y.dump(output, f)

                # We've written, so no need to loop anymore
                break

    @classmethod
    def read_config(
        cls,
        collision_system: str,
        number: int,
        specialization: ProductionSpecialization,
        track_skim_config_filename: Optional[Path] = None,
    ) -> "ProductionSettings":
        track_skim_config = _read_full_config(track_skim_config_filename)
        config = track_skim_config["productions"][collision_system][number]

        return cls(
            collision_system=collision_system,
            number=number,
            config=config,
            specialization=specialization,
        )