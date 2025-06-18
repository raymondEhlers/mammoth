"""Convert the JETSCAPE-analysis YAML configuration files to a list for PR purposes.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
from typing import Any

import attrs
import yaml

logger = logging.getLogger(__name__)


_name_translation_map = {
    "pt_ch": "charged hadron RAA",
    "pt_pi0": "charged pion RAA",
    "IAA_pt": "IAA",
    "dphi": "acoplanarity (dphi)",
    "Dpt": "Fragmentation (Pt)",
    "Dz": "Fragmentation (z)",
    "ptd": "Dispersion (PtD)",
    "axis": "Jet-axis difference",
    "charge": "Jet charge",
    "ktg": "Groomed kt",
    "mg": "Groomed jet mass (Mg)",
    "rg ": "Groomed jet radius (Rg)",
    "tg": "Groomed jet radius (Rg)",
    "zg": "Groomed momentum fraction (zg)",
    "zr": "Subjet z",
    "angularity": "Generalized angularity",
    "mass": "Mass",
    "chjet": "charged-particle jet",
    "dihadron": "di-hadron correlations",
    "nsubjettiness": "N-subjettiness",
    "four": "four-particle cumulant",
    "sp": "scalar product",
    # These two are from Peter and probably need to be re-encoded since they don't match our usual conventions...
    "R25": "R=0.2/0.5 IAA ratio",
    "gammajet": "gamma-jet",
    "pizerojet": "pi-zero jet",
    # Fall back in case nothing else picked it up
    "pt": "RAA",
}


def pretty_print_name(name: str) -> str:
    """Translates encoded name into something more readable.

    Args:
        name_substr: Name split into it's substrings for parsing. We don't do this
            automatically because sometimes you need to e.g. remove the experiment name.

    Returns:
        Readable name
    """
    working_str = name
    for k, v in _name_translation_map.items():
        if k in name:
            working_str = working_str.replace(k, v)
    return working_str.replace("_", " ")


@attrs.define
class ObservableInfo:
    observable_class: str = attrs.field()
    name: str = attrs.field()
    config: dict[str, Any] = attrs.field()

    @property
    def identifier(self) -> str:
        return f"{self.observable_class}_{self.name}"

    @property
    def experiment(self) -> str:
        return self.name.split("_")[-1].upper()

    def display_name(self) -> str:
        """Pretty print of the observable name"""
        # -1 removes the experiment name
        return pretty_print_name("_".join(self.name.split("_")[:-1]))

    def inspire_hep_identifier(self) -> tuple[str, int]:
        """Extract InspireHEP identifier from the config if possible."""
        # Attempt to extract from the HEPdata filename.

        # Validation
        # We mostly don't care about the pp HEPdata - it's mostly about the AA
        if not ("hepdata" in self.config or "hepdata_AA" in self.config):
            msg = f"Cannot find HEPdata key for observable {self.identifier}"
            raise ValueError(msg)

        hepdata_key = "hepdata"
        if hepdata_key not in self.config:
            hepdata_key = "hepdata_AA"

        # Example: "HEPData-ins1127262-v2-root.root"
        hepdata = self.config[hepdata_key]
        _, hepdata_id, hepdata_version, *_ = hepdata.split("-")
        # Remove "ins"
        hepdata_id = hepdata_id.replace("ins", "")
        # Extract just the numerical version number
        hepdata_version = hepdata_version.replace("v", "")

        return hepdata_id, int(hepdata_version)

    def to_markdown(self, name_prefix: str | None = None) -> str:  # noqa: C901
        """Return a pretty, formatted markdown string for this observable."""
        display_name = self.display_name()
        if name_prefix is not None:
            display_name = f"{name_prefix} {display_name}"
        lines = [f"- **Name:** {display_name}", f"  - **Experiment:** {self.experiment}"]
        try:
            hep_id, hep_version = self.inspire_hep_identifier()
            lines.append(f"  - **InspireHEP ID:** {hep_id} (v{hep_version})")
        except Exception:
            lines.append("  - **InspireHEP ID:** Not found")
        implementation_status = "Work-in-progress"
        if self.config["enabled"]:
            implementation_status = "Implemented"
        lines.append(f"  - **Implementation status**: {implementation_status}")

        # Special handling for particular keys:
        parameters = ["jet_R", "kappa", "axis", "SoftDrop"]
        if any(parameter in self.config for parameter in parameters):
            lines.append("  - **Parameters**:")
            if "jet_R" in self.config:
                lines.append(f"     - Jet R: {self.config['jet_R']}")
            if "kappa" in self.config:
                lines.append(f"     - Angularity kappa: {self.config['kappa']}")
            if "axis" in self.config:
                lines.append("     - Jet-axis difference:")
                for parameters in self.config["axis"]:
                    description = f"{parameters['type']}"
                    if "grooming_settings" in parameters:
                        description += f", z_cut={parameters['grooming_settings']['zcut']}, beta={parameters['grooming_settings']['beta']}"
                    lines.append(f"       - {description}")
            if "SoftDrop" in self.config:
                lines.append("     - SoftDrop:")
                for parameters in self.config["SoftDrop"]:
                    lines.append(f"       - z_cut={parameters['zcut']}, beta={parameters['beta']}")
            if "dynamical_grooming_a" in self.config:
                lines.append("     - Dynamical grooming:")
                for param in self.config["dynamical_grooming_a"]:
                    lines.append(f"       - a = {param}")

        return "\n".join(lines)


def write_observables(observables: dict[str, dict[str, ObservableInfo]], stream: io.TextIO) -> bool:
    """Write the observables in a clear, organized, and readable format."""
    for sqrt_s in sorted(observables.keys()):
        stream.write(rf"# $\sqrt{{s_{{NN}}}}$ = {sqrt_s} GeV" + "\n\n")
        # Group by observable class
        class_to_obs = {}
        for obs in observables[sqrt_s].values():
            class_to_obs.setdefault(obs.observable_class, []).append(obs)
        for obs_class in sorted(class_to_obs.keys()):
            stream.write(f"## {pretty_print_name(obs_class)}\n\n")
            for obs in sorted(class_to_obs[obs_class], key=lambda o: o.name):
                # stream.write(obs.to_markdown(name_prefix=pretty_print_name(obs_class)))
                stream.write(obs.to_markdown())
                stream.write("\n")
            stream.write("\n")
        stream.write("\n")
    return True


def main(jetscape_analysis_config_path: Path) -> None:
    # Parameters
    sqrt_s_values = [200, 2760, 5020]

    # Read configuration files
    configs = {}
    observable_classes = {}
    for sqrt_s in sqrt_s_values:
        with (jetscape_analysis_config_path / f"STAT_{sqrt_s}.yaml").open() as f:
            configs[sqrt_s] = yaml.safe_load(f)

        observable_classes[sqrt_s] = []
        found_start_of_observables = False
        for k in configs[sqrt_s]:
            if "hadron" in k or found_start_of_observables:
                observable_classes[sqrt_s].append(k)
                found_start_of_observables = True

    # Now extract all of the observables
    observables = {}
    for sqrt_s, config in configs.items():
        observables[sqrt_s] = {}
        for observable_class in observable_classes[sqrt_s]:
            for observable_key in config[observable_class]:
                if "v2" in observable_key and observable_class != "dijet":
                    # need to go a level deeper for the v2 since they're nested...
                    for sub_observable_key in config[observable_class][observable_key]:
                        observable_info = config[observable_class][observable_key][sub_observable_key]

                        # Move the experiment to the end of the name to match the convention
                        *base_observable_name, experiment_name = observable_key.split("_")
                        observable_name = "_".join(base_observable_name)
                        observable_name += f"_{sub_observable_key}_{experiment_name}"

                        observables[sqrt_s][f"{observable_class}_{observable_key}_{sub_observable_key}"] = (
                            ObservableInfo(
                                observable_class=observable_class,
                                name=observable_name,
                                config=observable_info,
                            )
                        )
                else:
                    observable_info = config[observable_class][observable_key]
                    observables[sqrt_s][f"{observable_class}_{observable_key}"] = ObservableInfo(
                        observable_class=observable_class,
                        name=observable_key,
                        config=observable_info,
                    )

    # import IPython; IPython.embed()
    here = Path(__file__).parent
    output_file = here / Path("observables.md")

    with output_file.open("w") as f:
        write_observables(observables=observables, stream=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JETSCAPE-analysis YAML configuration files to a list for PR purposes."
    )
    parser.add_argument(
        "-c",
        "--jetscape-analysis-config",
        type=Path,
        help="Path to the jetscape-analysis config directory.",
        required=True,
    )
    args = parser.parse_args()

    main(jetscape_analysis_config_path=args.jetscape_analysis_config)
