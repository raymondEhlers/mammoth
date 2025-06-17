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
        # TODO(RJE): Implement
        return self.name

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
        _, hepdata_id, hepdata_version, _ = hepdata.split("-")
        # Remove "ins"
        hepdata_id = hepdata_id.replace("ins", "")
        # Extract just the numerical version number
        hepdata_version = hepdata_version.replace("v", "")

        return hepdata_id, int(hepdata_version)


def write_observables(observables: dict[str, dict[str, ObservableInfo]], stream: io.TextIO) -> bool:
    """Write the observables in a clear, organized, and readable format."""
    for sqrt_s in sorted(observables.keys()):
        stream.write(f"# sqrt(s) = {sqrt_s} GeV\n\n")
        # Group by observable class
        class_to_obs = {}
        for obs in observables[sqrt_s].values():
            class_to_obs.setdefault(obs.observable_class, []).append(obs)
        for obs_class in sorted(class_to_obs.keys()):
            stream.write(f"## {obs_class}\n\n")
            for obs in sorted(class_to_obs[obs_class], key=lambda o: o.name):
                stream.write(f"- **Name:** {obs.display_name()}\n")
                stream.write(f"  - **Experiment:** {obs.experiment}\n")
                # Try to get InspireHEP ID
                try:
                    hep_id, hep_version = obs.inspire_hep_identifier()
                    stream.write(f"  - **InspireHEP ID:** {hep_id} (v{hep_version})\n")
                except Exception:
                    stream.write("  - **InspireHEP ID:** Not found\n")
                # Optionally, print config keys
                stream.write(f"  - **Config keys:** {', '.join(obs.config.keys())}\n")
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
                observable_info = config[observable_class][observable_key]
                observables[sqrt_s][f"{observable_class}_{observable_key}"] = ObservableInfo(
                    observable_class=observable_class,
                    name=observable_key,
                    config=observable_info,
                )

    # import IPython; IPython.embed()
    output_file = Path("output.md")

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
