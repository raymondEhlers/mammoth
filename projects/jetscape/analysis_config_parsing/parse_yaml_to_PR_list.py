"""Convert the JETSCAPE-analysis YAML configuration files to a list for PR purposes.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import argparse
import io
import itertools
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

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
class Observable:
    sqrt_s: float = attrs.field()
    observable_class: str = attrs.field()
    name: str = attrs.field()
    config: dict[str, Any] = attrs.field()

    @property
    def identifier(self) -> str:
        return f"{self.observable_class}_{self.name}"

    @property
    def internal_name_without_experiment(self) -> str:
        *name, _ = self.name.split("_")
        return "_".join(name)

    @property
    def experiment(self) -> str:
        return self.name.split("_")[-1].upper()

    @property
    def display_name(self) -> str:
        """Pretty print of the observable name.

        It's fairly ad-hoc, but at least gives us something to work with.
        """
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

    def parameters(self) -> tuple[list[dict[str, str]], list[dict[str, int]]]:
        """The parameters that are relevant to the observable.

        Note:
            The bin indices are not meant to be comprehensive - just those that are
            relevant for the observable.

        Returns:
            Parameters, bin indices associated with the parameters (e.g. pt).
        """
        _parameters = BaseParameters.construct_parameters(observable=self, config=self.config)
        if "jet_R" in self.config:
            _parameters.update(JetParameters.construct_parameters(observable=self, config=self.config))
        # TODO(RJE): Handle trigger appropriately...

        return _parameters

    def format_parameters_for_printing(self, parameters: dict[str, Any]) -> dict[str, str]:
        output_parameters = BaseParameters.format_parameters_for_printing(parameters)
        if "jet_R" in self.config:
            output_parameters.update(JetParameters.format_parameters_for_printing(parameters))
        # TODO(RJE): Handle trigger appropriately...

        missing_keys = set(parameters).difference(set(output_parameters))
        if missing_keys:
            logger.warning(f"missing formatting for {missing_keys}")

        # NOTE: Wrapped in dict to avoid leaking the defaultdict
        return dict(output_parameters)

    def generate_parameters(self, parameters: dict[str, list[Any]]) -> Iterator[tuple[str, dict[str, int]]]:
        """Generate combinations of parameters that are relevant to the observable.

        Note:
            The bin indices are not meant to be comprehensive - just those that are
            relevant for the observable.

        Returns:
            Description of parameters, bins associated with the parameters (e.g. pt).
        """
        # Add indices before each parameters:
        # e.g. "pt": [[1, 2], [2, 3]] -> [(0, [1,2]), (1, [2,3])]
        # parameters_with_indices = {
        #    p: [(i, v) for i, v in enumerate(values)] for p, values in parameters.items()
        # }
        # TODO(RJE): Grooming parameters are mutually exclusive, so we need to handle them one-by-one
        # grooming_parameters
        # We get the same combinations, but also with the indices.
        indices = {p: list(range(len(values))) for p, values in parameters.items()}
        # Get all combinations
        combinations = itertools.product(*parameters.values())
        indices_combinations = itertools.product(*indices.values())

        # And then return them labeled by the parameter name
        # We want: ({"pt": [], ...], {"pt": }})
        yield from zip(
            (dict(zip(parameters.keys(), values, strict=True)) for values in combinations),
            (dict(zip(parameters.keys(), values, strict=True)) for values in indices_combinations),
            strict=True,
        )

    def to_markdown(self, name_prefix: str | None = None) -> str:  # noqa: C901
        """Return a pretty, formatted markdown string for this observable."""
        display_name = self.display_name
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

    def to_csv_like(self, separator: str = "\t") -> str:
        """Convert observable into a csv-like entries.

        If there are multiple parameters that would justify multiple entries,
        then multiple lines are provided.

        Args:
            separator: Separator used to differentiate fields. Default: `\t`.
        Returns:
            Lines formatted suitably for a csv-like.
        """


def write_observables(observables: dict[str, dict[str, Observable]], stream: io.TextIO) -> bool:
    """Write the observables in a clear, organized, and readable format.

    Args:
        observables: All observables
        stream: Output stream
    Returns:
        True if successful.
    """
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


class ParameterGroup(Protocol):
    """Group of parameters.

    This is an interface that each group of parameters must implement.
    """

    def construct_parameters(observable: Observable, config: dict[str, Any]) -> dict[str, list[Any]]: ...

    def format_parameters_for_printing(parameters: dict[str, list[Any]]) -> dict[str, list[str]]: ...


class BaseParameters:
    def construct_parameters(observable: Observable, config: dict[str, Any]) -> dict[str, list[Any]]:
        base_parameters = {}
        # Centrality
        base_parameters["centrality"] = [tuple(v) for v in config["centrality"]]
        return base_parameters

    def format_parameters_for_printing(parameters: dict[str, list[Any]]) -> dict[str, list[str]]:
        # output_parameters = collections.defaultdict(list)
        output_parameters = {}
        # Centrality
        cent_low, cent_high = parameters["centrality"]
        output_parameters["centrality"] = f"{cent_low}-{cent_high}%"

        return output_parameters


class PtParameters:
    def construct_parameters(observable: Observable, config: dict[str, Any]) -> dict[str, list[Any]]:
        values = []
        if "pt" in config:
            pt_values = config["pt"]
            values = [(pt_low, pt_high) for pt_low, pt_high in itertools.pairwise(pt_values)]
        elif "pt_min" in config:
            values = [(config["pt_min"], -1)]

        if values:
            # Wrap it in a "pt" key to handle it similarly to the other parameters
            return {"pt": values}
        # Nothing to construct out of this
        return {}

    def format_parameters_for_printing(parameters: dict[str, list[Any]]) -> dict[str, list[str]]:
        # output_parameters = collections.defaultdict(list)
        output_parameters = {}
        if "pt" in parameters:
            pt_low, pt_high = parameters["pt"]
            if pt_high == -1:
                output_parameters["pt"] = f"pt > {pt_low}"
            else:
                output_parameters["pt"] = f"{pt_low} < pt < {pt_high}"
        # return _propagate_rest_of_parameters(output_parameters=output_parameters, parameters=parameters)
        return output_parameters


class JetParameters:
    def construct_parameters(observable: Observable, config: dict[str, Any]) -> dict[str, list[Any]]:
        values = {}

        # Handle standard cases first
        # Standardize parameter names
        parameter_names = {
            "jet_R": "jet_R",
            "axis": "axis",
            "kappa": "kappa",
            "r": "subjet_zr",
            "SoftDrop": "soft_drop",
            "dynamical_grooming_a": "dynamical_grooming",
        }
        for input_name, output_name in parameter_names.items():
            if input_name in config:
                values[output_name] = config[input_name]
        # Finally, handle special cases:
        # i.e. the pt
        values.update(PtParameters.construct_parameters(observable=observable, config=config))

        return values

    def format_parameters_for_printing(parameters: dict[str, list[Any]]) -> dict[str, list[str]]:
        # output_parameters = collections.defaultdict(list)
        output_parameters = {}
        # Jet R
        if "jet_R" in parameters:
            output_parameters["jet_R"] = f"R={parameters['jet_R']}"
        # pt
        if "pt" in parameters:
            output_parameters.update(PtParameters.format_parameters_for_printing(parameters=parameters))
        # Axis
        if "axis" in parameters:
            param = parameters["axis"]
            description = f"{param['type']}"
            if "grooming_settings" in param:
                description += (
                    f", SD z_cut={param['grooming_settings']['zcut']}, beta={param['grooming_settings']['beta']}"
                )
            output_parameters["axis"] = description
        # Kappa
        if "kappa" in parameters:
            output_parameters["kappa"] = f"ang. kappa={parameters['kappa']}"
        # Subjet z
        if "subjet_zr" in parameters:
            output_parameters["subjet_zr"] = f"Subjet r={parameters['subjet_zr']}"
        # Grooming
        # Soft Drop
        if "soft_drop" in parameters:
            # output_parameters["soft_drop"].extend(f"SD z_cut={param['zcut']}, beta={param['beta']}" for param in parameters["soft_drop"])
            param = parameters["soft_drop"]
            output_parameters["soft_drop"] = f"SD z_cut={param['zcut']}, beta={param['beta']}"
        # DyG
        if "dynamical_grooming" in parameters:
            # output_parameters["dynamical_grooming"].extend(f"DyG a={a}" for a in parameters["dynamical_grooming"])
            output_parameters["dynamical_grooming"] = f"DyG a={parameters['dynamical_grooming']}"

        return output_parameters


def _propagate_rest_of_parameters(
    output_parameters: dict[str, list[str]], parameters: dict[str, list[Any]]
) -> dict[str, list[str]]:
    # Ensure we keep the remaining keys!
    formatted_keys = list(output_parameters)
    for k, v in parameters.items():
        if k not in formatted_keys:
            output_parameters[k] = v
    return output_parameters


def write_observable_names_csv(observables: dict[str, dict[str, Observable]], stream: io.TextIO) -> bool:  # noqa: C901
    """Write the observables in a CSV format for help with organizing HEPdata table names.

    Args:
        observables: All observables
        stream: Output stream
    Returns:
        True if successful.
    """
    for sqrt_s in sorted(observables.keys()):
        output_line_base = f"{sqrt_s}"
        # Group by observable class
        class_to_obs: dict[str, list[Observable]] = {}
        for obs in observables[sqrt_s].values():
            class_to_obs.setdefault(obs.observable_class, []).append(obs)
        for obs_class_name, obs_class in class_to_obs.items():
            for obs in sorted(obs_class, key=lambda o: o.name):
                base_values = [
                    output_line_base,
                    obs_class_name,
                    obs.internal_name_without_experiment,
                    obs.display_name,
                    obs.experiment,
                ]
                columns_to_print_separately = ["centrality"]
                full_set_of_parameters = obs.parameters()
                for parameters, indices in obs.generate_parameters(full_set_of_parameters):
                    formatted_parameters = obs.format_parameters_for_printing(parameters)
                    values = [*base_values]
                    # Print these parameters separately
                    for c in columns_to_print_separately:
                        v = formatted_parameters.pop(c)
                        if v:
                            values.append(v)
                    # Give a heads up if there's nothing else to include.
                    if not formatted_parameters:
                        formatted_parameters = {"_": "None"}
                    # And then put the rest in the parameters field
                    values.append(", ".join(formatted_parameters.values()))
                    # Next, add in the responsible person field, which will always be empty
                    values.append("")
                    # Followed by the pp spectra histogram, if available
                    # NOTE: Since this is pp, we'll ignore many of the parameters, such as centrality.
                    pt_suffix = ""
                    if len(full_set_of_parameters.get("pt", {})) > 1:
                        pt_suffix = f"_pt{indices['pt']}"
                    logger.info(f"{pt_suffix=}")

                    # Suffix
                    suffix = ""
                    # TODO(RJE): Refactor from here..
                    # Jet R
                    if "jet_R" in parameters:
                        suffix = f"_R{parameters['jet_R']}"
                    # Grooming
                    # Soft drop
                    if "soft_drop" in parameters:
                        suffix += f"_zcut{parameters['soft_drop']['zcut']}_beta{parameters['soft_drop']['beta']}"
                    if "dynamical_grooming" in parameters:
                        suffix += f"_a{parameters['dynamical_grooming']}"
                    main_suffix = ""
                    # Jet-axis difference
                    if "axis" in parameters:
                        main_suffix += f"_{parameters['axis']['type']}"
                    # Generalized angularities
                    if "kappa" in parameters:
                        main_suffix += f"_k{parameters['kappa']}"
                    # Subjet z
                    if "r" in parameters:
                        main_suffix += f"_r{parameters['r']}"
                    logger.info(f"{main_suffix=}")
                    # ENDTODO

                    for system in ["pp", "AA"]:
                        # system dir name
                        value_to_store = f"Missing {system} dir"
                        for hepdata_dir_name in [
                            f"hepdata_{system}_dir{suffix}",
                            f"hepdata_{system}_dir{suffix}{pt_suffix}",
                            f"hepdata_{system}_dir",
                        ]:
                            # logger.info(f"{hepdata_dir_name=}")
                            if hepdata_dir_name in obs.config:
                                hepdata_dir = obs.config[hepdata_dir_name]
                                if isinstance(hepdata_dir, list):
                                    if indices["centrality"] >= len(hepdata_dir):
                                        value_to_store = "Cent index out of range"
                                    else:
                                        value_to_store = hepdata_dir[indices["centrality"]]
                                else:
                                    value_to_store = hepdata_dir
                        values.append(value_to_store)

                        # system graph name
                        value_to_store = f"Missing {system} graph name"
                        for hepdata_graph_name in [
                            f"hepdata_{system}_gname{suffix}",
                            f"hepdata_{system}_gname{suffix}{pt_suffix}",
                            f"hepdata_{system}_gname",
                        ]:
                            # First try the standard name
                            if hepdata_graph_name in obs.config:
                                hepdata_graph = obs.config[hepdata_graph_name]
                                if isinstance(hepdata_graph, list):
                                    if indices["centrality"] >= len(hepdata_graph):
                                        value_to_store = "Cent index out of range"
                                    else:
                                        value_to_store = hepdata_graph[indices["centrality"]]
                                else:
                                    value_to_store = hepdata_graph
                        values.append(value_to_store)
                    # AA spectra distribution
                    # system dir name
                    value_to_store = f"Missing {system} distribution dir"
                    for hepdata_dir_name in [
                        f"hepdata_{system}_distribution_dir{suffix}",
                        f"hepdata_{system}_distribution_dir{suffix}{pt_suffix}",
                        f"hepdata_{system}_distribution_dir",
                    ]:
                        # logger.info(f"{hepdata_dir_name=}")
                        if hepdata_dir_name in obs.config:
                            hepdata_dir = obs.config[hepdata_dir_name]
                            if isinstance(hepdata_dir, list):
                                if indices["centrality"] >= len(hepdata_dir):
                                    value_to_store = "Cent index out of range"
                                else:
                                    value_to_store = hepdata_dir[indices["centrality"]]
                            else:
                                value_to_store = hepdata_dir
                    values.append(value_to_store)

                    # system graph name
                    value_to_store = f"Missing {system} distribution graph name"
                    for hepdata_graph_name in [
                        f"hepdata_{system}_distribution_gname{suffix}",
                        f"hepdata_{system}_distribution_gname{suffix}{pt_suffix}",
                        f"hepdata_{system}_distribution_gname",
                    ]:
                        # First try the standard name
                        if hepdata_graph_name in obs.config:
                            hepdata_graph = obs.config[hepdata_graph_name]
                            if isinstance(hepdata_graph, list):
                                if indices["centrality"] >= len(hepdata_graph):
                                    value_to_store = "Cent index out of range"
                                else:
                                    value_to_store = hepdata_graph[indices["centrality"]]
                            else:
                                value_to_store = hepdata_graph
                    values.append(value_to_store)

                    stream.write("\t".join([*values, ""]) + "\n")


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
                observables[sqrt_s][f"{observable_class}_{observable_key}"] = Observable(
                    sqrt_s=sqrt_s,
                    observable_class=observable_class,
                    name=observable_key,
                    config=observable_info,
                )

    # import IPython; IPython.embed()
    here = Path(__file__).parent

    # PR list
    output_file = here / Path("observables.md")
    with output_file.open("w") as f:
        write_observables(observables=observables, stream=f)

    # List for help with assigning HEPdata tables to observables
    output_file_hepdata_list = here / Path("HEPdata_list.tsv")
    with output_file_hepdata_list.open("w") as f:
        f.write(
            "# sqrt_s\tobservable class\tobservable name\tdisplay name\texperiment\tcentrality\tparameters\tperson\tHEPdata pp spectra table\tHEPdata pp spectra entry\tHEPdata AA/pp ratio table\tHEPdata AA/pp ratio entry\tHEPdata AA spectra table\tHEPdata AA spectra table"
            + "\n"
        )
        # HEPdata pp spectra table	HEPdata pp spectra entry	HEPdata AA spectra table	HEPdata AA spectra entry	HEPdata AA/pp ratio table	HEPdata AA/pp ratio entry
        write_observable_names_csv(observables=observables, stream=f)


if __name__ == "__main__":
    import mammoth.helpers

    mammoth.helpers.setup_logging(level=logging.DEBUG)

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
