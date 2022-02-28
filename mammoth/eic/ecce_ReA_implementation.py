
import logging
from typing import Any, Dict, List, Mapping, Sequence

import attr
import hist
import numpy as np

logger = logging.getLogger(__name__)


@attr.define
class AnalysisConfig:
    jet_R_values: List[float]
    jet_types: List[str]
    regions: List[str]
    variables: List[str]


@attr.frozen
class JetParameters:
    _jet_R: float
    jet_type: str
    region: str
    observable: str
    variable: str
    variation: int
    n_PDF_name: str

    @property
    def jet_R(self) -> str:
        return f"{round(self._jet_R * 100):03}"

    @property
    def jet_R_value(self) -> float:
        return self._jet_R

    def name(self, n_PDF_label: str) -> str:
        return f"jetR{self.jet_R}_{self.jet_type}_{self.region}_{self.observable}_{self.variable}_{n_PDF_label}"

    @property
    def name_ep(self) -> str:
        return self.name(n_PDF_label="ep")

    @property
    def name_eA(self) -> str:
        return self.name(n_PDF_label=f"eA_variation{self.variation}")
        #return self.name(n_PDF_label=f"eA")

    def __str__(self) -> str:
        return self.name_eA



def scale_jets(input_hists: Dict[str, Dict[str, hist.Hist]],
               sim_config: Any,
               analysis_config: AnalysisConfig,
               cross_section: float,
               expected_luminosities: Mapping[str, float]) -> Dict[JetParameters, hist.Hist]:
    scaled_hists = {}

    # Supports both ep and eA
    for input_spec in sim_config.input_specs:
        # Define these for convenience so we don't have to mess with the loop variables too much
        hists = input_hists[input_spec.n_PDF_name]
        pdf_name = input_spec.n_PDF_name

        expected_luminosity = expected_luminosities[pdf_name if pdf_name == "ep" else "eA"]
        #scaled_hists[f"{pdf_name}_scaled"] = {}
        scaled_hists[f"{pdf_name}"] = {}

        for jet_R in analysis_config.jet_R_values:
            for jet_type in analysis_config.jet_types:
                for region in analysis_config.regions:
                    for variable in analysis_config.variables:
                        for variation in input_spec.variations:
                            parameters_spectra = JetParameters(
                                jet_R=jet_R, jet_type=jet_type, region=region, observable="spectra", variable=variable, variation=variation, n_PDF_name=pdf_name
                            )

                            # First, scale by the cross section, which we need to do in all cases
                            h_scaled = hists[parameters_spectra.name_eA if pdf_name != "ep" else parameters_spectra.name_ep] * cross_section

                            # Now we need to account for the projected luminosity.
                            # However, the overall relative scaling between the ep and eA will be messed up if we scale them directly
                            # by their projected luminosity. So what we want to do is reduce the size of the errors based on the projected
                            # luminosity while keeping the overall scale (values) fixed
                            # For all those complicated tests, it measures that we just need to scale the variance by 1 / expected_luminosity,
                            # which propagates to a 1/sqrt(expected_luminosity) in the error.
                            h_scaled.variances()[:] = h_scaled.variances() / expected_luminosity

                            # Store the new hist
                            scaled_hists[pdf_name][parameters_spectra.name_eA if pdf_name != "ep" else parameters_spectra.name_ep] = h_scaled

    return scaled_hists
