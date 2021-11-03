
import logging
from typing import Any, Dict, List, Mapping, Sequence

import attr
import hist
import numpy as np

logger = logging.getLogger(__name__)


@attr.s
class AnalysisConfig:
    jet_R_values: List[float] = attr.ib()
    jet_types: List[str] = attr.ib()
    regions: List[str] = attr.ib()
    variables: List[str] = attr.ib()


@attr.s(frozen=True)
class JetParameters:
    _jet_R: float = attr.ib()
    jet_type: str = attr.ib()
    region: str = attr.ib()
    observable: str = attr.ib()
    variable: str = attr.ib()
    variation: int = attr.ib()
    n_PDF_name: str = attr.ib()

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
    # Start with ep, so we can get the counts
    #counts_without_lumi = {}
    pdf_name = "ep"
    hists = input_hists["ep"]
    expected_luminosity = expected_luminosities["ep"]

    scaled_hists = {}
    #scaled_hists["ep_scaled"] = {}
    scaled_hists["ep"] = {}
    for jet_R in analysis_config.jet_R_values:
        for jet_type in analysis_config.jet_types:
            for region in analysis_config.regions:
                for variable in analysis_config.variables:
                    for variation in [0]:
                        parameters_spectra = JetParameters(jet_R=jet_R, jet_type=jet_type, region=region, observable="spectra", variable=variable, variation=variation, n_PDF_name=pdf_name)
                        h_temp = hists[parameters_spectra.name_ep]
                        h = h_temp * cross_section
                        #counts_without_lumi[parameters_spectra.name_ep] = h

                        # TODO: Describe... (old: Now, scale the ep errors by the increased number of counts)
                        h_scaled_ep = h_temp * cross_section
                        h_scaled_ep.variances()[:] = h_scaled_ep.variances() / expected_luminosity
                        #values = h_scaled_ep.values()
                        #h_scaled_ep.variances()[:] = np.divide(
                        #    h_scaled_ep.variances(),
                        #    ((values ** 2) * expected_luminosity),
                        #    out=np.zeros_like(values),
                        #    where=values!=0,
                        #)
                        #scaled_hists["ep_scaled"][parameters_spectra.name_ep] = h_scaled_ep
                        scaled_hists["ep"][parameters_spectra.name_ep] = h_scaled_ep

                        ## Now, scale the ep errors by the increased number of counts
                        #h_ep_scaled = h_temp * cross_section * expected_luminosity
                        #h_ep_scaled.values()[:] = h.values().copy()
                        ##scaled_hists["ep_scaled"][parameters_spectra.name_ep] = h_ep_scaled
                        #scaled_hists["ep"][parameters_spectra.name_ep] = h_ep_scaled

                        #c = h_ep_scaled.values()
                        #c *= expected_luminosity
                        #print(f"c: {c}")
                        #h_ep_scaled.variances()[:] = h_ep_scaled.variances() * expected_luminosity
                        #h_ep_scaled.variances()[:] = 1 / (np.sqrt(c) ** 2)
                        #h_ep_scaled.variances()[:] = 1 / (np.sqrt(c) ** 2)
                        #h_ep_scaled.variances()[:] = np.divide(1, np.sqrt(c) ** 2, out=np.zeros_like(c), where=b!=0)

    #for pdf_name, hists in input_hists.items():
    for input_spec in sim_config.input_specs:
        # Define these for convenience so we don't have to mess with the loop variables too much
        hists = input_hists[input_spec.n_PDF_name]
        pdf_name = input_spec.n_PDF_name

        if pdf_name == "ep":
            continue
        expected_luminosity = expected_luminosities["eA"]
        #scaled_hists[f"{pdf_name}_scaled"] = {}
        scaled_hists[f"{pdf_name}"] = {}

        for jet_R in analysis_config.jet_R_values:
            for jet_type in analysis_config.jet_types:
                for region in analysis_config.regions:
                    for variable in analysis_config.variables:
                        for variation in input_spec.variations:
                            parameters_spectra = JetParameters(jet_R=jet_R, jet_type=jet_type, region=region, observable="spectra", variable=variable, variation=variation, n_PDF_name=pdf_name
                            )

                            # First, scale by the cross section, which we need to do in all cases
                            h_scaled_eA = hists[parameters_spectra.name_eA] * cross_section

                            # Now we need to account for the projected luminosity.
                            # However, the overall relative scaling between the ep and eA will be messed up if we scale them directly
                            # by their projected luminosity. So what we want to do is reduce the size of the errors based on the projected
                            # luminosity while keeping the overall scale (values) fixed
                            # For all those complicated tests, it measures that we just need to scale the variance by 1 / expected_luminosity,
                            # which propagates to a 1/sqrt(expected_luminosity) in the error.
                            h_scaled_eA.variances()[:] = h_scaled_eA.variances() / expected_luminosity

                            # Store the new hist
                            scaled_hists[pdf_name][parameters_spectra.name_eA] = h_scaled_eA

    return scaled_hists
