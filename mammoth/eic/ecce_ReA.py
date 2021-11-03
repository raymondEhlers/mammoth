#!/usr/bin/env python

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import attr
import cycler
import hist
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot as pb
import seaborn as sns
import uproot
from mammoth import helpers
from mammoth.eic import base as ecce_base
from mammoth.eic import ecce_ReA_implementation
from mammoth.eic.ecce_ReA_implementation import JetParameters
from pachyderm import binned_data

pb.configure()

logger = logging.getLogger(__name__)


@attr.s(frozen=True)
class InputSpec:
    n_PDF_name: str = attr.ib()
    n_variations: int = attr.ib()
    base_filename: str = attr.ib(default="output_JetObservables")

    @property
    def variations(self) -> Iterable[int]:
        return range(0, self.n_variations)

    @property
    def filename(self) -> Path:
        filename = self.base_filename
        if self.n_PDF_name != "ep":
            filename = f"{filename}_{self.n_PDF_name}"
        p = Path(filename).with_suffix(".root")
        return p


@attr.s
class SimulationConfig:
    dataset_spec: ecce_base.DatasetSpecPythia = attr.ib()
    input_specs: Sequence[InputSpec] = attr.ib()
    jet_algorithm: str = attr.ib()
    input_dir: Path = attr.ib()
    _output_dir: Path = attr.ib()

    def setup(self) -> bool:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        return True

    @property
    def output_dir(self) -> Path:
        return self._output_dir / f"{self.dataset_spec.electron_beam_energy}x{self.dataset_spec.proton_beam_energy}_{self.jet_algorithm}"


def _load_results(config: SimulationConfig, input_specs: Sequence[InputSpec]) -> Dict[str, Dict[str, hist.Hist]]:
    output_hists = {}
    for spec in input_specs:
        logger.info(f"Loading hists from {config.input_dir / spec.filename}")
        # Temprorarily only look at the main variation to avoid it taking forever to load.
        output_hists[spec.n_PDF_name] = ecce_base.load_hists(config.input_dir / spec.filename,
                                                             filters=[f"variation{i}" for i in spec.variations] if spec.n_PDF_name != "ep" else None)
        #output_hists[spec.n_PDF_name] = ecce_base.load_hists(config.input_dir / spec.filename)
        # Convert to hist.Hist
        for k, v in output_hists[spec.n_PDF_name].items():
            output_hists[spec.n_PDF_name][k] = v.to_hist()

    return output_hists


def _calculate_ReA(ep_hists: Dict[str, hist.Hist], eA_hists: Dict[str, hist.Hist], parameters: JetParameters, rebin_factor: int = 2) -> hist.Hist:
    ep_hist = binned_data.BinnedData.from_existing_data(ep_hists[parameters.name_ep])
    eA_hist = binned_data.BinnedData.from_existing_data(eA_hists[parameters.name_eA])
    #return hist.Hist((eA_hist / ep_hist).to_boost_histogram() * 1/79.0)[::hist.rebin(2)] / 2.0
    return hist.Hist((eA_hist / ep_hist).to_boost_histogram())[::hist.rebin(rebin_factor)] / (rebin_factor * 1.0)
    #return hist.Hist((eA_hist / ep_hist).to_boost_histogram())[::hist.rebin(5)] / 5.0


def calculate_ReA(input_hists: Dict[str, Dict[str, hist.Hist]],
                  sim_config: SimulationConfig,
                  analysis_config: ecce_ReA_implementation.AnalysisConfig,
                  ) -> Dict[JetParameters, hist.Hist]:
    ReA_hists = {}

    for input_spec in sim_config.input_specs:
        if input_spec.n_PDF_name == "ep":
            continue

        ReA_hists[input_spec.n_PDF_name] = {}
        for jet_R in analysis_config.jet_R_values:
            for jet_type in analysis_config.jet_types:
                for region in analysis_config.regions:
                    for variable in analysis_config.variables:
                        for variation in input_spec.variations:
                            parameters_spectra = JetParameters(jet_R=jet_R, jet_type=jet_type, region=region, observable="spectra", variable=variable, variation=variation, n_PDF_name=input_spec.n_PDF_name)
                            parameters_ReA = JetParameters(jet_R=jet_R, jet_type=jet_type, region=region, observable="ReA", variable=variable, variation=variation, n_PDF_name=input_spec.n_PDF_name)
                            ReA_hists[input_spec.n_PDF_name][parameters_ReA] = _calculate_ReA(
                                ep_hists=input_hists["ep"],
                                eA_hists=input_hists[input_spec.n_PDF_name],
                                parameters=parameters_spectra,
                            )

    return ReA_hists


def calculate_double_ratio(ReA_hists: Dict[str, Dict[str, hist.Hist]],
                           sim_config: SimulationConfig,
                           analysis_config: ecce_ReA_implementation.AnalysisConfig,
                           rebin_factor: int = 1,
                           ) -> Dict[JetParameters, hist.Hist]:
    double_ratio_hists = {}

    for input_spec in sim_config.input_specs:
        if input_spec.n_PDF_name == "ep":
            continue

        double_ratio_hists[input_spec.n_PDF_name] = {}
        for variable in analysis_config.variables:
            for jet_type in analysis_config.jet_types:
                for region in analysis_config.regions:
                    for variation in input_spec.variations:
                        # Retrieve the relevant hists
                        fixed_region_ReA_hists = {
                            k: v
                            for k, v in ReA_hists[input_spec.n_PDF_name].items() if k.region == region and k.jet_type == jet_type and k.variable == variable and k.variation == variation
                        }

                        # Then, find the ratio reference
                        for k, v in fixed_region_ReA_hists.items():
                            #logger.info(f"eta ranges: {_jet_eta_range(region=region, jet_R=k.jet_R_value)}")
                            if k.jet_R_value == 1.0:
                                ref = binned_data.BinnedData.from_existing_data(v) / _jet_eta_range(region=region, jet_R = k.jet_R_value)

                        # And finally, divide and store the relevant hists.
                        for k, v in fixed_region_ReA_hists.items():
                            if k.jet_R_value == 1.0:
                                continue
                            # hist doesn't divide hists properly, so go through binned_data
                            #logger.info(f"Storing double ratio for {str(k)}")
                            double_ratio_hists[input_spec.n_PDF_name][k] = hist.Hist(
                                (
                                    (binned_data.BinnedData.from_existing_data(v) / _jet_eta_range(region=region, jet_R=k.jet_R_value)) / ref
                                ).to_boost_histogram()[::hist.rebin(rebin_factor)] / (rebin_factor * 1.0))

    return double_ratio_hists

def _calculate_nominal_variations(variation_hists: Dict[str, Dict[str, hist.Hist]], nominal_hist: hist.Hist) -> bool:
    lower = np.zeros(len(nominal_hist.values()))
    upper = np.zeros(len(nominal_hist.values()))
    for k, v in variation_hists.items():
        difference = nominal_hist.values() - v.values()
        lower = np.minimum(lower, difference)
        upper = np.maximum(upper, difference)

    if not nominal_hist.metadata:
        nominal_hist.metadata = {}
    nominal_hist.metadata["n_PDF_lower"] = lower
    nominal_hist.metadata["n_PDF_upper"] = upper

    return True


_okabe_ito_colors = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    #"#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]

_jet_R_to_color_index = {0.3: 0, 0.5: 1, 0.8: 2, 1.0: 3}


def _plot_multiple_R(hists: Mapping[JetParameters, hist.Hist], is_ReA_related: bool, plot_config: pb.PlotConfig, output_dir: Path) -> None:
    #with sns.color_palette("Set2"):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_prop_cycle(cycler.cycler(color=_okabe_ito_colors))

    for k, v in hists.items():
        logger.info(f"plotting {k}")
        p = ax.errorbar(
            v.axes[0].centers,
            v.values(),
            xerr=v.axes[0].widths / 2,
            yerr=np.sqrt(v.variances()),
            linestyle="",
            label=f"$R = {round(int(k.jet_R) / 100, 2):01}$",
            marker="d",
            markersize=6,
            zorder=5,
        )
        if v.metadata and "n_PDF_lower" in v.metadata and "n_PDF_upper" in v.metadata:
            # Plot the error band
            ax.fill_between(
                v.axes[0].centers,
                # + because the values are negative for the lower
                v.values() + v.metadata["n_PDF_lower"],
                v.values() + v.metadata["n_PDF_upper"],
                color=p[0].get_color(),
                alpha=0.4,
                zorder=2,
                edgecolor="None",
            )

    if is_ReA_related:
        ax.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Labeling and presentation
    plot_config.apply(fig=fig, ax=ax)
    # A few additional tweaks.
    #ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    # ax_ratio.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.2))

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def _plot_n_PDF_variations(hists: Mapping[JetParameters, hist.Hist], is_ReA_related: bool, plot_config: pb.PlotConfig, output_dir: Path) -> None:
    #with sns.color_palette("Set2"):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_prop_cycle(cycler.cycler(color=_okabe_ito_colors))

    jet_R_values = set([k.jet_R_value for k in hists])
    labeled_jet_R = {j: False for j in jet_R_values}

    for k, v in hists.items():
        extra_kwargs = {}
        if not labeled_jet_R[k.jet_R_value]:
            labeled_jet_R[k.jet_R_value] = True
            extra_kwargs = {
                "label": f"$R = {round(int(k.jet_R) / 100, 2):01}$",
            }

        logger.info(f"plotting {k}")
        ax.plot(
            v.axes[0].centers,
            v.values(),
            linestyle="-",
            linewidth=2,
            color=_okabe_ito_colors[_jet_R_to_color_index[k.jet_R_value]],
            alpha=0.075,
            marker="",
            **extra_kwargs,
        )

    if is_ReA_related:
        ax.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Labeling and presentation
    plot_config.apply(fig=fig, ax=ax)
    # A few additional tweaks.
    #ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    # ax_ratio.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.2))

    # Ensure the legend is visible
    # See: https://stackoverflow.com/a/42403471/12907985
    for lh in ax.get_legend().legendHandles:
        lh.set_alpha(1)

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def _plot_true_vs_det_level_ReA(true_hists: Mapping[JetParameters, hist.Hist], det_hists: Mapping[JetParameters, hist.Hist], plot_config: pb.PlotConfig, output_dir: Path) -> None:
    #with sns.color_palette("Set2"):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_prop_cycle(cycler.cycler(color=_okabe_ito_colors))

    for hists in [true_hists, det_hists]:
        for k, v in hists.items():
            logger.info(f"plotting {k}")
            ax.errorbar(
                v.axes[0].centers,
                v.values(),
                xerr=v.axes[0].widths / 2,
                yerr=np.sqrt(v.variances()),
                linestyle="",
                #label=f"$R = {round(int(k.jet_R) / 100, 2):01}$",
                label=k.jet_type.replace("_", " "),
                marker="d",
                markersize=6,
            )

    ax.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Labeling and presentation
    plot_config.apply(fig=fig, ax=ax)
    # A few additional tweaks.
    #ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    # ax_ratio.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.2))

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


_regions_acceptance = {
    "forward": [1.5, 3.5],
    "mid_rapidity": [-1.5, 1.5],
    "backward": [-3.5, -1.5],
}


def _jet_eta_range(region: str, jet_R: float) -> float:
    low, high = _regions_acceptance[region]
    if region == "forward":
        high -= jet_R
    if region == "backward":
        low += jet_R
    return high - low


def dataset_spec_display_label(d: ecce_base.DatasetSpecPythia) -> str:
    return f"{d.generator.upper()} {d.electron_beam_energy}x{d.proton_beam_energy}, ${d.q2_display}$"


def plot_ReA(sim_config: SimulationConfig, analysis_config: ecce_ReA_implementation.AnalysisConfig, input_hists: Dict[str, Dict[str, hist.Hist]],
             cross_section: float, scale_jets_by_expected_luminosity: bool = False, expected_luminosities: Mapping[str, float] = None) -> None:
    scaled_hists = {}
    input_spectra_hists = input_hists
    if scale_jets_by_expected_luminosity:
        logger.info("Scaling jet spectra by expected luminosity")
        # Replaces the input spectra hists with the scaled hists
        input_spectra_hists = ecce_ReA_implementation.scale_jets(
            input_hists=input_hists,
            sim_config=sim_config,
            analysis_config=analysis_config,
            cross_section=cross_section, expected_luminosities=expected_luminosities,
        )

        #import IPython; IPython.start_ipython(user_ns={**globals(),**locals()})

    # Calculate ReA
    ReA_hists = calculate_ReA(
        input_hists=input_spectra_hists,
        sim_config=sim_config,
        analysis_config=analysis_config,
   )

    # Calculate ReA double ratio
    ReA_double_ratio_hists = calculate_double_ratio(
        ReA_hists=ReA_hists,
        sim_config=sim_config,
        analysis_config=analysis_config,
    )

    # First, print raw spectra. Print all R on the same figure
    for n_PDF_name in input_spectra_hists:
        for variable in analysis_config.variables:
            for jet_type in analysis_config.jet_types:
                for region in analysis_config.regions:
                    # TEMP
                    #continue
                    # ENDTEMP
                    # Spectra of fixed variable, jet type, and region, varying as a function of R
                    # Intentionally only look at the variation 0 case
                    spectra_hists = {}
                    for jet_R in analysis_config.jet_R_values:
                        _parameters_spectra = JetParameters(jet_R=jet_R, jet_type=jet_type, region=region, observable="spectra", variable=variable, variation=0, n_PDF_name=n_PDF_name)
                        spectra_hists[_parameters_spectra] = input_spectra_hists[n_PDF_name][_parameters_spectra.name_eA if n_PDF_name != "ep" else _parameters_spectra.name_ep]

                    variable_label = ""
                    x_range = (5, 50)
                    if variable == "pt":
                        variable_label = r"_{\text{T}}"
                        x_range = (5, 25)
                    text = "ECCE Simulation"
                    text += "\n" + dataset_spec_display_label(d=sim_config.dataset_spec)
                    text += "\n" + r"anti-$k_{\text{T}}$ jets"
                    if region == "forward":
                        text += "\n" + r"$1.5 < \eta < 3.5 - R$"
                    if region == "mid_rapidity":
                        text += "\n" + r"$-1.5 < \eta < 1.5$"
                    _plot_multiple_R(
                        hists=spectra_hists,
                        is_ReA_related=False,
                        plot_config=pb.PlotConfig(
                            name=n_PDF_name + "_" + next(iter(spectra_hists)).name_eA.replace("jetR030_", ""),
                            panels=pb.Panel(
                                    axes=[
                                        pb.AxisConfig("x", label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$", font_size=22, range=x_range),
                                        pb.AxisConfig(
                                            "y",
                                            label=r"$\frac{\text{d}^{2}\sigma}{\text{d}\eta\text{d}p" + variable_label + r"^{\text{jet}}}\:(\text{GeV}/c)$",
                                            log=True,
                                            font_size=22,
                                        ),
                                    ],
                                    text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                                    legend=pb.LegendConfig(location="lower left", font_size=22),
                                ),
                            figure=pb.Figure(edge_padding=dict(left=0.125, bottom=0.1)),
                        ),
                        output_dir=sim_config.output_dir,
                    )

    try:
        # Plot spectra variations
        for input_spec in sim_config.input_specs:
            if input_spec.n_variations > 1 and input_spec.n_PDF_name != "ep":
                for variable in analysis_config.variables:
                    for jet_type in analysis_config.jet_types:
                        for region in analysis_config.regions:
                            for jet_R in analysis_config.jet_R_values:
                                variation_hists = {}
                                for variation in input_spec.variations:
                                    _parameters_spectra = JetParameters(jet_R=jet_R, jet_type=jet_type, region=region,
                                                                        observable="spectra", variable=variable, variation=variation, n_PDF_name=input_spec.n_PDF_name)
                                    variation_hists[_parameters_spectra] = input_spectra_hists[input_spec.n_PDF_name][_parameters_spectra.name_eA if input_spec.n_PDF_name != "ep" else _parameters_spectra.name_ep]

                                variable_label = ""
                                x_range = (5, 50)
                                if variable == "pt":
                                    variable_label = r"_{\text{T}}"
                                    x_range = (5, 25)
                                text = "ECCE Simulation"
                                text += "\n" + dataset_spec_display_label(d=sim_config.dataset_spec)
                                text += "\n" + f"$R$ = {jet_R}" + r" anti-$k_{\text{T}}$ jets"
                                if region == "forward":
                                    text += "\n" + r"$1.5 < \eta < 3.5 - R$"
                                if region == "mid_rapidity":
                                    text += "\n" + r"$-1.5 < \eta < 1.5$"
                                variations_index = next(iter(variation_hists)).name_eA.find("_variation")
                                _plot_n_PDF_variations(
                                    hists=variation_hists,
                                    is_ReA_related=False,
                                    plot_config=pb.PlotConfig(
                                        # [:variations_index] removes the variations number, since we'll show all variations here
                                        name=input_spec.n_PDF_name + "_" + next(iter(variation_hists)).name_eA[:variations_index] + "_variations",
                                        panels=pb.Panel(
                                                axes=[
                                                    pb.AxisConfig("x", label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$", font_size=22, range=x_range),
                                                    pb.AxisConfig(
                                                        "y",
                                                        label=r"$\frac{\text{d}^{2}\sigma}{\text{d}\eta\text{d}p" + variable_label + r"^{\text{jet}}}\:(\text{GeV}/c)$",
                                                        log=True,
                                                        font_size=22,
                                                    ),
                                                ],
                                                text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                                                legend=pb.LegendConfig(location="lower left", font_size=22),
                                            ),
                                        figure=pb.Figure(edge_padding=dict(left=0.125, bottom=0.1)),
                                    ),
                                    output_dir=sim_config.output_dir,
                                )
    except Exception as e:
        logger.info(f"Plotting n_PDF_variations failed with {e}")
        import IPython; IPython.start_ipython(user_ns={**globals(),**locals()})

    # Calculate band for nominal variation
    try:
        for input_spec in sim_config.input_specs:
            if input_spec.n_variations > 1 and input_spec.n_PDF_name != "ep":
                for variable in analysis_config.variables:
                    for jet_type in analysis_config.jet_types:
                        for region in analysis_config.regions:
                            for jet_R in analysis_config.jet_R_values:
                                variation_hists = {}
                                nominal_hist = None
                                for variation in input_spec.variations:
                                    _parameters_ReA = JetParameters(jet_R=jet_R, jet_type=jet_type, region=region,
                                                                    observable="ReA", variable=variable, variation=variation, n_PDF_name=input_spec.n_PDF_name)
                                    variation_hists[_parameters_ReA] = ReA_hists[input_spec.n_PDF_name][_parameters_ReA]
                                    if variation == 0:
                                        nominal_hist = variation_hists[_parameters_ReA]

                                _calculate_nominal_variations(
                                    variation_hists=variation_hists,
                                    nominal_hist=nominal_hist,
                                )
                                #logger.info(f"nominal_hist.metadata: {nominal_hist.metadata}")
                                #_temp = JetParameters(jet_R=jet_R, jet_type=jet_type, region=region,
                                #                      observable="ReA", variable=variable, variation=0, n_PDF_name=input_spec.n_PDF_name)
                                #logger.info(f"nominal_hist in array.metadata: {variation_hists[_temp].metadata}")
    except Exception as e:
        logger.info(f"Error band calculation for ReA failed with {e}")
        import IPython; IPython.start_ipython(user_ns={**globals(),**locals()})

    for input_spec in sim_config.input_specs:
        if input_spec.n_PDF_name == "ep":
            continue

        for variable in analysis_config.variables:
            for jet_type in analysis_config.jet_types:
                for region in analysis_config.regions:
                    # ReA of fixed variable, jet type, and region, varying as a function of R
                    fixed_region_ReA_hists = {
                        k: v
                        for k, v in ReA_hists[input_spec.n_PDF_name].items() if k.region == region and k.jet_type == jet_type and k.variable == variable and k.variation == 0
                    }
                    variable_label = ""
                    x_range = (5, 50)
                    if variable == "pt":
                        variable_label = r"_{\text{T}}"
                        x_range = (5, 25)
                    text = "ECCE Simulation"
                    text += "\n" + dataset_spec_display_label(d=sim_config.dataset_spec)
                    text += "\n" + r"anti-$k_{\text{T}}$ jets"
                    if region == "forward":
                        text += "\n" + r"$1.5 < \eta < 3.5 - R$"
                    if region == "mid_rapidity":
                        text += "\n" + r"$-1.5 < \eta < 1.5$"
                    _plot_multiple_R(
                        hists=fixed_region_ReA_hists,
                        is_ReA_related=True,
                        plot_config=pb.PlotConfig(
                            name=input_spec.n_PDF_name + "_" + next(iter(fixed_region_ReA_hists)).name_eA.replace("jetR030_", ""),
                            panels=pb.Panel(
                                    axes=[
                                        pb.AxisConfig("x", label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$", font_size=22, range=x_range),
                                        pb.AxisConfig(
                                            "y",
                                            label=r"$R_{\text{eA}}$",
                                            range=(0, 1.4),
                                            font_size=22,
                                        ),
                                    ],
                                    text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                                    legend=pb.LegendConfig(location="lower left", font_size=22),
                                ),
                            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.1)),
                        ),
                        output_dir=sim_config.output_dir,
                    )

    try:
        # ReA variations
        for input_spec in sim_config.input_specs:
            if input_spec.n_variations > 1 and input_spec.n_PDF_name != "ep":
                for variable in analysis_config.variables:
                    for jet_type in analysis_config.jet_types:
                        for region in analysis_config.regions:
                            variation_hists = {}
                            for jet_R in analysis_config.jet_R_values:
                                for variation in input_spec.variations:
                                    _parameters_ReA = JetParameters(jet_R=jet_R, jet_type=jet_type, region=region,
                                                                    observable="ReA", variable=variable, variation=variation, n_PDF_name=input_spec.n_PDF_name)
                                    variation_hists[_parameters_ReA] = ReA_hists[input_spec.n_PDF_name][_parameters_ReA]

                            variable_label = ""
                            x_range = (5, 50)
                            if variable == "pt":
                                variable_label = r"_{\text{T}}"
                                x_range = (5, 25)
                            text = "ECCE Simulation"
                            text += "\n" + dataset_spec_display_label(d=sim_config.dataset_spec)
                            text += "\n" + r" anti-$k_{\text{T}}$ jets"
                            if region == "forward":
                                text += "\n" + r"$1.5 < \eta < 3.5 - R$"
                            if region == "mid_rapidity":
                                text += "\n" + r"$-1.5 < \eta < 1.5$"
                            variations_index = next(iter(fixed_region_ReA_hists)).name_eA.replace("jetR030_", "").find("_variation")
                            _plot_n_PDF_variations(
                                hists=variation_hists,
                                is_ReA_related=True,
                                plot_config=pb.PlotConfig(
                                    # [:variations_index] removes the variations number, since we'll show all variations here
                                    name=input_spec.n_PDF_name + "_" + next(iter(fixed_region_ReA_hists)).name_eA.replace("jetR030_", "")[:variations_index] + "_variations",
                                    panels=pb.Panel(
                                            axes=[
                                                pb.AxisConfig("x", label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$", font_size=22, range=x_range),
                                                pb.AxisConfig(
                                                    "y",
                                                    label=r"$R_{\text{eA}}$",
                                                    range=(0, 1.4),
                                                    font_size=22,
                                                ),
                                            ],
                                            text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                                            legend=pb.LegendConfig(location="lower left", font_size=22),
                                        ),
                                    figure=pb.Figure(edge_padding=dict(left=0.125, bottom=0.1)),
                                ),
                                output_dir=sim_config.output_dir,
                            )
    except Exception as e:
        logger.info(f"Plotting n_PDF_variations for ReA failed with {e}")
        import IPython; IPython.start_ipython(user_ns={**globals(),**locals()})

    # Calculate double ratio error band for nominal variation
    try:
        for input_spec in sim_config.input_specs:
            if input_spec.n_variations > 1 and input_spec.n_PDF_name != "ep":
                for variable in analysis_config.variables:
                    for jet_type in analysis_config.jet_types:
                        for region in analysis_config.regions:
                            # -1 to skip R = 1.0, which isn't valid for the ratio
                            for jet_R in analysis_config.jet_R_values[:-1]:
                                variation_hists = {}
                                nominal_hist = None
                                for variation in input_spec.variations:
                                    _parameters_ReA = JetParameters(jet_R=jet_R, jet_type=jet_type, region=region,
                                                                    observable="ReA", variable=variable, variation=variation, n_PDF_name=input_spec.n_PDF_name)
                                    variation_hists[_parameters_ReA] = ReA_double_ratio_hists[input_spec.n_PDF_name][_parameters_ReA]
                                    if variation == 0:
                                        nominal_hist = variation_hists[_parameters_ReA]

                                _calculate_nominal_variations(
                                    variation_hists=variation_hists,
                                    nominal_hist=nominal_hist,
                                )
                                #logger.info(f"nominal_hist.metadata: {nominal_hist.metadata}")
                                #_temp = JetParameters(jet_R=jet_R, jet_type=jet_type, region=region,
                                #                      observable="ReA", variable=variable, variation=0, n_PDF_name=input_spec.n_PDF_name)
                                #logger.info(f"nominal_hist in array.metadata: {variation_hists[_temp].metadata}")
    except Exception as e:
        logger.info(f"Error band calculation for ReA failed with {e}")
        import IPython; IPython.start_ipython(user_ns={**globals(),**locals()})

    # Plot double ratios
    for input_spec in sim_config.input_specs:
        if input_spec.n_PDF_name == "ep":
            continue

        for variable in analysis_config.variables:
            for jet_type in analysis_config.jet_types:
                for region in analysis_config.regions:
                    double_ratio_hists = {
                        k: v
                        for k, v in ReA_double_ratio_hists[input_spec.n_PDF_name].items() if k.region == region and k.jet_type == jet_type and k.variable == variable and k.variation == 0
                    }

                    variable_label = ""
                    x_range = (5, 50)
                    if variable == "pt":
                        variable_label = r"_{\text{T}}"
                        x_range = (5, 25)
                    text = "ECCE Simulation"
                    text += "\n" + dataset_spec_display_label(d=sim_config.dataset_spec)
                    text += "\n" + r"anti-$k_{\text{T}}$ jets"
                    if region == "forward":
                        text += "\n" + r"$1.5 < \eta < 3.5 - R$"
                    if region == "mid_rapidity":
                        text += "\n" + r"$-1.5 < \eta < 1.5$"

                    _plot_multiple_R(
                        hists=double_ratio_hists,
                        is_ReA_related=True,
                        plot_config=pb.PlotConfig(
                            name=input_spec.n_PDF_name + "_" + next(iter(double_ratio_hists)).name_eA.replace("jetR030_", "") + "_ratio",
                            panels=pb.Panel(
                                    axes=[
                                        pb.AxisConfig("x", label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$", font_size=22, range=x_range),
                                        pb.AxisConfig(
                                            "y",
                                            label=r"$R_{\text{eA}} / R_{\text{eA}}|_{R=1.0}$",
                                            range=(0, 1.4),
                                            font_size=22,
                                        ),
                                    ],
                                    text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                                    legend=pb.LegendConfig(location="lower left", font_size=22),
                                ),
                            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.1)),
                        ),
                        output_dir=sim_config.output_dir,
                    )

    try:
        # Double ratio variations
        for input_spec in sim_config.input_specs:
            if input_spec.n_variations > 1 and input_spec.n_PDF_name != "ep":
                for variable in analysis_config.variables:
                    for jet_type in analysis_config.jet_types:
                        for region in analysis_config.regions:
                            variation_hists = {}
                            # -1 to skip R = 1.0, which isn't valid for the ratio
                            for jet_R in analysis_config.jet_R_values[:-1]:
                                for variation in input_spec.variations:
                                    _parameters_ReA = JetParameters(jet_R=jet_R, jet_type=jet_type, region=region,
                                                                    observable="ReA", variable=variable, variation=variation, n_PDF_name=input_spec.n_PDF_name)
                                    variation_hists[_parameters_ReA] = ReA_double_ratio_hists[input_spec.n_PDF_name][_parameters_ReA]

                            variable_label = ""
                            x_range = (5, 50)
                            if variable == "pt":
                                variable_label = r"_{\text{T}}"
                                x_range = (5, 25)
                            text = "ECCE Simulation"
                            text += "\n" + dataset_spec_display_label(d=sim_config.dataset_spec)
                            text += "\n" + r"anti-$k_{\text{T}}$ jets"
                            if region == "forward":
                                text += "\n" + r"$1.5 < \eta < 3.5 - R$"
                            if region == "mid_rapidity":
                                text += "\n" + r"$-1.5 < \eta < 1.5$"
                            variations_index = next(iter(double_ratio_hists)).name_eA.replace("jetR030_", "").find("_variation")
                            _plot_n_PDF_variations(
                                hists=variation_hists,
                                is_ReA_related=True,
                                plot_config=pb.PlotConfig(
                                    # [:variations_index] removes the variations number, since we'll show all variations here
                                    name=input_spec.n_PDF_name + "_" + next(iter(double_ratio_hists)).name_eA.replace("jetR030_", "") + "_ratio_variations",
                                    panels=pb.Panel(
                                            axes=[
                                                pb.AxisConfig("x", label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$", font_size=22, range=x_range),
                                                pb.AxisConfig(
                                                    "y",
                                                    label=r"$R_{\text{eA}} / R_{\text{eA}}|_{R=1.0}$",
                                                    range=(0, 1.4),
                                                    font_size=22,
                                                ),
                                            ],
                                            text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                                            legend=pb.LegendConfig(location="lower left", font_size=22),
                                        ),
                                    figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.1)),
                                ),
                                output_dir=sim_config.output_dir,
                            )
    except Exception as e:
        logger.info(f"Plotting n_PDF_variations for double ratio failed with {e}")
        import IPython; IPython.start_ipython(user_ns={**globals(),**locals()})

    # Compare true vs det level to see the importance of unfolding
    for input_spec in sim_config.input_specs:
        if input_spec.n_PDF_name == "ep":
            continue

        for jet_type_true, jet_type_det in zip(["true_full", "true_charged"], ["calo", "charged"]):
            for variable in analysis_config.variables:
                for region in analysis_config.regions:
                    for jet_R in analysis_config.jet_R_values:
                        true_hists = {
                            k: v
                            for k, v in ReA_hists[input_spec.n_PDF_name].items() if k.region == region and k.jet_type == jet_type_true and k.variable == variable and k.jet_R_value == jet_R and k.variation == 0
                        }
                        det_hists = {
                            k: v
                            for k, v in ReA_hists[input_spec.n_PDF_name].items() if k.region == region and k.jet_type == jet_type_det and k.variable == variable and k.jet_R_value == jet_R and k.variation == 0
                        }

                        if not true_hists or not det_hists:
                            logger.info(f"Couldn't find any hists for {jet_type_true}, {jet_type_det}, {variable}, {region}, {jet_R} and variation 0. Continiuing")
                            continue

                        variable_label = ""
                        x_range = (5, 50)
                        if variable == "pt":
                            variable_label = r"_{\text{T}}"
                            x_range = (5, 25)
                        text = "ECCE Simulation"
                        text += "\n" + dataset_spec_display_label(d=sim_config.dataset_spec)
                        text += "\n" + r"anti-$k_{\text{T}}$ jets"
                        if region == "forward":
                            text += "\n" + r"$1.5 < \eta < 3.5 - R$"
                        if region == "mid_rapidity":
                            text += "\n" + r"$-1.5 < \eta < 1.5$"

                        _plot_true_vs_det_level_ReA(
                            true_hists=true_hists,
                            det_hists=det_hists,
                            plot_config=pb.PlotConfig(
                                #name=next(iter(fixed_region_ReA_hists)).name_eA.replace("jetR030_", "") + "_ratio",
                                name=input_spec.n_PDF_name + "_" + next(iter(true_hists)).name_eA + "_ReA_true_vs_det",
                                panels=pb.Panel(
                                        axes=[
                                            pb.AxisConfig("x", label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$", font_size=22, range=x_range),
                                            pb.AxisConfig(
                                                "y",
                                                label=r"$R_{\text{eA}}$",
                                                range=(0, 1.4),
                                                font_size=22,
                                            ),
                                        ],
                                        text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                                        legend=pb.LegendConfig(location="lower left", font_size=22),
                                    ),
                                figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.1)),
                            ),
                            output_dir=sim_config.output_dir,
                        )

    import IPython; IPython.start_ipython(user_ns={**globals(),**locals()})
    #import IPython; IPython.embed()


def run() -> None:
    helpers.setup_logging()

    #import warnings
    #warnings.filterwarnings("error")

    # Settings
    scale_jets_by_expected_luminosity = True
    analysis_config = ecce_ReA_implementation.AnalysisConfig(
        jet_R_values=[0.3, 0.5, 0.8, 1.0],
        #jet_types=["charged", "calo", "true_charged", "true_full"],
        # Full analysis
        #regions = ["forward", "backward", "mid_rapidity"],
        #variables = ["pt", "p"],
        # More minimal for speed + testing
        jet_types=["charged"],
        regions=["forward"],
        variables=["p"],
    )

    # Setup
    dataset_spec = ecce_base.DatasetSpecPythia(
        site="production",
        generator="pythia8",
        electron_beam_energy=10, proton_beam_energy=100,
        q2_selection=[100],
        label="",
    )
    # Setup I/O dirs
    #label = "fix_variable_shadowing"
    label = "min_p_cut_with_tracklets_EPPS"
    #label = "min_p_cut_with_tracklets_nNNPDF"
    base_dir = Path(f"/Volumes/data/eic/ReA/current_best_knowledge/{str(dataset_spec)}")
    input_dir = base_dir / label
    output_dir = base_dir / "plots" / label
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Fix scaling. It needs to scale the fractional error to get the right scaling, probably
    # TODO: Add the nPDF label to the plot since it seems to matter a good deal

    config = SimulationConfig(
        dataset_spec=dataset_spec,
        jet_algorithm="anti_kt",
        input_specs=[
            InputSpec("ep", n_variations=1),
            # EPPS
            # For testing
            InputSpec("EPPS16nlo_CT14nlo_Au197", n_variations=2),
            # Full set of variations
            #InputSpec("EPPS16nlo_CT14nlo_Au197", n_variations=97),
            # nNNPDF
            # For testing
            #InputSpec("nNNPDF20_nlo_as_0118_Au197", n_variations=1),
            #InputSpec("nNNPDF20_nlo_as_0118_Au197", n_variations=5),
            # Full set of variations
            #InputSpec("nNNPDF20_nlo_as_0118_Au197", n_variations=250),
        ],
        input_dir=input_dir,
        output_dir=output_dir,
    )
    config.setup()

    # Inputs
    # From the evaluator files, in pb (pythia provides in mb, but then it's change to pb during the conversion to HepMC2)
    _pb_to_fb = 1e3
    _cross_sections = {
        "production-pythia8-10x100-q2-100": 1322.52 * _pb_to_fb,
        "production-pythia8-10x100-q2-1-to-100": 470921.71 * _pb_to_fb,
    }
    # 1 year in fb^{-1}
    _luminosity_projections = {
        "ep": 10,
        # Scaling according to the recommendations
        "eA": 10 * 1.0/197,
    }

    logger.info(f"Analyzing {label}")
    input_hists = _load_results(
        config=config,
        input_specs=config.input_specs
    )

    plot_ReA(
        sim_config=config,
        analysis_config=analysis_config,
        input_hists=input_hists,
        cross_section=_cross_sections[str(dataset_spec)],
        expected_luminosities=_luminosity_projections,
        scale_jets_by_expected_luminosity=scale_jets_by_expected_luminosity,
    )


if __name__ == "__main__":
    run()
