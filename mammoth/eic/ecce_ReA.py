#!/usr/bin/env python

import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence

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
from pachyderm import binned_data

pb.configure()

logger = logging.getLogger(__name__)


@attr.s(frozen=True)
class InputSpec:
    n_PDF_name: str = attr.ib()
    base_filename: str = attr.ib(default="output_JetObservables")

    @property
    def filename(self) -> Path:
        filename = self.base_filename
        if self.n_PDF_name != "ep":
            filename = f"{filename}_{self.n_PDF_name}"
        p = Path(filename).with_suffix(".root")
        return p


@attr.s
class SimulationConfig:
    electron_beam_energy: int = attr.ib()
    proton_beam_energy: int = attr.ib()
    input_specs: Sequence[InputSpec] = attr.ib()
    jet_algorithm: str = attr.ib()
    input_dir: Path = attr.ib()
    _output_dir: Path = attr.ib()

    def setup(self) -> bool:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        return True

    @property
    def output_dir(self) -> Path:
        return self._output_dir / f"{self.electron_beam_energy}x{self.proton_beam_energy}_{self.jet_algorithm}"



def _load_results(config: SimulationConfig, input_specs: Sequence[InputSpec]) -> Dict[str, Dict[str, hist.Hist]]:
    output_hists = {}
    for spec in input_specs:
        logger.info(f"Loading hists from {config.input_dir / spec.filename}")
        output_hists[spec.n_PDF_name] = ecce_base.load_hists(config.input_dir / spec.filename)

    return output_hists


@attr.s(frozen=True)
class JetParameters:
    _jet_R: float = attr.ib()
    jet_type: str = attr.ib()
    region: str = attr.ib()
    observable: str = attr.ib()
    variable: str = attr.ib()
    n_PDF_name: str = attr.ib()

    @property
    def jet_R(self) -> str:
        return f"{round(self._jet_R * 100):03}"

    def name(self, n_PDF_label: str) -> str:
        return f"jetR{self.jet_R}_{self.jet_type}_{self.region}_{self.observable}_{self.variable}_{n_PDF_label}"

    @property
    def name_ep(self) -> str:
        return self.name(n_PDF_label="ep")

    @property
    def name_eA(self) -> str:
        return self.name(n_PDF_label="eA")

    def __str__(self) -> str:
        return self.name_eA


def _calculate_ReA(ep_hists: Dict[str, hist.Hist], eA_hists: Dict[str, hist.Hist], parameters: JetParameters) -> hist.Hist:
    ep_hist = binned_data.BinnedData.from_existing_data(ep_hists[parameters.name_ep])
    eA_hist = binned_data.BinnedData.from_existing_data(eA_hists[parameters.name_eA])
    return hist.Hist((eA_hist / ep_hist).to_boost_histogram() * 1/79.0)[::hist.rebin(2)] / 2.0
    #return hist.Hist((eA_hist / ep_hist).to_boost_histogram())[::hist.rebin(5)] / 5.0


def calculate_ReA(output_hists: Dict[str, Dict[str, hist.Hist]],
                  input_n_PDF_names: Sequence[str],
                  jet_R_values: Sequence[float],
                  jet_types: Sequence[str],
                  regions: Sequence[str],
                  variables: Sequence[str],
                  ) -> Dict[JetParameters, hist.Hist]:
    RAA_hists = {}
    n_PDF_names = [name for name in input_n_PDF_names if name != "ep"]

    for n_PDF_name in n_PDF_names:
        for jet_R in jet_R_values:
            for jet_type in jet_types:
                for region in regions:
                    for variable in variables:
                        parameters_spectra = JetParameters(jet_R=jet_R, jet_type=jet_type, region=region, observable="spectra", variable=variable, n_PDF_name=n_PDF_name)
                        parameters_RAA = JetParameters(jet_R=jet_R, jet_type=jet_type, region=region, observable="RAA", variable=variable, n_PDF_name=n_PDF_name)
                        RAA_hists[parameters_RAA] = _calculate_ReA(
                            ep_hists=output_hists["ep"],
                            eA_hists=output_hists[n_PDF_name],
                            parameters=parameters_spectra,
                        )

    return RAA_hists


_okabe_ito_colors = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]


def _plot_ReA_multiple_R(hists: Mapping[JetParameters, hist.Hist], plot_config: pb.PlotConfig, output_dir: Path) -> None:

    #with sns.color_palette("Set2"):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_prop_cycle(cycler.cycler(color=_okabe_ito_colors))

    for k, v in hists.items():
        logger.info(f"plotting {k}")
        ax.errorbar(
            v.axes[0].centers,
            v.values(),
            xerr=v.axes[0].widths / 2,
            yerr=np.sqrt(v.variances()),
            linestyle="",
            label=f"$R = {round(int(k.jet_R) / 100, 2):01}$",
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


def _plot_ReA_ratio_multiple_R(hists: Mapping[JetParameters, hist.Hist], plot_config: pb.PlotConfig, output_dir: Path) -> None:
    #with sns.color_palette("Set2"):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_prop_cycle(cycler.cycler(color=_okabe_ito_colors))

    for k, v in hists.items():
        logger.info(f"plotting {k}")
        ax.errorbar(
            v.axes[0].centers,
            v.values(),
            xerr=v.axes[0].widths / 2,
            yerr=np.sqrt(v.variances()),
            linestyle="",
            label=f"$R = {round(int(k.jet_R) / 100, 2):01}$",
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


def plot_ReA(config: SimulationConfig, output_hists: Dict[str, Dict[str, hist.Hist]]) -> None:

    jet_R_values = [0.3, 0.5, 0.8, 1.0]
    jet_types = ["charged", "calo", "true_charged", "true_full"]
    regions = ["forward", "backward", "mid_rapidity"]
    variables = ["pt", "p"]

    RAA_hists = calculate_ReA(
        output_hists=output_hists, input_n_PDF_names=[k for k in output_hists],
        jet_R_values=jet_R_values, jet_types=jet_types,
        regions=regions, variables=variables,
   )

    #for k, v in RAA_hists.items():
    # TODO: Fill in text...
    for variable in ["p", "pt"]:
        for jet_type in jet_types:
            for region in ["forward", "mid_rapidity"]:
                fixed_region_ReA_hists = {
                    k: v
                    for k, v in RAA_hists.items() if k.region == region and k.jet_type == jet_type and k.variable == variable
                }
                variable_label = ""
                if variable == "pt":
                    variable_label = r"_{\text{T}}"
                text = "ECCE Simulation"
                text += "\n" + "PYTHIA8 10x100, $Q^{2} > 100$"
                text += "\n" + r"anti-$k_{\text{T}}$ jets"
                if region == "forward":
                    text += "\n" + r"$1.5 < \eta < 3.5$"
                if region == "mid_rapidity":
                    text += "\n" + r"$-1.5 < \eta < 1.5$"
                _plot_ReA_multiple_R(
                    hists=fixed_region_ReA_hists,
                    plot_config=pb.PlotConfig(
                        name=next(iter(fixed_region_ReA_hists)).name_eA.replace("jetR030_", ""),
                        panels=pb.Panel(
                                axes=[
                                    pb.AxisConfig("x", label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$", font_size=22, range=(0, 50)),
                                    pb.AxisConfig(
                                        "y",
                                        label=r"$R_{\text{eA}}$",
                                        range=(0, 1.4),
                                        font_size=22,
                                    ),
                                ],
                                text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                                legend=pb.LegendConfig(location="center right", font_size=22),
                            ),
                        figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.1)),
                    ),
                    output_dir=config.output_dir,
                )

                # Calculate ratio
                for k, v in fixed_region_ReA_hists.items():
                    if k.jet_R == "100":
                        ref = binned_data.BinnedData.from_existing_data(v)

                fixed_region_ReA_ratio_hists = {}
                for k, v in fixed_region_ReA_hists.items():
                    if k.jet_R == "100":
                        continue
                    fixed_region_ReA_ratio_hists[k] = hist.Hist((binned_data.BinnedData.from_existing_data(v) / ref).to_boost_histogram()[::hist.rebin(2)] / 2.0)

                _plot_ReA_ratio_multiple_R(
                    hists=fixed_region_ReA_ratio_hists,
                    plot_config=pb.PlotConfig(
                        name=next(iter(fixed_region_ReA_hists)).name_eA.replace("jetR030_", "") + "_ratio",
                        panels=pb.Panel(
                                axes=[
                                    pb.AxisConfig("x", label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$", font_size=22, range=(0, 50)),
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
                    output_dir=config.output_dir,
                )

    import IPython; IPython.embed()


if __name__ == "__main__":
    helpers.setup_logging()

    # Setup
    electron_beam_energy = 10
    proton_beam_energy = 100
    production = "production-pythia8-10x100-q2-100"
    input_dir = Path(f"/Volumes/data/eic/ReA/2021-10-15/{production}")
    output_dir = Path(f"/Volumes/data/eic/ReA/2021-10-15/plots/{production}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(
        electron_beam_energy=electron_beam_energy,
        proton_beam_energy=proton_beam_energy,
        jet_algorithm="anti_kt",
        input_specs=[
            InputSpec("ep"),
            InputSpec("EPPS16nlo_CT14nlo_Au197")
        ],
        input_dir=input_dir,
        output_dir=output_dir,
    )
    config.setup()

    output_hists = _load_results(
        config=config,
        input_specs=config.input_specs
    )

    plot_ReA(
        config=config,
        output_hists=output_hists,
    )
