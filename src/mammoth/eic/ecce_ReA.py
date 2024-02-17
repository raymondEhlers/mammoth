from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import attrs
import cycler
import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot as pb
from pachyderm import binned_data

from mammoth import helpers
from mammoth.eic import base as ecce_base
from mammoth.eic import ecce_ReA_implementation
from mammoth.eic.ecce_ReA_implementation import JetParameters

pb.configure()

logger = logging.getLogger(__name__)


@attrs.frozen
class InputSpec:
    n_PDF_name: str
    n_variations: int
    base_filename: str = attrs.field(default="output_JetObservables")

    @property
    def variations(self) -> Iterable[int]:
        return range(self.n_variations)

    @property
    def filename(self) -> Path:
        filename = self.base_filename
        if self.n_PDF_name != "ep":
            filename = f"{filename}_{self.n_PDF_name}"
        return Path(filename).with_suffix(".root")


_n_PDF_name_display_name = {
    "EPPS16nlo_CT14nlo_Au197": "EPPS16 NLO, CT14 NLO",
    "nNNPDF20_nlo_as_0118_Au197": "nNNPDF 2.0 NLO, NNPDF 3.1 NNLO",
}


@attrs.define
class SimulationConfig:
    dataset_spec: ecce_base.DatasetSpecPythia
    input_specs: Sequence[InputSpec]
    jet_algorithm: str
    input_dir: Path
    _output_dir: Path

    def setup(self) -> bool:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        return True

    @property
    def output_dir(self) -> Path:
        return (
            self._output_dir
            / f"{self.dataset_spec.electron_beam_energy}x{self.dataset_spec.proton_beam_energy}_{self.jet_algorithm}"
        )


def _load_results(config: SimulationConfig, input_specs: Sequence[InputSpec]) -> dict[str, dict[str, hist.Hist]]:
    output_hists = {}
    for spec in input_specs:
        logger.info(f"Loading hists from {config.input_dir / spec.filename}")
        output_hists[spec.n_PDF_name] = ecce_base.load_hists(
            config.input_dir / spec.filename,
            filters=[f"variation{i}" for i in spec.variations] if spec.n_PDF_name != "ep" else None,
            require_ends_with_in_filter=True,
        )
        # output_hists[spec.n_PDF_name] = ecce_base.load_hists(config.input_dir / spec.filename)
        # Convert to hist.Hist
        for k, v in output_hists[spec.n_PDF_name].items():
            output_hists[spec.n_PDF_name][k] = v.to_hist()  # type: ignore[attr-defined]

    return output_hists


def _calculate_ReA(
    ep_hists: dict[str, hist.Hist],
    eA_hists: dict[str, hist.Hist],
    parameters: JetParameters,
    narrow_rebin_factor: int = 2,
    wide_rebin_factor: int = 5,
    transition_for_binning: int = 10,
) -> hist.Hist:
    ep_hist = binned_data.BinnedData.from_existing_data(ep_hists[parameters.name_ep])
    eA_hist = binned_data.BinnedData.from_existing_data(eA_hists[parameters.name_eA])

    # We have our ReA, so ideally we'd be able to pass a variable binning. However, that doesn't exist yet,
    # so we a dumb thing and simple thing:
    # 1. rebin with two widths: narrow and wide
    # 2. merge the two histograms together at some bin, taking the narrow below and the wide above
    res = hist.Hist((eA_hist / ep_hist).to_boost_histogram())
    narrow_rebin = res[: complex(0, transition_for_binning) : hist.rebin(narrow_rebin_factor)] / (  # type: ignore[misc,operator]
        narrow_rebin_factor * 1.0
    )  # type: ignore[misc,operator]
    wide_rebin = res[complex(0, transition_for_binning) :: hist.rebin(wide_rebin_factor)] / (wide_rebin_factor * 1.0)  # type: ignore[misc,operator]

    bin_edges = np.concatenate([narrow_rebin.axes[0].edges, wide_rebin.axes[0].edges[1:]])  # type: ignore[union-attr]
    values = np.concatenate([narrow_rebin.values(), wide_rebin.values()])  # type: ignore[union-attr]
    variances = np.concatenate([narrow_rebin.variances(), wide_rebin.variances()])  # type: ignore[union-attr]

    combined = hist.Hist(
        binned_data.BinnedData(axes=[bin_edges], values=values, variances=variances).to_boost_histogram()
    )

    return combined  # noqa: RET504


def calculate_ReA(
    input_hists: dict[str, dict[str, hist.Hist]],
    sim_config: SimulationConfig,
    analysis_config: ecce_ReA_implementation.AnalysisConfig,
    narrow_rebin_factor: int = 2,
    wide_rebin_factor: int = 5,
) -> dict[str, dict[JetParameters, hist.Hist]]:
    ReA_hists: dict[str, dict[JetParameters, hist.Hist]] = {}

    for input_spec in sim_config.input_specs:
        if input_spec.n_PDF_name == "ep":
            continue

        ReA_hists[input_spec.n_PDF_name] = {}
        for jet_R in analysis_config.jet_R_values:
            for jet_type in analysis_config.jet_types:
                for region in analysis_config.regions:
                    for variable in analysis_config.variables:
                        for variation in input_spec.variations:
                            parameters_spectra = JetParameters(
                                jet_R=jet_R,
                                jet_type=jet_type,
                                region=region,
                                observable="spectra",
                                variable=variable,
                                variation=variation,
                                n_PDF_name=input_spec.n_PDF_name,
                            )
                            parameters_ReA = JetParameters(
                                jet_R=jet_R,
                                jet_type=jet_type,
                                region=region,
                                observable="ReA",
                                variable=variable,
                                variation=variation,
                                n_PDF_name=input_spec.n_PDF_name,
                            )
                            ReA_hists[input_spec.n_PDF_name][parameters_ReA] = _calculate_ReA(
                                ep_hists=input_hists["ep"],
                                eA_hists=input_hists[input_spec.n_PDF_name],
                                parameters=parameters_spectra,
                                narrow_rebin_factor=2 if variable == "pt" else narrow_rebin_factor,
                                wide_rebin_factor=5 if variable == "pt" else wide_rebin_factor,
                            )

    return ReA_hists


def calculate_double_ratio(
    ReA_hists: dict[str, dict[JetParameters, hist.Hist]],
    sim_config: SimulationConfig,
    analysis_config: ecce_ReA_implementation.AnalysisConfig,
    rebin_factor: int = 1,
) -> dict[str, dict[JetParameters, hist.Hist]]:
    double_ratio_hists: dict[str, dict[JetParameters, hist.Hist]] = {}

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
                            for k, v in ReA_hists[input_spec.n_PDF_name].items()
                            if k.variable == variable
                            and k.jet_type == jet_type
                            and k.region == region
                            and k.variation == variation
                        }

                        # Then, find the ratio reference
                        reference: binned_data.BinnedData | None = None
                        for k, v in fixed_region_ReA_hists.items():
                            # logger.info(f"eta ranges: {_jet_rapidity_range(region=region, jet_R=k.jet_R_value)}")
                            if k.jet_R_value == 1.0:
                                # NOTE: We don't want to normalize by the jet rapidity range because it's already divide out in the ReA
                                reference = binned_data.BinnedData.from_existing_data(v)

                        # Double check that we've found a reference for this case
                        assert reference is not None

                        # And finally, divide and store the relevant hists.
                        for k, v in fixed_region_ReA_hists.items():
                            if k.jet_R_value == 1.0:
                                continue
                            # hist doesn't divide hists properly, so go through binned_data
                            # logger.info(f"Storing double ratio for {str(k)}")
                            # NOTE: We don't want to normalize by the jet rapidity range because it's already divide out in the ReA
                            double_ratio_hists[input_spec.n_PDF_name][k] = hist.Hist(
                                ((binned_data.BinnedData.from_existing_data(v)) / reference).to_boost_histogram()[
                                    :: hist.rebin(rebin_factor)  # type: ignore[misc]
                                ]
                                / (rebin_factor * 1.0)
                            )  # type: ignore[misc]

                        # import IPython; IPython.start_ipython(user_ns={**globals(),**locals()})

    return double_ratio_hists


def _calculate_nominal_variations(variation_hists: dict[JetParameters, hist.Hist], nominal_hist: hist.Hist) -> bool:
    # Collect all of the differences from all of the variations
    differences_list = []
    for _k, v in variation_hists.items():
        differences_list.append(nominal_hist.values() - v.values())

    # transpose so that each row is now a single set of values
    differences = np.transpose(np.array(differences_list))
    # Sort for each p value
    differences = np.sort(differences, axis=1)
    # Could extract this many ways, but this works and is simple
    n_variations = differences.shape[1]
    # We want to remove the top and bottom 5% percent
    remove_up_to_this_index = round(n_variations / 20)
    # NOTE: If we end up rounding down to 0, it won't work. In that case,
    # there's nothing to be done.
    if remove_up_to_this_index != 0:
        # Remove the top and bottom 5%, and keep the rest
        differences = differences[:, remove_up_to_this_index:-remove_up_to_this_index]

    lower = np.min(differences, axis=1)
    upper = np.max(differences, axis=1)

    if not nominal_hist.metadata:
        nominal_hist.metadata = {}
    nominal_hist.metadata["n_PDF_lower"] = lower
    nominal_hist.metadata["n_PDF_upper"] = upper

    return True


_okabe_ito_colors = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    # "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]

_jet_R_to_color_index = {0.3: 0, 0.5: 1, 0.8: 2, 1.0: 3}


def _plot_spectra_2D(hist: hist.Hist, plot_config: pb.PlotConfig, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7.5))

    # Determine the normalization range
    z_axis_range = {
        # "vmin": h_proj.values[h_proj.values > 0].min(),
        "vmin": max(1e-4, hist.values()[hist.values() > 0].min()),
        "vmax": hist.values().max(),
        # "vmax": 1,
    }

    # Plot
    mesh = ax.pcolormesh(
        hist.axes[0].edges.T,
        hist.axes[1].edges.T,
        hist.values().T,
        norm=mpl.colors.LogNorm(**z_axis_range),
    )
    fig.colorbar(mesh, pad=0.02)

    # Labeling and presentation
    plot_config.apply(fig=fig, ax=ax)
    # A few additional tweaks.
    # ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    # ax_ratio.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.2))

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def _plot_multiple_R(
    hists: Mapping[JetParameters, hist.Hist], is_ReA_related: bool, plot_config: pb.PlotConfig, output_dir: Path
) -> None:
    # with sns.color_palette("Set2"):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_prop_cycle(cycler.cycler(color=_okabe_ito_colors))

    for k, v in hists.items():
        logger.info(f"plotting {k}")
        p = ax.errorbar(
            v.axes[0].centers,
            v.values(),
            xerr=v.axes[0].widths / 2,
            yerr=np.sqrt(v.variances()),  # type: ignore[arg-type]
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
    # ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    # ax_ratio.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.2))

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def _plot_n_PDF_variations(
    hists: Mapping[JetParameters, hist.Hist], is_ReA_related: bool, plot_config: pb.PlotConfig, output_dir: Path
) -> None:
    # with sns.color_palette("Set2"):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_prop_cycle(cycler.cycler(color=_okabe_ito_colors))

    jet_R_values = {k.jet_R_value for k in hists}
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
    # ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    # ax_ratio.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.2))

    # Ensure the legend is visible
    # See: https://stackoverflow.com/a/42403471/12907985
    for lh in ax.get_legend().legendHandles:
        lh.set_alpha(1)

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def _plot_true_vs_det_level_ReA(
    true_hists: Mapping[JetParameters, hist.Hist],
    det_hists: Mapping[JetParameters, hist.Hist],
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    # with sns.color_palette("Set2"):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_prop_cycle(cycler.cycler(color=_okabe_ito_colors))

    for hists in [true_hists, det_hists]:
        for k, v in hists.items():
            logger.info(f"plotting {k}")
            ax.errorbar(
                v.axes[0].centers,
                v.values(),
                xerr=v.axes[0].widths / 2,
                yerr=np.sqrt(v.variances()),  # type: ignore[arg-type]
                linestyle="",
                # label=f"$R = {round(int(k.jet_R) / 100, 2):01}$",
                label=k.jet_type.replace("_", " "),
                marker="d",
                markersize=6,
            )

    ax.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Labeling and presentation
    plot_config.apply(fig=fig, ax=ax)
    # A few additional tweaks.
    # ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    # ax_ratio.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.2))

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


_regions_acceptance = {
    "forward": [1.5, 3.5],
    "mid_rapidity": [-1.5, 1.5],
    "backward": [-3.5, -1.5],
}


def _jet_rapidity_range(region: str, jet_R: float) -> float:
    low, high = _regions_acceptance[region]
    if region == "forward":
        high -= jet_R
    if region == "backward":
        low += jet_R
    return high - low


def dataset_spec_display_label(d: ecce_base.DatasetSpecPythia) -> str:
    return f"{d.generator.upper()} {d.electron_beam_energy}x{d.proton_beam_energy}, ${d.q2_display}$"


def expected_luminosities_display_text(expected_luminosities: Mapping[str, float]) -> str:
    entries = []
    for n_PDF_name in ["ep", "eA"]:
        entries.append(
            rf"$\mathcal{{L}}^{{\text{{int}}}}_{{{n_PDF_name}}} = {round(expected_luminosities[n_PDF_name], 2)}\:\text{{fb}}^{{-1}}$"
        )
    return "Projected: " + ", ".join(entries)


_jet_type_display_label = {
    "charged": "charged-particle jets",
    "true_charged": "true charged-particle jets",
    "calo": "calorimeter jets",
    "true_full": "true jets",
}


def plot_ReA(  # noqa: C901
    sim_config: SimulationConfig,
    analysis_config: ecce_ReA_implementation.AnalysisConfig,
    input_hists: dict[str, dict[str, hist.Hist]],
    cross_section: float,
    scale_jets_by_expected_luminosity: bool = False,
    expected_luminosities: Mapping[str, float] | None = None,
    skip_slow_2D_plots: bool = False,
) -> None:
    input_spectra_hists = input_hists
    # Validation
    # Help out mypy
    assert expected_luminosities is not None
    if scale_jets_by_expected_luminosity:
        logger.info("Scaling jet spectra by expected luminosity")
        # Replaces the input spectra hists with the scaled hists
        input_spectra_hists = ecce_ReA_implementation.scale_jets(
            input_hists=input_hists,
            sim_config=sim_config,
            analysis_config=analysis_config,
            cross_section=cross_section,
            expected_luminosities=expected_luminosities,
        )

        # import IPython; IPython.start_ipython(user_ns={**globals(),**locals()})

    ###############################################
    # Basic QA: jet momentum as a function of x, Q2
    ###############################################
    # NOTE: We only plot the nominal variation. It's good enough
    if not skip_slow_2D_plots:
        for n_PDF_name in input_spectra_hists:
            for variable in analysis_config.variables:
                for jet_type in analysis_config.jet_types:
                    for region in analysis_config.regions:
                        for jet_R in analysis_config.jet_R_values:
                            # Q2 vs spectra of fixed variable, jet type, region, and R
                            _parameters_spectra = JetParameters(
                                # Have to hack the variable name here because I wasn't careful enough in the definition
                                jet_R=jet_R,
                                jet_type=jet_type,
                                region=region,
                                observable="spectra",
                                variable=f"{variable}_Q2",
                                variation=0,
                                n_PDF_name=n_PDF_name,
                            )
                            h = input_hists[n_PDF_name][
                                _parameters_spectra.name_eA if n_PDF_name != "ep" else _parameters_spectra.name_ep
                            ]

                            variable_label = ""
                            x_range = (0, 50)
                            if variable == "pt":
                                variable_label = r"_{\text{T}}"
                                x_range = (0, 25)
                            text = "ECCE Simulation"
                            text += "\n" + dataset_spec_display_label(d=sim_config.dataset_spec)
                            if n_PDF_name != "ep":
                                text += "\n" + _n_PDF_name_display_name[n_PDF_name]
                            text += (
                                "\n" + f"$R$ = {jet_R}" + r" anti-$k_{\text{T}}$ " + _jet_type_display_label[jet_type]
                            )
                            if region == "forward":
                                text += "\n" + r"$1.5 < y < 3.5 - R$"
                            if region == "mid_rapidity":
                                text += "\n" + r"$-1.5 < y < 1.5$"
                            name = _parameters_spectra.name_eA if n_PDF_name != "ep" else _parameters_spectra.name_ep
                            logger.info(f"Plotting {n_PDF_name}, {name} for Q2")
                            _plot_spectra_2D(
                                hist=h,
                                plot_config=pb.PlotConfig(
                                    name=f"{n_PDF_name}_{name}_Q2",
                                    panels=pb.Panel(
                                        axes=[
                                            pb.AxisConfig(
                                                "x",
                                                label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$",
                                                font_size=22,
                                                range=x_range,
                                            ),
                                            pb.AxisConfig(
                                                "y",
                                                label=r"$Q^{2}\:(\text{GeV}^{2})$",
                                                log=True,
                                                font_size=22,
                                                range=(45, 1100),
                                            ),
                                        ],
                                        text=pb.TextConfig(x=0.97, y=0.03, text=text, font_size=22),
                                        # legend=pb.LegendConfig(location="lower left", font_size=22),
                                    ),
                                    figure=pb.Figure(edge_padding={"left": 0.125, "bottom": 0.1}),
                                ),
                                output_dir=sim_config.output_dir,
                            )
                            # x vs spectra of fixed variable, jet type, region, and R
                            _parameters_spectra = JetParameters(
                                # Have to hack the variable name here because I wasn't careful enough in the definition
                                jet_R=jet_R,
                                jet_type=jet_type,
                                region=region,
                                observable="spectra",
                                variable=f"{variable}_x",
                                variation=0,
                                n_PDF_name=n_PDF_name,
                            )
                            h = input_hists[n_PDF_name][
                                _parameters_spectra.name_eA if n_PDF_name != "ep" else _parameters_spectra.name_ep
                            ]

                            variable_label = ""
                            x_range = (0, 50)
                            if variable == "pt":
                                variable_label = r"_{\text{T}}"
                                x_range = (0, 25)
                            text = "ECCE Simulation"
                            text += "\n" + dataset_spec_display_label(d=sim_config.dataset_spec)
                            if n_PDF_name != "ep":
                                text += "\n" + _n_PDF_name_display_name[n_PDF_name]
                            text += (
                                "\n" + f"$R$ = {jet_R}" + r" anti-$k_{\text{T}}$ " + _jet_type_display_label[jet_type]
                            )
                            if region == "forward":
                                text += "\n" + r"$1.5 < y < 3.5 - R$"
                            if region == "mid_rapidity":
                                text += "\n" + r"$-1.5 < y < 1.5$"
                            name = _parameters_spectra.name_eA if n_PDF_name != "ep" else _parameters_spectra.name_ep
                            logger.info(f"Plotting {n_PDF_name}, {name} for x")
                            _plot_spectra_2D(
                                hist=h,
                                plot_config=pb.PlotConfig(
                                    name=f"{n_PDF_name}_{name}_x",
                                    panels=pb.Panel(
                                        axes=[
                                            pb.AxisConfig(
                                                "x",
                                                label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$",
                                                font_size=22,
                                                range=x_range,
                                            ),
                                            pb.AxisConfig(
                                                "y",
                                                label=r"$x$",
                                                log=True,
                                                font_size=22,
                                                range=(8e-3, 1),
                                            ),
                                        ],
                                        text=pb.TextConfig(x=0.97, y=0.03, text=text, font_size=22),
                                        # legend=pb.LegendConfig(location="lower left", font_size=22),
                                    ),
                                    figure=pb.Figure(edge_padding={"left": 0.125, "bottom": 0.1}),
                                ),
                                output_dir=sim_config.output_dir,
                            )

    ##############################
    # Calculate derived quantities
    ##############################
    # Calculate ReA
    ReA_hists = calculate_ReA(
        input_hists=input_spectra_hists,
        sim_config=sim_config,
        analysis_config=analysis_config,
        narrow_rebin_factor=5,
        wide_rebin_factor=10,
    )
    # Calculate ReA double ratio
    ReA_double_ratio_hists = calculate_double_ratio(
        ReA_hists=ReA_hists,
        sim_config=sim_config,
        analysis_config=analysis_config,
    )

    ##########################################################
    # First, print raw spectra.
    ##########################################################
    # Plot all R on the same figure
    for n_PDF_name in input_spectra_hists:
        for variable in analysis_config.variables:
            for jet_type in analysis_config.jet_types:
                for region in analysis_config.regions:
                    # TEMP: Possibility to skip over this to save time
                    # continue
                    # ENDTEMP
                    # Spectra of fixed variable, jet type, and region, varying as a function of R
                    # Intentionally only look at the variation 0 case
                    spectra_hists = {}
                    for jet_R in analysis_config.jet_R_values:
                        _parameters_spectra = JetParameters(
                            jet_R=jet_R,
                            jet_type=jet_type,
                            region=region,
                            observable="spectra",
                            variable=variable,
                            variation=0,
                            n_PDF_name=n_PDF_name,
                        )
                        # NOTE: Since we're intentionally taking the unscaled hists regardless of whether
                        #       we enabled scaling. (we never want to scale up the values by luminosity because
                        #       then we'll end up on different scales in the ReA. This is why we only scale the errors).
                        #       So we need to scale here by cross section so we can talk about well defined spectra
                        spectra_hists[_parameters_spectra] = (
                            input_hists[n_PDF_name][
                                _parameters_spectra.name_eA if n_PDF_name != "ep" else _parameters_spectra.name_ep
                            ]
                            * cross_section
                        )

                    variable_label = ""
                    x_range = (5, 50)
                    if variable == "pt":
                        variable_label = r"_{\text{T}}"
                        x_range = (5, 25)
                    text = "ECCE Simulation"
                    text += "\n" + dataset_spec_display_label(d=sim_config.dataset_spec)
                    if n_PDF_name != "ep":
                        text += "\n" + _n_PDF_name_display_name[n_PDF_name]
                    text += "\n" + r"anti-$k_{\text{T}}$ " + _jet_type_display_label[jet_type]
                    if region == "forward":
                        text += "\n" + r"$1.5 < y < 3.5 - R$"
                    if region == "mid_rapidity":
                        text += "\n" + r"$-1.5 < y < 1.5$"
                    _plot_multiple_R(
                        hists=spectra_hists,
                        is_ReA_related=False,
                        plot_config=pb.PlotConfig(
                            name=n_PDF_name + "_" + next(iter(spectra_hists)).name_eA.replace("jetR030_", ""),
                            panels=pb.Panel(
                                axes=[
                                    pb.AxisConfig(
                                        "x",
                                        label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$",
                                        font_size=22,
                                        range=x_range,
                                    ),
                                    pb.AxisConfig(
                                        "y",
                                        label=r"$\frac{\text{d}^{2}\sigma}{\text{d}y\text{d}p"
                                        + variable_label
                                        + r"^{\text{jet}}}\:(\text{fb}\:c/\text{GeV})$",
                                        log=True,
                                        # Reduce this range for p to make to easier to see
                                        range=(1e8, 1e11) if variable == "p" else None,
                                        font_size=22,
                                    ),
                                ],
                                text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                                legend=pb.LegendConfig(location="lower left", font_size=22),
                            ),
                            figure=pb.Figure(edge_padding={"left": 0.125, "bottom": 0.1}),
                        ),
                        output_dir=sim_config.output_dir,
                    )

    ##########################################################
    # Plot spectra variations
    ##########################################################
    for input_spec in sim_config.input_specs:
        if input_spec.n_variations > 1 and input_spec.n_PDF_name != "ep":
            for variable in analysis_config.variables:
                for jet_type in analysis_config.jet_types:
                    for region in analysis_config.regions:
                        for jet_R in analysis_config.jet_R_values:
                            variation_hists = {}
                            for variation in input_spec.variations:
                                _parameters_spectra = JetParameters(
                                    jet_R=jet_R,
                                    jet_type=jet_type,
                                    region=region,
                                    observable="spectra",
                                    variable=variable,
                                    variation=variation,
                                    n_PDF_name=input_spec.n_PDF_name,
                                )
                                # NOTE: Since we're intentionally taking the unscaled hists regardless of whether
                                #       we enabled scaling. (we never want to scale up the values by luminosity because
                                #       then we'll end up on different scales in the ReA. This is why we only scale the errors).
                                #       So we need to scale here by cross section so we can talk about well defined spectra
                                variation_hists[_parameters_spectra] = (
                                    input_hists[input_spec.n_PDF_name][
                                        _parameters_spectra.name_eA
                                        if input_spec.n_PDF_name != "ep"
                                        else _parameters_spectra.name_ep
                                    ]
                                    * cross_section
                                )

                            variable_label = ""
                            x_range = (5, 50)
                            if variable == "pt":
                                variable_label = r"_{\text{T}}"
                                x_range = (5, 25)
                            text = "ECCE Simulation"
                            text += "\n" + dataset_spec_display_label(d=sim_config.dataset_spec)
                            if input_spec.n_PDF_name != "ep":
                                text += "\n" + _n_PDF_name_display_name[n_PDF_name]
                            text += (
                                "\n" + f"$R$ = {jet_R}" + r" anti-$k_{\text{T}}$ " + _jet_type_display_label[jet_type]
                            )
                            if region == "forward":
                                text += "\n" + r"$1.5 < y < 3.5 - R$"
                            if region == "mid_rapidity":
                                text += "\n" + r"$-1.5 < y < 1.5$"
                            variations_index = next(iter(variation_hists)).name_eA.find("_variation")
                            _plot_n_PDF_variations(
                                hists=variation_hists,
                                is_ReA_related=False,
                                plot_config=pb.PlotConfig(
                                    # [:variations_index] removes the variations number, since we'll show all variations here
                                    name=input_spec.n_PDF_name
                                    + "_"
                                    + next(iter(variation_hists)).name_eA[:variations_index]
                                    + "_variations",
                                    panels=pb.Panel(
                                        axes=[
                                            pb.AxisConfig(
                                                "x",
                                                label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$",
                                                font_size=22,
                                                range=x_range,
                                            ),
                                            pb.AxisConfig(
                                                "y",
                                                label=r"$\frac{\text{d}^{2}\sigma}{\text{d}y\text{d}p"
                                                + variable_label
                                                + r"^{\text{jet}}}\:(\text{fb}\:c/\text{GeV})$",
                                                log=True,
                                                # Reduce this range for p to make to easier to see
                                                range=(1e8, 1e11) if variable == "p" else None,
                                                font_size=22,
                                            ),
                                        ],
                                        text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                                        legend=pb.LegendConfig(location="lower left", font_size=22),
                                    ),
                                    figure=pb.Figure(edge_padding={"left": 0.125, "bottom": 0.1}),
                                ),
                                output_dir=sim_config.output_dir,
                            )

    #########################################
    # Calculate band for PDF variation of ReA
    #########################################
    # NOTE: The results are stored in the metadata of the nominal variation
    for input_spec in sim_config.input_specs:
        if input_spec.n_variations > 1 and input_spec.n_PDF_name != "ep":
            for variable in analysis_config.variables:
                for jet_type in analysis_config.jet_types:
                    for region in analysis_config.regions:
                        for jet_R in analysis_config.jet_R_values:
                            variation_hists = {}
                            nominal_hist = None
                            for variation in input_spec.variations:
                                _parameters_ReA = JetParameters(
                                    jet_R=jet_R,
                                    jet_type=jet_type,
                                    region=region,
                                    observable="ReA",
                                    variable=variable,
                                    variation=variation,
                                    n_PDF_name=input_spec.n_PDF_name,
                                )
                                variation_hists[_parameters_ReA] = ReA_hists[input_spec.n_PDF_name][_parameters_ReA]
                                if variation == 0:
                                    nominal_hist = variation_hists[_parameters_ReA]

                            # Help out mypy...
                            assert nominal_hist is not None

                            _calculate_nominal_variations(
                                variation_hists=variation_hists,
                                nominal_hist=nominal_hist,
                            )

    ####################################
    # Plot ReA for nominal PDF variation
    ####################################
    # Multiple R are plotted on a single figure
    for input_spec in sim_config.input_specs:
        if input_spec.n_PDF_name == "ep":
            continue

        for variable in analysis_config.variables:
            for jet_type in analysis_config.jet_types:
                for region in analysis_config.regions:
                    # ReA of fixed variable, jet type, and region, varying as a function of R
                    fixed_region_ReA_hists = {
                        k: v
                        for k, v in ReA_hists[input_spec.n_PDF_name].items()
                        if k.region == region and k.jet_type == jet_type and k.variable == variable and k.variation == 0
                    }
                    variable_label = ""
                    x_range = (5, 50)
                    if variable == "pt":
                        variable_label = r"_{\text{T}}"
                        x_range = (5, 25)
                    text = "ECCE Simulation"
                    text += "\n" + dataset_spec_display_label(d=sim_config.dataset_spec)
                    if input_spec.n_PDF_name != "ep":
                        text += "\n" + _n_PDF_name_display_name[n_PDF_name]
                    text += "\n" + r"anti-$k_{\text{T}}$ " + _jet_type_display_label[jet_type]
                    if region == "forward":
                        text += "\n" + r"$1.5 < y < 3.5 - R$"
                    if region == "mid_rapidity":
                        text += "\n" + r"$-1.5 < y < 1.5$"
                    _plot_multiple_R(
                        hists=fixed_region_ReA_hists,
                        is_ReA_related=True,
                        plot_config=pb.PlotConfig(
                            name=input_spec.n_PDF_name
                            + "_"
                            + next(iter(fixed_region_ReA_hists)).name_eA.replace("jetR030_", ""),
                            panels=pb.Panel(
                                axes=[
                                    pb.AxisConfig(
                                        "x",
                                        label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$",
                                        font_size=22,
                                        range=x_range,
                                    ),
                                    pb.AxisConfig(
                                        "y",
                                        label=r"$R_{\text{eA}}$",
                                        range=(0, 1.4),
                                        font_size=22,
                                    ),
                                ],
                                text=[
                                    pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                                    pb.TextConfig(
                                        x=0.97,
                                        y=0.03,
                                        text=expected_luminosities_display_text(expected_luminosities),
                                        font_size=22,
                                    ),
                                ],
                                legend=pb.LegendConfig(location="lower left", font_size=22),
                            ),
                            figure=pb.Figure(edge_padding={"left": 0.12, "bottom": 0.1}),
                        ),
                        output_dir=sim_config.output_dir,
                    )

    #############################################
    # Plot ReA variations
    #############################################
    # Again, plotting multiple R on a single figure
    for input_spec in sim_config.input_specs:
        if input_spec.n_variations > 1 and input_spec.n_PDF_name != "ep":
            for variable in analysis_config.variables:
                for jet_type in analysis_config.jet_types:
                    for region in analysis_config.regions:
                        variation_hists = {}
                        for jet_R in analysis_config.jet_R_values:
                            for variation in input_spec.variations:
                                _parameters_ReA = JetParameters(
                                    jet_R=jet_R,
                                    jet_type=jet_type,
                                    region=region,
                                    observable="ReA",
                                    variable=variable,
                                    variation=variation,
                                    n_PDF_name=input_spec.n_PDF_name,
                                )
                                variation_hists[_parameters_ReA] = ReA_hists[input_spec.n_PDF_name][_parameters_ReA]

                        variable_label = ""
                        x_range = (5, 50)
                        if variable == "pt":
                            variable_label = r"_{\text{T}}"
                            x_range = (5, 25)
                        text = "ECCE Simulation"
                        text += "\n" + dataset_spec_display_label(d=sim_config.dataset_spec)
                        if input_spec.n_PDF_name != "ep":
                            text += "\n" + _n_PDF_name_display_name[n_PDF_name]
                        text += "\n" + r"anti-$k_{\text{T}}$ " + _jet_type_display_label[jet_type]
                        if region == "forward":
                            text += "\n" + r"$1.5 < y < 3.5 - R$"
                        if region == "mid_rapidity":
                            text += "\n" + r"$-1.5 < y < 1.5$"
                        variations_index = (
                            next(iter(variation_hists)).name_eA.replace("jetR030_", "").find("_variation")
                        )
                        _plot_n_PDF_variations(
                            hists=variation_hists,
                            is_ReA_related=True,
                            plot_config=pb.PlotConfig(
                                # [:variations_index] removes the variations number, since we'll show all variations here
                                name=input_spec.n_PDF_name
                                + "_"
                                + next(iter(variation_hists)).name_eA.replace("jetR030_", "")[:variations_index]
                                + "_variations",
                                panels=pb.Panel(
                                    axes=[
                                        pb.AxisConfig(
                                            "x",
                                            label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$",
                                            font_size=22,
                                            range=x_range,
                                        ),
                                        pb.AxisConfig(
                                            "y",
                                            label=r"$R_{\text{eA}}$",
                                            range=(0, 1.4),
                                            font_size=22,
                                        ),
                                    ],
                                    text=[
                                        pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                                        pb.TextConfig(
                                            x=0.97,
                                            y=0.03,
                                            text=expected_luminosities_display_text(expected_luminosities),
                                            font_size=22,
                                        ),
                                    ],
                                    legend=pb.LegendConfig(location="lower left", font_size=22),
                                ),
                                figure=pb.Figure(edge_padding={"left": 0.125, "bottom": 0.1}),
                            ),
                            output_dir=sim_config.output_dir,
                        )

    ######################################################
    # Calculate double ratio error band for PDF variations
    ######################################################
    # NOTE: The results are stored in the metadata of the nominal variation
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
                                _parameters_ReA = JetParameters(
                                    jet_R=jet_R,
                                    jet_type=jet_type,
                                    region=region,
                                    observable="ReA",
                                    variable=variable,
                                    variation=variation,
                                    n_PDF_name=input_spec.n_PDF_name,
                                )
                                variation_hists[_parameters_ReA] = ReA_double_ratio_hists[input_spec.n_PDF_name][
                                    _parameters_ReA
                                ]
                                if variation == 0:
                                    nominal_hist = variation_hists[_parameters_ReA]

                            # Help out mypy...
                            assert nominal_hist is not None

                            _calculate_nominal_variations(
                                variation_hists=variation_hists,
                                nominal_hist=nominal_hist,
                            )
                            # logger.info(f"nominal_hist.metadata: {nominal_hist.metadata}")
                            # _temp = JetParameters(jet_R=jet_R, jet_type=jet_type, region=region,
                            #                      observable="ReA", variable=variable, variation=0, n_PDF_name=input_spec.n_PDF_name)
                            # logger.info(f"nominal_hist in array.metadata: {variation_hists[_temp].metadata}")

    ######################################################
    # Plot double ratios
    ######################################################
    for input_spec in sim_config.input_specs:
        if input_spec.n_PDF_name == "ep":
            continue

        for variable in analysis_config.variables:
            for jet_type in analysis_config.jet_types:
                for region in analysis_config.regions:
                    double_ratio_hists = {
                        k: v
                        for k, v in ReA_double_ratio_hists[input_spec.n_PDF_name].items()
                        if k.region == region and k.jet_type == jet_type and k.variable == variable and k.variation == 0
                    }

                    variable_label = ""
                    x_range = (5, 50)
                    if variable == "pt":
                        variable_label = r"_{\text{T}}"
                        x_range = (5, 25)
                    text = "ECCE Simulation"
                    text += "\n" + dataset_spec_display_label(d=sim_config.dataset_spec)
                    if input_spec.n_PDF_name != "ep":
                        text += "\n" + _n_PDF_name_display_name[n_PDF_name]
                    text += "\n" + r"anti-$k_{\text{T}}$ " + _jet_type_display_label[jet_type]
                    if region == "forward":
                        text += "\n" + r"$1.5 < y < 3.5 - R$"
                    if region == "mid_rapidity":
                        text += "\n" + r"$-1.5 < y < 1.5$"

                    _plot_multiple_R(
                        hists=double_ratio_hists,
                        is_ReA_related=True,
                        plot_config=pb.PlotConfig(
                            name=input_spec.n_PDF_name
                            + "_"
                            + next(iter(double_ratio_hists)).name_eA.replace("jetR030_", "")
                            + "_ratio",
                            panels=pb.Panel(
                                axes=[
                                    pb.AxisConfig(
                                        "x",
                                        label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$",
                                        font_size=22,
                                        range=x_range,
                                    ),
                                    pb.AxisConfig(
                                        "y",
                                        label=r"$R_{\text{eA}}(R) / R_{\text{eA}}(R=1.0)$",
                                        range=(0.5, 1.4),
                                        font_size=22,
                                    ),
                                ],
                                text=[
                                    pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                                    pb.TextConfig(
                                        x=0.97,
                                        y=0.03,
                                        text=expected_luminosities_display_text(expected_luminosities),
                                        font_size=22,
                                    ),
                                ],
                                legend=pb.LegendConfig(location="lower left", font_size=22),
                            ),
                            figure=pb.Figure(edge_padding={"left": 0.12, "bottom": 0.1}),
                        ),
                        output_dir=sim_config.output_dir,
                    )

    ######################################################
    # Plot double ratio nPDF variations
    ######################################################
    try:
        for input_spec in sim_config.input_specs:
            if input_spec.n_variations > 1 and input_spec.n_PDF_name != "ep":
                for variable in analysis_config.variables:
                    for jet_type in analysis_config.jet_types:
                        for region in analysis_config.regions:
                            variation_hists = {}
                            # -1 to skip R = 1.0, which isn't valid for the ratio
                            for jet_R in analysis_config.jet_R_values[:-1]:
                                for variation in input_spec.variations:
                                    _parameters_ReA = JetParameters(
                                        jet_R=jet_R,
                                        jet_type=jet_type,
                                        region=region,
                                        observable="ReA",
                                        variable=variable,
                                        variation=variation,
                                        n_PDF_name=input_spec.n_PDF_name,
                                    )
                                    variation_hists[_parameters_ReA] = ReA_double_ratio_hists[input_spec.n_PDF_name][
                                        _parameters_ReA
                                    ]

                            variable_label = ""
                            x_range = (5, 50)
                            if variable == "pt":
                                variable_label = r"_{\text{T}}"
                                x_range = (5, 25)
                            text = "ECCE Simulation"
                            text += "\n" + dataset_spec_display_label(d=sim_config.dataset_spec)
                            if input_spec.n_PDF_name != "ep":
                                text += "\n" + _n_PDF_name_display_name[n_PDF_name]
                            text += "\n" + r"anti-$k_{\text{T}}$ " + _jet_type_display_label[jet_type]
                            if region == "forward":
                                text += "\n" + r"$1.5 < y < 3.5 - R$"
                            if region == "mid_rapidity":
                                text += "\n" + r"$-1.5 < y < 1.5$"
                            variations_index = (
                                next(iter(variation_hists)).name_eA.replace("jetR030_", "").find("_variation")
                            )
                            _plot_n_PDF_variations(
                                hists=variation_hists,
                                is_ReA_related=True,
                                plot_config=pb.PlotConfig(
                                    # [:variations_index] removes the variations number, since we'll show all variations here
                                    name=input_spec.n_PDF_name
                                    + "_"
                                    + next(iter(variation_hists)).name_eA.replace("jetR030_", "")
                                    + "_ratio_variations",
                                    panels=pb.Panel(
                                        axes=[
                                            pb.AxisConfig(
                                                "x",
                                                label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$",
                                                font_size=22,
                                                range=x_range,
                                            ),
                                            pb.AxisConfig(
                                                "y",
                                                label=r"$R_{\text{eA}}(R) / R_{\text{eA}}(R=1.0)$",
                                                range=(0, 1.4),
                                                font_size=22,
                                            ),
                                        ],
                                        text=[
                                            pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                                            pb.TextConfig(
                                                x=0.97,
                                                y=0.03,
                                                text=expected_luminosities_display_text(expected_luminosities),
                                                font_size=22,
                                            ),
                                        ],
                                        legend=pb.LegendConfig(location="lower left", font_size=22),
                                    ),
                                    figure=pb.Figure(edge_padding={"left": 0.12, "bottom": 0.1}),
                                ),
                                output_dir=sim_config.output_dir,
                            )
    except Exception as e:
        logger.info(f"Plotting n_PDF_variations for double ratio failed with {e}")
        import IPython

        IPython.start_ipython(user_ns={**globals(), **locals()})  # type: ignore[no-untyped-call]

    ######################################################
    # Compare true vs det level to see the importance of unfolding
    ######################################################
    for input_spec in sim_config.input_specs:
        if input_spec.n_PDF_name == "ep":
            continue

        for jet_type_true, jet_type_det in [("true_full", "calo"), ("true_charged", "charged")]:
            for variable in analysis_config.variables:
                for region in analysis_config.regions:
                    for jet_R in analysis_config.jet_R_values:
                        true_hists = {
                            k: v
                            for k, v in ReA_hists[input_spec.n_PDF_name].items()
                            if k.region == region
                            and k.jet_type == jet_type_true
                            and k.variable == variable
                            and k.jet_R_value == jet_R
                            and k.variation == 0
                        }
                        det_hists = {
                            k: v
                            for k, v in ReA_hists[input_spec.n_PDF_name].items()
                            if k.region == region
                            and k.jet_type == jet_type_det
                            and k.variable == variable
                            and k.jet_R_value == jet_R
                            and k.variation == 0
                        }

                        if not true_hists or not det_hists:
                            logger.info(
                                f"Couldn't find any hists for {jet_type_true}, {jet_type_det}, {variable}, {region}, {jet_R} and variation 0. Continuing"
                            )
                            continue

                        variable_label = ""
                        x_range = (5, 50)
                        if variable == "pt":
                            variable_label = r"_{\text{T}}"
                            x_range = (5, 25)
                        text = "ECCE Simulation"
                        text += "\n" + dataset_spec_display_label(d=sim_config.dataset_spec)
                        if input_spec.n_PDF_name != "ep":
                            text += "\n" + _n_PDF_name_display_name[n_PDF_name]
                        text += "\n" + r"anti-$k_{\text{T}}$ " + _jet_type_display_label[jet_type]
                        if region == "forward":
                            text += "\n" + r"$1.5 < y < 3.5 - R$"
                        if region == "mid_rapidity":
                            text += "\n" + r"$-1.5 < y < 1.5$"

                        _plot_true_vs_det_level_ReA(
                            true_hists=true_hists,
                            det_hists=det_hists,
                            plot_config=pb.PlotConfig(
                                # name=next(iter(fixed_region_ReA_hists)).name_eA.replace("jetR030_", "") + "_ratio",
                                name=input_spec.n_PDF_name + "_" + next(iter(true_hists)).name_eA + "_ReA_true_vs_det",
                                panels=pb.Panel(
                                    axes=[
                                        pb.AxisConfig(
                                            "x",
                                            label=r"$p" + variable_label + r"^{\text{jet}}\:(\text{GeV}/c)$",
                                            font_size=22,
                                            range=x_range,
                                        ),
                                        pb.AxisConfig(
                                            "y",
                                            label=r"$R_{\text{eA}}$",
                                            range=(0, 1.4),
                                            font_size=22,
                                        ),
                                    ],
                                    text=[
                                        pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                                        pb.TextConfig(
                                            x=0.97,
                                            y=0.03,
                                            text=expected_luminosities_display_text(expected_luminosities),
                                            font_size=22,
                                        ),
                                    ],
                                    legend=pb.LegendConfig(location="lower left", font_size=22),
                                ),
                                figure=pb.Figure(edge_padding={"left": 0.12, "bottom": 0.1}),
                            ),
                            output_dir=sim_config.output_dir,
                        )

    import IPython

    IPython.start_ipython(user_ns={**globals(), **locals()})  # type: ignore[no-untyped-call]
    # import IPython; IPython.embed()


def run() -> None:
    helpers.setup_logging()

    # import warnings
    # warnings.filterwarnings("error")

    # Settings
    scale_jets_by_expected_luminosity = True
    skip_slow_2D_plots = False
    analysis_config = ecce_ReA_implementation.AnalysisConfig(
        jet_R_values=[0.3, 0.5, 0.8, 1.0],
        # jet_types=["charged", "calo", "true_charged", "true_full"],
        # Full analysis
        # regions = ["forward", "backward", "mid_rapidity"],
        # variables = ["pt", "p"],
        # More minimal for speed + testing
        # NOTE: For the future, the number of hists is usually too large to load all of the into memory at once.
        #       So instead, load some of them at a time, and take it in steps. One could do this with a shell script, etc.
        #       (or carefully clear the memory in python). However, the easiest thing to do so have has been to deal
        #       with it by hand.
        jet_types=["charged", "true_charged"],
        # jet_types=["calo", "true_full"],
        regions=["forward"],
        variables=["p", "pt"],
    )

    # Setup
    dataset_spec = ecce_base.DatasetSpecPythia(
        # site="production",
        site="cades",
        generator="pythia8",
        electron_beam_energy=10,
        proton_beam_energy=100,
        q2_selection=[100],
        label="",
    )
    # Setup I/O dirs
    # label = "fix_variable_shadowing"
    # label = "min_p_cut_with_tracklets_EPPS"
    # label = "min_p_cut_with_tracklets_nNNPDF"
    label = "min_p_cut_EPPS"
    base_dir = Path(f"/Volumes/data/eic/ReA/current_best_knowledge/{dataset_spec!s}")
    input_dir = base_dir / label
    output_dir = base_dir / "plots" / label
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(
        dataset_spec=dataset_spec,
        jet_algorithm="anti_kt",
        input_specs=[
            InputSpec("ep", n_variations=1),
            # EPPS
            # For testing
            # InputSpec("EPPS16nlo_CT14nlo_Au197", n_variations=2),
            # Full set of variations
            InputSpec("EPPS16nlo_CT14nlo_Au197", n_variations=97),
            # nNNPDF
            # For testing
            # InputSpec("nNNPDF20_nlo_as_0118_Au197", n_variations=1),
            # InputSpec("nNNPDF20_nlo_as_0118_Au197", n_variations=15),
            # Full set of variations
            # InputSpec("nNNPDF20_nlo_as_0118_Au197", n_variations=250),
        ],
        input_dir=input_dir,
        output_dir=output_dir,
    )
    config.setup()

    # Inputs
    # From the evaluator files, in pb (pythia provides in mb, but then it's change to pb during the conversion to HepMC2)
    _pb_to_fb = 1e3
    _cross_sections = {}
    for site in ["production", "cades"]:
        _cross_sections.update(
            {
                f"{site}-pythia8-10x100-q2-100": 1322.52 * _pb_to_fb,
                f"{site}-pythia8-10x100-q2-1-to-100": 470921.71 * _pb_to_fb,
            }
        )
    # 1 year in fb^{-1}
    _luminosity_projections = {
        "ep": 10.0,
        # Scaling according to the recommendations
        "eA": 10 * 1.0 / 197,
    }

    logger.info(f"Analyzing {label}")
    input_hists = _load_results(config=config, input_specs=config.input_specs)

    plot_ReA(
        sim_config=config,
        analysis_config=analysis_config,
        input_hists=input_hists,
        cross_section=_cross_sections[str(dataset_spec)],
        expected_luminosities=_luminosity_projections,
        scale_jets_by_expected_luminosity=scale_jets_by_expected_luminosity,
        skip_slow_2D_plots=skip_slow_2D_plots,
    )


if __name__ == "__main__":
    run()
