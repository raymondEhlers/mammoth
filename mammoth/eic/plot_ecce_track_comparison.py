
import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import cycler
import hist
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot as pb
from mammoth.eic import run_ecce_analysis

pb.configure()

logger = logging.getLogger(__name__)


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


def get_eta_label(eta_range: Tuple[float, float]) -> str:
    return fr"${eta_range[0]} < \eta < {eta_range[1]}$"


def _plot_tracking_comparison(input_specs: Sequence[run_ecce_analysis.DatasetSpec], input_spec_labels: Mapping[str, str], hists: Mapping[str, Mapping[str, hist.Hist]],
                              all_regions: Sequence[Tuple[float, float]], regions_index: Sequence[int],
                              plot_config: pb.PlotConfig, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_prop_cycle(cycler.cycler(
        color=_okabe_ito_colors,
    ))

    for i_input_spec, input_spec in enumerate(input_specs):
        input_spec_hists = hists[str(input_spec)]
        markers = ["s", "o", "X", "D", "*"]
        for i_eta_index, (eta_index, h) in enumerate(input_spec_hists.items()):
            #m = h.values() > -1e6
            #values = h.values()[m]
            #errors = np.sqrt(h.variances()[m])
            #bin_centers = h.axes[0].centers[m]
            #bin_widths = h.axes[0].widths[m]
            #indices_with_values = np.where(m)[0]
            #if len(indices_with_values) == 0:
            #    logger.info(f"No valid values. Skipping {eta_index}, {str(input_spec)}")
            #    continue
            #else:
            #    # Inefficient, but I don't really care here - it doesn't need to be that fast.
            #    groups = [(k, sum(1 for i in g)) for k,g in itertools.groupby(m)]
            #    # Max is three groups: Falses, followed by Trues, followed by Falses
            #    if len(groups) > 3:
            #        logger.warning(f"Can't slice in a continuous range for {eta_index}, {str(input_spec)}. Groups: {groups}")
            #    # NOTE: Have to explicitly convert to int because they have an explicit isinstance on int, and apparently np.int64 doesn't count...
            #    s = slice(int(indices_with_values[0]), int(indices_with_values[-1] + 1))
            #h_sliced = h[s]
            logger.info(f"plotting eta_index: {eta_index}, {str(input_spec)}")
            #ax.errorbar(
            #    h_sliced.axes[0].centers, h_sliced.values(),
            #    yerr=np.sqrt(h_sliced.variances()),
            #    label=str(input_spec)
            #)
            #logger.info(f"h.values(): {h.values()}")
            ax.errorbar(
                h.axes[0].centers,
                h.values(),
                xerr=h.axes[0].widths / 2,
                yerr=np.sqrt(h.variances()),
                label=input_spec_labels[str(input_spec)] if len(input_spec_hists) == 1 else get_eta_label(all_regions[eta_index]),
                marker=markers[i_eta_index if len(input_spec_hists) > 1 else i_input_spec],
                linestyle="",
                markersize=8,
                #alpha=0.7,
            )

    if "_mean" in plot_config.name:
        ax.axhline(y=0, color="black", linestyle="dashed", zorder=1)

    # Labeling and presentation
    plot_config.apply(fig=fig, ax=ax)
    # A few additional tweaks.
    #ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    # ax_ratio.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.2))

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_tracking_comparison(input_specs: Sequence[run_ecce_analysis.DatasetSpec], input_spec_labels: Mapping[str, str], output_hists: Dict[str, Dict[str, hist.Hist]],
                             hist_name_template: str, plot_name: str,
                             all_regions: Sequence[Tuple[float, float]],
                             regions_index: Sequence[int], regions_label: str, y_range: Tuple[float, float], y_label: str,
                             selected_particle: str,
                             text: str,
                             output_dir: Path,
                             plot_jets: bool = False,
                             x_label: str = r"$p^{\text{MC}}\:(\text{GeV}/c)$",
                             x_range: Tuple[float, float] = (0.1, 30)) -> None:
    logger.info(f"hist_name_template: {hist_name_template}")
    logger.info(f"input_specs: {str(input_specs[0])}")

    hists = {}
    for input_spec in input_specs:
        hists[str(input_spec)] = {}
        for index in regions_index:
            temp_name = hist_name_template.format(particle=selected_particle.capitalize(), eta_region_index=index)
            #logger.info(f"{str(input_spec)}, {index}, template_name: {temp_name}")
            hists[str(input_spec)][index] = output_hists[str(input_spec)][temp_name]

    _plot_tracking_comparison(
        input_specs=input_specs,
        input_spec_labels=input_spec_labels,
        #hists = {
        #    str(input_spec): {
        #        index: output_hists[str(input_spec)][hist_name_template.format(particle=selected_particle.capitalize(), eta_region_index=index)]
        #        for index in regions_index
        #    }
        #    for input_spec in input_specs
        #},
        hists=hists,
        all_regions=all_regions,
        regions_index=regions_index,
        plot_config=pb.PlotConfig(
            name=f"{plot_name}_{regions_label}_{'_'.join([str(v) for v in regions_index])}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=x_label, font_size=22, log=(plot_jets is False), range=x_range),
                    pb.AxisConfig(
                        "y",
                        label=y_label,
                        range=y_range,
                        font_size=22,
                    ),
                ],
                text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                legend=pb.LegendConfig(location="upper left", font_size=22),
            ),
            figure=pb.Figure(edge_padding=dict(left=0.10 if "width" in plot_name else 0.13, bottom=0.10)),
        ),
        output_dir=output_dir,
    )

