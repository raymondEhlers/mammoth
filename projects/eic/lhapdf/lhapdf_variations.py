from __future__ import annotations

import logging
from pathlib import Path

import lhapdf
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot as pb

from mammoth import helpers

logger = logging.getLogger(__name__)

pb.configure()

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

_display_name = {
    "EPPS16nlo_CT14nlo_Au197": "EPPS16 NLO / CT14 NLO",
    "nNNPDF20_nlo_as_0118_Au197": "nNNPDF 2.0 NLO / NNPDF 3.1 NNLO",
}


def run(n_variations: int = 97) -> None:  # noqa: ARG001
    q2 = 100
    x = np.logspace(-4, 0, 100, endpoint=False)
    output_dir = Path()

    for struck_quark_pid, struck_quark_label in [(lhapdf.UP, "u"), (lhapdf.DOWN, "d")]:
        values = {}

        for (
            n_PDF_name,
            proton_pdf_name,
            _n_variations,
        ) in [  # ("nNNPDF20_nlo_as_0118_Au197", "NNPDF40_nnlo_as_01180", 250)]:
            ("nNNPDF20_nlo_as_0118_Au197", "nNNPDF20_nlo_as_0118_N1", 250),
            ("EPPS16nlo_CT14nlo_Au197", "CT14nlo", 97),
        ]:
            values[n_PDF_name] = {}
            proton_pdf = lhapdf.mkPDF(proton_pdf_name)
            first_loop = True
            logger.info(
                f"proton_pdf {proton_pdf_name} limits: x=({proton_pdf.xMin}, {proton_pdf.xMax}), Q2=({proton_pdf.q2Min}, {proton_pdf.q2Max})"
            )
            for variation in range(_n_variations):
                n_pdf = lhapdf.mkPDF(n_PDF_name, variation)
                if first_loop:
                    logger.info(
                        f"n_pdf {n_PDF_name} limits: x=({n_pdf.xMin}, {n_pdf.xMax}), Q2=({n_pdf.q2Min}, {n_pdf.q2Max})"
                    )
                    first_loop = False

                temp_values = []
                for x_val in x:
                    weightNPDF = n_pdf.xfxQ2(struck_quark_pid, x_val, q2)
                    weightPDF = proton_pdf.xfxQ2(struck_quark_pid, x_val, q2)
                    ratio = weightNPDF / weightPDF
                    # print(x_val, ratio)
                    temp_values.append(weightNPDF / weightPDF)

                values[n_PDF_name][variation] = temp_values

        # text = "EPPS16nlo - ${}^{197}Au$"
        text = "${}^{197}$Au"
        text += ", " + rf"$Q^{{2}} = {q2}\:\text{{GeV}}^{2}$, {struck_quark_label} quark"
        plot_config = pb.PlotConfig(
            name=f"variations_{struck_quark_label}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$x$", font_size=22, log=True),
                    pb.AxisConfig(
                        "y",
                        label=r"nPDF/pPDF",
                        font_size=22,
                        range=(0, 1.4) if struck_quark_label == "u" else (0, 3.0),
                    ),
                ],
                text=pb.TextConfig(x=0.03, y=0.97, text=text, font_size=22),
                legend=pb.LegendConfig(location="lower left", font_size=22),
            ),
            figure=pb.Figure(edge_padding={"left": 0.125, "bottom": 0.1}),
        )

        # with sns.color_palette("Set2"):
        fig, ax = plt.subplots(figsize=(10, 7.5))

        for i, n_PDF_name in enumerate(values):
            needs_label = True
            for k, v in values[n_PDF_name].items():
                extra_args = {}
                if needs_label:
                    extra_args = {
                        "label": _display_name[n_PDF_name],
                    }
                    needs_label = False
                logger.info(f"plotting {k}")
                ax.plot(
                    x,
                    v,
                    linestyle="-",
                    linewidth=2,
                    # color=_okabe_ito_colors[i * 2],
                    color=_okabe_ito_colors[i * 3 + 2],
                    alpha=0.075,
                    marker="",
                    **extra_args,
                )

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


if __name__ == "__main__":
    helpers.setup_logging()
    run()
