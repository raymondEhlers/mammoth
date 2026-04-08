"""Convert skim to histograms for further analysis

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from pathlib import Path

import awkward as ak
import hist
import numpy as np
import uproot
import vector

from mammoth.framework.io import output_utils

logger = logging.getLogger(__name__)
vector.register_awkward()


def define_base_histograms(levels: list[str]) -> dict[str, hist.Hist[hist.storage.Weight]]:
    """Basic histograms for the analysis."""

    hists = {}

    # Spectra
    for level in levels:
        # Leading, subleading spectra
        hists[f"{level}_leading_jet_spectra"] = hist.Hist(
            hist.axis.Regular(131, 9.5, 140.5, label="leading_jet_pt"), storage=hist.storage.Weight()
        )
        hists[f"{level}_subleading_jet_spectra"] = hist.Hist(
            hist.axis.Regular(131, 9.5, 140.5, label="subleading_jet_pt"), storage=hist.storage.Weight()
        )
        # N constituents as function of jet pt
        hists[f"{level}_n_constituents_jet_pt"] = hist.Hist(
            hist.axis.Regular(131, 9.5, 140.5, label="leading_jet_pt"),
            hist.axis.Regular(131, 9.5, 140.5, label="subleading_jet_pt"),
            hist.axis.Regular(50, -0.5, 49, label="$n_{const}^{lead}$"),
            hist.axis.Regular(50, -0.5, 49, label="$n_{const}^{sublead}$"),
            storage=hist.storage.Weight(),
        )

    return hists


def load_data(
    skim_directory: Path,
    level_name: str,
) -> tuple[ak.Array, ak.Array, ak.Array]:
    logger.info(f"Loading skim data from {skim_directory}")
    arr = ak.from_parquet(skim_directory)

    # Extract jets (n.b. despite the name in the array, these jets aren't actually pt sorted, so it's just pt_1, pt_2
    leading_jet = arr[f"{level_name}_leading_jet"]
    subleading_jet = arr[f"{level_name}_subleading_jet"]

    # Use scale_factor when available
    scale_factors = arr["scale_factor"] if "scale_factor" in ak.fields(arr) else np.ones(len(leading_jet))

    return leading_jet, subleading_jet, scale_factors


def skim_to_histograms(
    skim_directory: Path,
) -> None:
    # Setup
    level_name = "data"

    # NOTE: Could also consider dask_awkward. But for now, this is good enough.
    leading_jet, subleading_jet, scale_factors = load_data(skim_directory=skim_directory, level_name=level_name)

    # Define hists
    hists = {}
    hists.update(define_base_histograms(levels=[level_name]))

    # import IPython

    # IPython.embed()

    # Fill hists
    # Spectra
    hists[f"{level_name}_leading_jet_spectra"].fill(
        ak.flatten(leading_jet.pt, axis=None),
        weight=scale_factors,
    )
    hists[f"{level_name}_subleading_jet_spectra"].fill(
        ak.flatten(leading_jet.pt, axis=None),
        weight=scale_factors,
    )

    # (leading, subleading) N constituents as a function of (leading, subleading) jet pt
    # NOTE: I can do multiple weights with: hist.storage.MultiWeight(3), where 3 = number of weights.
    #       See e.g.: https://github.com/scikit-hep/boost-histogram/pull/1008 (called MultiCell)
    hists[f"{level_name}_n_constituents_jet_pt"].fill(
        ak.flatten(leading_jet.pt, axis=None),
        ak.flatten(subleading_jet.pt, axis=None),
        ak.flatten(ak.num(leading_jet.constituents, axis=1), axis=None),
        ak.flatten(ak.num(subleading_jet.constituents, axis=1), axis=None),
        weight=scale_factors,
    )

    # Write histograms
    output_hist_filename = skim_directory / "hists" / "analysis"
    output_hist_filename.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(output_hist_filename) as f:
        output_utils.write_hists_to_file(hists=hists, f=f)


if __name__ == "__main__":
    import mammoth.helpers

    mammoth.helpers.setup_logging(level=logging.INFO)

    skim_to_histograms(skim_directory=Path("trains/pp/0063/skim"))
