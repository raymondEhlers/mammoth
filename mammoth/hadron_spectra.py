""" Calculate hadron spectra for a given input

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Type, TypeVar

import attr
import awkward as ak
import boost_histogram as bh
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pachyderm.plot
import particle
import uproot
from pachyderm import binned_data

from mammoth import base, parse_ascii


pachyderm.plot.configure()


@attr.s
class ReferenceData:
    data: binned_data.BinnedData = attr.ib()
    _stat_errors: binned_data.BinnedData = attr.ib()
    _sys_errors: binned_data.BinnedData = attr.ib()

    @property
    def values(self) -> np.ndarray:
        return self.data.values

    @property
    def stat_errors(self) -> np.ndarray:
        return self._stat_errors.errors

    @property
    def sys_errors(self) -> np.ndarray:
        return self._sys_errors.errors


def load_reference_data() -> Dict[str, binned_data.BinnedData]:
    # Setup
    reference_data = {}
    return_data = {}
    input_data_path = Path("inputData")

    # STAR Reference Data:
    # 200 GeV charged pions - https://inspirehep.net/literature/709170
    # I average the measurements and add the statistical and systematic errors in quadrature
    with uproot.open(input_data_path / "STAR" / "HEPData-ins709170-v1-Table_2.root") as f:
        print(f.keys())
        data = binned_data.BinnedData.from_existing_data(f["Table 2"][f"Hist1D_y1"])
        stat_errors = binned_data.BinnedData.from_existing_data(f["Table 2"][f"Hist1D_y1_e1"])
        sys_errors = binned_data.BinnedData.from_existing_data(f["Table 2"][f"Hist1D_y1_e2"])
        reference_data["star_pi_plus"] = ReferenceData(
            data = data, stat_errors = stat_errors, sys_errors = sys_errors
        )

    with uproot.open(input_data_path / "STAR" / "HEPData-ins709170-v1-Table_7.root") as f:
        data = binned_data.BinnedData.from_existing_data(f["Table 7"][f"Hist1D_y1"])
        stat_errors = binned_data.BinnedData.from_existing_data(f["Table 7"][f"Hist1D_y1_e1"])
        sys_errors = binned_data.BinnedData.from_existing_data(f["Table 7"][f"Hist1D_y1_e2"])
        reference_data["star_pi_minus"] = ReferenceData(
            data = data, stat_errors = stat_errors, sys_errors = sys_errors
        )

    return_data["STAR $\pi^{\pm}$"] = binned_data.BinnedData(
        axes=reference_data["star_pi_plus"].data.axes[0].bin_edges,
        values=(reference_data["star_pi_plus"].values + reference_data["star_pi_minus"].values) / 2,
        # Add statistical and systematics errors in quadrature.
        variances=(
            reference_data["star_pi_plus"].stat_errors ** 2 + reference_data["star_pi_minus"].stat_errors ** 2
            + reference_data["star_pi_plus"].sys_errors ** 2 + reference_data["star_pi_minus"].sys_errors ** 2
        ),
    )

    return return_data


def setup() -> None:
    parse_ascii.parse_to_parquet(
        base_output_filename="skim/output.parquet",
        store_only_necessary_columns=True,
        input_filename=f"final_state_hadrons.out",
        events_per_chunk=10000,
        #max_chunks=1,
    )


def analyze(output_dir: Path, reference_data: Mapping[str, binned_data.BinnedData]) -> None:
    hist_pt = bh.Histogram(bh.axis.Regular(50, 0, 10), storage=bh.storage.Weight())

    # Load array
    n_events = 0
    for filename in output_dir.glob("*.parquet"):
        #arrays = ak.with_name(ak.from_parquet(filename), "LorentzVector")
        arrays = ak.from_parquet(filename)
        n_events += len(arrays)
        arrays["m"] = base.determine_masses_from_events(arrays)
        arrays = base.LorentzVectorArray.from_awkward_ptetaphim(arrays)

        # Particle selections
        # Drop neutrinos.
        #arrays = arrays[(np.abs(arrays["particle_ID"]) != 12) & (np.abs(arrays["particle_ID"]) != 14) & (np.abs(arrays["particle_ID"]) != 16)]
        # And then we'll select by status codes.
        #all_status_codes = np.unique(ak.to_numpy(ak.flatten(arrays["status"])))
        #print(all_status_codes)

        # Select pions
        charged_pions_mask = base.build_PID_selection_mask(arrays, absolute_pids=[211])
        # Selection from STAR analysis.
        # NOTE: For now, we use eta since we don't want to construct the full object. Can do more later.
        #rapidity_mask = np.abs(arrays["eta"]) < 0.5
        rapidity_mask = np.abs(arrays.rapidity) < 0.5
        charged_pions = arrays[charged_pions_mask & rapidity_mask]

        # Subtract holes from hadrons
        # Not relevant for pp, so we skip it.

        # Fill the hists.
        hist_pt.fill(ak.flatten(charged_pions.pt))

    # Convert the histogram to a suitable form
    h_pt = binned_data.BinnedData.from_existing_data(hist_pt)
    # Normalize
    h_pt /= n_events

    # Plot, including reference data.
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        h_pt.axes[0].bin_centers,
        h_pt.values,
        yerr=h_pt.errors,
        marker=".",
        linestyle="",
        label="JS PP19 $\pi^{\pm}$",
    )

    # Reference data
    for label, data in reference_data.items():
        ax.errorbar(
            data.axes[0].bin_centers,
            data.values,
            yerr=data.errors,
            marker=".",
            linestyle="",
            label=label,
        )

    # Presentation
    ax.set_xlabel(r"$p_{\text{T}}\:(\text{GeV}/c)$")
    ax.set_yscale("log")
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig("rhicHadronSpectra.pdf")

    plt.close(fig)


if __name__ == "__main__":
    # Setup doesn't autodetect if it needs to run. Instead, it's up to the user.
    #setup()
    reference_data = load_reference_data()
    analyze(output_dir=Path("skim") / "events_per_chunk_10000", reference_data=reference_data)

