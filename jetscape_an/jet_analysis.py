""" Tasks related to jet finding.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import functools
from pathlib import Path
from typing import Any, Mapping, Tuple, Type, TypeVar

import awkward1 as ak
import boost_histogram as bh
import matplotlib
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pachyderm.plot
import particle
import pyfastjet as fj
from pachyderm import binned_data


pachyderm.plot.configure()
# Enable ticks on all sides
# Unfortunately, some of this is overriding the pachyderm plotting style.
# That will have to be updated eventually...
matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["xtick.minor.top"] = True
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["ytick.minor.right"] = True


class LorentzVectorCommon:
    """ Basic Lorentz Vector class for conveinence.

    Assumes metric: (+ - - -)

    """
    t: Any
    x: Any
    y: Any
    z: Any

    @property
    def pt(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)

    @property
    def eta(self):
        return np.arcsinh(self.z / self.pt)

    @property
    def phi(self):
        """

        Appears to be defined from [-pi, pi). PseudoJets are defined from [0, 2pi)
        """
        # NOTE: Could put it within [0, 2pi) with (take from fastjet::PseudoJet):
        #       if (_phi >= twopi) _phi -= twopi;
        #       if (_phi < 0)      _phi += twopi;
        return np.arctan2(self.y, self.x)


class LorentzVector(ak.Record, LorentzVectorCommon):  # type: ignore
    """ Basic Lorentz Vector class for conveinence.

    Assumes metric: (+ - - -)

    """
    t: float
    x: float
    y: float
    z: float
    ...


class LorentzVectorArray(ak.Array, LorentzVectorCommon):  # type: ignore
    t: ak.Array
    x: ak.Array
    y: ak.Array
    z: ak.Array

    @staticmethod
    def from_ptetaphim(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, m: np.ndarray) -> ak.Array:
        return ak.zip(
            {
                # magnitude of p = pt*cosh(eta)
                "t": np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2),
                "x": pt * np.cos(phi),
                "y": pt * np.sin(phi),
                "z": pt * np.sinh(eta),
            },
            with_name="LorentzVector",
        )


# Register behavior
ak.behavior["LorentzVector"] = LorentzVector
ak.behavior["*", "LorentzVector"] = LorentzVectorArray


def find_jets(array: ak.Array) -> ...:
    """ Find jets.

    """
    jet_defintion = fj.JetDefinition(fj.JetAlgorithm.antikt_algorithm, R = 0.4)
    for event in array:
        columns = ["px", "py", "pz", "E"]
        arr = ak.to_numpy(
            event[columns]
        )
        # Convert from recarray to standard array
        arr = arr.view(np.float64).reshape((len(event), len(columns)))

        cs = fj.ClusterSequence(
            pseudojets = arr,
            jet_definition = jet_defintion,
        )
        # Convert from pt, eta, phi, m -> standard LorentzVector
        jets = LorentzVectorArray.from_ptetaphim(
            **dict(zip(["pt", "eta", "phi", "m"], cs.to_numpy()))
        )
        print(jets.layout)
        print(ak.type(jets))
        sorted_by_pt = ak.argsort(jets.pt, ascending=True)
        jets = jets[sorted_by_pt]

        #jets = ak.zip(
        #    dict(zip(["x", "y", "z", "t"], cs.to_numpy())),
        #    with_name="LorentzVector",
        #)
        #jets = fj.sorted_by_pt(cs.inclusive_jets())
        #import IPython; IPython.embed()

        #print(jets.to_numpy())



@functools.lru_cache()
def _pdg_id_to_mass(pdg_id: int) -> float:
    """ Convert PDG ID to mass.

    We cache the result to speed it up.

    Args:
        pdg_id: PDG ID.
    Returns:
        Mass in MeV.
    """
    return particle.Particle.from_pdgid(pdg_id).mass

@nb.njit
def determine_mass(events: ak.Array, builder: ak.ArrayBuilder, pdg_id_to_mass: Mapping[int, float]) -> ak.Array:
    for event in events:
        builder.begin_list()
        for particle in event:
            builder.append(pdg_id_to_mass[particle["particle_ID"]])
        builder.end_list()

    #return builder.snapshot()


def find_jets_arr(array: ak.Array) -> ak.Array:
    """ Find jets.

    """
    # Particle selection
    # Drop neutrinos.
    new_array = array[(np.abs(array["particle_ID"]) != 12) & (np.abs(array["particle_ID"]) != 14) & (np.abs(array["particle_ID"]) != 16)]
    # Determine masses
    all_particle_IDs = np.unique(ak.to_numpy(ak.flatten(new_array["particle_ID"])))

    pdg_id_to_mass = nb.typed.Dict.empty(
        key_type=nb.core.types.int64,
        value_type=nb.core.types.float64,
    )
    for pdg_id in all_particle_IDs:
        pdg_id_to_mass[pdg_id] = _pdg_id_to_mass(pdg_id)
    #import IPython; IPython.embed()
    #pdg_id_to_mass = nb.typed.Dict({pdg_id: _pdg_id_to_mass(pdg_id) for pdg_id in all_particle_IDs})

    builder = ak.ArrayBuilder()
    determine_mass(events=new_array, builder=builder, pdg_id_to_mass=pdg_id_to_mass)
    #array["m"] = builder.snapshot()
    mass = builder.snapshot()

    new_array_lorentz = LorentzVectorArray.from_ptetaphim(new_array["pt"], new_array["eta"], new_array["phi"], mass)

    jet_defintion = fj.JetDefinition(fj.JetAlgorithm.antikt_algorithm, R = 0.4)
    area_definition = fj.AreaDefinition(fj.AreaType.passive_area, fj.GhostedAreaSpec(1, 1, 0.05))
    settings = fj.JetFinderSettings(jet_definition=jet_defintion, area_definition=area_definition)

    # Convert to an array that fj will recognize. Otherwise, the arguments won't match.
    # TODO: Handle this more gracfully...
    temp_array = ak.zip(
        {
            "E": new_array_lorentz["t"],
            "px": new_array_lorentz["x"],
            "py": new_array_lorentz["y"],
            "pz": new_array_lorentz["z"],
            # NOTE: Having status is okay, even though it's not part of the LorentzVector. Which is quite nice!
            "status": new_array["status"],
        },
        # TODO: This isn't quite right, but fine for now.
        with_name="LorentzVector",
    )
    print(temp_array.type)

    #jets = fj.find_jets(events=array.layout.Content, settings=settings)
    #jets = ak.Array(fj.find_jets(events=temp_array.layout, settings=settings))

    #import IPython; IPython.embed()

    #jets = ak.Array(fj.find_jets_awkward_test(events=temp_array.layout))

    jets = fj.find_jets(events=temp_array, settings=settings)
    #import IPython; IPython.embed()
    return jets


def particle_pt_by_status(arrays: ak.Array, pt_hat_bin: Tuple[int, int], base_output_dir: Path) -> None:
    """ Plot particle pt by status code.

    """
    # Setup
    output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Particle selection
    # Drop neutrinos.
    arrays = arrays[(np.abs(arrays["particle_ID"]) != 12) & (np.abs(arrays["particle_ID"]) != 14) & (np.abs(arrays["particle_ID"]) != 16)]

    all_status_codes = np.unique(ak.to_numpy(ak.flatten(arrays["status"])))
    n_total_particles = ak.sum(ak.num(arrays, axis=1))

    fig, ax = plt.subplots(figsize=(8, 6))

    for status_code in all_status_codes:
        print(f"status_code: {status_code}")
        mask = (arrays["status"] == status_code)
        pt_hist = bh.Histogram(bh.axis.Regular(100, 0, 50), storage=bh.storage.Weight())
        pt_hist.fill(ak.to_numpy(ak.flatten(arrays[mask]["pt"])))
        hist = binned_data.BinnedData.from_existing_data(pt_hist)

        fraction_of_particles = ak.sum(ak.num(arrays[mask], axis=1)) / n_total_particles
        print(f"Fraction of particles: {fraction_of_particles:.02g}")

        # Normalize
        hist /= hist.axes[0].bin_widths

        ax.errorbar(
            hist.axes[0].bin_centers,
            hist.values,
            xerr=hist.axes[0].bin_widths / 2,
            label=f"Status = {status_code}, {int(fraction_of_particles * 100)}\%",
            marker="o",
            linestyle="",
        )

    # Label
    ax.text(
        0.45,
        0.97,
        r"$\hat{p_{\text{T}}} =$ " + f"{pt_hat_bin[0]}-{pt_hat_bin[1]}",
        transform=ax.transAxes,
        horizontalalignment="left", verticalalignment="top", multialignment="left",
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"dN/d$p_{\text{T}}$ $(\text{GeV}/c)^{-1}$")
    ax.set_xlabel(r"$p_{\text{T}}$ (GeV/c)")
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(output_dir / f"pt_distribution_{pt_hat_bin[0]}_{pt_hat_bin[1]}.pdf")
    plt.close(fig)


def angular_distribution_around_jet(jets: ak.Array, arrays: ak.Array, pt_hat_bin: Tuple[int, int], base_output_dir: Path) -> None:
    # Setup
    output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    min_jet_pt = 20

    # Particle selection
    # Drop neutrinos.
    arrays = arrays[(np.abs(arrays["particle_ID"]) != 12) & (np.abs(arrays["particle_ID"]) != 14) & (np.abs(arrays["particle_ID"]) != 16)]
    all_status_codes = np.unique(ak.to_numpy(ak.flatten(arrays["status"])))

    # Jets selection
    # Keep jets with at least min_jet_pt GeV.
    jets = jets[jets.pt > min_jet_pt]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig_eta, ax_eta = plt.subplots(figsize=(8, 6))
    fig_phi, ax_phi = plt.subplots(figsize=(8, 6))

    hists = []
    for status_code in all_status_codes:
        # Select only particles of a particular status.
        particle_mask = (arrays["status"] == status_code)

        # Calculate all of the distances. Hopefully this doesn't run out of memory!
        comb_jets, comb_particles = ak.unzip(ak.cartesian([jets, arrays[particle_mask]]))
        delta_phi = ak.flatten(np.abs(comb_jets.phi - comb_particles.phi))
        delta_phi = ak.where(delta_phi > np.pi, (2 * np.pi) - delta_phi, delta_phi)
        delta_eta = ak.flatten(comb_jets.eta - comb_particles.eta)
        distance_from_jet = np.sqrt(delta_eta ** 2 + delta_phi ** 2)

        # Create, fill, and plot the histogram
        distance_hist = bh.Histogram(bh.axis.Regular(160, 0, 8), storage=bh.storage.Weight())
        distance_hist.fill(ak.to_numpy(distance_from_jet))
        hist = binned_data.BinnedData.from_existing_data(distance_hist)

        # Normalize
        # Bin widths
        hist /= hist.axes[0].bin_widths
        # N jets
        hist /= ak.sum(ak.num(jets, axis=1))

        ax.errorbar(
            hist.axes[0].bin_centers,
            hist.values,
            xerr=hist.axes[0].bin_widths / 2,
            label=f"Status = {status_code}",
            marker="o",
            linestyle="",
        )

        # Store for summary plot
        hists.append(hist)

        # Create, fill, and plot the histogram
        distance_from_jet = np.abs(delta_eta)
        distance_hist = bh.Histogram(bh.axis.Regular(160, 0, 8), storage=bh.storage.Weight())
        distance_hist.fill(ak.to_numpy(distance_from_jet))
        hist = binned_data.BinnedData.from_existing_data(distance_hist)

        # Normalize
        # Bin widths
        hist /= hist.axes[0].bin_widths
        # N jets
        hist /= ak.sum(ak.num(jets, axis=1))

        ax_eta.errorbar(
            hist.axes[0].bin_centers,
            hist.values,
            xerr=hist.axes[0].bin_widths / 2,
            label=f"Status = {status_code}",
            marker="o",
            linestyle="",
        )

        # Create, fill, and plot the histogram
        distance_from_jet = delta_phi
        distance_hist = bh.Histogram(bh.axis.Regular(80, 0, 4), storage=bh.storage.Weight())
        distance_hist.fill(ak.to_numpy(distance_from_jet))
        hist = binned_data.BinnedData.from_existing_data(distance_hist)

        # Normalize
        # Bin widths
        hist /= hist.axes[0].bin_widths
        # N jets
        hist /= ak.sum(ak.num(jets, axis=1))

        ax_phi.errorbar(
            hist.axes[0].bin_centers,
            hist.values,
            xerr=hist.axes[0].bin_widths / 2,
            label=f"Status = {status_code}",
            marker="o",
            linestyle="",
        )

    # Plot summary
    h_all = sum(hists)
    ax.errorbar(
        h_all.axes[0].bin_centers,
        h_all.values,
        xerr=h_all.axes[0].bin_widths / 2,
        label=f"Sum",
        marker="o",
        linestyle="",
    )

    # Label
    for a in [ax, ax_eta, ax_phi]:
        a.text(
            0.03,
            0.97,
            "R = 0.4 " + r"anti-$k_{\text{T}}$ jets"
            "\n" + r"$p_{\text{T}}^{\text{jet}} > " + fr"{min_jet_pt}\:\text{{GeV}}/c$"
            "\n" + r"$\hat{p_{\text{T}}} =$ " + f"{pt_hat_bin[0]}-{pt_hat_bin[1]}",
            transform=a.transAxes,
            horizontalalignment="left", verticalalignment="top", multialignment="left",
        )
        a.legend(loc="upper right", frameon=False)
    #ax.set_yscale("log")
    ax.set_ylabel(r"$1/N_{\text{jets}} \text{d}N/\text{d}R$")
    ax.set_xlabel(r"Distance from jet axis")
    ax_eta.set_ylabel(r"$1/N_{\text{jets}} \text{d}N/\text{d}\eta$")
    ax_eta.set_xlabel(r"$\eta$ from jet axis")
    ax_phi.set_ylabel(r"$1/N_{\text{jets}} \text{d}N/\text{d}\varphi$")
    ax_phi.set_xlabel(r"$\varphi$ from jet axis")

    fig.tight_layout()
    fig.savefig(output_dir / f"distance_from_jet_{pt_hat_bin[0]}_{pt_hat_bin[1]}.pdf")
    plt.close(fig)

    fig_eta.tight_layout()
    fig_eta.savefig(output_dir / f"eta_from_jet_{pt_hat_bin[0]}_{pt_hat_bin[1]}.pdf")
    plt.close(fig_eta)
    fig_phi.tight_layout()
    fig_phi.savefig(output_dir / f"phi_from_jet_{pt_hat_bin[0]}_{pt_hat_bin[1]}.pdf")
    plt.close(fig_phi)

if __name__ == "__main__":
    for pt_hat_bin in [(7, 9), (20, 25), (50, 55), (100, 110), (250, 260), (500, 550), (900, 1000)]:
        pt_hat_range = f"{pt_hat_bin[0]}_{pt_hat_bin[1]}"
        print(f"Running for pt hat range: {pt_hat_range}")
        events_per_chunk = 1000
        filename = f"JetscapeHadronListBin{pt_hat_range}"
        base_output_dir = Path("performance_studies")
        input_filename = Path("skim") / f"events_per_chunk_{events_per_chunk}" / f"{filename}_00.parquet"
        input_arrays = ak.from_parquet(input_filename)
        # We use some very different value to make it clear if something ever goes wrong.
        # NOTE: It's important to do this before constructing anything else. Otherwise it can
        #       mess up the awkward1 behaviors.
        fill_none_value = -9999
        input_arrays = ak.fill_none(input_arrays, fill_none_value)

        # Fully zip the arrays together.
        arrays = ak.zip(dict(zip(ak.fields(input_arrays), ak.unzip(input_arrays))), depth_limit=None)
        #arrays = ak.zip(
        #    {
        #        "particle_ID": input_arrays["particle_ID"],
        #        "E": input_arrays["E"],
        #        "px": input_arrays["px"],
        #        "py": input_arrays["py"],
        #        "pz": input_arrays["pz"],
        #    },
        #    depth_limit = None,
        #)
        #import IPython; IPython.embed()

        particle_pt_by_status(arrays=arrays, pt_hat_bin=pt_hat_bin, base_output_dir=base_output_dir)
        jets = find_jets_arr(array=arrays)
        angular_distribution_around_jet(jets=jets, arrays=arrays, pt_hat_bin=pt_hat_bin, base_output_dir=base_output_dir)
