""" Tasks related to jet finding.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import functools
from pathlib import Path
from typing import Any, Mapping, Tuple, Type, TypeVar

import awkward as ak
import boost_histogram as bh
import matplotlib
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pachyderm.plot
import particle
import pyfastjet as fj
from pachyderm import binned_data

from mammoth import base


pachyderm.plot.configure()
# Enable ticks on all sides
# Unfortunately, some of this is overriding the pachyderm plotting style.
# That will have to be updated eventually...
matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["xtick.minor.top"] = True
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["ytick.minor.right"] = True


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

    new_array_lorentz = base.LorentzVectorArray.from_ptetaphim(new_array["pt"], new_array["eta"], new_array["phi"], mass)

    jet_defintion = fj.JetDefinition(fj.JetAlgorithm.antikt_algorithm, R = 0.4)
    area_definition = fj.AreaDefinition(fj.AreaType.passive_area, fj.GhostedAreaSpec(1, 1, 0.05))
    settings = fj.JetFinderSettings(jet_definition=jet_defintion, area_definition=area_definition)

    # Convert to an array that fj will recognize. Otherwise, the arguments won't match.
    # TODO: Handle this more gracefully... (This was more of a problem in the past. Now, it just looks unnecessary, but not worth changing at the moment)
    temp_array = ak.zip(
        {
            "px": new_array_lorentz["px"],
            "py": new_array_lorentz["py"],
            "pz": new_array_lorentz["pz"],
            "E": new_array_lorentz["E"],
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


@nb.njit
def phi_minus_pi_to_pi(phi_array: ak.Array, builder: ak.ArrayBuilder) -> ak.ArrayBuilder:
    for event in phi_array:
        builder.begin_list()
        for particle_phi in event:
            if particle_phi > np.pi:
                builder.append(particle_phi - 2 * np.pi)
            elif particle_phi < -np.pi:
                builder.append(particle_phi + 2 * np.pi)
            else:
                builder.append(particle_phi)
        builder.end_list()

    return builder

@nb.njit
def get_constituents(array: ak.Array, event_constituent_indices: ak.Array, builder: ak.ArrayBuilder) -> ak.ArrayBuilder:
    """ This is a hack, and should be handled properly later...

    """
    for event_particles, jet_constituent_indices in zip(array, event_constituent_indices):
        builder.begin_list()
        for indices in jet_constituent_indices:
            builder.begin_list()
            for i in indices:
                builder.append(event_particles[i])
            builder.end_list()
        builder.end_list()

    return builder


def convert_local_phi(phi):
    if phi > np.pi:
        phi -= 2 * np.pi
    elif phi < -np.pi:
        phi += 2 * np.pi
    return phi


def angular_distribution_around_jet(jets: ak.Array, arrays: ak.Array, pt_hat_bin: Tuple[int, int], base_output_dir: Path) -> None:
    # Setup
    output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    min_jet_pt = 20

    # Particle selection
    # Drop neutrinos.
    arrays = arrays[(np.abs(arrays["particle_ID"]) != 12) & (np.abs(arrays["particle_ID"]) != 14) & (np.abs(arrays["particle_ID"]) != 16)]
    all_status_codes = np.unique(ak.to_numpy(ak.flatten(arrays["status"])))

    # Do the dumb thing to get phi in [-pi, pi) and keep awkward happy. There's got to be a better way...
    #px = arrays["pt"] * np.cos(arrays["phi"])
    #py = arrays["pt"] * np.sin(arrays["phi"])
    #arrays["phi"] = np.arctan2(py, px)
    # Should modify in place
    builder = ak.ArrayBuilder()
    arrays["phi"] = phi_minus_pi_to_pi(arrays["phi"], builder=builder).snapshot()

    # Jets selection
    # TODO: Move this conversion into the pyfastjet...
    jets["constituent_index"] = ak.values_astype(jets.constituent_indices, np.int32)
    # Keep jets with at least min_jet_pt GeV.
    jets = jets[jets.jets.pt > min_jet_pt]

    #import IPython; IPython.embed()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig_fj, ax_fj = plt.subplots(figsize=(8, 6))
    fig_eta, ax_eta = plt.subplots(figsize=(8, 6))
    fig_phi, ax_phi = plt.subplots(figsize=(8, 6))

    hists = []
    hists_fj = []
    for status_code in all_status_codes:
        # Select only particles of a particular status.
        particle_mask = (arrays["status"] == status_code)

        # Calculate all of the distances. Hopefully this doesn't run out of memory!
        #comb_jets, comb_particles = ak.unzip(ak.cartesian([jets.jets, get_constituents(arrays, jets.constituent_index, ak.ArrayBuilder()).snapshot()]))
        #comb_jets, comb_particles = ak.unzip(ak.cartesian([jets.jets, arrays[jets.constituent_index]]))
        comb_jets, comb_particles = ak.unzip(ak.cartesian([jets.jets, arrays[particle_mask]]))
        #import IPython; IPython.embed()
        builder = ak.ArrayBuilder()
        #delta_phi = ak.flatten(phi_minus_pi_to_pi(ak.flatten(comb_jets.phi, axis=1) - ak.flatten(comb_particles.phi, axis=1), builder=builder).snapshot())
        delta_phi = ak.flatten(phi_minus_pi_to_pi(comb_jets.phi - comb_particles.phi, builder=builder).snapshot())
        #delta_phi = ak.flatten(np.abs(comb_jets.phi - comb_particles.phi))
        #delta_phi = np.mod(delta_phi, 2 * np.pi) - np.pi
        #import IPython; IPython.embed()
        #delta_phi = ak.where(delta_phi > np.pi, (2 * np.pi) - delta_phi, delta_phi)
        #delta_phi = ak.where(delta_phi > np.pi, delta_phi - (2 * np.pi), delta_phi)
        #import IPython; IPython.embed()
        #delta_phi = ak.where(delta_phi < -np.pi, (2 * np.pi) + delta_phi, delta_phi)
        #delta_eta = ak.flatten(ak.flatten(comb_jets.eta, axis=1) - ak.flatten(comb_particles.eta, axis=1))
        delta_eta = ak.flatten(comb_jets.eta - comb_particles.eta)
        distance_from_jet = np.sqrt(delta_eta ** 2 + delta_phi ** 2)
        #mask = (delta_phi < 0.1) & (delta_eta < 0.1)
        #distance_from_jet = np.sqrt(delta_eta[mask] ** 2 + delta_eta[mask] ** 2)

        # Cross check deltaR calculation against fj
        # Just for the first event
        jets_fj = [fj.PseudoJet(j.px, j.py, j.pz, j.E) for j in jets.jets[0]]
        particles_fj = [
            fj.PseudoJet(p.px, p.py, p.pz, p.E) for p in
            base.LorentzVectorArray.from_ptetaphie(arrays[particle_mask].pt, arrays[particle_mask].eta, arrays[particle_mask].phi, arrays[particle_mask].E)[0]
        ]
        # Compare for the first jet for simplicity
        fj_distance_lead_jet_to_all_particles = [jets_fj[0].delta_R(p) for p in particles_fj]
        # NOTE: PseudoJet.delta_R uses rapidity, not eta!!
        #fj_delta_eta = [jets_fj[0].eta - p.eta for p in particles_fj]
        fj_delta_eta = [jets_fj[0].rap - p.rap for p in particles_fj]
        #fj_delta_phi = [convert_local_phi(jets_fj[0].phi_std - p.phi_std) for p in particles_fj]
        fj_delta_phi = [np.abs(jets_fj[0].phi - p.phi) for p in particles_fj]
        fj_delta_phi = [2*np.pi - dphi if dphi > np.pi else dphi for dphi in fj_delta_phi]
        fj_distance_by_hand = [np.sqrt(eta ** 2 + phi ** 2) for eta, phi in zip(fj_delta_eta, fj_delta_phi)]
        local_delta_eta = [jets.jets[0][0].eta - p.eta for p in arrays[particle_mask][0]]
        local_delta_phi = [convert_local_phi(jets.jets[0][0].phi - p.phi) for p in arrays[particle_mask][0]]
        local_distance = [np.sqrt(eta ** 2 + phi ** 2) for eta, phi in zip(local_delta_eta, local_delta_phi)]

        # Create, fill, and plot the histogram
        distance_hist = bh.Histogram(bh.axis.Regular(160, 0, 8), storage=bh.storage.Weight())
        #distance_hist = bh.Histogram(bh.axis.Regular(20, 0, 1), storage=bh.storage.Weight())
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

        # Compare to fastjet...
        # NOTE: It will be slightly different because fj use rapidity instead of eta, but the general shape should
        #       be similar, I think.
        #jets_fj = [[fj.PseudoJet(j.x, j.y, j.z, j.t) for j in _jets_in_event] for _jets_in_event in jets.jets]
        #particles_fj = [
        #    [fj.PseudoJet(p.x, p.y, p.z, p.t) for p in
        #    LorentzVectorArray.from_ptetaphie(particles.pt, particles.eta, particles.phi, particles.E)]
        #    for particles in arrays[particle_mask]
        #]

        # This is slow, but probably easier and safer...
        #output = np.zeros(ak.sum(ak.num(jets, axis=1)) * ak.sum(ak.num(arrays[particle_mask], axis=1)), dtype=np.float64)
        counter = 0

        # About to calculate fj distances...
        # Cut to only 50 for technical reasons.
        distance_hist = bh.Histogram(bh.axis.Regular(160, 0, 8), storage=bh.storage.Weight())
        for jets_in_event, particles_in_event in zip(jets.jets[:50], base.LorentzVectorArray.from_ptetaphie(arrays[particle_mask].pt, arrays[particle_mask].eta, arrays[particle_mask].phi, arrays[particle_mask].E)[:50]):
            for jet in jets_in_event:
                jet_fj = fj.PseudoJet(jet.px, jet.py, jet.pz, jet.E)
                if counter % 10000 == 0:
                    print(f"Counter: {counter}")
                for p in particles_in_event:
                    particle_fj = fj.PseudoJet(p.px, p.py, p.pz, p.E)
                    distance_hist.fill(jet_fj.delta_R(particle_fj))
                    #output[counter] = jet_fj.delta_R(particle_fj)
                    counter += 1

        # Plot fj comparison
        # Create, fill, and plot the histogram
        #distance_hist.fill(output)
        hist = binned_data.BinnedData.from_existing_data(distance_hist)

        # Normalize
        # Bin widths
        hist /= hist.axes[0].bin_widths
        # N jets
        hist /= ak.sum(ak.num(jets, axis=1))

        ax_fj.errorbar(
            hist.axes[0].bin_centers,
            hist.values,
            xerr=hist.axes[0].bin_widths / 2,
            label=f"Status = {status_code}",
            marker="o",
            linestyle="",
        )

        hists_fj.append(hist)

        # Create, fill, and plot the eta distance histogram
        distance_hist = bh.Histogram(bh.axis.Regular(160, 0, 8), storage=bh.storage.Weight())
        distance_hist.fill(ak.to_numpy(delta_eta))
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
        distance_hist = bh.Histogram(bh.axis.Regular(80, 0, 4), storage=bh.storage.Weight())
        distance_hist.fill(ak.to_numpy(delta_phi))
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

        fig_eta_phi, ax_eta_phi = plt.subplots(figsize=(8, 6))

        # Delta eta, delta phi
        distance_from_jet = delta_phi
        distance_hist = bh.Histogram(bh.axis.Regular(80, -4, 4), bh.axis.Regular(80, -3 * np.pi, 3 * np.pi), storage=bh.storage.Weight())
        distance_hist.fill(ak.to_numpy(delta_eta), ak.to_numpy(delta_phi))
        hist = binned_data.BinnedData.from_existing_data(distance_hist)
        hist.values[hist.values == 0] = np.nan
        z_axis_range = {
            # "vmin": h_proj.values[h_proj.values > 0].min(),
            # Don't show values at 0.
            "vmin": 1,
            # Account for the possibility of having no values.
            "vmax": np.nanmax(hist.values) if (hist.values).any() else 1.0,
        }
        # Plot
        mesh = ax_eta_phi.pcolormesh(
            hist.axes[0].bin_edges.T, hist.axes[1].bin_edges.T, hist.values.T, norm=matplotlib.colors.Normalize(**z_axis_range),
        )
        fig_eta_phi.colorbar(mesh, pad=0.02)

        fig_eta_phi.tight_layout()
        fig_eta_phi.savefig(output_dir / f"delta_eta_phi_from_jet_{pt_hat_bin[0]}_{pt_hat_bin[1]}_status_{status_code}.pdf")
        plt.close(fig_eta_phi)

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
    # For fj comparison
    h_fj_all = sum(hists_fj)
    ax_fj.errorbar(
        h_fj_all.axes[0].bin_centers,
        h_fj_all.values,
        xerr=h_fj_all.axes[0].bin_widths / 2,
        label=f"Sum",
        marker="o",
        linestyle="",
    )

    # Label
    for a in [ax, ax_fj, ax_eta, ax_phi]:
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
    ax_fj.set_ylabel(r"$1/N_{\text{jets}} \text{d}N/\text{d}R$")
    ax_fj.set_xlabel(r"Distance from jet axis")
    ax_eta.set_ylabel(r"$1/N_{\text{jets}} \text{d}N/\text{d}\eta$")
    ax_eta.set_xlabel(r"$\eta$ from jet axis")
    ax_phi.set_ylabel(r"$1/N_{\text{jets}} \text{d}N/\text{d}\varphi$")
    ax_phi.set_xlabel(r"$\varphi$ from jet axis")

    fig.tight_layout()
    fig.savefig(output_dir / f"distance_from_jet_{pt_hat_bin[0]}_{pt_hat_bin[1]}.pdf")
    plt.close(fig)
    fig_fj.tight_layout()
    fig_fj.savefig(output_dir / f"distance_from_jet_fj_{pt_hat_bin[0]}_{pt_hat_bin[1]}.pdf")
    plt.close(fig_fj)

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
