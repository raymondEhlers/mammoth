""" Analyze JETSCAPE events.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import functools
import operator
from typing import Mapping, Optional, Sequence, Tuple
from pathlib import Path

import awkward1 as ak
import numba as nb
import numpy as np
import particle
import pyfastjet as fj
from pachyderm import binned_data

from jetscape_an import base


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

    return builder


def build_PID_selection(particles: ak.Array, absolute_pids: Optional[Sequence[int]] = None, single_pids: Optional[Sequence[int]] = None):
    # Validation
    if absolute_pids is None:
        absolute_pids = []
    if single_pids is None:
        single_pids = []

    pid_cuts = [
        np.abs(particles["particle_ID"]) == pid for pid in absolute_pids
    ]
    pid_cuts.extend([
        particles["particle_ID"] == pid for pid in single_pids
    ])
    return functools.reduce(operator.and_, pid_cuts)


def add_masses_to_array(events: ak.Array) -> ak.Array:
    # Add the masses based on the PDG code.
    # First, determine all possible PDG codes, and then retrieve their masses for a lookup table.
    all_particle_IDs = np.unique(ak.to_numpy(ak.flatten(events["particle_ID"])))
    # NOTE: We need to use this special typed dict with numba.
    pdg_id_to_mass = nb.typed.Dict.empty(
        key_type=nb.core.types.int64,
        value_type=nb.core.types.float64,
    )
    # As far as I can tell, we can't fill the dict directly on initialization, so we have to loop over entires.
    for pdg_id in all_particle_IDs:
        pdg_id_to_mass[pdg_id] = _pdg_id_to_mass(pdg_id)
    # It's important that we snapshot it!
    return determine_mass(events=events, builder=ak.ArrayBuilder(), pdg_id_to_mass=pdg_id_to_mass).snapshot()

@nb.njit
def _delta_phi(phi_a: float, phi_b: float, output_range: Tuple[float, float] = (-np.pi, np.pi)) -> float:
    # Ensure that they're in the same range.
    phi_a = phi_a % (2 * np.pi)
    phi_b = phi_b % (2 * np.pi)
    delta_phi = phi_a - phi_b
    if delta_phi < output_range[0]:
        delta_phi += (2 * np.pi)
    elif delta_phi > output_range[1]:
        delta_phi -= (2 * np.pi)
    return delta_phi

@nb.njit
def _delta_R(eta_one: float, phi_one: float, eta_two: float, phi_two: float) -> float:
    return np.sqrt(_delta_phi(phi_one, phi_two) ** 2 + (eta_one - eta_two) ** 2)


@nb.njit
def _subtract_holes_from_jets_pt(jets_pt: ak.Array, jets_eta: ak.Array, jets_phi: ak.Array,
                              particles_holes_pt: ak.Array, particles_holes_eta: ak.Array, particles_holes_phi: ak.Array,
                              jet_R: float, builder: ak.ArrayBuilder) -> ak.Array:
    for jets_pt_in_event, jets_eta_in_event, jets_phi_in_event, holes_pt_in_event, holes_eta_in_event, holes_phi_in_event in \
            zip(jets_pt, jets_eta, jets_phi, particles_holes_pt, particles_holes_eta, particles_holes_phi):
        builder.begin_list()
        for jet_pt, jet_eta, jet_phi in zip(jets_pt_in_event, jets_eta_in_event, jets_phi_in_event):
            for p_pt, p_eta, p_phi in zip(holes_pt_in_event, holes_eta_in_event, holes_phi_in_event):
                if _delta_R(eta_one=jet_eta, phi_one=jet_phi, eta_two=p_eta, phi_two=p_phi) < jet_R:
                    jet_pt -= p_pt
            builder.append(jet_pt)
        builder.end_list()

    return builder

@nb.njit
def _calculate_leading_track_cut_mask(constituent_indices: ak.Array, particles_pt: ak.Array, particles_id: ak.Array, leading_track_cut: float, charged_particle_PIDs: Sequence[int], builder: ak.ArrayBuilder):
    for event_constituent_indices, event_particles_pt, event_particles_id in zip(constituent_indices, particles_pt, particles_id):
        builder.begin_list()
        for jet_constituent_indices in event_constituent_indices:
            passed = False
            for i in jet_constituent_indices:
                if event_particles_pt[i] > leading_track_cut:
                    for possible_PID in charged_particle_PIDs:
                        if event_particles_id[i] == possible_PID:
                            passed = True
                            break
                    # Make sure that we break all the way out
                    if passed:
                        break
            builder.append(passed)
        builder.end_list()

    return builder


def run(particles: ak.Array) -> None:

    # Particle setup and selection
    # Remove neutrinos.
    particles = particles[(np.abs(particles["particle_ID"]) != 12) & (np.abs(particles["particle_ID"]) != 14) & (np.abs(particles["particle_ID"]) != 16)]
    # Add the masses based on the PDG code.
    # First, determine all possible PDG codes, and then retrieve their masses for a lookup table.
    all_particle_IDs = np.unique(ak.to_numpy(ak.flatten(particles["particle_ID"])))
    # NOTE: We need to use this special typed dict with numba.
    pdg_id_to_mass = nb.typed.Dict.empty(
        key_type=nb.core.types.int64,
        value_type=nb.core.types.float64,
    )
    # As far as I can tell, we can't fill the dict directly on initialization, so we have to loop over entires.
    for pdg_id in all_particle_IDs:
        pdg_id_to_mass[pdg_id] = _pdg_id_to_mass(pdg_id)
    # Finally, store the result in the existing particles array.
    # It's important that we snapshot it!
    particles["m"] = add_masses_to_array(events=particles)
    # Convert to the expected LorentzVector format
    #particles = base.LorentzVectorArray.from_awkward_ptetaphim(particles["pt"], particles["eta"], particles["phi"], particles["m"])
    #particles = base.LorentzVectorArray.from_awkward_ptetaphim(particles)

    # Divide into signals and holes
    particles_signal = particles[particles["status"] == 0]
    particles_holes = particles[particles["status"] < 0]

    # Hadron RAA.
    # First, experiment defined particle selections:
    # Includes: (e-, mu-, pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-)
    _default_charged_hadron_PID = [11, 13, 211, 321, 2212, 3222, 3112, 3312, 3334]
    # NOTE: We don't want to include the holes.
    # ATLAS:
    # Charged hadrons: (pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-)
    # NOTE: They apparently exclude: e-, mu- (11, 13). Quoting from the AN:
    #       Since the main observables in this analysis are the charged hadron spectra, leptons arising from the decays of heavy vector bosons
    #       are excluded from the measured spectra.  Tracks forming part of reconstructedmuons  are  identified  and  the  contribution  from
    #       stable  leptons  is  subtracted  twice  from  the  measuredspectra, assuming that electrons contribute the same as muons.
    atlas_charged_hadrons = particles_signal[build_PID_selection(particles_signal, absolute_pids=_default_charged_hadron_PID[2:])]
    # Eta selection
    atlas_charged_hadrons = atlas_charged_hadrons[np.abs(atlas_charged_hadrons.eta) < 2.5]
    # ALICE:
    # Charged hadrons: Primary charged particles (w/ mean proper lifetime τ larger than 1 cm/c )
    # Pratically, that means: (e-, mu-, pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-)
    alice_charged_hadrons = particles_signal[build_PID_selection(particles_signal, absolute_pids=_default_charged_hadron_PID)]
    # Eta selection
    alice_charged_hadrons = alice_charged_hadrons[np.abs(alice_charged_hadrons.eta) < 0.8]
    # CMS:
    # Charged hadrons: (e-, mu-, pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-)
    cms_charged_hadrons = particles_signal[build_PID_selection(particles_signal, absolute_pids=_default_charged_hadron_PID)]
    # Eta selection
    cms_charged_hadrons = cms_charged_hadrons[np.abs(cms_charged_hadrons.eta) < 1.0]

    # Jet finding
    # Setup
    jet_R = 0.4
    jet_defintion = fj.JetDefinition(fj.JetAlgorithm.antikt_algorithm, R = jet_R)
    area_definition = fj.AreaDefinition(fj.AreaType.passive_area, fj.GhostedAreaSpec(1, 1, 0.05))
    settings = fj.JetFinderSettings(jet_definition=jet_defintion, area_definition=area_definition)
    # Jet finding is only performed on signal particles.
    #import IPython; IPython.embed()
    # For some reason, I need to do the LorentzVectorArray conversion here...
    particles_signal = base.LorentzVectorArray.from_awkward_ptetaphim(particles_signal)
    res = fj.find_jets(events=particles_signal, settings=settings)
    jets = res.jets
    constituent_indices = res.constituent_indices

    # Subtract holes from jet pt.
    jets["pt_subtracted"] = _subtract_holes_from_jets_pt(
        jets_pt=jets.pt, jets_eta=jets.eta, jets_phi=jets.phi,
        particles_holes_pt=particles_holes.pt, particles_holes_eta=particles_holes.eta, particles_holes_phi=particles_holes.phi,
        jet_R=jet_R, builder=ak.ArrayBuilder()
    ).snapshot()

    # Select jets
    # ATLAS
    atlas_jets = jets[np.abs(jets.rapidity) < 2.8]
    # ALICE
    alice_jets_eta_mask = np.abs(jets.eta) < 0.7 - jet_R
    alice_jets = jets[alice_jets_eta_mask]
    # Apply leading track cut (ie. charged hadron).
    # Assuming R = 0.4
    leading_track_cut_mask = _calculate_leading_track_cut_mask(
        constituent_indices=constituent_indices[alice_jets_eta_mask],
        particles_pt=particles_signal.pt,
        particles_id=particles_signal.particle_ID,
        leading_track_cut=7,
        charged_particle_PIDs=nb.typed.List(_default_charged_hadron_PID),
        builder=ak.ArrayBuilder(),
    ).snapshot()
    alice_jets = alice_jets[leading_track_cut_mask]
    # CMS
    cms_jets = jets[np.abs(jets.eta) < 2.0]

    import IPython; IPython.embed()


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

        run(particles = arrays)

