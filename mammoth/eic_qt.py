""" Analyze Pythia6 e-p events.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from typing import Dict, Mapping, Sequence, Tuple
from pathlib import Path

import awkward as ak
import boost_histogram as bh
import numba as nb
import numpy as np
import pyfastjet as fj
import uproot
from pachyderm import binned_data, yaml

from mammoth import base


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


def run(particles: ak.Array, hists: Mapping[str, bh.Histogram]) -> None:
    # Particle setup and selection
    # Select only final state particles, which according to AliGenPythiaPlus, are 1. Note that this excludes some semi-stable
    # particles, but we don't have that info on hand, and I don't think this should have a huge impact.
    particles = particles[particles["status"] == 1]
    # Remove neutrinos.
    particles = particles[(np.abs(particles["particle_ID"]) != 12) & (np.abs(particles["particle_ID"]) != 14) & (np.abs(particles["particle_ID"]) != 16)]

    # Add mass for calculating four vectors (if needed).
    # particles["m"] = base.determine_masses_from_events(events=particles)
    # Convert to the expected LorentzVector format
    #particles = base.LorentzVectorArray.from_awkward_ptetaphim(particles["pt"], particles["eta"], particles["phi"], particles["m"])
    #particles = base.LorentzVectorArray.from_awkward_ptetaphim(particles)

    # Require at least one electron.
    at_least_one_electron = (ak.count_nonzero(particles["particle_ID"] == 11, axis=1) > 0)
    particles = particles[at_least_one_electron]

    # Find our electrons for comparison
    electrons = particles[particles["particle_ID"] == 11]
    leading_electrons = base.LorentzVectorArray.from_awkward_ptetaphie(
        electrons[ak.argmax(electrons.pt, axis=1, keepdims=True)]
    )

    # TODO: Cleanup everything below. It has a bunch of unneeded crap

    ## Divide into signals and holes
    #particles_signal = particles[particles["status"] == 0]
    #particles_holes = particles[particles["status"] < 0]

    ## Hadron RAA.
    ## First, experiment defined particle selections:
    ## Includes: (e-, mu-, pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-)
    #_default_charged_hadron_PID = [11, 13, 211, 321, 2212, 3222, 3112, 3312, 3334]
    ## NOTE: We don't want to include the holes.
    ## ATLAS:
    ## Charged hadrons: (pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-)
    ## NOTE: They apparently exclude: e-, mu- (11, 13). Quoting from the AN:
    ##       Since the main observables in this analysis are the charged hadron spectra, leptons arising from the decays of heavy vector bosons
    ##       are excluded from the measured spectra.  Tracks forming part of reconstructedmuons  are  identified  and  the  contribution  from
    ##       stable  leptons  is  subtracted  twice  from  the  measuredspectra, assuming that electrons contribute the same as muons.
    #atlas_charged_hadrons = particles_signal[base.build_PID_selection_mask(particles_signal, absolute_pids=_default_charged_hadron_PID[2:])]
    ## Eta selection
    #atlas_charged_hadrons = atlas_charged_hadrons[np.abs(atlas_charged_hadrons.eta) < 2.5]
    ## ALICE:
    ## Charged hadrons: Primary charged particles (w/ mean proper lifetime τ larger than 1 cm/c )
    ## Pratically, that means: (e-, mu-, pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-)
    #alice_charged_hadrons = particles_signal[base.build_PID_selection_mask(particles_signal, absolute_pids=_default_charged_hadron_PID)]
    ## Eta selection
    #alice_charged_hadrons = alice_charged_hadrons[np.abs(alice_charged_hadrons.eta) < 0.8]
    ## CMS:
    ## Charged hadrons: (e-, mu-, pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-)
    #cms_charged_hadrons = particles_signal[base.build_PID_selection_mask(particles_signal, absolute_pids=_default_charged_hadron_PID)]
    ## Eta selection
    #cms_charged_hadrons = cms_charged_hadrons[np.abs(cms_charged_hadrons.eta) < 1.0]

    # Jet finding
    # Setup
    jet_R = 1.0
    jet_defintion = fj.JetDefinition(fj.JetAlgorithm.antikt_algorithm, R = jet_R)
    area_definition = fj.AreaDefinition(fj.AreaType.passive_area, fj.GhostedAreaSpec(1, 1, 0.05))
    settings = fj.JetFinderSettings(jet_definition=jet_defintion, area_definition=area_definition)
    # Jet finding is only performed on signal particles.
    #import IPython; IPython.embed()
    # For some reason, I need to do the LorentzVectorArray conversion here...
    particles = base.LorentzVectorArray.from_awkward_ptetaphie(particles)
    res = fj.find_jets(events=particles, settings=settings)
    jets = res.jets
    constituent_indices = res.constituent_indices

    # Jet selection
    jets = jets[(jets.eta > 1 + jet_R) & (jets.eta < 4 - jet_R)]
    # No jet pt cut for now...

    # Calculate qt
    qt = np.sqrt((leading_electrons[:, np.newaxis].px + jets.px) ** 2 + (leading_electrons[:, np.newaxis].py + jets.py) ** 2)

    #import IPython; IPython.embed()

    try:
        hists["qt"].fill(ak.flatten(jets.pt, axis=None), ak.flatten(qt, axis=None))
    except ValueError as e:
        print(f"Womp womp: {e}")
        import IPython; IPython.embed()

    # # Subtract holes from jet pt.
    # jets["pt_subtracted"] = _subtract_holes_from_jets_pt(
    #     jets_pt=jets.pt, jets_eta=jets.eta, jets_phi=jets.phi,
    #     particles_holes_pt=particles_holes.pt, particles_holes_eta=particles_holes.eta, particles_holes_phi=particles_holes.phi,
    #     jet_R=jet_R, builder=ak.ArrayBuilder()
    # ).snapshot()

    # # Select jets
    # # ATLAS
    # atlas_jets = jets[np.abs(jets.rapidity) < 2.8]
    # # ALICE
    # alice_jets_eta_mask = np.abs(jets.eta) < 0.7 - jet_R
    # alice_jets = jets[alice_jets_eta_mask]
    # # Apply leading track cut (ie. charged hadron).
    # # Assuming R = 0.4
    # leading_track_cut_mask = _calculate_leading_track_cut_mask(
    #     constituent_indices=constituent_indices[alice_jets_eta_mask],
    #     particles_pt=particles_signal.pt,
    #     particles_id=particles_signal.particle_ID,
    #     leading_track_cut=7,
    #     charged_particle_PIDs=nb.typed.List(_default_charged_hadron_PID),
    #     builder=ak.ArrayBuilder(),
    # ).snapshot()
    # alice_jets = alice_jets[leading_track_cut_mask]
    # # CMS
    # cms_jets = jets[np.abs(jets.eta) < 2.0]

    # import IPython; IPython.embed()


def setup() -> Dict[str, bh.Histogram]:
    hists = {}
    hists["qt"] = bh.Histogram(bh.axis.Regular(100, 0, 100), bh.axis.Regular(50, 0, 10), storage=bh.storage.Weight())

    return hists


if __name__ == "__main__":
    # Setup
    #input_file = Path("/alf/data/rehlers/eic/pythia6/writeTree_1000000.root")
    input_file = Path("/Volumes/data/eic/writeTree_1000000.root")
    hists = setup()

    #for i, arrays in enumerate(uproot.iterate(f"{input_file}:tree", step_size="200 MB")):
    for i, arrays in enumerate(uproot.iterate(f"{input_file}:tree", step_size="100 MB"), start=1):
        print(f"Processing iter {i}")
        # Drop event_ID so it doesn't mess with selections later.
        # It may not actually be included, but we remove it to be thorough.
        arrays = arrays[[k for k in ak.fields(arrays) if k != "event_ID"]]
        run(particles=arrays, hists=hists)

    print("Done. Writing hist...")
    # Write out...
    y = yaml.yaml(modules_to_register=[binned_data])
    h = binned_data.BinnedData.from_existing_data(hists["qt"])
    with open("qt.yaml", "w") as f:
        y.dump([h], f)

    import IPython; IPython.embed()

