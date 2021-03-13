""" Analyze Pythia6 e-p events.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from typing import Dict, Mapping, Sequence, Tuple
from pathlib import Path

import awkward as ak
import boost_histogram as bh
import numpy as np
import pyfastjet as fj
import uproot
from pachyderm import binned_data, yaml

from mammoth import base

def run(event_properties: ak.Array,
        particles: ak.Array,
        jet_R: float,
        jet_eta_limits: Tuple[float, float],
        min_Q2: float,
        x_limits: Tuple[float, float],
        hists: Mapping[str, bh.Histogram]) -> None:
    # The outgoing parton always seems to be in index 7 (pythia index #8)
    # Need to be retrieved immediately because it will be cut in the "status" cut.
    outgoing_partons = particles[:, 7]

    # Particle setup and selection
    # Select only final state particles, which according to AliGenPythiaPlus, are 1. Note that this excludes some semi-stable
    # particles, but we don't have that info on hand, and I don't think this should have a huge impact.
    particles = particles[particles["status"] == 1]
    # Remove neutrinos.
    particles = particles[(np.abs(particles["particle_ID"]) != 12) & (np.abs(particles["particle_ID"]) != 14) & (np.abs(particles["particle_ID"]) != 16)]
    # To avoid anything that's too soft, require E of at least 50 MeV.
    particles = particles[particles.E > 0.005]

    # Potentially only select charged hadrons...
    # Charged hadrons: Primary charged particles (w/ mean proper lifetime τ larger than 1 cm/c )
    # Pratically, that means: (e-, mu-, pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-)
    #_default_charged_hadron_PID = [11, 13, 211, 321, 2212, 3222, 3112, 3312, 3334]
    #charged_hadrons = particles[base.build_PID_selection_mask(particles, absolute_pids=_default_charged_hadron_PID)]

    # Event level cuts.
    # Require at least one electron.
    at_least_one_electron = (ak.count_nonzero(particles["particle_ID"] == 11, axis=1) > 0)
    particles = particles[at_least_one_electron]
    outgoing_partons = outgoing_partons[at_least_one_electron]
    event_properties = event_properties[at_least_one_electron]
    # Q2 and x selections are made during generation.
    ## q2 and x selection
    #min_Q2_selection = event_properties["q2"] > min_Q2
    ## x1 is the electron because it is the projectile.
    #x_selection = (event_properties["x1"] > x_limits[0]) & (event_properties["x2"] < x_limits[1])
    #particles = particles[min_Q2_selection & x_selection]
    #event_properties = event_properties[min_Q2_selection & x_selection]

    # Convert the outgoing partons to LorentzVectors.
    outgoing_partons = base.LorentzVectorArray.from_awkward_ptetaphie(outgoing_partons)

    # Find our electrons for comparison
    electrons_mask = (particles["particle_ID"] == 11)
    # Need to concretely select one of the variables for the mask to work properly.
    electrons_pt = ak.mask(particles.pt, electrons_mask)
    leading_electrons_mask = ak.argmax(electrons_pt, axis=1, keepdims=True)
    leading_electrons = base.LorentzVectorArray.from_awkward_ptetaphie(
        #electrons[ak.argmax(electrons.pt, axis=1, keepdims=True)]
        particles[leading_electrons_mask]
    )

    # We want to remove all of the leading electrons from our particles for jet finding.
    # We have the leading electrons indices, but we need a way to remove them.
    # We do this by keeping particles with a local_index that doesn't match the argmax from the electron.
    # This can't possibility be the best way to do this, but it seems to work, so I'll just take it...
    particles_without_leading_electron_mask = ak.firsts(
        ak.local_index(particles.pt) != leading_electrons_mask[:, np.newaxis],
        axis=-1
    )
    # There is nothing empty, so filling none just to get rid of the "?"
    particles = ak.fill_none(
        particles[particles_without_leading_electron_mask],
        -99999,
    )

    # Jet finding
    # Setup
    jet_defintion = fj.JetDefinition(fj.JetAlgorithm.antikt_algorithm, R = jet_R)
    area_definition = fj.AreaDefinition(fj.AreaType.passive_area, fj.GhostedAreaSpec(1, 1, 0.05))
    settings = fj.JetFinderSettings(jet_definition=jet_defintion, area_definition=area_definition)
    # Run the jet finder
    particles = base.LorentzVectorArray.from_awkward_ptetaphie(particles)
    res = fj.find_jets(events=particles, settings=settings)
    jets = res.jets
    constituent_indices = res.constituent_indices

    # Jet selection
    # Select forward jets.
    jets = jets[(jets.eta > jet_eta_limits[0] + jet_R) & (jets.eta < jet_eta_limits[1] - jet_R)]
    # Take only the leading jet.
    jets = ak.firsts(jets)

    # Calculate qt
    qt = np.sqrt((leading_electrons[:, np.newaxis].px + jets.px) ** 2 + (leading_electrons[:, np.newaxis].py + jets.py) ** 2)

    # NOTE: A small number of events won't have jets in the acceptance. So we need to make sure the
    #       event level properties match. Since qt uses the jets explicitly, it will pick this up naturally.
    #       The outgoing_partons also broadcats with qt, so it doesn't need any modifications here.
    event_properties = event_properties[~ak.is_none(jets)]

    # Print the means out of curiosity. Saved in histograms below.
    print(f"Mean Q2: {ak.mean(event_properties['q2'])}")
    print(f"Mean x: {ak.mean(event_properties['x2'])}")

    try:
        hists["qt"].fill(ak.flatten(jets.pt, axis=None), ak.flatten(qt, axis=None))
        hists["qt_pt_jet"].fill(ak.flatten(jets.p, axis=None), ak.flatten(qt / jets.pt, axis=None))
        hists["qt_pt_electron"].fill(ak.flatten(jets.p, axis=None), ak.flatten(qt / leading_electrons[:, np.newaxis].pt, axis=None))
        hists["qt_pt_parton"].fill(ak.flatten(jets.p, axis=None), ak.flatten(qt / outgoing_partons[:, np.newaxis].pt, axis=None))
        hists["q2"].fill(ak.flatten(jets.p, axis=None), sample=ak.flatten(event_properties.q2, axis=None))
        hists["x"].fill(ak.flatten(jets.p, axis=None), sample=ak.flatten(event_properties.x2, axis=None))
    except ValueError as e:
        print(f"Womp womp: {e}")
        import IPython; IPython.embed()


def setup() -> Dict[str, bh.Histogram]:
    hists = {}
    hists["qt"] = bh.Histogram(bh.axis.Regular(100, 0, 100), bh.axis.Regular(50, 0, 10), storage=bh.storage.Weight())
    hists["qt_pt_jet"] = bh.Histogram(bh.axis.Regular(30, 0, 300), bh.axis.Regular(20, 0, 1), storage=bh.storage.Weight())
    hists["qt_pt_electron"] = bh.Histogram(bh.axis.Regular(30, 0, 300), bh.axis.Regular(20, 0, 1), storage=bh.storage.Weight())
    hists["qt_pt_parton"] = bh.Histogram(bh.axis.Regular(30, 0, 300), bh.axis.Regular(20, 0, 1), storage=bh.storage.Weight())
    #hists["q2"] = bh.Histogram(bh.axis.Regular(30, 0, 300), bh.axis.Regular(100, 0, 1000), storage=bh.storage.Weight())
    #hists["x"] = bh.Histogram(bh.axis.Regular(30, 0, 300), bh.axis.Regular(100, 0, 1), storage=bh.storage.Weight())
    hists["q2"] = bh.Histogram(bh.axis.Regular(6, 0, 300), storage=bh.storage.WeightedMean())
    hists["x"] = bh.Histogram(bh.axis.Regular(6, 0, 300), storage=bh.storage.WeightedMean())

    return hists


if __name__ == "__main__":
    # Setup
    jet_R = 0.7
    jet_eta_limits = (1.1, 3.5)
    # As of 13 March 2021, we don't set the Q2 and x limits here. Instead, we set them in the simulation.
    # It appears to be much more efficient that way.
    min_Q2 = 100
    x_limits = (0.05, 0.8)
    #input_file = Path("/alf/data/rehlers/eic/pythia6/writeTree_1000000.root")
    #input_file = Path("/Volumes/data/eic/prod/writeTree_nevents_200_x_q2_index_0.root")
    #input_file = Path("/Volumes/data/eic/prod/writeTree_nevents_*_p_trigger_*_x_0.1_1_q2_100_index_*.root")
    #output_dir = Path("output") / "eic_qt_test"
    input_file = Path("/Volumes/data/eic/prod/writeTree_nevents_*_p_trigger_*_x_0.1_1_q2_*_index_*.root")
    output_dir = Path("output") / "eic_qt_test_all_q2_cuts"
    output_dir.mkdir(parents=True, exist_ok=True)

    hists = setup()

    results = []
    for i, arrays in enumerate(uproot.iterate(f"{input_file}:tree", step_size="100 MB"), start=1):
        print(f"Processing iter {i}")
        # Split into event level and particle level properties. This makes working with the data
        # (slicing, etc) much easier.
        event_property_names = ["event_ID", "x1", "x2", "q2"]
        event_properties = arrays[[k for k in ak.fields(arrays) if k in event_property_names]]
        particles = arrays[[k for k in ak.fields(arrays) if k not in event_property_names]]
        results.append(
            run(
                event_properties=event_properties,
                particles=particles,
                jet_R=jet_R,
                jet_eta_limits=jet_eta_limits,
                min_Q2=min_Q2,
                x_limits=x_limits,
                hists=hists
            )
        )

    # Do some projections here to ensure that we get them right. We won't get them right later
    # because we convert the boost histogram type implicitly when converting to binned_data.
    means = {}
    p_ranges = [(100, 150), (150, 200), (200, 250), (100, 250), (0, 300)]
    for p_range in p_ranges:
        means[p_range] = {
            "x": hists["x"][bh.loc(p_range[0]):bh.loc(p_range[1]):bh.sum].value,
            "Q2": hists["q2"][bh.loc(p_range[0]):bh.loc(p_range[1]):bh.sum].value,
        }

    print("Done. Writing hist + info...")
    # Write out...
    y = yaml.yaml(modules_to_register=[binned_data])
    #h = binned_data.BinnedData.from_existing_data(hists["qt"])
    with open(output_dir / "qt.yaml", "w") as f:
        output = {k: binned_data.BinnedData.from_existing_data(v) for k, v in hists.items()}
        output["means"] = means
        y.dump(output, f)

    import IPython; IPython.embed()

