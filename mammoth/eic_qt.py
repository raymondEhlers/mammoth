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
        min_q2: float,
        x_limits: Tuple[float, float],
        hists: Mapping[str, bh.Histogram]) -> None:
    # Particle setup and selection
    # Select only final state particles, which according to AliGenPythiaPlus, are 1. Note that this excludes some semi-stable
    # particles, but we don't have that info on hand, and I don't think this should have a huge impact.
    particles = particles[particles["status"] == 1]
    # Remove neutrinos.
    particles = particles[(np.abs(particles["particle_ID"]) != 12) & (np.abs(particles["particle_ID"]) != 14) & (np.abs(particles["particle_ID"]) != 16)]

    # Potentially only select charged hadrons...
    # Charged hadrons: Primary charged particles (w/ mean proper lifetime τ larger than 1 cm/c )
    # Pratically, that means: (e-, mu-, pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-)
    #_default_charged_hadron_PID = [11, 13, 211, 321, 2212, 3222, 3112, 3312, 3334]
    #charged_hadrons = particles[base.build_PID_selection_mask(particles, absolute_pids=_default_charged_hadron_PID)]

    # Event level cuts.
    # Require at least one electron.
    at_least_one_electron = (ak.count_nonzero(particles["particle_ID"] == 11, axis=1) > 0)
    particles = particles[at_least_one_electron]
    event_properties = event_properties[at_least_one_electron]
    # q2 and x selection
    min_q2_selection = event_properties["q2"] > min_q2
    # x1 is the electron because it is the projectile.
    x_selection = (event_properties["x1"] > x_limits[0]) & (event_properties["x2"] < x_limits[1])
    particles = particles[min_q2_selection & x_selection]
    event_properties = event_properties[min_q2_selection & x_selection]

    # Find our electrons for comparison
    electrons = particles[particles["particle_ID"] == 11]
    leading_electrons = base.LorentzVectorArray.from_awkward_ptetaphie(
        electrons[ak.argmax(electrons.pt, axis=1, keepdims=True)]
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
    # No jet pt cut for now...

    # Calculate qt
    qt = np.sqrt((leading_electrons[:, np.newaxis].px + jets.px) ** 2 + (leading_electrons[:, np.newaxis].py + jets.py) ** 2)
    # TODO: Does a jet pt cut matter here?

    try:
        hists["qt"].fill(ak.flatten(jets.pt, axis=None), ak.flatten(qt, axis=None))
        hists["qt_pt"].fill(ak.flatten(jets.p, axis=None), ak.flatten(qt / jets.pt, axis=None))
        hists["qt_pte"].fill(ak.flatten(jets.p, axis=None), ak.flatten(qt / leading_electrons[:, np.newaxis].pt, axis=None))
    except ValueError as e:
        print(f"Womp womp: {e}")
        import IPython; IPython.embed()


def setup() -> Dict[str, bh.Histogram]:
    hists = {}
    hists["qt"] = bh.Histogram(bh.axis.Regular(100, 0, 100), bh.axis.Regular(50, 0, 10), storage=bh.storage.Weight())
    hists["qt_pt"] = bh.Histogram(bh.axis.Regular(30, 0, 300), bh.axis.Regular(20, 0, 1), storage=bh.storage.Weight())
    hists["qt_pte"] = bh.Histogram(bh.axis.Regular(30, 0, 300), bh.axis.Regular(20, 0, 1), storage=bh.storage.Weight())

    return hists


if __name__ == "__main__":
    # Setup
    jet_R = 0.7
    jet_eta_limits = (1.1, 3.5)
    min_q2 = 100
    x_limits = (0.05, 0.8)
    #input_file = Path("/alf/data/rehlers/eic/pythia6/writeTree_1000000.root")
    #input_file = Path("/Volumes/data/eic/writeTree_1000000.root")
    input_file = Path("/Volumes/data/eic/writeTree_1e6.root")
    hists = setup()

    for i, arrays in enumerate(uproot.iterate(f"{input_file}:tree", step_size="100 MB"), start=1):
        print(f"Processing iter {i}")
        # Split into event level and particle level properties. This makes working with the data
        # (slicing, etc) much easier.
        event_property_names = ["event_ID", "x1", "x2", "q2"]
        event_properties = arrays[[k for k in ak.fields(arrays) if k in event_property_names]]
        particles = arrays[[k for k in ak.fields(arrays) if k not in event_property_names]]
        run(
            event_properties=event_properties,
            particles=particles,
            jet_R=jet_R,
            jet_eta_limits=jet_eta_limits,
            min_q2=min_q2,
            x_limits=x_limits,
            hists=hists
        )

    print("Done. Writing hist...")
    # Write out...
    output_dir = Path("output") / "eic_qt"
    y = yaml.yaml(modules_to_register=[binned_data])
    #h = binned_data.BinnedData.from_existing_data(hists["qt"])
    with open(output_dir / "qt.yaml", "w") as f:
        y.dump({k: binned_data.BinnedData.from_existing_data(v) for k, v in hists.items()}, f)

    import IPython; IPython.embed()

