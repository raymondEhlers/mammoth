"""Analyze Pythia6 e-p events.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import awkward as ak
import boost_histogram as bh
import numpy as np
import uproot
import vector
from pachyderm import binned_data, yaml

from mammoth.framework import jet_finding


def jet_R_to_str(jet_R: float) -> str:
    return f"jet_R_{jet_R * 100:03g}"


def run(
    event_properties: ak.Array,
    particles: ak.Array,
    jet_R_values: Sequence[float],
    jet_eta_limits: tuple[float, float],
    min_Q2: float,  # noqa: ARG001
    x_limits: tuple[float, float],  # noqa: ARG001
    hists: Mapping[str, Mapping[str, bh.Histogram]],
) -> None:
    # The outgoing parton always seems to be in index 7 (pythia index #8)
    # Need to be retrieved immediately because it will be cut in the "status" cut.
    outgoing_partons = particles[:, 7]

    # Particle setup and selection
    # Select only final state particles, which according to AliGenPythiaPlus, are 1. Note that this excludes some semi-stable
    # particles, but we don't have that info on hand, and I don't think this should have a huge impact.
    particles = particles[particles["status"] == 1]
    # Remove neutrinos.
    particles = particles[
        (np.abs(particles["particle_ID"]) != 12)
        & (np.abs(particles["particle_ID"]) != 14)
        & (np.abs(particles["particle_ID"]) != 16)
    ]
    # To avoid anything that's too soft, require E of at least 50 MeV.
    particles = particles[particles.E > 0.05]

    # Potentially only select charged hadrons...
    # Charged hadrons: Primary charged particles (w/ mean proper lifetime τ larger than 1 cm/c )
    # Practically, that means: (e-, mu-, pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-)
    # _default_charged_hadron_PID = [11, 13, 211, 321, 2212, 3222, 3112, 3312, 3334]
    # charged_hadrons = particles[base.build_PID_selection_mask(particles, absolute_pids=_default_charged_hadron_PID)]

    # Event level cuts.
    # Require at least one electron.
    at_least_one_electron = ak.count_nonzero(particles["particle_ID"] == 11, axis=1) > 0
    particles = particles[at_least_one_electron]
    outgoing_partons = outgoing_partons[at_least_one_electron]
    event_properties = event_properties[at_least_one_electron]
    # Q2 and x selections are made during generation.
    ## q2 and x selection
    # min_Q2_selection = event_properties["q2"] > min_Q2
    ## x1 is the electron because it is the projectile.
    # x_selection = (event_properties["x1"] > x_limits[0]) & (event_properties["x2"] < x_limits[1])
    # particles = particles[min_Q2_selection & x_selection]
    # event_properties = event_properties[min_Q2_selection & x_selection]

    # Convert the outgoing partons to LorentzVectors.
    outgoing_partons = vector.Array(outgoing_partons)

    # Find our electrons for comparison
    electrons_mask = particles["particle_ID"] == 11
    # Need to concretely select one of the variables for the mask to work properly.
    electrons_pt = ak.mask(particles.pt, electrons_mask)
    leading_electrons_mask = ak.argmax(electrons_pt, axis=1, keepdims=True)
    leading_electrons = particles[leading_electrons_mask]

    # We want to remove all of the leading electrons from our particles for jet finding.
    # We have the leading electrons indices, but we need a way to remove them.
    # We do this by keeping particles with a local_index that doesn't match the argmax from the electron.
    # This can't possibility be the best way to do this, but it seems to work, so I'll just take it...
    particles_without_leading_electron_mask = ak.firsts(
        ak.local_index(particles.pt) != leading_electrons_mask[:, np.newaxis], axis=-1
    )
    # There is nothing empty, so filling none just to get rid of the "?"
    particles = ak.fill_none(
        particles[particles_without_leading_electron_mask],
        -99999,
    )

    # Jet finding
    # Setup
    particles = vector.Array(particles)
    for jet_R in jet_R_values:
        print(f"Jet R: {jet_R}")  # noqa: T201
        # Run the jet finder
        jets = jet_finding.find_jets(
            particles=particles,
            jet_finding_settings=jet_finding.JetFindingSettings(
                R=jet_R,
                algorithm="anti_kt",
                pt_range=jet_finding.pt_range(),
                eta_range=jet_finding.eta_range(jet_R=jet_R, fiducial_acceptance=False, eta_min=-4.0, eta_max=4.0),
            ),
        )

        # Jet selection
        # Select forward jets.
        jets_eta_mask = (jets.eta > jet_eta_limits[0] + jet_R) & (jets.eta < jet_eta_limits[1] - jet_R)
        jets = jets[jets_eta_mask]
        constituent_indices = jets.constituent_indices[jets_eta_mask]

        # Take only the leading jet.
        jets = ak.firsts(jets)
        constituent_indices = ak.firsts(constituent_indices)

        # Calculate qt
        qt = np.sqrt(
            (leading_electrons[:, np.newaxis].px + jets.px) ** 2 + (leading_electrons[:, np.newaxis].py + jets.py) ** 2
        )

        # NOTE: A small number of events won't have jets in the acceptance. So we need to make sure the
        #       event level properties match. Since qt uses the jets explicitly, it will pick this up naturally.
        #       The outgoing_partons also broadcast with qt, so it doesn't need any modifications here.
        event_properties_R_dependent = event_properties[~ak.is_none(jets)]

        # Print the means out of curiosity. Saved in histograms below.
        print(f"Mean Q2: {ak.mean(event_properties_R_dependent['q2'])}")  # noqa: T201
        print(f"Mean x: {ak.mean(event_properties_R_dependent['x2'])}")  # noqa: T201

        try:
            jet_R_str = jet_R_to_str(jet_R)
            hists[jet_R_str]["jet_p"].fill(ak.flatten(jets.p, axis=None))
            hists[jet_R_str]["jet_pt"].fill(ak.flatten(jets.p, axis=None), ak.flatten(jets.pt, axis=None))
            hists[jet_R_str]["jet_multiplicity"].fill(
                ak.flatten(jets.p, axis=None), ak.flatten(ak.num(constituent_indices, axis=1), axis=None)
            )
            hists[jet_R_str]["qt"].fill(ak.flatten(jets.pt, axis=None), ak.flatten(qt, axis=None))
            hists[jet_R_str]["qt_pt_jet"].fill(ak.flatten(jets.p, axis=None), ak.flatten(qt / jets.pt, axis=None))
            hists[jet_R_str]["qt_pt_electron"].fill(
                ak.flatten(jets.p, axis=None), ak.flatten(qt / leading_electrons[:, np.newaxis].pt, axis=None)
            )
            hists[jet_R_str]["qt_pt_parton"].fill(
                ak.flatten(jets.p, axis=None), ak.flatten(qt / outgoing_partons[:, np.newaxis].pt, axis=None)
            )
            hists[jet_R_str]["q2"].fill(
                ak.flatten(jets.p, axis=None), sample=ak.flatten(event_properties_R_dependent.q2, axis=None)
            )
            hists[jet_R_str]["x"].fill(
                ak.flatten(jets.p, axis=None), sample=ak.flatten(event_properties_R_dependent.x2, axis=None)
            )
        except ValueError as e:
            print(f"Womp womp: {e}")  # noqa: T201
            import IPython

            IPython.embed()  # type: ignore[no-untyped-call]


def setup_hists() -> dict[str, bh.Histogram]:
    hists = {}
    hists["jet_p"] = bh.Histogram(bh.axis.Regular(600, 0, 300), storage=bh.storage.Weight())
    hists["jet_pt"] = bh.Histogram(
        bh.axis.Regular(30, 0, 300), bh.axis.Regular(200, 0, 50), storage=bh.storage.Weight()
    )
    hists["jet_multiplicity"] = bh.Histogram(
        bh.axis.Regular(30, 0, 300), bh.axis.Regular(30, 0, 30), storage=bh.storage.Weight()
    )
    hists["qt"] = bh.Histogram(bh.axis.Regular(100, 0, 100), bh.axis.Regular(200, 0, 10), storage=bh.storage.Weight())
    hists["qt_pt_jet"] = bh.Histogram(
        bh.axis.Regular(30, 0, 300), bh.axis.Regular(100, 0, 1), storage=bh.storage.Weight()
    )
    hists["qt_pt_electron"] = bh.Histogram(
        bh.axis.Regular(30, 0, 300), bh.axis.Regular(100, 0, 1), storage=bh.storage.Weight()
    )
    hists["qt_pt_parton"] = bh.Histogram(
        bh.axis.Regular(30, 0, 300), bh.axis.Regular(100, 0, 1), storage=bh.storage.Weight()
    )
    # hists["q2"] = bh.Histogram(bh.axis.Regular(30, 0, 300), bh.axis.Regular(100, 0, 1000), storage=bh.storage.Weight())
    # hists["x"] = bh.Histogram(bh.axis.Regular(30, 0, 300), bh.axis.Regular(100, 0, 1), storage=bh.storage.Weight())
    hists["q2"] = bh.Histogram(bh.axis.Regular(6, 0, 300), storage=bh.storage.WeightedMean())
    hists["x"] = bh.Histogram(bh.axis.Regular(6, 0, 300), storage=bh.storage.WeightedMean())

    return hists


if __name__ == "__main__":
    # Setup
    jet_R_values = [0.5, 0.7, 1.0]
    jet_eta_limits = (1.1, 3.5)
    # As of 13 March 2021, we don't set the Q2 and x limits here. Instead, we set them in the simulation.
    # It appears to be much more efficient that way.
    min_Q2 = 100
    x_limits = (0.05, 0.8)
    # input_file = Path("/alf/data/rehlers/eic/pythia6/writeTree_1000000.root")
    # input_file = Path("/Volumes/data/eic/prod/writeTree_nevents_200_x_q2_index_0.root")
    # input_file = Path("/Volumes/data/eic/prod/writeTree_nevents_*_p_trigger_*_x_0.1_1_q2_100_index_*.root")
    # output_dir = Path("output") / "eic_qt_test"
    input_file = Path("/Volumes/data/eic/prod/writeTree_nevents_*_p_trigger_*_x_0.1_1_q2_*_index_*.root")
    output_dir = Path("output") / "eic_qt_all_q2_cuts_narrow_bins_jet_R_dependence"
    output_dir.mkdir(parents=True, exist_ok=True)

    hists = {}
    for jet_R in jet_R_values:
        hists[jet_R_to_str(jet_R)] = setup_hists()

    for i, arrays in enumerate(uproot.iterate(f"{input_file}:tree", step_size="100 MB"), start=1):
        print(f"Processing iter {i}")  # noqa: T201
        # Split into event level and particle level properties. This makes working with the data
        # (slicing, etc) much easier.
        event_property_names = ["event_ID", "x1", "x2", "q2"]
        event_properties = arrays[[k for k in ak.fields(arrays) if k in event_property_names]]
        particles = arrays[[k for k in ak.fields(arrays) if k not in event_property_names]]
        run(
            event_properties=event_properties,
            particles=particles,
            jet_R_values=jet_R_values,
            jet_eta_limits=jet_eta_limits,
            min_Q2=min_Q2,
            x_limits=x_limits,
            hists=hists,
        )

    # Do some projections here to ensure that we get them right. We won't get them right later
    # because we convert the boost histogram type implicitly when converting to binned_data.
    means: dict[str, dict[tuple[int, int], dict[str, float]]] = {}
    p_ranges = [(100, 150), (150, 200), (200, 250), (100, 250), (0, 300)]
    for jet_R in jet_R_values:
        jet_R_str = jet_R_to_str(jet_R)
        means[jet_R_str] = {}
        for p_range in p_ranges:
            means[jet_R_str][p_range] = {
                "x": hists[jet_R_str]["x"][bh.loc(p_range[0]) : bh.loc(p_range[1]) : bh.sum].value,  # type: ignore[union-attr,misc]
                "Q2": hists[jet_R_str]["q2"][bh.loc(p_range[0]) : bh.loc(p_range[1]) : bh.sum].value,  # type: ignore[union-attr,misc]
            }

    print("Done. Writing hist + info...")  # noqa: T201
    # Write out...
    y = yaml.yaml(modules_to_register=[binned_data])
    # h = binned_data.BinnedData.from_existing_data(hists["qt"])
    with (output_dir / "qt.yaml").open("w") as f:
        output: dict[str, Any] = {}
        for jet_R_str, output_hists in hists.items():
            output[jet_R_str] = {k: binned_data.BinnedData.from_existing_data(v) for k, v in output_hists.items()}
        output["means"] = means
        y.dump(output, f)

    import IPython

    IPython.embed()  # type: ignore[no-untyped-call]
