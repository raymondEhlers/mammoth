""" Tests for jet finding

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import pytest  # noqa: F401

import awkward as ak
import numpy as np
import vector

from mammoth.framework import jet_finding

def test_jet_finding_basic_single_event() -> None:
    """ Basic jet finding test with a single event. """
    # Setup
    vector.register_awkward()

    # an event with three particles:   px    py  pz      E
    input_particles = ak.zip(
        {
            "px": [[99.0, 4.0, -99.0]],
            "py": [[0.1, -0.1, 0]],
            "pz": [[0, 0, 0]],
            "E": [[100.0, 5.0, 99.0]],
        },
        with_name="Momentum4D",
    )
    print(f"input particles array type: {ak.type(input_particles)}")
    jets = jet_finding.find_jets(particles=input_particles, jet_R=0.7, algorithm="anti-kt", area_settings=jet_finding.AREA_AA)

    expected_jets = ak.zip(
        {
            "px": [[103.0, -99.0]],
            "py": [[0.0, 0.0]],
            "pz": [[0.0, 0.0]],
            "E": [[105.0, 99.0]],
        },
        with_name="Momentum4D",
    )

    print(f"input_particles: {input_particles.to_list()}")
    print(f"jets: {jets.to_list()}")
    print(f"expected_jets: {expected_jets.to_list()}")

    # Check four momenta
    assert all([np.allclose(np.asarray(measured.px), np.asarray(expected.px))
                and np.allclose(np.asarray(measured.py), np.asarray(expected.py))
                and np.allclose(np.asarray(measured.pz), np.asarray(expected.pz))
                and np.allclose(np.asarray(measured.E), np.asarray(expected.E))
                for event, event_expected in zip(jets, expected_jets) for measured, expected in zip(event, event_expected)])

    # only for testing - we want to see any fastjet warnings
    #assert False


def test_jet_finding_basic_multiple_events() -> None:
    """ Basic jet finding test with for multiple events. """
    # Setup
    vector.register_awkward()

    # an event with three particles
    # First event is the standard fastjet test particles,
    # while the second event is the standard with px <-> py.
    # The output jets should be the same as well, but with px <-> py.
    input_particles = ak.zip(
        {
            "px": [
                [99.0, 4.0, -99.0],
                [0.1, -0.1, 0],
            ],
            "py": [
                [0.1, -0.1, 0],
                [99.0, 4.0, -99.0],
            ],
            "pz": [
                [0, 0, 0],
                [0, 0, 0],
            ],
            "E": [
                [100.0, 5.0, 99.0],
                [100.0, 5.0, 99.0],
            ],
        },
        with_name="Momentum4D",
    )
    print(f"input particles array type: {ak.type(input_particles)}")
    jets = jet_finding.find_jets(particles=input_particles, jet_R=0.7, algorithm="anti-kt", area_settings=jet_finding.AREA_AA)

    expected_jets = ak.zip(
        {
            "px": [
                [103.0, -99.0],
                [0.0, 0.0],
            ],
            "py": [
                [0.0, 0.0],
                [103.0, -99.0],
            ],
            "pz": [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            "E": [
                [105.0, 99.0],
                [105.0, 99.0],
            ],
        },
        with_name="Momentum4D",
    )

    print(f"input_particles: {input_particles.to_list()}")
    print(f"jets: {jets.to_list()}")
    print(f"expected_jets: {expected_jets.to_list()}")

    # Check four momenta
    assert all([np.allclose(np.asarray(measured.px), np.asarray(expected.px))
                and np.allclose(np.asarray(measured.py), np.asarray(expected.py))
                and np.allclose(np.asarray(measured.pz), np.asarray(expected.pz))
                and np.allclose(np.asarray(measured.E), np.asarray(expected.E))
                for event, event_expected in zip(jets, expected_jets) for measured, expected in zip(event, event_expected)])

    # only for testing - we want to see any fastjet warnings
    #assert False


def test_jet_finding_with_subtraction_multiple_events() -> None:
    """ Jet finding with subtraction for multiple events. """
    # Setup
    vector.register_awkward()

    # an event with three particles
    # First event is the standard fastjet test particles,
    # while the second event is the standard with px <-> py.
    # The output jets should be the same as well, but with px <-> py.
    input_particles = ak.zip(
        {
            "px": [
                [99.0, 4.0, -99.0],
                [0.1, -0.1, 0],
            ],
            "py": [
                [0.1, -0.1, 0],
                [99.0, 4.0, -99.0],
            ],
            "pz": [
                [0, 0, 0],
                [0, 0, 0],
            ],
            "E": [
                [100.0, 5.0, 99.0],
                [100.0, 5.0, 99.0],
            ],
        },
        with_name="Momentum4D",
    )
    print(f"input particles array type: {ak.type(input_particles)}")
    jets = jet_finding.find_jets(particles=input_particles, jet_R=0.7,
                                 algorithm="anti-kt",
                                 area_settings=jet_finding.AREA_AA,
                                 background_subtraction=True)

    expected_jets = ak.zip(
        {
            "px": [
                [103.0, -99.0],
                [0.0, 0.0],
            ],
            "py": [
                [0.0, 0.0],
                [103.0, -99.0],
            ],
            "pz": [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            "E": [
                [105.0, 99.0],
                [105.0, 99.0],
            ],
        },
        with_name="Momentum4D",
    )

    print(f"input_particles: {input_particles.to_list()}")
    print(f"jets: {jets.to_list()}")
    print(f"expected_jets: {expected_jets.to_list()}")

    # Check four momenta
    assert all([np.allclose(np.asarray(measured.px), np.asarray(expected.px))
                and np.allclose(np.asarray(measured.py), np.asarray(expected.py))
                and np.allclose(np.asarray(measured.pz), np.asarray(expected.pz))
                and np.allclose(np.asarray(measured.E), np.asarray(expected.E))
                for event, event_expected in zip(jets, expected_jets) for measured, expected in zip(event, event_expected)])

    # only for testing - we want to see any fastjet warnings
    #assert False


def test_jet_finding_with_constituent_subtraction_does_something_multiple_events() -> None:
    """ Jet finding with constituent subtraction modifies the jets somehow for multiple events.

    NOTE:
        This doesn't test that CS gives a particular expected result - just that it modifies the jets.
        This is because there's no simple reference. So we have to validate elsewhere.
    """
    # Setup
    vector.register_awkward()

    # an event with three particles
    # First event is the standard fastjet test particles,
    # while the second event is the standard with px <-> py.
    # The output jets should be the same as well, but with px <-> py.
    input_particles = ak.zip(
        {
            "px": [
                [99.0, 4.0, -99.0],
                [0.1, -0.1, 0],
            ],
            "py": [
                [0.1, -0.1, 0],
                [99.0, 4.0, -99.0],
            ],
            "pz": [
                [0, 0, 0],
                [0, 0, 0],
            ],
            "E": [
                [100.0, 5.0, 99.0],
                [100.0, 5.0, 99.0],
            ],
        },
        with_name="Momentum4D",
    )
    print(f"input particles array type: {ak.type(input_particles)}")
    jets = jet_finding.find_jets(particles=input_particles, jet_R=0.7,
                                 algorithm="anti-kt",
                                 area_settings=jet_finding.AREA_AA,
                                 constituent_subtraction=jet_finding.ConstituentSubtractionSettings(
                                     r_max=0.25,
                                     alpha=1.0,
                                 ))

    expected_jets = ak.zip(
        {
            "px": [
                [103.0, -99.0],
                [0.0, 0.0],
            ],
            "py": [
                [0.0, 0.0],
                [103.0, -99.0],
            ],
            "pz": [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            "E": [
                [105.0, 99.0],
                [105.0, 99.0],
            ],
        },
        with_name="Momentum4D",
    )

    print(f"input_particles: {input_particles.to_list()}")
    print(f"jets: {jets.to_list()}")
    print(f"expected_jets: {expected_jets.to_list()}")

    # Check four momenta
    # Here, we expect it _not_ to agree with the "expected jets" from above because
    # consitutent subtraction has modified the four vectors. We don't have a simple
    # and convenient reference, so we effectively require that it is changed _somehow_
    # by the constituent subtraction. It will have to be validated elsewhere.
    assert not all([np.allclose(np.asarray(measured.px), np.asarray(expected.px))
                    and np.allclose(np.asarray(measured.py), np.asarray(expected.py))
                    and np.allclose(np.asarray(measured.pz), np.asarray(expected.pz))
                    and np.allclose(np.asarray(measured.E), np.asarray(expected.E))
                    for event, event_expected in zip(jets, expected_jets) for measured, expected in zip(event, event_expected)])

    # only for testing - we want to see any fastjet warnings
    #assert False