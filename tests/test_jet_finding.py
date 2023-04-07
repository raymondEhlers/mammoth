""" Tests for jet finding

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
import pytest  # noqa: F401
from typing import Any

import awkward as ak
import numpy as np
import vector

from mammoth.framework import jet_finding

logger = logging.getLogger(__name__)


def test_jet_finding_basic_single_event(caplog: Any) -> None:
    """ Basic jet finding test with a single event. """
    # Setup
    caplog.set_level(logging.DEBUG)
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
    logger.info(f"input particles array type: {ak.type(input_particles)}")
    jets = jet_finding.find_jets(
        particles=input_particles,
        jet_finding_settings=jet_finding.JetFindingSettings(
            R=0.7, algorithm="anti-kt",
            area_settings=jet_finding.AreaAA(),
            pt_range=jet_finding.pt_range(),
            eta_range=jet_finding.eta_range(jet_R=0.7, fiducial_acceptance=False, eta_min=-5., eta_max=5.),
        )
    )

    expected_jets = ak.zip(
        {
            "px": [[103.0, -99.0]],
            "py": [[0.0, 0.0]],
            "pz": [[0.0, 0.0]],
            "E": [[105.0, 99.0]],
        },
        with_name="Momentum4D",
    )

    logger.info(f"input_particles: {input_particles.to_list()}")
    logger.info(f"jets: {jets.to_list()}")
    logger.info(f"expected_jets: {expected_jets.to_list()}")

    # Check four momenta
    assert all([np.allclose(np.asarray(measured.px), np.asarray(expected.px))
                and np.allclose(np.asarray(measured.py), np.asarray(expected.py))
                and np.allclose(np.asarray(measured.pz), np.asarray(expected.pz))
                and np.allclose(np.asarray(measured.E), np.asarray(expected.E))
                for event, event_expected in zip(jets, expected_jets) for measured, expected in zip(event, event_expected)])

    # only for testing - we want to see any fastjet warnings
    #assert False


@pytest.mark.parametrize("calculate_area", [True, False])
@pytest.mark.parametrize("algorithm", ["anti_kt", "generalized_kt"])
def test_jet_finding_basic_multiple_events(caplog: Any, calculate_area: bool, algorithm: str) -> None:
    """ Basic jet finding test with for multiple events. """
    # Setup
    caplog.set_level(logging.DEBUG)
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
            "index": [
                [1, 2, 3],
                [4, 5, 6],
            ],
        },
        with_name="Momentum4D",
    )
    logger.info(f"input particles array type: {ak.type(input_particles)}")
    extra_jet_finding_settings = {}
    if algorithm == "generalized_kt":
        extra_jet_finding_settings = {
            "additional_algorithm_parameter": 0.5,
        }
    jets = jet_finding.find_jets(
        particles=input_particles,
        jet_finding_settings=jet_finding.JetFindingSettings(
            R=0.7, algorithm=algorithm,
            area_settings=jet_finding.AreaAA() if calculate_area else None,
            pt_range=jet_finding.pt_range(),
            eta_range=jet_finding.eta_range(jet_R=0.7, fiducial_acceptance=False, eta_min=-5., eta_max=5.),
            **extra_jet_finding_settings,
        )
    )

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

    logger.info(f"input_particles: {input_particles.to_list()}")
    logger.info(f"jets: {jets.to_list()}")
    logger.info(f"expected_jets: {expected_jets.to_list()}")

    # Check four momenta
    assert all([np.allclose(np.asarray(measured.px), np.asarray(expected.px))
                and np.allclose(np.asarray(measured.py), np.asarray(expected.py))
                and np.allclose(np.asarray(measured.pz), np.asarray(expected.pz))
                and np.allclose(np.asarray(measured.E), np.asarray(expected.E))
                for event, event_expected in zip(jets, expected_jets) for measured, expected in zip(event, event_expected)])

    # Check that we've handled the area properly
    if calculate_area:
        assert "area" in ak.fields(jets)
    else:
        assert "area" not in ak.fields(jets)

    # only for testing - we want to see any fastjet warnings
    #assert False


@pytest.mark.parametrize("separate_background_particles_arg", [True, False], ids=["Standard", "Separate background particles argument"])
@pytest.mark.parametrize("use_custom_user_index", [True, False])
def test_jet_finding_with_subtraction_multiple_events(caplog: Any, separate_background_particles_arg: bool, use_custom_user_index: bool) -> None:
    """ Jet finding with subtraction for multiple events. """
    # Setup
    caplog.set_level(logging.DEBUG)
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
            "user_index" if use_custom_user_index else "index": [
                [4, -5, 6],
                [7, -8, 9],
            ],
        },
        with_name="Momentum4D",
    )
    logger.info(f"input particles array type: {ak.type(input_particles)}")
    extra_kwargs = {}
    if separate_background_particles_arg:
        extra_kwargs = {"background_particles": input_particles}

    jets = jet_finding.find_jets(
        particles=input_particles,
        jet_finding_settings=jet_finding.JetFindingSettings(
            R=0.7, algorithm="anti-kt",
            area_settings=jet_finding.AreaAA(),
            pt_range=jet_finding.pt_range(),
            eta_range=jet_finding.eta_range(jet_R=0.7, fiducial_acceptance=False, eta_min=-5., eta_max=5.),
        ),
        background_subtraction=jet_finding.BackgroundSubtraction(
            type=jet_finding.BackgroundSubtractionType.rho,
            estimator=jet_finding.JetMedianBackgroundEstimator(
                jet_finding_settings=jet_finding.JetMedianJetFindingSettings()
            ),
            subtractor=jet_finding.RhoSubtractor(),
        ),
        **extra_kwargs,
    )

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

    logger.info(f"input_particles: {input_particles.to_list()}")
    logger.info(f"jets: {jets.to_list()}")
    logger.info(f"expected_jets: {expected_jets.to_list()}")

    # Check four momenta
    assert all([np.allclose(np.asarray(measured.px), np.asarray(expected.px))
                and np.allclose(np.asarray(measured.py), np.asarray(expected.py))
                and np.allclose(np.asarray(measured.pz), np.asarray(expected.pz))
                and np.allclose(np.asarray(measured.E), np.asarray(expected.E))
                for event, event_expected in zip(jets, expected_jets) for measured, expected in zip(event, event_expected)])

    # only for testing - we want to see any fastjet warnings
    #assert False


def test_jet_finding_with_constituent_subtraction_does_something_multiple_events(caplog: Any) -> None:
    """ Jet finding with constituent subtraction modifies the jets somehow for multiple events.

    NOTE:
        This doesn't test that CS gives a particular expected result - just that it modifies the jets.
        This is because there's no simple reference. So we have to validate elsewhere.
    """
    # Setup
    caplog.set_level(logging.DEBUG)
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
            "index": [
                [1, 2, 3],
                [4, 5, 6],
            ],
        },
        with_name="Momentum4D",
    )
    logger.info(f"input particles array type: {ak.type(input_particles)}")

    jets = jet_finding.find_jets(
        particles=input_particles,
        jet_finding_settings=jet_finding.JetFindingSettings(
            R=0.7, algorithm="anti-kt",
            area_settings=jet_finding.AreaAA(),
            pt_range=jet_finding.pt_range(),
            eta_range=jet_finding.eta_range(jet_R=0.7, fiducial_acceptance=False, eta_min=-5., eta_max=5.),
        ),
        background_subtraction=jet_finding.BackgroundSubtraction(
            type=jet_finding.BackgroundSubtractionType.event_wise_constituent_subtraction,
            estimator=jet_finding.JetMedianBackgroundEstimator(
                jet_finding_settings=jet_finding.JetMedianJetFindingSettings()
            ),
            subtractor=jet_finding.ConstituentSubtractor(
                r_max=0.25, alpha=1.0
            ),
        ),
    )

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

    logger.debug(f"input_particles: {input_particles.to_list()}")
    logger.debug(f"jets: {jets.to_list()}")
    logger.debug(f"expected_jets: {expected_jets.to_list()}")

    # Check four momenta
    # Here, we expect it _not_ to agree with the "expected jets" from above because
    # constituent subtraction has modified the four vectors. We don't have a simple
    # and convenient reference, so we effectively require that it is changed _somehow_
    # by the constituent subtraction. It will have to be validated elsewhere.
    assert not all([np.allclose(np.asarray(measured.px), np.asarray(expected.px))
                    and np.allclose(np.asarray(measured.py), np.asarray(expected.py))
                    and np.allclose(np.asarray(measured.pz), np.asarray(expected.pz))
                    and np.allclose(np.asarray(measured.E), np.asarray(expected.E))
                    for event, event_expected in zip(jets, expected_jets) for measured, expected in zip(event, event_expected)])

    # only for testing - we want to see any fastjet warnings
    #assert False


def test_negative_energy_recombiner(caplog: Any) -> None:
    """ Jet finding with negative energy recombiner for multiple events. """
    # Setup
    caplog.set_level(logging.DEBUG)
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
            "user_index": [
                [4, -5, 6],
                [7, -8, 9],
            ],
        },
        with_name="Momentum4D",
    )
    logger.info(f"input particles array type: {ak.type(input_particles)}")
    #extra_kwargs = {}

    jets = jet_finding.find_jets(
        particles=input_particles,
        jet_finding_settings=jet_finding.JetFindingSettings(
            R=0.7, algorithm="anti-kt",
            # Use pp settings to speed it up. I don't think I need much detail (and as of April 2023,
            # AA equally works the same)
            area_settings=jet_finding.AreaPP(),
            pt_range=jet_finding.pt_range(),
            eta_range=jet_finding.eta_range(jet_R=0.7, fiducial_acceptance=False, eta_min=-5., eta_max=5.),
            recombiner=jet_finding.NegativeEnergyRecombiner(
                identifier_index=-123456
            ),
        ),
    )

    # NOTE: These values were extracted by running the code! Since these weren't calculated independently,
    #       this is more of a regression test than a full integration test.
    expected_jets = ak.zip(
        {
            "px": [
                [-99.0, 95],
                [0.0, 0.2],
            ],
            "py": [
                [0.0, 0.2],
                [-99.0, 95],
            ],
            "pz": [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            "E": [
                [99.0, 95.0],
                [99.0, 95.0],
            ],
        },
        with_name="Momentum4D",
    )

    logger.info(f"input_particles: {input_particles.to_list()}")
    logger.info("jets:")
    jets[["px", "py", "pz", "E"]].show()
    logger.info(f"expected_jets: {expected_jets.to_list()}")

    # Check four momenta
    assert all([np.allclose(np.asarray(measured.px), np.asarray(expected.px))
                and np.allclose(np.asarray(measured.py), np.asarray(expected.py))
                and np.allclose(np.asarray(measured.pz), np.asarray(expected.pz))
                and np.allclose(np.asarray(measured.E), np.asarray(expected.E))
                for event, event_expected in zip(jets, expected_jets) for measured, expected in zip(event, event_expected)])

    # only for testing - we want to see any fastjet warnings
    #assert False