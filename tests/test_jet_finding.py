"""Tests for jet finding

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
from typing import Any

import awkward as ak
import numpy as np
import pytest
import vector

from mammoth.framework import jet_finding

logger = logging.getLogger(__name__)


def test_find_constituent_indices_via_user_index(caplog: Any) -> None:
    # Setup
    caplog.set_level(logging.DEBUG)

    _user_index = ak.Array([[4, -5, 6], [7, -8, 9]])
    _constituents_user_index_awkward = ak.Array([[[4, -5], [6]], [[-8, 7], [9]]])
    _constituent_indices_awkward = jet_finding.find_constituent_indices_via_user_index(
        event_structured_particles_ref_user_index=_user_index,
        event_structured_jets_constituents_user_index=_constituents_user_index_awkward,
    )

    assert _constituent_indices_awkward.to_list() == [[[0, 1], [2]], [[1, 0], [2]]]


def test_find_unsubtracted_constituent_index_from_subtracted_index_via_user_index(caplog: Any) -> None:
    """Test relating the unsubtracted constituent index to the"""
    # Setup
    caplog.set_level(logging.DEBUG)

    # NOTE: This user_index permutations are probably more general than we'll see in data, but better to test fully
    _user_index = ak.Array([[1, -2, 3, 4], [6, -5, 4], [9, 7, -8]])
    _subtracted_index_to_unsubtracted_user_index_awkward = ak.Array([[1, -2, 3, 4], [4, -5, 6], [7, -8, 9]])

    result = jet_finding.find_unsubtracted_constituent_index_from_subtracted_index_via_user_index(
        user_indices=_user_index,
        subtracted_index_to_unsubtracted_user_index=_subtracted_index_to_unsubtracted_user_index_awkward,
    )

    assert result.to_list() == [[0, 1, 2, 3], [2, 1, 0], [1, 2, 0]]


def test_find_unsubtracted_constituent_index_from_subtracted_index_via_user_index_with_empty_events(
    caplog: Any,
) -> None:
    """Test relating the unsubtracted constituent index to the"""
    # Setup
    caplog.set_level(logging.DEBUG)

    # NOTE: This user_index permutations are probably more general than we'll see in data, but better to test fully
    _user_index = ak.Array([[1, -2, 3, 4], [], [6, -5, 4], [9, 7, -8]])
    _subtracted_index_to_unsubtracted_user_index_awkward = ak.Array([[1, -2, 3, 4], [], [4, -5, 6], [7, -8, 9]])

    result = jet_finding.find_unsubtracted_constituent_index_from_subtracted_index_via_user_index(
        user_indices=_user_index,
        subtracted_index_to_unsubtracted_user_index=_subtracted_index_to_unsubtracted_user_index_awkward,
    )

    assert result.to_list() == [[0, 1, 2, 3], [], [2, 1, 0], [1, 2, 0]]


def test_calculate_user_index_with_encoded_sign_info(caplog: Any) -> None:
    """Test calculating a custom user_index where we encode sign info."""
    # Setup
    caplog.set_level(logging.DEBUG)

    # An event with three particles
    # First event is the standard fastjet test particles,
    # while the second event is the standard with px <-> py.
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
            "source_index": [
                [1, 2, 3],
                [4, 5, 6],
            ],
        },
        with_name="Momentum4D",
    )
    values_to_encode = ak.Array([[1, 1, 0], [1, 0, 1]])
    mask_to_encode_with_negative = values_to_encode == 0

    res = jet_finding.calculate_user_index_with_encoded_sign_info(
        particles=input_particles,
        mask_to_encode_with_negative=mask_to_encode_with_negative,
    )

    assert res.to_list() == [[0, 1, -2], [0, -1, 2]]


def test_calculate_user_index_with_encoded_sign_info_detect_error(caplog: Any) -> None:
    """Test calculating a custom user_index where we encode sign info."""
    # Setup
    caplog.set_level(logging.DEBUG)

    # An event with three particles
    # First event is the standard fastjet test particles,
    # while the second event is the standard with px <-> py.
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
            "source_index": [
                [1, 2, 3],
                [4, 5, 6],
            ],
        },
        with_name="Momentum4D",
    )
    values_to_encode = ak.Array([[1, 1, 0], [0, 1, 1]])
    mask_to_encode_with_negative = values_to_encode == 0

    with pytest.raises(ValueError, match="contain index of 0"):
        jet_finding.calculate_user_index_with_encoded_sign_info(
            particles=input_particles,
            mask_to_encode_with_negative=mask_to_encode_with_negative,
        )


def test_jet_finding_basic_single_event(caplog: Any) -> None:
    """Basic jet finding test with a single event."""
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
            R=0.7,
            algorithm="anti-kt",
            area_settings=jet_finding.AreaAA(),
            pt_range=jet_finding.pt_range(),
            eta_range=jet_finding.eta_range(jet_R=0.7, fiducial_acceptance=False, eta_min=-5.0, eta_max=5.0),
        ),
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
    assert all(
        np.allclose(np.asarray(measured.px), np.asarray(expected.px))
        and np.allclose(np.asarray(measured.py), np.asarray(expected.py))
        and np.allclose(np.asarray(measured.pz), np.asarray(expected.pz))
        and np.allclose(np.asarray(measured.E), np.asarray(expected.E))
        for event, event_expected in zip(jets, expected_jets, strict=True)
        for measured, expected in zip(event, event_expected, strict=True)
    )

    # only for testing - we want to see any fastjet warnings
    # assert False


@pytest.mark.parametrize("calculate_area", [True, False])
@pytest.mark.parametrize("algorithm", ["anti_kt", "generalized_kt"])
def test_jet_finding_basic_multiple_events(caplog: Any, calculate_area: bool, algorithm: str) -> None:
    """Basic jet finding test with for multiple events."""
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
            "source_index": [
                [1, 2, 3],
                [4, 5, 6],
            ],
        },
        with_name="Momentum4D",
    )
    logger.info(f"input particles array type: {ak.type(input_particles)}")
    extra_jet_finding_settings: dict[str, Any] = {}
    if algorithm == "generalized_kt":
        extra_jet_finding_settings = {
            "additional_algorithm_parameter": 0.5,
        }
    jets = jet_finding.find_jets(
        particles=input_particles,
        jet_finding_settings=jet_finding.JetFindingSettings(
            R=0.7,
            algorithm=algorithm,
            area_settings=jet_finding.AreaAA() if calculate_area else None,
            pt_range=jet_finding.pt_range(),
            eta_range=jet_finding.eta_range(jet_R=0.7, fiducial_acceptance=False, eta_min=-5.0, eta_max=5.0),
            **extra_jet_finding_settings,
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

    logger.info(f"input_particles: {input_particles.to_list()}")
    logger.info(f"jets: {jets.to_list()}")
    logger.info(f"expected_jets: {expected_jets.to_list()}")

    # Check four momenta
    assert all(
        np.allclose(np.asarray(measured.px), np.asarray(expected.px))
        and np.allclose(np.asarray(measured.py), np.asarray(expected.py))
        and np.allclose(np.asarray(measured.pz), np.asarray(expected.pz))
        and np.allclose(np.asarray(measured.E), np.asarray(expected.E))
        for event, event_expected in zip(jets, expected_jets, strict=True)
        for measured, expected in zip(event, event_expected, strict=True)
    )

    # Check that we've handled the area properly
    if calculate_area:
        assert "area" in ak.fields(jets)
    else:
        assert "area" not in ak.fields(jets)

    # only for testing - we want to see any fastjet warnings
    # assert False


@pytest.mark.parametrize(
    "separate_background_particles_arg", [True, False], ids=["Standard", "Separate background particles argument"]
)
@pytest.mark.parametrize("use_custom_user_index", [True, False])
def test_jet_finding_with_rho_subtraction_and_multiple_events(
    caplog: Any, separate_background_particles_arg: bool, use_custom_user_index: bool
) -> None:
    """Jet finding with subtraction for multiple events.

    Note:
        These include empty events, so we can check that the offsets are correctly calculated.

    Note that the subtraction doesn't do anything here because rho is 0 (i.e. since there aren't
    enough jets to make a meaningful background). It's just testing that the software interface
    works when subtraction is enabled.
    """
    # Setup
    caplog.set_level(logging.DEBUG)
    vector.register_awkward()

    # An event with three particles
    # First event is the standard fastjet test particles,
    # the second is an empty event,
    # while the third event is the standard with px <-> py.
    # The output jets should be the same as well, but with px <-> py.
    additional_fields = {}
    if use_custom_user_index:
        additional_fields["user_index"] = [
            [4, -5, 6],
            [],
            [7, -8, 9],
        ]
    input_particles = ak.zip(
        {
            "px": [
                [99.0, 4.0, -99.0],
                [],
                [0.1, -0.1, 0],
            ],
            "py": [
                [0.1, -0.1, 0],
                [],
                [99.0, 4.0, -99.0],
            ],
            "pz": [
                [0, 0, 0],
                [],
                [0, 0, 0],
            ],
            "E": [
                [100.0, 5.0, 99.0],
                [],
                [100.0, 5.0, 99.0],
            ],
            "source_index": [
                [4, 5, 6],
                [],
                [7, 8, 9],
            ],
            **additional_fields,
        },
        with_name="Momentum4D",
    )
    logger.info(f"input particles array type: {ak.type(input_particles)}")
    extra_kwargs = {}
    if separate_background_particles_arg:
        # NOTE: We'll include testing out having any empty background event too
        # NOTE: We intentionally don't want to match all of the empty events with
        #       the input particles to try to test for further issues.
        #       (This is to say, event 3 is empty when the input particles are not)
        more_fields = {}
        if use_custom_user_index:
            more_fields["user_index"] = [
                [4, -5, 6],
                [],
                [],
            ]
        background_particles = ak.zip(
            {
                "px": [
                    [99.0, 4.0, -99.0],
                    [],
                    [],
                ],
                "py": [
                    [0.1, -0.1, 0],
                    [],
                    [],
                ],
                "pz": [
                    [0, 0, 0],
                    [],
                    [],
                ],
                "E": [
                    [100.0, 5.0, 99.0],
                    [],
                    [],
                ],
                "source_index": [
                    [4, 5, 6],
                    [],
                    [],
                ],
                **more_fields,
            },
            with_name="Momentum4D",
        )
        extra_kwargs = {"background_particles": background_particles}

    jets = jet_finding.find_jets(
        particles=input_particles,
        jet_finding_settings=jet_finding.JetFindingSettings(
            R=0.7,
            algorithm="anti-kt",
            area_settings=jet_finding.AreaAA(),
            pt_range=jet_finding.pt_range(),
            eta_range=jet_finding.eta_range(jet_R=0.7, fiducial_acceptance=False, eta_min=-5.0, eta_max=5.0),
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
                [],
                [0.0, 0.0],
            ],
            "py": [
                [0.0, 0.0],
                [],
                [103.0, -99.0],
            ],
            "pz": [
                [0.0, 0.0],
                [],
                [0.0, 0.0],
            ],
            "E": [
                [105.0, 99.0],
                [],
                [105.0, 99.0],
            ],
        },
        with_name="Momentum4D",
    )

    logger.info(f"input_particles: {input_particles.to_list()}")
    logger.info(f"jets: {jets.to_list()}")
    logger.info(f"expected_jets: {expected_jets.to_list()}")

    # Check four momenta
    assert all(
        np.allclose(np.asarray(measured.px), np.asarray(expected.px))
        and np.allclose(np.asarray(measured.py), np.asarray(expected.py))
        and np.allclose(np.asarray(measured.pz), np.asarray(expected.pz))
        and np.allclose(np.asarray(measured.E), np.asarray(expected.E))
        for event, event_expected in zip(jets, expected_jets, strict=True)
        for measured, expected in zip(event, event_expected, strict=True)
    )

    # only for testing - we want to see any fastjet warnings
    # assert False


@pytest.mark.parametrize("use_custom_user_index", [True, False])
def test_jet_finding_with_constituent_subtraction_does_something_multiple_events(
    caplog: Any, use_custom_user_index: bool
) -> None:
    """Jet finding with constituent subtraction modifies the jets somehow for multiple events.

    NOTE:
        This doesn't really properly test the CS subtraction since rho will be 0 (and mocking up
        a consistent background such that rho is non-zero would be a pain). So here, we don't
        look for a particular result - just that the jets are modified in some way (which appears
        to be mostly due to the transition of the input particles -> subtracted, even though it
        shouldn't have any real impact). In any case, there is no simple reference, so we have to
        validate in other ways in other tests (i.e. using the track skim validation).
    """
    # Setup
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.INFO, logger="numba")
    vector.register_awkward()

    # an event with three particles
    # First event is the standard fastjet test particles
    # the second event is empty,
    # and the third event is the standard with px <-> py
    # The output jets should be similar, but with px <-> py.
    additional_fields = {}
    if use_custom_user_index:
        additional_fields["user_index"] = [
            [4, -5, 6],
            [],
            [7, -8, 9],
        ]
    input_particles = ak.zip(
        {
            "px": [
                [99.0, 4.0, -99.0],
                [],
                [0.1, -0.1, 0],
            ],
            "py": [
                [0.1, -0.1, 0],
                [],
                [99.0, 4.0, -99.0],
            ],
            "pz": [
                [0, 0, 0],
                [],
                [0, 0, 0],
            ],
            "E": [
                [100.0, 5.0, 99.0],
                [],
                [100.0, 5.0, 99.0],
            ],
            "source_index": [
                [1, 2, 3],
                [],
                [4, 5, 6],
            ],
            **additional_fields,
        },
        with_name="Momentum4D",
    )
    logger.info(f"input particles array type: {ak.type(input_particles)}")

    jets = jet_finding.find_jets(
        particles=input_particles,
        jet_finding_settings=jet_finding.JetFindingSettings(
            R=0.7,
            algorithm="anti-kt",
            area_settings=jet_finding.AreaAA(random_seed=jet_finding.VALIDATION_MODE_RANDOM_SEED),
            pt_range=jet_finding.pt_range(),
            eta_range=jet_finding.eta_range(jet_R=0.7, fiducial_acceptance=False, eta_min=-5.0, eta_max=5.0),
        ),
        background_subtraction=jet_finding.BackgroundSubtraction(
            type=jet_finding.BackgroundSubtractionType.event_wise_constituent_subtraction,
            estimator=jet_finding.JetMedianBackgroundEstimator(
                jet_finding_settings=jet_finding.JetMedianJetFindingSettings()
            ),
            subtractor=jet_finding.ConstituentSubtractor(r_max=0.25, alpha=1.0),
        ),
    )

    expected_jets = ak.zip(
        {
            "px": [
                [103.0, -99.0],
                [],
                [0.0, 0.0],
            ],
            "py": [
                [0.0, 0.0],
                [],
                [103.0, -99.0],
            ],
            "pz": [
                [0.0, 0.0],
                [],
                [0.0, 0.0],
            ],
            "E": [
                [105.0, 99.0],
                [],
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
    # constituent subtraction has modified the four vectors.
    # NOTE: Since rho is 0, the biggest change appears to be just due to rewriting
    #       the constituents from the input particles to the subtracted constituents.
    #       However, at some level, this is fine - we just want to see that things have
    #       changed. Since we don't have a simple and convenient reference, it will have
    #       to be validated elsewhere via the track skim.
    # NOTE: Why are thy changed at all? I **think** this is due to CS renormalizing particles
    #       to be massless. If we check, input_particles.E**2 != input_particles.p**2, but
    #       jets.constituents.E ** 2 == jets.constituents.p ** 2, as expected. So I think
    #       that's the reason!
    assert not all(
        np.allclose(np.asarray(measured.px), np.asarray(expected.px))
        and np.allclose(np.asarray(measured.py), np.asarray(expected.py))
        and np.allclose(np.asarray(measured.pz), np.asarray(expected.pz))
        and np.allclose(np.asarray(measured.E), np.asarray(expected.E))
        for event, event_expected in zip(jets, expected_jets, strict=True)
        for measured, expected in zip(event, event_expected, strict=True)
    )

    if use_custom_user_index:
        # NOTE: The order of the constituents appears to be susceptible to whether there are ghosts included or not,
        #       so we figured out the right assignments by hand, and then just adjusted the order as needed based on
        #       what result the code returned. Hopefully this will be reasonably repeatable and stable.
        expected_user_index = [[[-5, 4], [6]], [], [[7, -8], [9]]]
        assert expected_user_index == jets.constituents.user_index.to_list()

    # only for testing - we want to see any fastjet warnings
    # assert False


def test_negative_energy_recombiner(caplog: Any) -> None:
    """Jet finding with negative energy recombiner for multiple events."""
    # Setup
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.INFO, logger="numba")
    vector.register_awkward()

    # an event with three particles
    # First event is the standard fastjet test particles,
    # while the second event is the standard with px <-> py.
    # The output jets should be the same as well, but with px <-> py.
    input_particles = ak.zip(
        {
            "px": [
                [99.0, 2.0, 4.0, -99.0],
                [0.1, -0.1, 0],
            ],
            "py": [
                [0.1, 0.0, -0.1, 0],
                [99.0, 4.0, -99.0],
            ],
            "pz": [
                [0, 0, 0, 0],
                [0, 0, 0],
            ],
            "E": [
                [100.0, 2.0, 5.0, 99.0],
                [100.0, 5.0, 99.0],
            ],
            "user_index": [
                [4, -8, -5, 6],
                [7, -8, 9],
            ],
        },
        with_name="Momentum4D",
    )
    logger.info(f"input particles array type: {ak.type(input_particles)}")
    # extra_kwargs = {}

    jets = jet_finding.find_jets(
        particles=input_particles,
        jet_finding_settings=jet_finding.JetFindingSettings(
            R=0.7,
            algorithm="anti-kt",
            # Use pp settings to speed it up. I don't think I need much detail (and as of April 2023,
            # AA equally works the same)
            area_settings=jet_finding.AreaPP(),
            pt_range=jet_finding.pt_range(),
            eta_range=jet_finding.eta_range(jet_R=0.7, fiducial_acceptance=False, eta_min=-5.0, eta_max=5.0),
            recombiner=jet_finding.NegativeEnergyRecombiner(identifier_index=-123456),
        ),
    )

    # NOTE: These values were extracted by running the code! Since these weren't calculated independently,
    #       this is more of a regression test than a full integration test.
    expected_jets = ak.zip(
        {
            "px": [
                [-99.0, 93],
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
                [99.0, 93.0],
                [99.0, 95.0],
            ],
        },
        with_name="Momentum4D",
    )

    logger.info(f"input_particles: {input_particles.to_list()}")
    logger.info("jets:")
    import io

    _s = io.StringIO()
    jets[["px", "py", "pz", "E"]].show(stream=_s)
    logger.info(f"{_s}")
    logger.info(f"expected_jets: {expected_jets.to_list()}")

    # Check four momenta
    assert all(
        np.allclose(np.asarray(measured.px), np.asarray(expected.px))
        and np.allclose(np.asarray(measured.py), np.asarray(expected.py))
        and np.allclose(np.asarray(measured.pz), np.asarray(expected.pz))
        and np.allclose(np.asarray(measured.E), np.asarray(expected.E))
        for event, event_expected in zip(jets, expected_jets, strict=True)
        for measured, expected in zip(event, event_expected, strict=True)
    )

    # Check user indices of constituents.
    # These should be passed through, so we want to verify that actually happened correctly.
    # NOTE: The order of the constituents appears to be susceptible to whether there are ghosts included or not,
    #       so we figured out the right assignments, and then just adjusted the order as needed. Hopefully this will
    #       be reasonably repeatable and stable.
    expected_user_index = [[[6], [-5, 4, -8]], [[9], [7, -8]]]
    assert jets.constituents.user_index.to_list() == expected_user_index

    # only for testing - we want to see any fastjet warnings
    # assert False


def test_reclustering(caplog: Any) -> None:
    """Test reclustering

    Requires running the jet finding code first, and then reclustering together afterwards.

    Note:
        This will be more of an integration test, since there's limits to what I'm willing
        to calculate for a splitting tree
    """
    # Setup
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.INFO, logger="numba")
    vector.register_awkward()

    # an event with three particles
    # First event is an event that I made up. It consists of:
    """
    Splittings tree with energies (which are approximate) as label
    Splittings generation:
    1.  2.      3.      4.
    --------------------------
    200 -> 150  -> 100  -> 105
                        -> -5 (a hole)
                -> 50
        -> 50   -> 25
                -> 25

    We then include a second jet rotated by 90 degrees from the first just to
    double check that one doesn't interfere with the other.
    """
    # while the second event is the standard with px <-> py.
    # The output jets should be the same as well, but with px <-> py.
    # NOTE: We put the hole nearest to the 105.0 particle to ensure that it's stably selected
    #       for combination together in the e.g. Cambridge/Aachen case.
    input_particles = ak.zip(
        {
            "px": [
                [50.0, 105.0, 25.0, 25.0, 5.0, 0.0, 0.1, 0.19],
                [0.5, -0.5, 4.75, 5.25, -0.05],
            ],
            "py": [
                [
                    0.5,
                    -0.5,
                    4.75,
                    5.25,
                    -0.05,
                    49.0,
                    100.0,
                    49.0,
                ],
                [50.0, 105.0, 25.0, 25.0, 5.0],
            ],
            "pz": [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            "E": [
                [50.0, 105.0, 25.0, 25.0, 5.0, 49.0, 100.0, 49.0],
                [50.0, 105.0, 25.0, 25.0, 5.0],
            ],
            "source_index": [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [6, 7, 8, 9, 10],
            ],
        },
        with_name="Momentum4D",
    )
    # Encode some hole like user_index structure
    # Here, 0 corresponds to the hole particle
    values_to_encode = ak.Array([[1, 1, 1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 0]])
    mask_to_encode_with_negative = values_to_encode == 0
    user_index = jet_finding.calculate_user_index_with_encoded_sign_info(
        particles=input_particles,
        mask_to_encode_with_negative=mask_to_encode_with_negative,
    )
    input_particles["user_index"] = user_index
    logger.info(f"input particles array type: {ak.type(input_particles)}")

    jets = jet_finding.find_jets(
        particles=input_particles,
        jet_finding_settings=jet_finding.JetFindingSettings(
            R=0.7,
            algorithm="anti-kt",
            # Use pp settings to speed it up. I don't think I need much detail (and as of April 2023,
            # AA equally works the same)
            # area_settings=jet_finding.AreaPP(),
            area_settings=None,
            pt_range=jet_finding.pt_range(),
            eta_range=jet_finding.eta_range(jet_R=0.7, fiducial_acceptance=False, eta_min=-5.0, eta_max=5.0),
            recombiner=jet_finding.NegativeEnergyRecombiner(identifier_index=-123456),
        ),
    )

    # NOTE: These values were extracted by running the code! Since these weren't calculated independently,
    #       this is more of a regression test than a full integration test.
    expected_jets = ak.zip(
        {
            "px": [
                [200.0, 0.29],
                [10.05],
            ],
            "py": [
                [10.05, 198.0],
                [200.0],
            ],
            "pz": [
                [0.0, 0.0],
                [0.0],
            ],
            "E": [
                [200.0, 198.0],
                [200.0],
            ],
        },
        with_name="Momentum4D",
    )

    logger.info(f"input_particles: {input_particles.to_list()}")
    logger.info("jets:")
    import io

    _s = io.StringIO()
    jets[["px", "py", "pz", "E"]].show(stream=_s)
    logger.info(f"{_s}")
    logger.info(f"expected_jets: {expected_jets.to_list()}")

    # Check four momenta
    assert all(
        np.allclose(np.asarray(measured.px), np.asarray(expected.px))
        and np.allclose(np.asarray(measured.py), np.asarray(expected.py))
        and np.allclose(np.asarray(measured.pz), np.asarray(expected.pz))
        and np.allclose(np.asarray(measured.E), np.asarray(expected.E))
        for event, event_expected in zip(jets, expected_jets, strict=True)
        for measured, expected in zip(event, event_expected, strict=True)
    )

    # Check additional label fields
    # 1. user index of constituents.
    # These should be passed through, so we want to verify that actually happened correctly.
    # NOTE: The order of the constituents appears to be susceptible to whether there are ghosts included or not,
    #       so we figured out the right assignments, and then just adjusted the order as needed. Hopefully this will
    #       be reasonably repeatable and stable.
    expected_user_index = [[[0, 1, -4, 2, 3], [7, 5, 6]], [[0, 1, -4, 2, 3]]]
    assert jets.constituents.user_index.to_list() == expected_user_index
    # 2. The source index
    # These should be passed through - nothing fancy.
    # The order is dictated by how fastjet clusters, as above with the user index.
    expected_source_index = [[[1, 2, 5, 3, 4], [8, 6, 7]], [[6, 7, 10, 8, 9]]]
    assert jets.constituents.source_index.to_list() == expected_source_index

    ####################################
    # Now that we've completed the jet finding, we go onto the reclustering
    ####################################
    logger.info("Reclustering jets...")
    # First, using the user_index
    reclustering_jets = jet_finding.recluster_jets(
        jets=jets,
        jet_finding_settings=jet_finding.ReclusteringJetFindingSettings(
            area_settings=jet_finding.AreaSubstructure(random_seed=jet_finding.VALIDATION_MODE_RANDOM_SEED),
            recombiner=jet_finding.NegativeEnergyRecombiner(identifier_index=-123456),
        ),
        store_recursive_splittings=True,
    )
    # And then without the user_index
    jets_without_user_index = ak.copy(jets)
    # Remove the "user_index" field
    del jets_without_user_index["constituents", "user_index"]
    # And do the reclustering without the user_index
    reclustering_jets_without_user_index = jet_finding.recluster_jets(
        jets=jets_without_user_index,
        jet_finding_settings=jet_finding.ReclusteringJetFindingSettings(
            area_settings=jet_finding.AreaSubstructure(random_seed=jet_finding.VALIDATION_MODE_RANDOM_SEED),
        ),
        store_recursive_splittings=True,
    )
    logger.info(f"reclustering_jets: {reclustering_jets.to_list()}")
    logger.info(f"reclustering_jets_without_user_index: {reclustering_jets_without_user_index.to_list()}")

    # Check the substructure properties.
    # Note that these are just extracted from the calculation and check for regression...
    # We could check more properties, but the kt is enough, since it's directly impacted by the
    # subtraction vs addition of the negative energy particles.
    expected_kt = ak.Array(
        [
            [
                [10.031998634338379, 0.7249953746795654, 0.026190893724560738, 0.49901485443115234],
                [0.15711446106433868, 0.048999983817338943],
            ],
            [[10.031998634338379, 0.7249953746795654, 0.026190893724560738, 0.49901485443115234]],
        ]
    )
    assert ak.all(reclustering_jets.jet_splittings.kt == expected_kt)

    # Check the constituent indices to ensure that we they provide the expected results.
    # It's easy to e.g. get the overall jet correct but then mess up the constituents,
    # so we need to check carefully.
    # The expected values are somewhat calculated by hand and somewhat empirical
    expected_constituent_indices = ak.Array(
        [
            [[[0, 1, 2], [3, 4], [1, 2], [0], [1], [2], [4], [3]], [[1, 2], [0], [2], [1]]],
            [[[0, 1, 2], [3, 4], [1, 2], [0], [1], [2], [4], [3]]],
        ]
    )
    assert reclustering_jets.subjets.constituent_indices.to_list() == expected_constituent_indices.to_list()

    # And compare between the reclustering with and without the user_index
    kt_splittings_are_close = ak.isclose(
        reclustering_jets.jet_splittings.kt, reclustering_jets_without_user_index.jet_splittings.kt
    )
    # The reasoning for the expected values is as follows:
    # For the first jet,
    # - The 0th splitting is impacted by the NegativeEnergyRecombiner, so it's different
    # - The 1st splitting is similarly impacted because it the negative particle is still contained in one of the constituent subjet
    # - The 2nd splitting contains the negative particle, but since the kt is calculated between the two subjets (of which one is the
    #    negative particle), the splitting calculation is not impacted by the subtraction rather than addition (ie. but the preceding
    #    subjets are) I think this is the right behavior.
    # - The 3rd splitting doesn't contain any negative particles, so it should trivially agree.
    # For the second jet, there are no negative particles, so the kt splittings should agree.
    expected_kt_splittings_are_close = ak.Array(
        [[[False, False, True, True], [True, True]], [[False, False, True, True]]]
    )
    assert kt_splittings_are_close.to_list() == expected_kt_splittings_are_close.to_list()

    # only for testing - we want to see any fastjet warnings
    # assert False
