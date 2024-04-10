"""Tests for the EEC analysis using ALICE parameters

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from typing import Any

import awkward as ak
import pytest  # noqa: F401
import vector

from mammoth.eec import analyze_chunk

vector.register_awkward()
logger = logging.getLogger(__name__)


def test_calculate_weight_for_plotting(caplog: Any) -> None:
    # Setup
    caplog.set_level(logging.INFO)

    particles = ak.zip(
        {
            "pt": [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4],
                [1, 2, 3, 4, 5],
            ],
            "eta": [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            "phi": [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            "m": [
                [0.13957018, 0.13957018, 0.13957018, 0.13957018, 0.13957018],
                [0.13957018, 0.13957018, 0.13957018, 0.13957018],
                [0.13957018, 0.13957018, 0.13957018, 0.13957018, 0.13957018],
                [0.13957018, 0.13957018, 0.13957018, 0.13957018],
                [0.13957018, 0.13957018, 0.13957018, 0.13957018, 0.13957018],
            ],
            "charge": [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            "pdg": [
                [211, 211, 211, 211, 211],
                [211, 211, 211, 211],
                [211, 211, 211, 211, 211],
                [211, 211, 211, 211],
                [211, 211, 211, 211, 211],
            ],
        },
        with_name="Momentum4D",
    )
    particles.type.show()

    left, right = ak.unzip(ak.combinations(particles, 2))
    res = analyze_chunk._calculate_weight_for_plotting_two_particle_correlator(
        left=left,
        right=right,
        trigger_pt_event_wise=particles.pt[:, 0],
        momentum_weight_exponent=1,
    )

    logger.info(res)
    # This was mostly a test for whether copilot could generate unit tests, but no such luck.
    # I don't want to calculate the weights by hand, so I'll just check that the code runs.
