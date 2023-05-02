"""Tests for analysis objects.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

import logging
import tempfile
from pathlib import Path

import pytest  # noqa: F401
from pachyderm import yaml

from mammoth.framework.analysis import objects as analysis_objects

logger = logging.getLogger(__name__)

_here = Path(__file__).parent
_track_skim_base_path = _here / "track_skim_validation"


def test_round_trip_serialization() -> None:
    """Check serialization, leveraging the existing track skim validation scale factors for convenience."""

    # Start with a consistent input
    # Basically reimplements `analysis_objects.read_extracted_scale_factors`, but we do it
    # separately to ensure that we're in full control
    path = Path(_track_skim_base_path / "input" / "LHC20g4_AOD_2640_scale_factors.yaml")
    y = yaml.yaml(classes_to_register=[analysis_objects.ScaleFactor])
    with path.open() as f:
        scale_factors: dict[int, analysis_objects.ScaleFactor] = y.load(f)

    with tempfile.NamedTemporaryFile() as output_file:
        y.dump(scale_factors, output_file)
        output_file.seek(0)
        round_trip_scale_factors = y.load(output_file)

    assert scale_factors == round_trip_scale_factors
