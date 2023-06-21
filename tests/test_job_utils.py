"""Tests for job_utils

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from typing import Any

import pytest  # noqa: F841
from mammoth.job_utils import hours_in_walltime

def test_hours_in_walltime(caplog: Any) -> None:
    assert hours_in_walltime("01:30:00") == 1
    assert hours_in_walltime("00:45:00") == 0
    assert hours_in_walltime("10:00:00") == 10
    assert hours_in_walltime("24:00:00") == 24
    assert hours_in_walltime("100:00:00") == 100