"""Task related functionality

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

import logging
from typing import Any

import attrs
import hist

logger = logging.getLogger(__name__)


@attrs.frozen
class Output:
    success: bool
    message: str
    collision_system: str
    hists: dict[str, hist.Hist] = attrs.field(factory=dict)
    results: dict[str, Any] = attrs.field(factory=dict)
