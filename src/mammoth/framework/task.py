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

    def print(self) -> None:
        """Print the message to the logger.

        Note:
            This is implemented as a convenience function since it otherwise
            won't evaluate newlines in the message. We don't implement __str__
            or __repr__ since those require an additional function call (ie. explicit print).
            This is all just for convenience.
        """
        logger.info(self.message)
