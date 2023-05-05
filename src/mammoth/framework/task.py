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
    production_identifier: str
    collision_system: str
    success: bool
    message: str
    hists: dict[str, hist.Hist] = attrs.field(factory=dict, kw_only=True, repr=lambda value: str(list(value.keys())))
    results: dict[str, Any] = attrs.field(factory=dict, kw_only=True)
    metadata: dict[str, Any] = attrs.field(factory=dict, kw_only=True)

    def print(self) -> None:
        """Print the message to the logger.

        Note:
            This is implemented as a convenience function since it otherwise
            won't evaluate newlines in the message. We don't implement __str__
            or __repr__ since those require an additional function call (ie. explicit print).
            This is all just for convenience.
        """
        logger.info(f"collision_system={self.collision_system}, success={self.success}, identifier={self.production_identifier}")
        # NOTE: Evaluate the message separately to ensure that newlines are evaluated.
        logger.info(self.message)
