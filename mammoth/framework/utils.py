
""" Helpers and utilities.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from typing import Optional

import attr

@attr.s(frozen=True)
class Range:
    min: Optional[float] = attr.ib()
    max: Optional[float] = attr.ib()


