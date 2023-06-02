"""Steering for jobs

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Protocol

from mammoth.framework import sources
from mammoth.framework import task as framework_task
from mammoth.framework.io import output_utils

logger = logging.getLogger(__name__)


class CustomizeParameters(Protocol):
    def __call__(
            self,
            *,
            task_settings: framework_task.Settings,
            **analysis_arguments: Any,
        ) -> dict[str, Any]:
        ...

