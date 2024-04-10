"""Logging objects and utilities for use with c++ extensions

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from typing import Any


class StreamLogger:
    """Logger that can be used in place of a stream.

    From: https://stackoverflow.com/a/66209331/12907985
    """

    def __init__(self, logger: Any, level: int):
        self.logger = logger
        self.log_level = level
        self.buf: list[str] = []

    def write(self, msg: str) -> None:
        if msg.endswith("\n"):
            # Python 3.9+
            # self.buf.append(msg.removesuffix('\n'))
            # Before python 3.9
            self.buf.append(msg.rstrip("\n"))
            self.logger.log(self.log_level, "".join(self.buf))
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self) -> None:
        pass


# NOTE: We intentionally use "mammoth" rather than "mammoth_cpp" since we want to merge
#       the stream of log messages in with the jet finding module there.
#       We will of course need to update the path if it's moved.
jet_finding_logger_stdout = StreamLogger(logging.getLogger("mammoth.framework.jet_finding"), logging.DEBUG)
jet_finding_logger_stderr = StreamLogger(logging.getLogger("mammoth.framework.jet_finding"), logging.WARNING)
