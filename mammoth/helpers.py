"""A variety of helper functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from datetime import datetime
from typing import Optional, Sequence

import attr
import rich
from rich.logging import RichHandler


logging.getLogger(__name__)


@attr.s
class LogMessage:
    """Stores a log message for logging later.

    Since parsl logging configuration is broken (namely, we can't configure all modules),
    we need a way to store log messages, and then log them later.

    Attributes:
        _source: Source of the message. Usually should be `__name__` in the module which
            generated the message.
        level: Log level. Must be a valid level.
        message: Message to log.
    """

    _source: str = attr.ib()
    level: str = attr.ib()
    message: str = attr.ib()

    def log(self) -> None:
        """Log the message."""
        _logger = logging.getLogger(self._source)
        getattr(_logger, self.level)(self.message)


class RichModuleNameHandler(RichHandler):
    """Renders the module name instead of the log path."""

    def render(
        self,
        *,
        record: logging.LogRecord,
        traceback: Optional[rich.traceback.Traceback],
        message_renderable: "rich.console.ConsoleRenderable",
    ) -> "rich.console.ConsoleRenderable":
        """Render log for display.

        Args:
            record (LogRecord): logging Record.
            traceback (Optional[Traceback]): Traceback instance or None for no Traceback.
            message_renderable (ConsoleRenderable): Renderable (typically Text) containing log message contents.
        Returns:
            ConsoleRenderable: Renderable to display log.
        """
        # STAT modifications
        path = record.name
        # END modifications
        level = self.get_level_text(record)
        time_format = None if self.formatter is None else self.formatter.datefmt
        log_time = datetime.fromtimestamp(record.created)

        log_renderable = self._log_render(
            self.console,
            [message_renderable] if not traceback else [message_renderable, traceback],
            log_time=log_time,
            time_format=time_format,
            level=level,
            path=path,
            line_no=record.lineno,
            link_path=record.pathname if self.enable_link_path else None,
        )
        return log_renderable


def setup_logging(
    level: int = logging.INFO,
    stored_messages: Optional[Sequence[LogMessage]] = None,
    aggressively_quiet_parsl_logging: bool = False,
) -> bool:
    """Configure logging.

    NOTE:
        Don't call this before parsl has loaded a config! Otherwise, you'll be inundated with
        irrelevant log messages. Hopefully this will be fixed in parsl eventually.

    Args:
        level: Logging level. Default: "INFO".
        stored_messages: Messages stored that we can't log because we had to wait for the parsl
            initialization. Default: None.
        aggressively_quiet_parsl_logging: Aggressively quiet parsl logging. Default: False since
            it usually isn't required, but kept around in case.

    Returns:
        True if logging was set up successfully.
    """
    if not stored_messages:
        stored_messages = []

    FORMAT = "%(message)s"
    logging.basicConfig(level=level, format=FORMAT, datefmt="[%X]", handlers=[RichModuleNameHandler()])
    # Generally, parsl's logger configuration doesn't work for a number of modules because
    # they're not in the parsl namespace, but it's still worth trying to make it quieter.
    logging.getLogger("parsl").setLevel(logging.WARNING)
    # If we need to do more, it's possible, but rather messy. It's stored here for posterity,
    # but would need to be enabled manually.
    if aggressively_quiet_parsl_logging:
        for name in logging.root.manager.loggerDict:
            if name.startswith("parsl") or name.startswith("database_monitoring"):
                logging.getLogger(name).setLevel(logging.WARNING)
    # For sanity when using IPython
    logging.getLogger("parso").setLevel(logging.INFO)

    # Log the stored up messages.
    for message in stored_messages:
        message.log()

    return True
