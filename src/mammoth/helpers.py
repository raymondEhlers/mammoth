"""A variety of helper functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

import attrs
import parsl
import rich
import rich.progress
from rich.console import Console
from rich.logging import RichHandler
from rich_argparse import RichHelpFormatter as RichHelpFormatter

logger = logging.getLogger(__name__)

# We need a consistent console object to set everything up properly
# (Namely, logging with the progress bars), so we define it here.
rich_console = Console()


@attrs.define
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

    _source: str
    level: str
    message: str

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
        traceback: rich.traceback.Traceback | None,
        message_renderable: rich.console.ConsoleRenderable,
    ) -> rich.console.ConsoleRenderable:
        """Render log for display.

        Args:
            record (LogRecord): logging Record.
            traceback (Optional[Traceback]): Traceback instance or None for no Traceback.
            message_renderable (ConsoleRenderable): Renderable (typically Text) containing log message contents.
        Returns:
            ConsoleRenderable: Renderable to display log.
        """
        # START modifications (originally for STAT)
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
        return log_renderable  # noqa: RET504


def progress_bar() -> rich.progress.Progress:
    """Define rich progress bar with my preferred parameters"""
    return rich.progress.Progress(
        # The first four columns are from:
        # `*Progress.get_default_columns()`
        # However, I wanted a wider BarColumn, so I coped the definitions and modified the width
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.BarColumn(bar_width=None),
        rich.progress.TaskProgressColumn(),
        rich.progress.TimeRemainingColumn(),
        rich.progress.MofNCompleteColumn(),
        "Elapsed:",
        rich.progress.TimeElapsedColumn(),
        console=rich_console,
        # Refresh relatively slowly and estimate speed over a long period since our tasks
        # are usually rather long running
        refresh_per_second=1,
        speed_estimate_period=600,
        expand=True,
    )


def setup_logging_and_parsl(
    parsl_config: parsl.Config,
    level: int = logging.INFO,
    stored_messages: Sequence[LogMessage] | None = None,
) -> parsl.DataFlowKernel:
    """Configure logging and setup the parsl config.

    We need a separate function because otherwise parsl will spam log messages.
    Here, we use some tricks to keep the messages in check.

    Args:
        parsl_config: Parsl configuration.
        level: Logging level. Default: "INFO".
        stored_messages: Messages stored that we can't log because we had to wait for the parsl
            initialization. Default: None.

    Returns:
        The parsl DataFlowKernel created from the config
    """
    if not stored_messages:
        stored_messages = []

    # First, setup logging at critical level. This will ensure that the parsl spamming
    # loggers # will only log at that level (it's still unclear how they're initialized,
    # but # they seem to inherit this value, and won't change it if the level changes
    # later. Which is the source of all of the problems).
    res = setup_logging(level=logging.CRITICAL)
    if not res:
        _msg = "Failed to setup logging. Wat?"
        raise RuntimeError(_msg)

    # Next, load the parsl config
    dfk = parsl.load(parsl_config)
    # Finally, set the root logger to what we actually wanted in the first place.
    logging.getLogger().setLevel(level)
    logging.getLogger().handlers[0].setLevel(level)
    # Just in case, try to reset parsl to a reasonable level. It probably won't work,
    # but it doesn't hurt.
    logging.getLogger("parsl").setLevel(logging.WARNING)
    # And then log the stored messages, so they have a chance to emit at the desired level
    for message in stored_messages:
        message.log()

    return dfk


def setup_logging(
    level: int = logging.INFO,
    stored_messages: Sequence[LogMessage] | None = None,
) -> bool:
    """Configure logging.

    NOTE:
        Don't call this before parsl has loaded a config! Otherwise, you'll be permanently
        inundated with irrelevant log messages. Hopefully this will be fixed in parsl eventually.
        Until then, use `setup_logging_and_parsl` instead.

    Args:
        level: Logging level. Default: "INFO".
        stored_messages: Messages stored that we can't log because we had to wait for the parsl
            initialization. Default: None.

    Returns:
        True if logging was set up successfully.
    """
    if not stored_messages:
        stored_messages = []

    # First, setup logging
    FORMAT = "%(message)s"
    # NOTE: For shutting up parsl, it's import to set the level both in the logging setup
    #       as well as in the handler.
    logging.basicConfig(
        level=level,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichModuleNameHandler(level=level, console=rich_console, rich_tracebacks=True)],
    )

    # Capture warnings into logging
    logging.captureWarnings(True)

    # Quiet down some loggers for sanity
    # Generally, parsl's logger configuration doesn't work for a number of modules because
    # they're not in the parsl namespace, but it's still worth trying to make it quieter.
    # NOTE: There appear to be some further issues with parsl logging beyond just this.
    logging.getLogger("parsl").setLevel(logging.WARNING)
    # For sanity when using IPython
    logging.getLogger("parso").setLevel(logging.INFO)

    # Log the stored up messages.
    for message in stored_messages:
        message.log()

    return True
