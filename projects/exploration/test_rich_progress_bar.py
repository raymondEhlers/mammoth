from __future__ import annotations

import logging
import time

from rich.progress import Progress

from mammoth import helpers

logger = logging.getLogger(__name__)

def test_rich_progress_bar() -> None:
    stored_messages = [
        helpers.LogMessage(
            __name__,
            "info",
            "Stored message at info level",
        )
    ]
    rich_console = helpers.rich_console

    logger.warning("First warning")

    gen = range(10)

    with Progress(console=rich_console, refresh_per_second=10) as progress:
        track_results = progress.add_task(total=10, description="Processing results...")
        for r in gen:
            logger.info(f"log info: {r}")
            logger.warning(f"log warning: {r}")
            progress.console.print(f"progress object: {r}")
            progress.update(track_results, advance=1)
            time.sleep(0.5)

    logger.info("afterwards...")

if __name__ == "__main__":
    test_rich_progress_bar()
