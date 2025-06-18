from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def import_pythia() -> Any:
    found_pythia = False
    try:
        import pythia8mc as pythia8  # type: ignore[import-not-found]

        found_pythia = True
    except ImportError:
        logger.debug("Did not find pythia8mc - keep trying for local version.")

    try:
        import pythia8  # type: ignore[import-not-found] # pyright: ignore [reportMissingImports]

        found_pythia = True
    except ImportError:
        logger.debug("Did not find pythia8")

    if not found_pythia:
        msg = """Could not find pythia8 or pythia8mc. Did you install it yet?

If not, please install with something like: `pip install pythia8mc`."""
        raise ImportError(msg)

    return pythia8
