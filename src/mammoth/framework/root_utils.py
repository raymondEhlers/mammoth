"""(Avoid if possible!) Utilities for ROOT based analyses.

**Please, please, please reconsider using ROOT!** There are rare instances where it's absolutely required,
but the purpose of mammoth was precisely to avoid all of the bullshit associated with ROOT. So avoid as much
as possible!

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def import_ROOT(n_cores: int | None = None, force_reinitialization: bool = False) -> Any:
    """Import ROOT and configure as necessary to avoid a deadlock.

    Args:
        n_cores: Number of cores to use. If None, it will use the default value.
        force_reinitialize: Force reinitialize of the ROOT settings.

    Returns:
        The ROOT module.
    """
    import ROOT  # pyright: ignore[reportMissingImports]

    # NOTE: This is really important to avoid a deadlock (appears to be on taking the gil according to lldb).
    #       In principle, it's redundant after the first import, but calling anything on the ROOT module deadlocks
    #       it's really annoying for debugging! So we just always call it.
    ROOT.gROOT.SetBatch(True)

    # If ROOT is not in batch mode, we know that it's the first important, and we should fully configure it.
    # We keep track of this because we don't want to keep changing the n_core initialization (or having to pass the value around).
    # This way, we can import once with the right setting, and then just keep going with the existing settings.
    if not ROOT.IsImplicitMTEnabled() or force_reinitialization:
        # NOTE: The default argument in ROOT for n_cores for EnableImplicitMT is __0__. Passing 1 implies a thread pool of one!
        if n_cores is None:
            n_cores = 0

        logger.info(f"import_ROOT: Enabling ImplicitMT with {n_cores=}")
        ROOT.EnableImplicitMT(n_cores)

    return ROOT
