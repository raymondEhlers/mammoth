
"""Base functionality for ECCE analyses

.. codeuathor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Dict

import hist
import uproot

logger = logging.getLogger(__name__)


def load_hists(filename: Path, filter: str = "") -> Dict[str, hist.Hist]:
    """Load histograms from a flat root file

    Note:
        The typing is lying a bit here, but it's good enough - we can convert later,
        and there's no obvious typing for uproot hists since they're generally dynamically

    """
    hists = {}
    with uproot.open(filename) as f:
        for k in f.keys(cycle=False):
            if filter and filter not in k:
                continue
            hists[k] = f[k]

    return hists

