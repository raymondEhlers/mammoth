
"""Base functionality for ECCE analyses

.. codeuathor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import attr
import hist
import uproot

logger = logging.getLogger(__name__)


def load_hists(filename: Path, filter: str = "", filters: Sequence[str] = None) -> Dict[str, hist.Hist]:
    """Load histograms from a flat root file

    Note:
        The typing is lying a bit here, but it's good enough - we can convert later,
        and there's no obvious typing for uproot hists since they're generally dynamically

    """
    if filters is None:
        filters = []
    hists = {}
    logger.debug(f"Loading {filename} with filter: {filter}, filters: {filters}")
    with uproot.open(filename) as f:
        for k in f.keys(cycle=False):
            if filter and filter not in k:
                continue
            if filters and all(f not in k for f in filters):
                continue
            hists[k] = f[k]

    return hists



@attr.s(frozen=True)
class DatasetSpec:
    site: str = attr.ib()
    label: str = attr.ib()

    @property
    def identifier(self) -> str:
        return ""

    def __str__(self) -> str:
        s = f"{self.site}-{self.identifier}"
        if self.label:
            s += f"-{self.label}"
        return s


@attr.s(frozen=True)
class DatasetSpecSingleParticle(DatasetSpec):
    particle: str = attr.ib()
    momentum_selection: List[float] = attr.ib()

    @property
    def identifier(self) -> str:
        return f"single{self.particle.capitalize()}-p-{self.momentum_selection[0]:g}-to-{self.momentum_selection[1]:g}"


@attr.s(frozen=True)
class DatasetSpecPythia(DatasetSpec):
    generator: str = attr.ib()
    electron_beam_energy: int = attr.ib()
    proton_beam_energy: int = attr.ib()
    _q2_selection: List[int] = attr.ib()

    @property
    def q2(self) -> str:
        if len(self._q2_selection) == 2:
            return f"q2-{self._q2_selection[0]}-to-{self._q2_selection[1]}"
        elif len(self._q2_selection) == 1:
            return f"q2-{self._q2_selection[0]}"
        return ""

    @property
    def q2_display(self) -> str:
        if len(self._q2_selection) == 2:
            return fr"{self._q2_selection[0]} < Q^{{2}} < {self._q2_selection[1]}"
        elif len(self._q2_selection) == 1:
            return fr"Q^{{2}} > {self._q2_selection[0]}"
        return ""

    @property
    def identifier(self) -> str:
        return f"{self.generator}-{self.electron_beam_energy}x{self.proton_beam_energy}-{self.q2}"

