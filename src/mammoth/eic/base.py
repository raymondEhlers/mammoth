
"""Base functionality for ECCE analyses

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import attr
import hist
import uproot

logger = logging.getLogger(__name__)


def load_hists(filename: Path, filter: Optional[str] = "", filters: Optional[Sequence[str]] = None, require_ends_with_in_filter: bool = False) -> Dict[str, hist.Hist]:
    """Load histograms from a flat root file

    Note:
        The typing is lying a bit here, but it's good enough - we can convert later,
        and there's no obvious typing for uproot hists since they're generally dynamically

    Note:
        Specify either filter or filters, but not both!

    Args:
        filename: Path to the root file.
        filter: Filter to apply to the keys. Default: "".
        filters: List of filters to apply to the keys. Default: [].
        require_ends_with_in_filter: Instead of testing with `in` for the filter, test
            with `endswith`. This can be convenient for requiring exact matches to reduce
            the number of hists extracted, at the expense of a more complicated or
            expansive filter. Default: False.

    Returns:
        Dict of hist name -> hist.
    """
    # Raise error if both are passed
    if filter and filters:
        raise ValueError(f"Please provide only a single filter, or a list of filters. Not both. filter: {filter}, filters: {filters}")
    # Ensure the filters list is always iterable
    if filters is None:
        filters = []
    # We always want to use the filters list. So if only a filter is passed, then put it into the filters list.
    # This reduces the number of code paths, which makes the code simpler
    if filter != "" and filter is not None:
        filters = [filter]

    hists = {}
    logger.info(f"Loading {filename} with filter: {filter}, filters: {filters}")
    with uproot.open(filename) as f:
        for k in f.keys(cycle=False):
            if filters:
                if require_ends_with_in_filter:
                    if all(not k.endswith(f) for f in filters):
                        continue
                else:
                    if all(f not in k for f in filters):
                        continue
            hists[k] = f[k]

    return hists


@attr.frozen
class DatasetSpec:
    site: str
    label: str

    @property
    def identifier(self) -> str:
        return ""

    def __str__(self) -> str:
        s = f"{self.site}-{self.identifier}"
        if self.label:
            s += f"-{self.label}"
        return s


@attr.frozen
class DatasetSpecSingleParticle(DatasetSpec):
    particle: str
    momentum_selection: List[float]

    @property
    def identifier(self) -> str:
        return f"single{self.particle.capitalize()}-p-{self.momentum_selection[0]:g}-to-{self.momentum_selection[1]:g}"


@attr.frozen
class DatasetSpecPythia(DatasetSpec):
    generator: str
    electron_beam_energy: int
    proton_beam_energy: int
    _q2_selection: List[int]

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
