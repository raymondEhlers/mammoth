""" Basic objects related to analysis.

These all build on various aspects of the framework, but are at a higher level than the basic framework
functionality itself.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL
"""

import logging
from pathlib import Path
from typing import Any, Dict, Type

import attr
import attrs
import pachyderm.yaml
from pachyderm import binned_data

logger = logging.getLogger(__name__)

# NOTE: If we use attr.frozen instead, the yaml serialization won't work. The reason isn't clear
#       as of Sept 2022, but there's no point in investigating further since the old method seems to work fine.
@attr.s(frozen=True)
class ScaleFactor:
    """Store scale factors for a particular pt hard bin.

    In the case of going event-by-event in pythia, we would scale by cross_section / n_trials
    for that bin. However, if we're going by the single histograms per output file, it gets a
    good deal more complicated. This calculation has evolved significantly once we thought about
    this carefully and ran a bunch of tests. The right answer is simply cross_section / n_trials_total
    where n_trials_total much be the n_trials for the _entire_ pt hard bin!

    Attributes:
        cross_section: Cross section.
        n_trials_total: Total number of trials from the whole pt hard bin.
    """

    # float cast to ensure that we get a standard float instead of an np.float
    cross_section: float = attrs.field(converter=float)
    n_trials_total: int = attrs.field(converter=int)
    n_entries: int = attrs.field(converter=int)
    n_accepted_events: int = attrs.field(converter=int)

    def value(self) -> float:
        """Value of the scale factor.

        Args:
            None.
        Returns:
            Scale factor calculated based on the extracted values.
        """
        return self.cross_section / self.n_trials_total

    @classmethod
    def from_hists(
        cls: Type["ScaleFactor"], n_accepted_events: int, n_entries: int, cross_section: Any, n_trials: Any
    ) -> "ScaleFactor":
        # Validation (ensure that hists are valid)
        # NOTE: Since we're using BinnedData here just to grab the values, it's not overly critical
        #       whether it's aware of the type of hist (regular vs profile, for example).
        h_cross_section = binned_data.BinnedData.from_existing_data(cross_section)
        h_n_trials = binned_data.BinnedData.from_existing_data(n_trials)

        # Find the first non-zero values bin.
        # argmax will return the index of the first instance of True.
        # NOTE: This isn't the true value of the pt hard bin because of indexing from 0.
        #       The true pt hat bin is 1-indexed.
        pt_hat_bin_index = (h_cross_section.values != 0).argmax(axis=0)

        return cls(
            cross_section=h_cross_section.values[pt_hat_bin_index],
            n_trials_total=h_n_trials.values[pt_hat_bin_index],
            n_entries=n_entries,
            n_accepted_events=n_accepted_events,
        )


def read_extracted_scale_factors(
    path: Path,
) -> Dict[int, float]:
    """Read extracted scale factors.

    Args:
        collision_system: Name of the collision system.
        dataset_name: Name of the dataset.

    Returns:
        Normalized scaled factors
    """
    # Validation
    path = Path(path)

    y = pachyderm.yaml.yaml(classes_to_register=[ScaleFactor])
    with path.open() as f:
        scale_factors: dict[int, Any] = y.load(f)

    if hasattr(scale_factors[next(iter(scale_factors))], "value"):
        # Standard track skim production
        return_scale_factors: dict[int, ScaleFactor] = {
            pt_hard_bin: v.value() for pt_hard_bin, v in scale_factors.items()
        }
    # We already have the map of [int, float], so just pass it on. For example,
    # from the HF_tree
    return scale_factors
