""" Input sources

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from typing import Any, Iterator, List, Mapping, Sequence, Tuple, Type
from typing_extensions import Protocol

import attr
import awkward as ak
import numpy as np

class Source(Protocol):
    metadata: Mapping[str, Any]

    def events(self) -> Iterator[Tuple[ak.Array, ak.Array]]:
        """

        Returns:
            Event info, particles
        """
        ...


@attr.s
class UprootSource:
    _filename: str = attr.ib()
    _columns: Sequence[str] = attr.ib(factory=list)
    metadata: Mapping[str, Any] = attr.ib(factory=dict)

    def events(self) -> Iterator[Tuple[ak.Array, ak.Array]]:
        ...


@attr.s
class ParquetSource:
    _filename: str = attr.ib()
    _columns: Sequence[str] = attr.ib(factory=list)
    metadata: Mapping[str, Any] = attr.ib(factory=dict)

    def events(self) -> Iterator[Tuple[ak.Array, ak.Array]]:
        yield ak.from_parquet(
            self._filename,
            columns=self._columns if self._columns else None,
        )


@attr.s
class JetscapeSource(ParquetSource):
    """ Jetscape source via Parquet file.

    Nothing needs to be done here.
    """
    ...


class GeneratorSource:
    ...


@attr.s
class ThermalBackground:
    """ Thermal Background

    Try quick prototype to ensure that it works with this approach.
    """
    parameters: List[float] = attr.ib()
    n_events: int = attr.ib()


@attr.s
class MultipleFileSource:
    _filenames: Sequence[str] = attr.ib()
    _source_type: Type[Source] = attr.ib()
    metadata: Mapping[str, Any] = attr.ib(factory=dict)
    ...

    def events(self) -> Iterator[Tuple[ak.Array, ak.Array]]:
        ...


def _contains_signal_and_background(instance: "MultipleSources", attribute: attr.Attribute[Mapping[str, int]], value: Mapping[str, int]) -> None:
    found_signal = False
    found_background = False
    for k in value.keys():
        if "signal" in k:
            found_signal = True
        if "background" in k:
            found_background = True
    if not found_signal:
        raise ValueError(f"Must contain at least one signal source. Found: {list(value.keys())}.")
    if not found_background:
        raise ValueError(f"Must contain at least one background source. Found: {list(value.keys())}.")


def _has_offset_per_source(instance: "MultipleSources", attribute: attr.Attribute[Mapping[str, int]], value: Mapping[str, int]) -> None:
    if set(instance._sources) != set(instance.source_index_identifiers):
        raise ValueError("Mismtach in sources and offsets. Sources: {list(instance._sources)}, offsets: {list(instance.source_index_identifiers)}")


@attr.s
class MultipleSources:
    """ Combine multiple data sources together.

    Think: Embedding into data, embedding into thermal model, etc.

    Attributes:
        _sources: Contains an arbitary number of sources.
        source_index_identifiers: Map contanining an integer identifier for each source.
        _particles_columns: Names of columns to include in the particles.
    """
    _sources: Mapping[str, Source] = attr.ib(validator=_contains_signal_and_background)
    source_index_identifiers: Mapping[str, int] = attr.ib(factory=dict, validator=[_contains_signal_and_background, _has_offset_per_source])
    _particles_columns: Sequence[str] = attr.ib(factory=lambda: ["px", "py", "pz", "E"])
    metadata: Mapping[str, Any] = attr.ib(factory=dict)

    def events(self) -> Iterator[Tuple[ak.Array, ak.Array]]:
        # Grab the events from the sources
        source_events = {k: v.events()() for k, v in self._sources.items()}

        # Add source IDs
        for k, v in source_events.items():
            # Need a way to get a column that's properly formatted, so we take a known good one,
            # and then set the value as appropriate.
            v["source_ID"] = ak.values_astype(
                v[self._particles_columns[0]] * 0 + self.source_index_identifiers[k],
                np.int16,
            )

        # Need to differentiate the event info keys from the particle keys.
        event_keys = {k: set(ak.keys(source_events[k])) for k in source_events}
        event_keys = {k: v.difference(self._particles_columns) for k, v in event_keys.items()}

        # Check if there are keys which we will overwrite with each other.
        shared_keys = {k: k1.union(k2) for k, k1 in event_keys.items() for k2 in event_keys if k1 != k2}
        # If there are shared keys, need to rename them to be unique.
        # TODO: For now, just raise a KeyError
        raise KeyError(f"Overlapping keys: {shared_keys}")

        event_info = ak.zip(
            {
                # TODO: I'm not sure these would combine cleanly...
                #       May need to ask a question on the awkward discussion board.
                ak.unzip(source.events()[event_keys[k]]) for k, source in source_events
            },
            depth = 1,
        )

        # TODO: This isn't right. It needs to append the two collections together.
        #       This should be done via awkward primivities.
        particles = ak.zip(
            {
                k: ak.concatenate(
                    [source[k] for source in source_events.values()],
                    axis=1
                )
                for k in self._particles_columns
            },
        )

        # TODO: This isn't right. What about part vs det vs hybrid level, for example?
        yield event_info, particles

