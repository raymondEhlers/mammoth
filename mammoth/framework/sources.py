""" Input sources

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from typing import Any, Iterable, List, Mapping, Sequence, Tuple, Type
from typing_extensions import Protocol

import attr
import awkward as ak
import numpy as np
import uproot

class Source(Protocol):
    """ Data source.

    Attributes:
        metadata: Source metadata.
    """
    metadata: Mapping[str, Any]

    def data(self) -> Iterable[ak.Array]:
        """ Return data from the source.

        Returns:
            Data in the form of awkward arrays.
        """
        ...


def _contains_required_uproot_fields(instance: "UprootSource", attribute: attr.Attribute[Mapping[str, int]], value: Mapping[str, Any]) -> None:
    """Require that the uproot source has sufficient metadata info to work with a file."""
    required_keys = set(["tree_name"])
    all_keys = set(value.keys())
    missing_keys = all_keys

    # Validate
    missing_keys = [k for k in required_keys if k not in value.keys()]
    if missing_keys:
        raise ValueError(f"Missing metadata keys: {missing_keys}")


@attr.s
class UprootSource:
    _filename: str = attr.ib()
    _columns: Sequence[str] = attr.ib(factory=list)
    _entry_range: Tuple[Optional[int], Optional[int]] = attr.ib(default=[None, None])
    metadata: Mapping[str, Any] = attr.ib(factory=dict, validator=[_contains_required_uproot_fields])

    def data(self) -> Iterable[ak.Array]:
        with uproot.open(self._filename) as f:
            tree = f[self.metadata["tree_name"]]

            # Add metadata
            self.metadata["n_entries"] = tree.num_entries
            if self._entry_range[0] is not None or self._entry_range[1] is not None:
                self.metadata["entry_start"] = self._entry_range[0]
                self.metadata["entry_stop"] = self._entry_range[1]

            # Add restricted start and stop entries if requested.
            reading_kwargs = {
                "expressions": self._columns if self._columns else None,
            }
            if self.metadata["entry_start"] and self.metadata["entry_stop"]:
                reading_kwargs.update({"entry_start": self.metadata["entry_start"], "entry_stop": self.metadata["entry_stop"]})

            yield tree.arrays(**reading_kwargs)


@attr.s
class ParquetSource:
    _filename: str = attr.ib()
    _columns: Sequence[str] = attr.ib(factory=list)
    metadata: Mapping[str, Any] = attr.ib(factory=dict)

    def data(self) -> Iterable[ak.Array]:
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
    _parameters: List[float] = attr.ib()
    _n_events: int = attr.ib()
    metadata: Mapping[str, Any] = attr.ib(factory=dict)


@attr.s
class MultipleFileSource:
    """ Source which is composed of multiple files.

    Args:
        filenames: Names of the files.
        source_type: Source to use with the filenames.
        metadata: Source metadata.
    """
    _filenames: Sequence[str] = attr.ib()
    _source_type: Type[Source] = attr.ib()
    metadata: Mapping[str, Any] = attr.ib(factory=dict)

    def data(self) -> Iterable[ak.Array]:
        for filename in self._filenames:
            _source = self._source_type(filename)
            yield _source


class ChunkSource:
    chunk_size: int = attr.ib()
    _source: MultipleFileSource = attr.ib()
    metadata: Mapping[str, Any] = attr.ib(factory=dict)

    def data(self) -> Iterable[ak.Array]:
        source_iter = self._source.data()
        remaining_data = None
        while True:
            if remaining_data:
                _data = remaining_data
                remaining_data = None
            else:
                _data = next(source_iter)

            if len(_data) == self.chunk_size:
                yield _data
            elif len(_data) < self.chunk_size:
                additional_chunks = []
                remaining_n_events = self.chunk_size - len(_data)
                for _more_data in source_iter:
                    remaining_n_events -= len(_more_data)
                    if remaining_n_events < 0:
                        # Slice the reamining data and store for the next iteration
                        additional_chunks.append(_more_data[:remaining_n_events])
                        remaining_data = _more_data[remaining_n_events:]
                        break
                    additional_chunks.append(_more_data)
                yield ak.concatenate(
                    [_data, *additional_chunks],
                    axis=0,
                )
            else:
                remaining_n_events = self.chunk_size - len(_data)
                remaining_data = _data[remaining_n_events:]
                yield _data[:remaining_n_events]


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

    def data(self) -> Iterable[ak.Array]:
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


@attr.s
class EmbeddedSourceTransform:
    """ Transform an embedded source

    """

    def transform(self, input: ak.Array) -> ak.Array:
        particles = ak.Array({
            "true": input[[]],
            "det_level": input[[]],
            "hybrid": input[[]],
        })
        return particles
