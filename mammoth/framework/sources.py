""" Input sources

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import collections.abc
import itertools
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Union
from typing_extensions import Protocol

import attr
import awkward as ak
import numpy as np
import uproot

from mammoth.framework import models, utils


class Source(Protocol):
    """ Data source.

    Attributes:
        metadata: Source metadata.
    """
    metadata: MutableMapping[str, Any]

    def __len__(self) -> int:
        """ Number of entries in the source. """
        ...

    def data(self) -> ak.Array:
        """ Return data from the source.

        Returns:
            Data in an awkward array.
        """
        ...


def _convert_range(entry_range: Union[utils.Range, Sequence[float]]) -> utils.Range:
    """Convert sequences to Range.

    Args:
        entry_range: Range of entries to be stored in a Range.
    Returns:
        Range
    """
    if isinstance(entry_range, utils.Range):
        return entry_range
    return utils.Range(*entry_range)


@attr.s
class UprootSource:
    _filename: Path = attr.ib(converter=Path)
    _tree_name: str = attr.ib()
    _columns: Sequence[str] = attr.ib(factory=list)
    _entry_range: utils.Range = attr.ib(converter=_convert_range, default=utils.Range(None, None))
    metadata: MutableMapping[str, Any] = attr.ib(factory=dict)

    def __len__(self) -> int:
        if "n_entries" in self.metadata:
            return int(self.metadata["n_entries"])
        raise ValueError("N entries not yet available.")

    def data(self) -> ak.Array:
        with uproot.open(self._filename) as f:
            tree = f[self.metadata["tree_name"]]

            # First, let's setup the arguments
            # Columns
            reading_kwargs: Dict[str, Any] = {
                "expressions": self._columns if self._columns else None,
            }
            # Add restricted start and stop entries if requested.
            # Only if we specify a start and stop do we actually pass it on to uproot.
            if self._entry_range.min and self._entry_range.max:
                reading_kwargs.update({"entry_start": self._entry_range.min, "entry_stop": self._entry_range.max})

            # Add metadata
            self.metadata["entry_start"] = self._entry_range.min if self._entry_range.min is not None else 0
            self.metadata["entry_stop"] = self._entry_range.max if self._entry_range.max is not None else tree.num_entries
            self.metadata["n_entries"] = self.metadata["entry_stop"] - self.metadata["entry_start"]

            return tree.arrays(**reading_kwargs)


def chunked_uproot_source(
    filename: Path,
    tree_name: str,
    chunk_size: int,
    columns: Optional[Sequence[str]] = None,
) -> List[UprootSource]:
    """ Create a set of uproot sources in chunks for a given filename.

    This is most likely to be the main interface.

    Returns:
        List of UprootSource configured with the provided properties.
    """
    sources = []
    if columns is None:
        columns = []
    with uproot.open(filename) as f:
        number_of_entries = f[tree_name].num_entries

        start = 0
        continue_iterating = True
        while continue_iterating:
            end = start + chunk_size
            # Ensure that we never ask for more entries than are in the file.
            if start + chunk_size > number_of_entries:
                end = number_of_entries
                continue_iterating = False
            # Store the start and stop for convenience.
            sources.append(
                UprootSource(
                    filename=filename,
                    tree_name=tree_name,
                    columns=columns,
                    entry_range=utils.Range(start, end),
                )
            )
            # Move up to the next iteration.
            start = end

    return sources


@attr.s
class ParquetSource:
    _filename: Path = attr.ib(converter=Path)
    _columns: Sequence[str] = attr.ib(factory=list)
    metadata: MutableMapping[str, Any] = attr.ib(factory=dict)

    def __len__(self) -> int:
        if "n_entries" in self.metadata:
            return int(self.metadata["n_entries"])
        raise ValueError("N entries not yet available.")

    def data(self) -> ak.Array:
        arrays = ak.from_parquet(
            self._filename,
            columns=self._columns if self._columns else None,
        )

        # Extract metadata
        self.metadata["entry_start"] = 0
        self.metadata["entry_stop"] = len(arrays)
        self.metadata["n_entries"] = self.metadata["entry_stop"] - self.metadata["entry_start"]

        return arrays


@attr.s
class JetscapeSource(ParquetSource):
    """ Jetscape source via Parquet file.

    Nothing needs to be done here.
    """
    ...

@attr.s
class PythiaSource:
    config: Path = attr.ib(converter=Path)
    chunk_size: int = attr.ib()
    metadata: MutableMapping[str, Any] = attr.ib(factory=dict)

    def __len__(self) -> int:
        return self.chunk_size

    def data(self) -> ak.Array:
        raise NotImplementedError("Working on it...")


@attr.s
class ThermalBackgroundExponential:
    """ Thermal background model from Leticia

    Assume thermal particles are massless.
    pt = x*exp(-x/pt_exponential_scale), from 0 to 400, at least 40000 sampling points
    eta = flat from -1 to 1, at least 200 sampling points
    phi = flat from -pi to pi, at least 700 sampling points

    pt exponential scale is 0.4 by default.

    The number of thermal particles is determined by a Gaussian. The parameters are:
    - central: mean = 2500, sigma = 500
    - semi-central: mean = 1000, sigma = 400

    """
    chunk_size: int = attr.ib()
    n_particles_per_event_mean: float = attr.ib()
    n_particles_per_event_sigma: float = attr.ib()
    pt_exponential_scale: float = attr.ib(default=0.4)
    metadata: MutableMapping[str, Any] = attr.ib(factory=dict)

    def __len__(self) -> int:
        return self.chunk_size

    def data(self) -> ak.Array:
        # Setup
        rng = np.random.default_rng()

        # Determine overall event parameters.
        # NOTE: This is effectively jagged, since the number of particles per event varies
        # NOTE: We round to integers because the number of particles must of course be an int.
        n_particles_per_event = np.rint(
            rng.normal(loc=self.n_particles_per_event_mean, scale=self.n_particles_per_event_sigma, size=self.chunk_size),
        ).astype(np.int32)
        # To help out with this effective jaggedness, we flatten everything, and then will unflatten with awkward.
        total_n_samples = int(np.sum(n_particles_per_event))

        # Sample the distributions.
        pt = models.sample_x_exp(n_samples=total_n_samples, scale=self.pt_exponential_scale, x_min=0, x_max=400)
        eta = rng.uniform(low=-1, high=1, size=total_n_samples)
        phi = rng.uniform(low=-np.pi, high=np.pi, size=total_n_samples)

        # Need this as an intermediary, so calculate it first
        pz = pt * np.sinh(eta)

        # Finally, add the particle structure at the end.
        return ak.unflatten(
            ak.Array({
                "px": pt * np.cos(phi),
                "py": pt * np.sin(phi),
                "pz": pz,
                "E": np.sqrt(pt ** 2 + pz ** 2),
            }),
            counts=n_particles_per_event,
        )

def _sources_to_list(sources: Union[Source, Sequence[Source]]) -> Sequence[Source]:
    if not isinstance(sources, collections.abc.Iterable):
        return [sources]
    return sources


@attr.s
class ChunkSource:
    _chunk_size: int = attr.ib()
    _sources: Sequence[Source] = attr.ib(converter=_sources_to_list)
    _repeat: bool = attr.ib(default=False)
    metadata: MutableMapping[str, Any] = attr.ib(factory=dict)

    def __len__(self) -> int:
        if "n_entries" in self.metadata:
            return int(self.metadata["n_entries"])
        raise ValueError("N entries not yet available.")

    def data(self) -> ak.Array:
        """ Retrieve data to satisfy the given chunk size.

        """
        return next(iter(self.data_iter()))

    def data_iter(self) -> Iterable[ak.Array]:
        if self._repeat:
            # See: https://stackoverflow.com/a/24225372/12907985
            source_iter = itertools.chain.from_iterable(itertools.repeat(self._sources))
        else:
            source_iter = iter(self._sources)
        remaining_data = None

        while True:
            if remaining_data:
                _data = remaining_data
                remaining_data = None
            else:
                _data = next(source_iter).data()

            # Regardless of where we end up, the number of entries must be equal to the chunk size
            self.metadata["n_entries"] = self._chunk_size

            # Now, figure out how to get all of the required data.
            if len(_data) == self._chunk_size:
                yield _data
            elif len(_data) < self._chunk_size:
                additional_chunks = []
                remaining_n_events = self._chunk_size - len(_data)
                for _more_data_source in source_iter:
                    _more_data = _more_data_source.data()
                    remaining_n_events -= len(_more_data)
                    if remaining_n_events < 0:
                        # Slice the remaining data and store for the next iteration
                        additional_chunks.append(_more_data[:remaining_n_events])
                        remaining_data = _more_data[remaining_n_events:]
                        break
                    additional_chunks.append(_more_data)
                yield ak.concatenate(
                    [_data, *additional_chunks],
                    axis=0,
                )
            else:
                remaining_n_events = self._chunk_size - len(_data)
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
    if set(instance._sources) != set(instance._source_index_identifiers):
        raise ValueError("Mismatch in sources and offsets. Sources: {list(instance._sources)}, offsets: {list(instance.source_index_identifiers)}")


@attr.s
class MultipleSources:
    """ Combine multiple data sources together.

    Think: Embedding into data, embedding into thermal model, etc.

    Attributes:
        _sources: Contains an arbitrary number of sources.
        source_index_identifiers: Map containing an integer identifier for each source.
        _particles_columns: Names of columns to include in the particles.
    """
    # _signal_source: ChunkSource = attr.ib()
    # _background_source: ChunkSource = attr.ib()
    _sources: Mapping[str, Source] = attr.ib(validator=_contains_signal_and_background)
    _source_index_identifiers: Mapping[str, int] = attr.ib(factory=dict, validator=[_contains_signal_and_background, _has_offset_per_source])
    _particles_columns: Sequence[str] = attr.ib(factory=lambda: ["px", "py", "pz", "E"])
    metadata: MutableMapping[str, Any] = attr.ib(factory=dict)

    def __len__(self) -> int:
        if "n_entries" in self.metadata:
            return int(self.metadata["n_entries"])
        raise ValueError("N entries not yet available.")

    def data(self) -> ak.Array:
        # Grab the events from the sources
        source_data = {k: v.data() for k, v in self._sources.items()}

        # Cross check that we have the right sizes for all data sources
        lengths = [len(v) for v in source_data.values()]
        if lengths.count(lengths[0]) != len(lengths):
            raise ValueError(f"Length of data doesn't match: {lengths}")

        # Add source IDs
        for k, v in source_data.items():
            # Need a way to get a column that's properly formatted, so we take a known good one,
            # and then set the value as appropriate.
            v["source_ID"] = ak.values_astype(
                v[self._particles_columns[0]] * 0 + self._source_index_identifiers[k],
                np.int16,
            )

        # Add metadata
        self.metadata["n_entries"] = lengths[0]

        return ak.Array(source_data)


#@attr.s
#class MultipleSources:
#    """ Combine multiple data sources together.
#
#    Think: Embedding into data, embedding into thermal model, etc.
#
#    Attributes:
#        _sources: Contains an arbitary number of sources.
#        source_index_identifiers: Map contanining an integer identifier for each source.
#        _particles_columns: Names of columns to include in the particles.
#    """
#    _sources: Mapping[str, Source] = attr.ib(validator=_contains_signal_and_background)
#    source_index_identifiers: Mapping[str, int] = attr.ib(factory=dict, validator=[_contains_signal_and_background, _has_offset_per_source])
#    _particles_columns: Sequence[str] = attr.ib(factory=lambda: ["px", "py", "pz", "E"])
#    metadata: MutableMapping[str, Any] = attr.ib(factory=dict)
#
#    def data(self) -> Iterable[ak.Array]:
#        # Grab the events from the sources
#        source_events = {k: v.events()() for k, v in self._sources.items()}
#
#        # Add source IDs
#        for k, v in source_events.items():
#            # Need a way to get a column that's properly formatted, so we take a known good one,
#            # and then set the value as appropriate.
#            v["source_ID"] = ak.values_astype(
#                v[self._particles_columns[0]] * 0 + self.source_index_identifiers[k],
#                np.int16,
#            )
#
#        # Need to differentiate the event info keys from the particle keys.
#        event_keys = {k: set(ak.keys(source_events[k])) for k in source_events}
#        event_keys = {k: v.difference(self._particles_columns) for k, v in event_keys.items()}
#
#        # Check if there are keys which we will overwrite with each other.
#        shared_keys = {k: k1.union(k2) for k, k1 in event_keys.items() for k2 in event_keys if k1 != k2}
#        # If there are shared keys, need to rename them to be unique.
#        # TODO: For now, just raise a KeyError
#        raise KeyError(f"Overlapping keys: {shared_keys}")
#
#        event_info = ak.zip(
#            {
#                # TODO: I'm not sure these would combine cleanly...
#                #       May need to ask a question on the awkward discussion board.
#                ak.unzip(source.events()[event_keys[k]]) for k, source in source_events
#            },
#            depth = 1,
#        )
#
#        # TODO: This isn't right. It needs to append the two collections together.
#        #       This should be done via awkward primivities.
#        particles = ak.zip(
#            {
#                k: ak.concatenate(
#                    [source[k] for source in source_events.values()],
#                    axis=1
#                )
#                for k in self._particles_columns
#            },
#        )
#
#        # TODO: This isn't right. What about part vs det vs hybrid level, for example?
#        yield event_info, particles


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
