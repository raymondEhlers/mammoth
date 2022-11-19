""" Input sources

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import collections.abc
import enum
import itertools
import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    Final,
    Generator,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from typing_extensions import Protocol

import attr
import awkward as ak
import numpy as np
import uproot

from mammoth.framework import models, utils

logger = logging.getLogger(__name__)


class ChunkSizeSentinel(enum.Enum):
    SOURCE_DEFAULT = object()
    FULL_SOURCE = object()
    # Specific examples that only work for some sources
    # Requires the users to specify a size
    FIXED_SIZE = object()
    # One file at a time
    SINGLE_FILE = object()


T_ChunkSize = Union[int, ChunkSizeSentinel]
T_GenData = Generator[ak.Array, Optional[T_ChunkSize], None]


class DelayedSource(Protocol):
    """Create a source via a function call.

    This is used for loading data, but it's much more convenient if it's available from the sources.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Source:
        ...


class SourceFromFilename(Protocol):
    """Create a source from a filename.

    This is effectively a specialization of a DelayedSource.

    This is used for loading data, but it's much more convenient if it's available from the sources.
    """

    def __call__(self, filename: Path) -> Source:
        ...


# Allows loading all chunks by picking a number larger than any possible (set of) file(s).
_FULL_SOURCE_SIZE: Final[int] = int(1e10)


class Source(Protocol):
    """Data source.

    Attributes:
        metadata: Source metadata.
    """

    _default_chunk_size: T_ChunkSize
    metadata: MutableMapping[str, Any]

    def gen_data(self, chunk_size: T_ChunkSize = ChunkSizeSentinel.SOURCE_DEFAULT) -> T_GenData:
        """A iterator over a fixed size of data from the source.

        Returns:
            Iterable containing chunk size data in an awkward array.
        """
        ...


class NoDataAvailableError(Exception):
    """Indicates that we somehow have no data available.

    Most likely, one of the trees was empty, or the tree was missing from the file entirely.
    """

    ...


def convert_sequence_to_range(entry_range: Union[utils.Range, Sequence[float]]) -> utils.Range:
    """Convert sequences to Range.

    Args:
        entry_range: Range of entries to be stored in a Range.
    Returns:
        Range
    """
    if isinstance(entry_range, utils.Range):
        return entry_range
    return utils.Range(*entry_range)


def _validate_chunk_size(chunk_size: T_ChunkSize, source_default_chunk_size: T_ChunkSize) -> int:
    # Initial validation
    if chunk_size is ChunkSizeSentinel.SOURCE_DEFAULT:
        chunk_size = source_default_chunk_size
        assert (
            chunk_size is not ChunkSizeSentinel.SOURCE_DEFAULT
        ), "The default chunk size for a source cannot be SOURCE_DEFAULT"

    # Full validation
    # Perform the rest of the validation now that we know the chunk size input
    if chunk_size is ChunkSizeSentinel.FULL_SOURCE or chunk_size is ChunkSizeSentinel.SINGLE_FILE:
        chunk_size = _FULL_SOURCE_SIZE
    if chunk_size is ChunkSizeSentinel.FIXED_SIZE:
        raise ValueError("User must provide a chunk size! There is no natural choice for this source.")
    if not isinstance(chunk_size, (int, np.integer)):
        raise ValueError(f"Unrecognized chunk size: {chunk_size}, {type(chunk_size)=}")

    return chunk_size


def generator_from_existing_data(
    data: ak.Array,
    chunk_size: T_ChunkSize,
    source_default_chunk_size: T_ChunkSize,
    warn_on_not_enough_data: bool = False,
) -> T_GenData:
    # Validation
    # Chunk size
    chunk_size = _validate_chunk_size(chunk_size=chunk_size, source_default_chunk_size=source_default_chunk_size)
    # Input data
    full_data_length = len(data)
    if full_data_length < chunk_size and warn_on_not_enough_data:
        logger.warning(f"Requested {chunk_size}, but only have {full_data_length}")

    while len(data) > 0:
        if len(data) < chunk_size:
            _result = yield data
            if _result is not None:
                # I think we don't want to raise an exception that will interfere with the generator
                # understanding that the generator is exhausted. For now (March 2022), we throw a warning
                # to keep an eye on this, but we don't do anything about it right now.
                # If it becomes an issue in the future, we could convert it into a ValueError.
                logger.warning(f"Requested new chunk of size {_result}, but we've already exhausted this source.")
            break
        else:
            _result = yield data[:chunk_size]

        # We want to make sure that we don't reuse any of the data.
        data = data[chunk_size:]

        # If we received a send argument, use this to update the chunk_size
        if _result is not None:
            chunk_size = _validate_chunk_size(chunk_size=_result, source_default_chunk_size=source_default_chunk_size)

    # And indicate we're done. Not necessarily required, but I want to be quite explicit here.
    return None


@attr.define
class UprootSource:
    _filename: Path = attr.field(converter=Path)
    _tree_name: str
    _columns: Sequence[str] = attr.Factory(list)
    _entry_range: utils.Range = attr.field(converter=convert_sequence_to_range, default=utils.Range(None, None))
    _default_chunk_size: Union[int, ChunkSizeSentinel] = attr.field(default=ChunkSizeSentinel.FULL_SOURCE)
    metadata: MutableMapping[str, Any] = attr.Factory(dict)

    def _data(self) -> ak.Array:
        with uproot.open(self._filename) as f:
            # Allow for star matching in tree_name
            if "*" in self._tree_name:
                logger.debug(f"Searching for tree name pattern {self._tree_name}")
                # Search for keys which contain the provided tree name. Very nicely, uproot already has this built-in
                _possible_tree_names = f.keys(cycle=False, filter_name=self._tree_name, filter_classname="TTree")
                if len(_possible_tree_names) != 1:
                    if len(_possible_tree_names) == 0:
                        raise NoDataAvailableError(
                            f"Missing tree name '{self._tree_name}'. Please check the file. Filename: {self._filename}"
                        )
                    else:
                        raise ValueError(
                            f"Ambiguous tree name '{self._tree_name}'. Please revise it as needed. Possible tree names: {_possible_tree_names}. Filename: {self._filename}"
                        )
                # We're good - let's keep going
                self._tree_name = _possible_tree_names[0]

            tree = f[self._tree_name]

            # First, let's setup the arguments
            # Columns
            reading_kwargs: Dict[str, Any] = {
                "expressions": self._columns if self._columns else None,
            }
            # Add restricted start and stop entries if requested.
            # Only if we specify a start and stop do we actually pass it on to uproot.
            # Check explicitly for not none because min could be 0 and still a valid range.
            if self._entry_range.min is not None and self._entry_range.max is not None:
                reading_kwargs.update(
                    {
                        "entry_start": self._entry_range.min,
                        "entry_stop": self._entry_range.max,
                    }
                )

            # Add metadata
            self.metadata["entry_start"] = self._entry_range.min if self._entry_range.min is not None else 0
            self.metadata["entry_stop"] = (
                self._entry_range.max if self._entry_range.max is not None else tree.num_entries
            )
            self.metadata["n_entries"] = self.metadata["entry_stop"] - self.metadata["entry_start"]

            return tree.arrays(**reading_kwargs)

    def gen_data(self, chunk_size: T_ChunkSize = ChunkSizeSentinel.SOURCE_DEFAULT) -> T_GenData:
        # NOTE: This is somewhat less efficient than it could be. We load all of the data and then chunk it afterwards.
        #       This isn't currently (March 2022) a big issue, but if that changes, we could rewrite this to become
        #       more efficient by reading and yielding chunks of the requested size directly from uproot.
        return generator_from_existing_data(
            data=self._data(),
            chunk_size=chunk_size,
            source_default_chunk_size=self._default_chunk_size,
            warn_on_not_enough_data=False,
        )


def define_multiple_sources_from_single_root_file(
    filename: Path,
    tree_name: str,
    chunk_size: int,
    columns: Optional[Sequence[str]] = None,
) -> List[UprootSource]:
    """Create a set of uproot sources in chunks for a given filename.

    This is only needed if the root file is so large that opening the whole thing at once creates memory issues.

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


@attr.define
class ParquetSource:
    _filename: Path = attr.field(converter=Path)
    _columns: Sequence[str] = attr.Factory(list)
    _default_chunk_size: Union[int, ChunkSizeSentinel] = attr.field(default=ChunkSizeSentinel.FULL_SOURCE)
    metadata: MutableMapping[str, Any] = attr.Factory(dict)

    def _data(self) -> ak.Array:
        arrays = ak.from_parquet(
            self._filename,
            columns=self._columns if self._columns else None,
        )

        # Extract metadata
        self.metadata["entry_start"] = 0
        self.metadata["entry_stop"] = len(arrays)
        self.metadata["n_entries"] = self.metadata["entry_stop"] - self.metadata["entry_start"]

        return arrays

    def gen_data(self, chunk_size: T_ChunkSize = ChunkSizeSentinel.SOURCE_DEFAULT) -> T_GenData:
        return generator_from_existing_data(
            data=self._data(),
            chunk_size=chunk_size,
            source_default_chunk_size=self._default_chunk_size,
        )


@attr.define
class ALICEFastSimTrackingEfficiency:
    """ALICE fast simulation based on tracking efficiency

    This is definitely a poor man's implementation, but it's fine for a first look.
    """

    particle_level_source: Source
    fast_sim_parameters: models.ALICEFastSimParameters
    _default_chunk_size: Union[int, ChunkSizeSentinel] = attr.field(default=ChunkSizeSentinel.FULL_SOURCE)
    metadata: MutableMapping[str, Any] = attr.Factory(dict)

    def _data(self, particle_level_data: ak.Array) -> ak.Array:
        # Setup
        rng = np.random.default_rng()

        self.metadata["n_entries"] = len(particle_level_data)

        efficiencies = models.alice_fast_sim_tracking_efficiency(
            track_pt=np.asarray(ak.flatten(particle_level_data["part_level"].pt, axis=-1)),
            track_eta=np.asarray(ak.flatten(particle_level_data["part_level"].eta, axis=-1)),
            event_activity=self.fast_sim_parameters.event_activity,
            period=self.fast_sim_parameters.period,
        )

        n_particles_per_event = ak.num(particle_level_data["part_level"], axis=1)
        total_n_particles = ak.sum(n_particles_per_event)

        # Drop values that are higher than the tracking efficiency
        random_values = rng.uniform(low=0.0, high=1.0, size=total_n_particles)
        drop_particles_mask = random_values > efficiencies
        # Since True will keep the particle, we need to invert this
        drop_particles_mask = ~drop_particles_mask

        # Unflatten so we can apply the mask to the existing particles
        drop_particles_mask = ak.unflatten(drop_particles_mask, n_particles_per_event)

        # Finally, add the particle structure at the end.
        # NOTE: We return the fast sim wrapped in the "det_level" key because the framework
        #       expects that every source has some kind of particle column name.
        # NOTE: We also return the "part_level" because it's convenient to have both
        #       together, even if it's in principle available elsewhere. We also include the event
        #       level info for the same reason. I think we're violating the separation of concerns
        #       a little bit, but it seems to work, so good enough for now.
        return ak.Array(
            {
                "det_level": particle_level_data["part_level"][drop_particles_mask],
                "part_level": particle_level_data["part_level"],
                # Include the rest of the non-particle related fields (ie. event level info)
                **{
                    k: v
                    for k, v in zip(ak.fields(particle_level_data), ak.unzip(particle_level_data))
                    if k not in ["det_level", "part_level"]
                },
            }
        )

    def gen_data(self, chunk_size: T_ChunkSize = ChunkSizeSentinel.SOURCE_DEFAULT) -> T_GenData:
        # The particle source should take care of the chunk size, so we don't worry about taking
        # extra care of it here.
        particle_level_iter = self.particle_level_source.gen_data(chunk_size=chunk_size)
        try:
            particle_level_data = next(particle_level_iter)
            while True:
                result = yield self._data(particle_level_data=particle_level_data)
                # Update the chunk size when requested
                if result is not None:
                    chunk_size = result
                particle_level_data = particle_level_iter.send(chunk_size)
        except StopIteration:
            pass


@attr.define
class PythiaSource:
    config: Path = attr.field(converter=Path)
    _default_chunk_size: Union[int, ChunkSizeSentinel] = attr.field(default=ChunkSizeSentinel.FIXED_SIZE)
    metadata: MutableMapping[str, Any] = attr.Factory(dict)

    def gen_data(self, chunk_size: T_ChunkSize = ChunkSizeSentinel.SOURCE_DEFAULT) -> T_GenData:
        raise NotImplementedError("Working on it...")


@attr.define
class ThermalModelParameters:
    mean: float
    sigma: float
    pt_exponential_scale: float = attr.field(default=0.4)


THERMAL_MODEL_SETTINGS = {
    "5020_central": ThermalModelParameters(mean=2500, sigma=500),
    "5020_semi_central": ThermalModelParameters(mean=1000, sigma=40),
}


@attr.define
class ThermalModelExponential:
    """Thermal background model from Leticia

    Assume thermal particles are massless.
    pt = x*exp(-x/pt_exponential_scale), from 0 to 400, at least 40000 sampling points
    eta = flat from -1 to 1, at least 200 sampling points
    phi = flat from -pi to pi, at least 700 sampling points

    pt exponential scale is 0.4 by default.

    The number of thermal particles is determined by a Gaussian. The parameters are:
    - central: mean = 2500, sigma = 500
    - semi-central: mean = 1000, sigma = 40

    """

    thermal_model_parameters: ThermalModelParameters
    _particle_column_prefix: str = attr.field(default="data")
    _stop_iterating_after_one_chunk: bool = attr.field(default=False)
    _default_chunk_size: Union[int, ChunkSizeSentinel] = attr.field(default=ChunkSizeSentinel.FIXED_SIZE)
    metadata: MutableMapping[str, Any] = attr.Factory(dict)

    def gen_data(self, chunk_size: T_ChunkSize = ChunkSizeSentinel.SOURCE_DEFAULT) -> T_GenData:
        # Setup
        rng = np.random.default_rng()

        while True:
            # Validation
            # We put it inside the for loop because it could change from loop to loop
            chunk_size = _validate_chunk_size(chunk_size=chunk_size, source_default_chunk_size=self._default_chunk_size)

            self.metadata["n_entries"] = chunk_size

            # Determine overall event parameters.
            # NOTE: This is effectively jagged, since the number of particles per event varies
            # NOTE: We round to integers because the number of particles must of course be an int.
            n_particles_per_event = np.rint(
                rng.normal(
                    loc=self.thermal_model_parameters.mean,
                    scale=self.thermal_model_parameters.sigma,
                    size=chunk_size,
                ),
            ).astype(np.int32)
            # To help out with this effective jaggedness, we flatten everything, and then will unflatten with awkward.
            total_n_samples = int(np.sum(n_particles_per_event))

            # Sample the distributions.
            pt = models.sample_x_exp(
                n_samples=total_n_samples,
                scale=self.thermal_model_parameters.pt_exponential_scale,
                x_min=0,
                x_max=400,
            )
            # eta = rng.uniform(low=-1, high=1, size=total_n_samples)
            # We want to match the ALICE TPC acceptance
            eta = rng.uniform(low=-0.9, high=0.9, size=total_n_samples)
            phi = rng.uniform(low=-np.pi, high=np.pi, size=total_n_samples)

            # Need this as an intermediary, so calculate it first
            pz = pt * np.sinh(eta)

            # Finally, add the particle structure at the end.
            # NOTE: We return it wrapped in the "data" key because the framework
            #       expects that every source has some kind of particle column name.
            _result = yield ak.Array(
                {
                    self._particle_column_prefix: ak.unflatten(
                        ak.Array(
                            {
                                "px": pt * np.cos(phi),
                                "py": pt * np.sin(phi),
                                "pz": pz,
                                "E": np.sqrt(pt**2 + pz**2),
                            }
                        ),
                        counts=n_particles_per_event,
                    )
                }
            )
            # Update the chunk size if necessary
            if _result is not None:
                chunk_size = _validate_chunk_size(
                    chunk_size=_result,
                    source_default_chunk_size=self._default_chunk_size,
                )

            # If we want to stop after one iteration, we need to do it now
            if self._stop_iterating_after_one_chunk:
                return None

    @classmethod
    def create_deferred_source(
        cls,
        thermal_model_parameters: ThermalModelParameters,
        particle_column_prefix: str = "data",
        stop_iterating_after_one_chunk: bool = False,
        default_chunk_size: T_ChunkSize = ChunkSizeSentinel.FIXED_SIZE,
    ) -> DelayedSource:
        """Create a FileSource with a closure such that all arguments are set except for the filename.

        Args:
            collision_system: The collision system of the data.
            default_chunk_size: The default chunk size to use when generating data.

        Returns:
            A Callable which takes the filename and creates the FileSource.
        """

        def wrap(*args: Any, **kwargs: Any) -> "ThermalModelExponential":
            return cls(
                thermal_model_parameters=thermal_model_parameters,
                particle_column_prefix=particle_column_prefix,
                stop_iterating_after_one_chunk=stop_iterating_after_one_chunk,
                default_chunk_size=default_chunk_size,
            )

        return wrap


def _sources_to_list(sources: Union[Source, Sequence[Source]]) -> Sequence[Source]:
    if not isinstance(sources, collections.abc.Iterable):
        return [sources]
    return sources


def _determine_request_chunk_size(_target_chunk_size: int, _accumulated_data_size: int) -> Tuple[int, bool]:
    """Adjust the requested chunk size according to the conditions of the request.

    Returns:
        requested chunk size, True if the requested chunk size went negative and was reset to the target.
            (if True, you probably want to skip requesting more data.)
    """
    # To start, we always want to update the request chunk size!
    # There are two possible cases:
    # 1. Received new target -> we want to use that for the baseline of the request.
    # 2. We've received a new request for a data, but without a new chunk size. In this case, the
    #    requested check size may be correct, but it may still be an older value (for example, if we
    #    requested less data due to accumulated data from the previous look), so we update it to
    #    the target chunk size (and then we'll adjust it below as needed). It's not always required to
    #    update it, but it doesn't hurt.
    _request_chunk_size = _target_chunk_size

    # If we have stored data and are not requesting the whole file, we should only request as much
    # as we need. That way, we minimize additional IO (eg. if we already have enough accumulated data
    # to fulfill the next target chunk size, there's no point in requesting more from the MultiSource).
    # NOTE: When we request the whole file, we shouldn't adjust the request chunk size. In practice,
    #       `_FULL_SOURCE_SIZE` is large enough that adjusting it wouldn't have a practical impact
    #       on that amount of data that we would receive. However, it cause any comparison to
    #       `_FULL_SOURCE_SIZE` to fail unexpectedly, which is undesirable.
    if _request_chunk_size != _FULL_SOURCE_SIZE and _accumulated_data_size > 0:
        _request_chunk_size -= _accumulated_data_size

    # We need to validate this condition
    # If we already have enough stored, there's no need to request any data.
    # In this case, there's no point in making a request, but we need to handle it explicitly here.
    # NOTE: Although we could use send(0), if the source is already exhausted, it would force a new
    #       source to load unnecessarily.
    _do_not_request_more_data = False
    if _request_chunk_size <= 0:
        # We want to signal to iterate with no additional data.
        _do_not_request_more_data = True
        # For this case, we need to ensure that we're prepared to request a non-zero chunk on the
        # next iteration (eg. if the source is exhausted). If we receive a new target chunk size on
        # that next iteration, this check wouldn't be necessary, but if we somehow do not, this update
        # to the request chunk size avoids getting stuck in a loop with no new data.
        _request_chunk_size = _target_chunk_size

    return _request_chunk_size, _do_not_request_more_data


@attr.define
class MultiSource:
    sources: Sequence[Source] = attr.field(converter=_sources_to_list)
    repeat: bool = attr.field(default=False)
    _default_chunk_size: Union[int, ChunkSizeSentinel] = attr.field(default=ChunkSizeSentinel.FULL_SOURCE)
    metadata: MutableMapping[str, Any] = attr.Factory(dict)

    def gen_data(self, chunk_size: T_ChunkSize = ChunkSizeSentinel.SOURCE_DEFAULT) -> T_GenData:  # noqa: C901
        # Validation
        if chunk_size is ChunkSizeSentinel.SOURCE_DEFAULT:
            chunk_size = self._default_chunk_size
        # Need to keep track of this separately because we handle the single file case separately.
        is_single_file_at_a_time = chunk_size is ChunkSizeSentinel.SINGLE_FILE
        chunk_size = _validate_chunk_size(chunk_size=chunk_size, source_default_chunk_size=self._default_chunk_size)

        if self.repeat:
            # See: https://stackoverflow.com/a/24225372/12907985
            source_iter: Iterator[Source] = itertools.chain.from_iterable(itertools.repeat(self.sources))
        else:
            source_iter = iter(self.sources)

        # Need to track of all of our progress while iterating
        # We need to hang on to any additional data that we may not use all of at the moment
        # (eg. on the current iteration)
        _accumulated_data = []
        _accumulated_data_size = 0
        # Target chunk size is the size of the chunk that we're trying to get to in this MultiSource
        _target_chunk_size = chunk_size
        # Request chunk size is the size of the next chunk to request. This doesn't necessarily need to be
        # equal to the target chunk size (eg. if we have some data left over from the previous yield).
        _request_chunk_size = chunk_size
        # This is the value that we receive back from the yield, which we need to pass on to our target
        # chunk size (and request chunk size) if we receive a new value.
        _next_chunk_size_request = None

        for i_source, source in enumerate(source_iter):
            # Helpful for debugging, but it's not general, so just keep around as a reminder
            # if source._collision_system == "PbPb":
            #     logger.warning("-----> New PbPb source!")
            logger.debug(f"New source in MultiSource: {source} ({i_source=}")
            try:
                # We're going to start working with the current source.
                # Grab an iterator and the first chunk of data from it.
                _current_data_iter = source.gen_data(chunk_size=_request_chunk_size)
                _current_data = next(_current_data_iter)
                while True:
                    # Need to determine the chunk size as an int to determine if we need more data
                    # If we're going one file at a time, the chunk size is correct by definition
                    if is_single_file_at_a_time:
                        _request_chunk_size = len(_current_data)
                    logger.debug(f"-> MultiSource source: {source}, {_target_chunk_size=}, {_request_chunk_size=}")

                    # Keep track of the size
                    # This is a bit of a lie until we actually yield the data because we may not yet have the full
                    # target chunk size, but since the user cannot interaction with the MultiSource until it yields
                    # (when it will actually have enough data to hit the target chunk size), so it's not an issue.
                    self.metadata["n_entries"] = _target_chunk_size

                    # Since bookkeeping is rather complicated depending on the various options and data + chunk sizes,
                    # we will always store the data, and then determine the right action from there after concatenation.
                    # NOTE: We test if there is any data in _current_data because we may have an empty set of arrays
                    #       due to already have enough data stored to fulfill the next request. If the _current_data
                    #       arrays are empty, we want to avoid concatenating empty arrays, so we skip them.
                    if len(_current_data) > 0:
                        _accumulated_data.append(_current_data)
                        _accumulated_data_size += len(_current_data)
                    else:
                        # logger.info("Didn't accumulate data because the current data is empty")
                        ...
                    logger.debug(f"--> {_accumulated_data_size=}, {_next_chunk_size_request=}, {_target_chunk_size=}, {_request_chunk_size=}")

                    # We need to figure out whether we have enough data or we need to accumulate more.
                    if _accumulated_data_size < _target_chunk_size:
                        # We need another iteration (on to the next source) to get more data.
                        # We request to update the chunk size to ensure that we don't request extra data
                        # First, we update the request to the target chunk size since we may not have requested
                        # the full data size.
                        # NOTE: We don't care if whether this thinks that we should request new data - we
                        #       know that our source is out of data because we didn't receive enough when we asked
                        #       for it, so we for sure need to break here.
                        _request_chunk_size, _ = _determine_request_chunk_size(
                            _target_chunk_size=_target_chunk_size,
                            _accumulated_data_size=_accumulated_data_size,
                        )
                        logger.debug("---> Not enough data left in this source - we need a new one!")
                        break
                    else:
                        # NOTE: This else statement could be removed (ie. we could just de-indent this
                        #       block) since we use a `break` in the if statement. I've left it here because
                        #       the additional indentation helps me separate concerns (ie. the code in this else
                        #       statement handles providing + cleaning up the data, while after the block handles
                        #       the next request for data).

                        # We have enough data - now we need to sort out what to do.
                        # First step is to concatenate the data, so we can operate on it more easily
                        # NOTE: In principle, we could always just use concatenate even if there's only
                        #       item in the list, but I don't know if there's a short circuit for a list
                        #       of one entry, and concatenate could potentially be quite expensive.
                        if len(_accumulated_data) > 1:
                            _data_to_yield = ak.concatenate(_accumulated_data, axis=0)
                        else:
                            _data_to_yield = _accumulated_data[0]
                        # logger.info(f"{_data_to_yield=}")

                        # Next, let's yield the data
                        logger.info(
                            f"MultiSource: Providing data from source #{i_source} with target chunk size {_target_chunk_size}."
                            f"\n{source=}\n{_accumulated_data_size=}, {_accumulated_data=},\n{_data_to_yield=}..."
                        )
                        _next_chunk_size_request = yield _data_to_yield[:_target_chunk_size]

                        # Cleanup the accumulated data. We use an explicit del to ensure that python knows that
                        # this variable isn't pointing to the data anymore.
                        del _accumulated_data
                        _accumulated_data = []
                        _accumulated_data_size = 0
                        # We need to deal with the possible leftover data
                        if len(_data_to_yield) > _target_chunk_size:
                            _accumulated_data.append(_data_to_yield[_target_chunk_size:])
                            _accumulated_data_size += (len(_data_to_yield) - _target_chunk_size)
                        # And then finally cleanup the remaining data that we yielded. Again, explicitly deleting
                        # to help out python and attempt to clean out the data earlier.
                        del _data_to_yield

                        # Helpful for debugging, but it's not general, so just keep around as a reminder
                        # if source._collision_system == "PbPb":
                        #     import IPython; IPython.embed()

                    # We're now done with providing data from the last iteration. Now we focus on properly handling
                    # the new data request and providing that requested data.
                    # First, update the chunk size if received a new one
                    # logger.info(f"=>{_next_chunk_size_request=}")
                    if _next_chunk_size_request is not None:
                        _target_chunk_size = _validate_chunk_size(
                            chunk_size=_next_chunk_size_request, source_default_chunk_size=source._default_chunk_size
                        )
                        # And don't forget to reset the request size for the next chunk
                        # (If we don't get into the if statement, then it's already None and doesn't need a reset)
                        _next_chunk_size_request = None

                    # Determine the requested chunk size, potentially adjusting it for existing accumulated data.
                    logger.debug(f"=> {source=}, {_target_chunk_size=}, {_request_chunk_size=}, {_next_chunk_size_request=}, {_accumulated_data_size=}, {_accumulated_data=}")
                    _request_chunk_size, _do_not_request_new_data = _determine_request_chunk_size(
                        _target_chunk_size=_target_chunk_size,
                        _accumulated_data_size=_accumulated_data_size,
                    )
                    logger.debug(f"==> Post updated of {_request_chunk_size=}")
                    if _do_not_request_new_data:
                        # We set the current data to an empty array to ensure that we never double count data. We
                        # shouldn't ever actually use this data, but we need something valid there. To maintain the
                        # correct structure of the awkward array, we use a zero length slice.
                        _current_data = _current_data[:0]
                        # logger.warning("==> request chunk size is negative. Setting current data to empty array and no need to request more data")
                        continue

                    # If we've gotten this far, it's time to keep going with the current source
                    logger.debug("===> About to request more data from the current source in the MultiSource")
                    _current_data = _current_data_iter.send(_request_chunk_size)
                    logger.debug("====> The source isn't exhausted yet.")

            except StopIteration:
                pass

        # We're out of data. Provide what's left if there is any
        # NOTE: In principle, we could always just use concatenate even if there's only
        #       item in the list, but I don't know if there's a short circuit for a list
        #       of one entry, and concatenate could potentially be quite expensive.
        logger.debug(f"Out of data in MultiSource. Remaining data size to provide: {_accumulated_data_size}, Last source: {source}...")
        if len(_accumulated_data) > 1:
            _next_chunk_size_request = yield ak.concatenate(_accumulated_data, axis=0)
        elif len(_accumulated_data) == 1:
            _next_chunk_size_request = yield _accumulated_data[0]
        # logger.info("Done!")
        # Done!


def _only_one_source(
    instance: "CombineSources",
    attribute: attr.Attribute[Mapping[str, int]],
    value: Mapping[str, int],
) -> None:
    if len(instance._constrained_size_source) != 1:
        raise ValueError(f"Only one constrained source allow! Provided: {instance._constrained_size_source}")


def _no_overlapping_keys(
    instance: "CombineSources",
    attribute: attr.Attribute[Mapping[str, int]],
    value: Mapping[str, int],
) -> None:
    if set(instance._constrained_size_source).intersection(set(instance._unconstrained_size_sources)):
        raise ValueError(
            f"Overlapping keys between constrained size and unconstrained size sources. Constrained size sources: {list(instance._constrained_size_source)}, unconstrained size sources: {list(instance._unconstrained_size_sources)}."
        )


def _contains_signal_and_background(
    instance: "CombineSources",
    attribute: attr.Attribute[Mapping[str, int]],
    value: Mapping[str, int],
) -> None:
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


def _has_offset_per_source(
    instance: "CombineSources",
    attribute: attr.Attribute[Mapping[str, int]],
    value: Mapping[str, int],
) -> None:
    if (set(instance._constrained_size_source) | set(instance._unconstrained_size_sources)) != set(
        instance._source_index_identifiers
    ):
        raise ValueError(
            f"Mismatch in sources and offsets. Constrained size sources: {list(instance._constrained_size_source)}, unconstrained sources: {list(instance._unconstrained_size_sources)}, offsets: {list(instance._source_index_identifiers)}"
        )


@attr.define
class CombineSources:
    """Combine multiple data sources together.

    Think: Embedding into data, embedding into thermal model, etc.

    Attributes:
        _constrained_size_source_name: Name of the constrained size source when constructing
            the combined data.
        _constrained_size_source: Source which provides the constraint on the size of
            the combined data.
        _unconstrained_size_sources: Sources which will provide data of a specified size.
            The size of these chunks is determined by the constrained size source and is
            set when retrieving the data.
        _source_index_identifiers: Map containing an integer identifier for each source.
    """

    _constrained_size_source: Mapping[str, Source] = attr.field(validator=[_only_one_source, _no_overlapping_keys])
    _unconstrained_size_sources: Mapping[str, Source] = attr.field(validator=[_no_overlapping_keys])
    _source_index_identifiers: Mapping[str, int] = attr.field(
        factory=dict,
        validator=[_contains_signal_and_background, _has_offset_per_source],
    )
    _default_chunk_size: T_ChunkSize = attr.field(default=ChunkSizeSentinel.SOURCE_DEFAULT)
    metadata: MutableMapping[str, Any] = attr.Factory(dict)

    # def data(self) -> ak.Array:
    #    return next(self.gen_data(chunk_size=ChunkSizeSentinel.FULL_SOURCE))

    def gen_data(self, chunk_size: T_ChunkSize = ChunkSizeSentinel.SOURCE_DEFAULT) -> T_GenData:
        if chunk_size is ChunkSizeSentinel.SOURCE_DEFAULT:
            chunk_size = self._default_chunk_size
        # Grab the iter from the constrained size source first
        constrained_size_source_name = next(iter(self._constrained_size_source))
        constrained_size_source_generator = self._constrained_size_source[constrained_size_source_name].gen_data(
            chunk_size=chunk_size
        )

        unconstrained_size_sources_generators: Dict[str, T_GenData] = {}

        for constrained_size_data in constrained_size_source_generator:
            determined_chunk_size = len(constrained_size_data)

            # We need data of a fixed size, so set that size for each of the other iterators.
            if not unconstrained_size_sources_generators:
                unconstrained_size_sources_generators = {
                    k: v.gen_data(chunk_size=determined_chunk_size) for k, v in self._unconstrained_size_sources.items()
                }
                # Once we've defined the generators, we need to grab the data from them.
                # NOTE: We have to handle this case separately from all future calls because we can't send
                #       a non-None value to a just-started generator. (We could also do a `send(None)` here,
                #       but that's just what `next()` does, except more concisely).
                unconstrained_size_sources_data = {k: next(v) for k, v in unconstrained_size_sources_generators.items()}
            else:
                # Using the existing generator, request the chunk size that we need.
                unconstrained_size_sources_data = {
                    k: v.send(determined_chunk_size) for k, v in unconstrained_size_sources_generators.items()
                }

            # Add metadata
            self.metadata["n_entries"] = determined_chunk_size

            # NOTE: We're safe to blindly combine these here because the class validates that there
            #       are no overlapping keys between the fixed size and chunked data.
            _result = yield ak.zip(
                {**{constrained_size_source_name: constrained_size_data}, **unconstrained_size_sources_data},
                depth_limit=1,
            )
            if _result is not None:
                # NOTE: If there was a need, we could update the constrained source.
                #       However, as of March 2022, I don't see a need.
                raise ValueError(
                    f"Cannot send value to CombineSources - it's already specified by the constrained source. Sent: {_result}"
                )
