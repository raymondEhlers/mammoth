"""Convert track skim to parquet, making it easier to use.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any, Generator, List, Optional, MutableMapping

import attrs
import awkward as ak

from mammoth.framework import sources

logger = logging.getLogger(__name__)


@attrs.frozen
class Columns:
    event_level: List[str]
    particle_level: List[str]

    @classmethod
    def create(cls, collision_system: str) -> "Columns":
        # First, event level properties
        event_level_columns = [
            "run_number",
            "trigger_bit_INT7",
        ]
        if collision_system == "PbPb":
            event_level_columns += [
                "centrality",
                "event_plane_V0M",
                "trigger_bit_central",
                "trigger_bit_semi_central",
            ]
        # Next, particle level columns
        _base_particle_columns = ["pt", "eta", "phi"]
        _MC_particle_columns = [
            "particle_ID",
            "label",
        ]
        particle_columns = [f"particle_data_{c}" for c in _base_particle_columns]
        # Pick up the extra columns in the case of pythia
        if collision_system == "pythia":
            particle_columns += [f"particle_data_{c}" for c in _MC_particle_columns]
            # We skip particle_ID for the detector level
            particle_columns.pop(particle_columns.index("particle_data_particle_ID"))
            # And then do the same for particle_gen
            particle_columns += [f"particle_gen_{c}" for c in _base_particle_columns]
            particle_columns += [f"particle_gen_{c}" for c in _MC_particle_columns]

        return cls(
            event_level=event_level_columns,
            particle_level=particle_columns,
        )


@attrs.define
class FileSource:
    _filename: Path = attrs.field(converter=Path)
    _collision_system: str = attrs.field()
    _default_chunk_size: sources.T_ChunkSize = attrs.field(default=sources.ChunkSizeSentinel.FULL_SOURCE)
    metadata: MutableMapping[str, Any] = attrs.Factory(dict)

    def gen_data(
        self, chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.SOURCE_DEFAULT
    ) -> Generator[ak.Array, Optional[sources.T_ChunkSize], None]:
        """A iterator over a fixed size of data from the source.

        Returns:
            Iterable containing chunk size data in an awkward array.
        """
        if "parquet" not in self._filename.suffix:
            columns = Columns.create(collision_system=self._collision_system)
            source: sources.Source = sources.UprootSource(
                filename=self._filename,
                tree_name="AliAnalysisTaskTrackSkim_*_tree",
                columns=columns.event_level + columns.particle_level,
            )
            return _transform_output(
                gen_data=source.gen_data(chunk_size=chunk_size),
                collision_system=self._collision_system,
            )
        else:
            source = sources.ParquetSource(
                filename=self._filename,
            )
            return source.gen_data(chunk_size=chunk_size)

    @classmethod
    def create_deferred_source(
        cls,
        collision_system: str,
        default_chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.FULL_SOURCE,
    ) -> sources.SourceFromFilename:
        """Create a FileSource with a closure such that all arguments are set except for the filename.

        Args:
            collision_system: The collision system of the data.
            default_chunk_size: The default chunk size to use when generating data.

        Returns:
            A Callable which takes the filename and creates the FileSource.
        """

        def wrap(filename: Path) -> "FileSource":
            return cls(
                filename=filename,
                collision_system=collision_system,
                default_chunk_size=default_chunk_size,
            )

        return wrap


def _transform_output(
    gen_data: Generator[ak.Array, Optional[sources.T_ChunkSize], None],
    collision_system: str,
) -> Generator[ak.Array, Optional[sources.T_ChunkSize], None]:
    _columns = Columns.create(collision_system=collision_system)

    # NOTE: If there are no accepted tracks, we don't bother storing the event.
    #       However, we attempt to preclude this at the AnalysisTask level by not filling events
    #       where there are no accepted tracks in the first collection.

    particle_data_columns = [c for c in _columns.particle_level if "particle_data" in c]
    try:
        data = next(gen_data)
        while True:
            if collision_system == "pythia":
                # NOTE: The return values are formatted in this manner to avoid unnecessary copies of the data.
                particle_gen_columns = [c for c in _columns.particle_level if "particle_gen" in c]
                _result = yield ak.Array(
                    {
                        "det_level": ak.zip(
                            dict(
                                zip(
                                    [c.replace("particle_data_", "") for c in list(particle_data_columns)],
                                    ak.unzip(data[particle_data_columns]),
                                )
                            )
                        ),
                        "part_level": ak.zip(
                            dict(
                                zip(
                                    [c.replace("particle_gen_", "") for c in list(particle_gen_columns)],
                                    ak.unzip(data[particle_gen_columns]),
                                )
                            )
                        ),
                        **dict(
                            zip(
                                _columns.event_level,
                                ak.unzip(data[_columns.event_level]),
                            )
                        ),
                    },
                )
            else:
                # NOTE: The return values are formatted in this manner to avoid unnecessary copies of the data.
                _result = yield ak.Array(
                    {
                        "data": ak.zip(
                            dict(
                                zip(
                                    [c.replace("particle_data_", "") for c in list(particle_data_columns)],
                                    ak.unzip(data[particle_data_columns]),
                                )
                            )
                        ),
                        **dict(
                            zip(
                                _columns.event_level,
                                ak.unzip(data[_columns.event_level]),
                            )
                        ),
                    },
                )
            # Update for next step
            data = gen_data.send(_result)
    except StopIteration:
        pass


def write_to_parquet(arrays: ak.Array, filename: Path, collision_system: str) -> bool:
    """Write the jagged track skim arrays to parquet.

    In this form, they should be ready to analyze.
    """
    # Ensure the directory exists
    filename.parent.mkdir(parents=True, exist_ok=True)

    # NOTE: If there are issues about reading the files due to arrays being too short, check that
    #       there are no empty events. Empty events apparently cause issues for byte stream split
    #       encoding: https://issues.apache.org/jira/browse/ARROW-13024
    #       Unfortunately, this won't become clear until reading is attempted.
    ak.to_parquet(
        array=arrays,
        destination=str(filename),
        compression="zstd",
        # Optimize the compression via improved encodings for floats and strings.
        # Conveniently, awkward 2.x will now select the right columns for each if simply set to `True`
        # Optimize for columns with anything other than floats
        parquet_dictionary_encoding=True,
        # Optimize for columns with floats
        parquet_byte_stream_split=True
    )

    return True


if __name__ == "__main__":
    from mammoth import helpers

    helpers.setup_logging(level=logging.INFO)

    # for collision_system in ["pythia"]:
    for collision_system in ["pp", "pythia", "PbPb"]:
        logger.info(f"Converting collision system {collision_system}")
        source = FileSource(
            filename=Path(f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults.root"),
            collision_system=collision_system,
        )
        arrays = next(source.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE))

        write_to_parquet(
            arrays=arrays,
            filename=Path(
                f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults_track_skim.parquet"
            ),
            collision_system=collision_system,
        )

    # For embedding, we need to go to the separate signal and background files themselves.
    for collision_system in ["pythia", "PbPb"]:
        logger.info(f"Converting collision system {collision_system} for embedding")
        source = FileSource(
            filename=Path(
                f"/software/rehlers/dev/mammoth/projects/framework/embedPythia/track_skim/{collision_system}/AnalysisResults.root"
            ),
            collision_system=collision_system,
        )
        arrays = next(source.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE))

        write_to_parquet(
            arrays=arrays,
            filename=Path(
                f"/software/rehlers/dev/mammoth/projects/framework/embedPythia/AnalysisResults_{collision_system}_track_skim.parquet"
            ),
            collision_system=collision_system,
        )
