"""Convert track skim to parquet, making it easier to use.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, MutableMapping

import attr
import awkward as ak

from mammoth.framework import sources

logger = logging.getLogger(__name__)


@attr.frozen
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
        _base_particle_columns = [
            "pt", "eta", "phi"
        ]
        _MC_particle_columns = [
            "particle_ID", "label",
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


@attr.define
class FileSource:
    _filename: Path = attr.field(converter=Path)
    _collision_system: str
    _default_chunk_size: sources.T_ChunkSize = attr.field(default=sources.ChunkSizeSentinel.FULL_SOURCE)
    metadata: MutableMapping[str, Any] = attr.Factory(dict)

    def gen_data(self, chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.SOURCE_DEFAULT) -> Generator[ak.Array, Optional[sources.T_ChunkSize], None]:
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
    def create_source_from_filename(cls,
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
    # Determine the types for improved compression when writing
    # Ideally, we would determine these dynamically, but it's unclear how to do this at
    # the moment with awkward, so for now we specify them by hand...
    # float_types = [np.float32, np.float64]
    # float_columns = list(self.output_dataframe.select_dtypes(include=float_types).keys())
    # other_columns = list(self.output_dataframe.select_dtypes(exclude=float_types).keys())
    # Typing info
    # For pp
    # In [1]: arrays.type
    # Out[1]: 703 * {"data": var * {"pt": float32, "eta": float32, "phi": float32}, "run_number": int32, "trigger_bit_INT7": bool}

    # Apparently, specifying use_byte_stream_split=True causes bool to try to encode with the
    # byte stream, even if we specify dictionary encoding (from parquet metadata, I guess it may
    # be because bool can't be dictionary encoded either?). There are a few possible workarounds:
    #
    # 1. Use `values_astype` to convert bool to the next smallest type -> unsigned byte. This works,
    #    but costs storage.
    # 2. Specify `use_dictionary=True` to default encode as dictionary, and then specify the byte stream
    #    split columns by hand. This also works, but since dictionary is preferred over the byte stream
    #    (according to the parquet docs), that list of byte stream split columns is basically meaningless.
    #    So this is equivalent to not using byte stream split at all, which isn't very helpful.
    # 3. Specify both the dictionary columns and byte split stream columns explicitly. This seems to work,
    #    provides good compression, and doesn't error on bool. So we use this option.

    # Columns to store as integers
    dictionary_encoded_columns = [
        "run_number",
        "trigger_bit_INT7",
    ]
    if collision_system == "pythia":
        # NOTE: Uses notation from arrow/parquet
        #       `list.item` basically gets us to an column in the list.
        #       This may be a little brittle, but let's see.
        # NOTE: Recall that we don't include `particle_ID` for det_level because it's all 0s.
        dictionary_encoded_columns += [
            "det_level.list.item.label",
            "part_level.list.item.label",
            "part_level.list.item.particle_ID",
        ]
    if collision_system == "PbPb":
        dictionary_encoded_columns += [
            "centrality",
            "event_plane_V0M",
            "trigger_bit_central",
            "trigger_bit_semi_central",
        ]

    # Columns to store as float
    first_collection_name = "data" if collision_system != "pythia" else "det_level"
    byte_stream_split_columns = [
        f"{first_collection_name}.list.item.pt",
        f"{first_collection_name}.list.item.eta",
        f"{first_collection_name}.list.item.phi",
    ]
    if collision_system == "pythia":
        byte_stream_split_columns += [
            "part_level.list.item.pt",
            "part_level.list.item.eta",
            "part_level.list.item.phi",
        ]

    # NOTE: If there are issues about reading the files due to arrays being too short, check that
    #       there are no empty events. Empty events apparently cause issues for byte stream split
    #       encoding: https://issues.apache.org/jira/browse/ARROW-13024
    #       Unfortunately, this won't become clear until reading is attempted.
    ak.to_parquet(
        array=arrays,
        where=filename,
        compression="zstd",
        # Use for anything other than floats
        use_dictionary=dictionary_encoded_columns,
        # Optimize for floats for the rest
        use_byte_stream_split=byte_stream_split_columns,
    )

    return True


if __name__ == "__main__":
    from mammoth import helpers
    helpers.setup_logging(level=logging.INFO)

    # for collision_system in ["pythia"]:
    for collision_system in ["pp", "pythia", "PbPb"]:
        logger.info(f"Converting collision system {collision_system}")
        source = FileSource(
            filename=Path(
                f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults.root"
            ),
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
