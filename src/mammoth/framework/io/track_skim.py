"""ALICE track skim

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from collections.abc import Generator, MutableMapping
from pathlib import Path
from typing import Any

import attrs
import awkward as ak

from mammoth.framework import sources

logger = logging.getLogger(__name__)


@attrs.frozen
class Columns:
    """
    NOTE:
        dict maps from name in the root file to the desired field name.
    """

    event_level: dict[str, str]
    particle_level: dict[str, str]

    @classmethod
    def create(cls, collision_system: str) -> Columns:
        # First, event level properties
        event_level_columns = {
            "run_number": "run_number",
            "trigger_bit_INT7": "trigger_bit_INT7",
        }
        if collision_system == "PbPb":
            event_level_columns.update(
                {
                    "centrality": "centrality",
                    "event_plane_V0M": "event_plane_V0M",
                    "trigger_bit_central": "trigger_bit_central",
                    "trigger_bit_semi_central": "trigger_bit_semi_central",
                }
            )
        # Next, particle level columns
        _base_particle_columns = {
            "{prefix}_pt": "pt",
            "{prefix}_eta": "eta",
            "{prefix}_phi": "phi",
        }
        _MC_particle_columns = {
            "{prefix}_particle_ID": "particle_ID",
            "{prefix}_label": "identifier",
        }
        particle_columns = {
            column.format(prefix="particle_data"): field_name for column, field_name in _base_particle_columns.items()
        }
        # Pick up the extra columns in the case of pythia
        # NOTE: This will cause issues if we ever do PbPb_MC without a fastsim. So far (June 2024), we haven't done this,
        #       but adding this note in case it becomes an issue in the future.
        if collision_system in ["pythia", "pp_MC", "PbPb_MC"]:
            particle_columns.update(
                {
                    column.format(prefix="particle_data"): field_name
                    for column, field_name in _MC_particle_columns.items()
                }
            )
            # NOTE: We skip particle_ID for the detector level, since it's not likely to be available
            del particle_columns["particle_data_particle_ID"]
            # And then do the same for particle_gen
            particle_columns.update(
                {
                    column.format(prefix="particle_gen"): field_name
                    for column, field_name in {**_base_particle_columns, **_MC_particle_columns}.items()
                }
            )

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
    ) -> Generator[ak.Array, sources.T_ChunkSize | None, None]:
        """A iterator over a fixed size of data from the source.

        Returns:
            Iterable containing chunk size data in an awkward array.
        """
        if "parquet" not in self._filename.suffix:
            columns = Columns.create(collision_system=self._collision_system)
            source: sources.Source = sources.UprootSource(
                filename=self._filename,
                # NOTE: We add a second star at the end of the default value because later versions of the track skim
                #       add a version (e.g. v3) at the end of the tree name. So without it, we would miss those.
                #       In the case of earlier versions, the extra wildcard won't cause any issues - it will
                #       still find the existing tree.
                tree_name=self.metadata.get("tree_name", "AliAnalysisTaskTrackSkim_*_tree*"),
                columns=list(columns.event_level) + list(columns.particle_level),
            )
            return _transform_output(
                gen_data=source.gen_data(chunk_size=chunk_size),
                collision_system=self._collision_system,
            )
        else:  # noqa: RET505
            source = sources.ParquetSource(
                filename=self._filename,
            )
            return source.gen_data(chunk_size=chunk_size)


def _transform_output(
    gen_data: Generator[ak.Array, sources.T_ChunkSize | None, None],
    collision_system: str,
) -> Generator[ak.Array, sources.T_ChunkSize | None, None]:
    _columns = Columns.create(collision_system=collision_system)

    # NOTE: If there are no accepted tracks, we don't bother storing the event.
    #       However, we attempt to preclude this at the AnalysisTask level by not filling events
    #       where there are no accepted tracks in the first collection.

    particle_data_columns = {c: v for c, v in _columns.particle_level.items() if "particle_data" in c}
    try:
        data = next(gen_data)
        while True:
            # NOTE: This will cause issues if we ever do PbPb_MC without a fastsim. So far (June 2024), we haven't done this,
            #       but adding this note in case it becomes an issue in the future.
            if collision_system in ["pythia", "pp_MC", "PbPb_MC"]:
                # NOTE: The return values are formatted in this manner to avoid unnecessary copies of the data.
                particle_gen_columns = {c: v for c, v in _columns.particle_level.items() if "particle_gen" in c}
                _result = yield ak.Array(
                    {
                        "det_level": ak.zip(
                            dict(
                                zip(
                                    list(particle_data_columns.values()),
                                    ak.unzip(data[list(particle_data_columns)]),
                                    strict=True,
                                )
                            )
                        ),
                        "part_level": ak.zip(
                            dict(
                                zip(
                                    list(particle_gen_columns.values()),
                                    ak.unzip(data[list(particle_gen_columns)]),
                                    strict=True,
                                )
                            )
                        ),
                        **dict(
                            zip(
                                list(_columns.event_level.values()),
                                ak.unzip(data[list(_columns.event_level)]),
                                strict=True,
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
                                    list(particle_data_columns.values()),
                                    ak.unzip(data[list(particle_data_columns)]),
                                    strict=True,
                                )
                            )
                        ),
                        **dict(
                            zip(
                                list(_columns.event_level.values()),
                                ak.unzip(data[list(_columns.event_level)]),
                                strict=True,
                            )
                        ),
                    },
                )
            # Update for next step
            data = gen_data.send(_result)
    except StopIteration:
        pass


def write_to_parquet(arrays: ak.Array, filename: Path) -> bool:
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
        parquet_byte_stream_split=True,
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
        )
