""" Parse JETSCAPE ascii input files in chunks.

.. codeauthor:: Raymond Ehlers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Generator, MutableMapping

import attrs
import awkward as ak

from mammoth.framework import sources
from mammoth.framework.io import _jetscape_parser

logger = logging.getLogger(__name__)

# TODO: Needs to be implemented, I think. Or I can use the standard Parquet reader, but I need to implement transform output!
@attrs.frozen
class Columns:
    """
    NOTE:
        dict maps from name in the root file to the desired field name.
    """
    event_level: dict[str, str]
    event_level_optional: dict[str, str]
    particle_level: dict[str, str]

    @classmethod
    def create(cls, collision_system: str) -> Columns:
        # First, event level properties
        # NOTE: Only some of these are meaningful in pp, but we usually produce them anyway, so we don't overthink it.
        event_level_columns = {
            "event_plane_angle": "event_plane_angle",
            "event_ID": "event_ID",
            "cross_section": "cross_section",
            "cross_section_error": "cross_section_error",
        }
        event_level_columns_optional = {
            "event_weight": "event_weight",
        }
        # Next, particle level columns
        particle_columns = {
            "px": "px",
            "py": "py",
            "pz": "pz",
            "E": "E",
            "particle_ID": "particle_ID",
            "status": "identifier",
        }
        return cls(
            event_level=event_level_columns,
            event_level_optional=event_level_columns_optional,
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
            # TODO: Implement JETSCAPE reader...
            source: sources.Source = sources.UprootSource(
                filename=self._filename,
                tree_name="AliAnalysisTaskTrackSkim_*_tree",
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

        def wrap(filename: Path) -> FileSource:
            return cls(
                filename=filename,
                collision_system=collision_system,
                default_chunk_size=default_chunk_size,
            )

        return wrap


def _transform_output(
    gen_data: Generator[ak.Array, sources.T_ChunkSize | None, None],
    collision_system: str,
) -> Generator[ak.Array, sources.T_ChunkSize | None, None]:
    ...

