""" Parse JETSCAPE ascii input files in chunks.

.. codeauthor:: Raymond Ehlers
"""

from __future__ import annotations

import logging
from collections.abc import Generator, MutableMapping
from pathlib import Path
from typing import Any

import attrs
import awkward as ak

from mammoth.framework import sources
from mammoth.framework.io import _jetscape_parser

logger = logging.getLogger(__name__)

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
    def create(cls) -> Columns:
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
    _default_chunk_size: sources.T_ChunkSize = attrs.field(default=int(50_000))
    metadata: MutableMapping[str, Any] = attrs.Factory(dict)

    def gen_data(
        self, chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.SOURCE_DEFAULT
    ) -> Generator[ak.Array, sources.T_ChunkSize | None, None]:
        """A iterator over a fixed size of data from the source.

        Returns:
            Iterable containing chunk size data in an awkward array.
        """
        if "parquet" not in self._filename.suffix:
            chunk_size = sources.validate_chunk_size(chunk_size=chunk_size, source_default_chunk_size=self._default_chunk_size)
            assert isinstance(chunk_size, int)
            parser_source = _jetscape_parser.read(
                filename=self._filename,
                events_per_chunk=chunk_size,
                parser=self.metadata.get("parser", "pandas"),
            )
            return _transform_output(
                gen_data=parser_source,
                source_default_chunk_size=self._default_chunk_size
            )
        else:  # noqa: RET505
            source = sources.ParquetSource(
                filename=self._filename,
            )
            if self.metadata.get("legacy_skim", False):
                # Legacy skims need additional transformation to confirm to expected source outputs.
                # NOTE: The renaming of the overall "particles" field to something else could be handled
                #       via the `rename_prefix` argument in load_data, but it's easier to do it here since
                #       we may also need to rename particle level fields.
                return _transform_output(
                    gen_data=source.gen_data(chunk_size=chunk_size),
                    source_default_chunk_size=self._default_chunk_size
                )
            return source.gen_data(chunk_size=chunk_size)


def _transform_output(
    gen_data: Generator[ak.Array, int | None, None],
    source_default_chunk_size: sources.T_ChunkSize,
) -> Generator[ak.Array, sources.T_ChunkSize | None, None]:
    _columns = Columns.create()
    event_columns = _columns.event_level

    try:
        data = next(gen_data)
        event_columns.update({
            k: v for k, v in _columns.event_level_optional.items() if k in ak.fields(data)
        })
        while True:
            # Reduce to the minimum required data.
            data = _jetscape_parser.full_events_to_only_necessary_columns_E_px_py_pz(arrays=data)
            _result = yield ak.Array({
                "part_level": ak.zip(
                    dict(
                        zip(
                            list(_columns.particle_level.values()),
                            ak.unzip(data["particles"][list(_columns.particle_level)]),
                        )
                    )
                ),
                **dict(
                    zip(
                        list(event_columns.values()),
                        ak.unzip(data[list(event_columns)]),
                    )
                ),
            })

            # Update for next step
            # Update the chunk size as needed.
            if _result is not None:
                _result = sources.validate_chunk_size(
                    chunk_size=_result, source_default_chunk_size=source_default_chunk_size,
                )
            # And then grab the next set of data
            data = gen_data.send(_result)

    except StopIteration:
        pass
