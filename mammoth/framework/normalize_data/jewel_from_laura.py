"""Convert JEWEL inputs from Laura + Raghav into expected awkward array format.

This is particularly focused on JEWEL w/ recoils simulations, which as of 5 December 2021,
is stored in: `/alf/data/laura/pc069/alice/thermal_ML/jewel_stuff`

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any, Dict, Generator, MutableMapping, Optional

import attrs
import awkward as ak

from mammoth.framework import sources, utils

logger = logging.getLogger(__name__)


@attrs.frozen
class Columns:
    event_level: Dict[str, str]
    particle_level: Dict[str, str]

    @classmethod
    def create(cls) -> "Columns":
        # For JEWEL, these were the only meaningful columns
        event_level_columns = {
            "mcweight": "event_weight",
        }
        particle_columns = {
            "partpT": "pt",
            # This maps the rapidity to pseudorapidity. This is wrong, strictly speaking,
            # but good enough for these purposes (especially because mapping to the rapidity in
            # the vector constructors is apparently not so trivial...)
            "party": "eta",
            "partphi": "phi",
            "partm": "m",
            "partc": "charge",
            "parts": "scattering_center"
        }

        return cls(
            event_level=event_level_columns,
            particle_level=particle_columns,
        )


@attrs.define
class FileSource:
    _filename: Path = attrs.field(converter=Path)
    _entry_range: utils.Range = attrs.field(converter=sources.convert_sequence_to_range, default=utils.Range(None, None))
    _default_chunk_size: sources.T_ChunkSize = attrs.field(default=sources.ChunkSizeSentinel.FULL_SOURCE)
    metadata: MutableMapping[str, Any] = attrs.Factory(dict)

    def gen_data(self, chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.SOURCE_DEFAULT) -> Generator[ak.Array, Optional[sources.T_ChunkSize], None]:
        """A iterator over a fixed size of data from the source.

        Returns:
            Iterable containing chunk size data in an awkward array.
        """
        columns = Columns.create()
        source = sources.UprootSource(
            filename=self._filename,
            tree_name="ParticleTree",
            entry_range=self._entry_range,
            columns=list(columns.event_level) + list(columns.particle_level),
        )
        return _transform_output(
            gen_data=source.gen_data(chunk_size=chunk_size),
        )


def _transform_output(
    gen_data: Generator[ak.Array, Optional[sources.T_ChunkSize], None],
) -> Generator[ak.Array, Optional[sources.T_ChunkSize], None]:
    _columns = Columns.create()

    try:
        data = next(gen_data)
        while True:
            # NOTE: The return values are formatted in this manner to avoid unnecessary copies of the data.
            _result = yield ak.Array({
                "part_level": ak.zip(
                    dict(
                        zip(
                            list(_columns.particle_level.values()),
                            ak.unzip(data[_columns.particle_level]),
                        )
                    )
                ),
                **dict(
                    zip(
                        list(_columns.event_level.values()),
                        ak.unzip(data[_columns.event_level]),
                    )
                ),
            })

            # Update for next step
            data = gen_data.send(_result)
    except StopIteration:
        pass
