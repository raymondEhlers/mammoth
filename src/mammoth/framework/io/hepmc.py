"""HepMC reader

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from collections.abc import Generator, MutableMapping
from pathlib import Path
from typing import Any

import attrs
import awkward as ak
import numpy as np
import numpy.typing as npt
import pyhepmc

from mammoth.framework import sources

logger = logging.getLogger(__name__)


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
        with pyhepmc.open(self._filename) as f:
            return _transform_output(
                hepmc_file=f, chunk_size=chunk_size, source_default_chunk_size=self._default_chunk_size
            )


@attrs.define
class IntermediateFields:
    px: list[npt.NDArray[np.float64]] = attrs.field(factory=list)
    py: list[npt.NDArray[np.float64]] = attrs.field(factory=list)
    pz: list[npt.NDArray[np.float64]] = attrs.field(factory=list)
    E: list[npt.NDArray[np.float64]] = attrs.field(factory=list)
    status: list[npt.NDArray[np.int32]] = attrs.field(factory=list)
    particle_ID: list[npt.NDArray[np.int32]] = attrs.field(factory=list)

    def add_particles(self, particles: pyhepmc._core.ParticlesAPI) -> None:
        self.px.append(particles.px)
        self.py.append(particles.py)
        self.pz.append(particles.pz)
        self.E.append(particles.E)
        self.status.append(particles.status)
        self.particle_ID.append(particles.pid)

    def clear(self) -> None:
        self.px.clear()
        self.py.clear()
        self.pz.clear()
        self.E.clear()
        self.status.clear()
        self.particle_ID.clear()

    def send(self) -> ak.Array:
        """Format the data to send and clear for the next iteration."""
        return_value = ak.Array(
            {
                "data": ak.zip(
                    {
                        "px": ak.concatenate(self.px, axis=0),
                        "py": ak.concatenate(self.py, axis=0),
                        "pz": ak.concatenate(self.pz, axis=0),
                        "E": ak.concatenate(self.E, axis=0),
                        "status": ak.concatenate(self.status, axis=0),
                        "particle_ID": ak.concatenate(self.particle_ID, axis=0),
                    }
                ),
                # Any event level quantities would go here
            }
        )
        self.clear()
        return return_value


def _transform_output(
    hepmc_file: pyhepmc.io.HepMCFile,
    chunk_size: sources.T_ChunkSize,
    source_default_chunk_size: sources.T_ChunkSize,
) -> Generator[ak.Array, sources.T_ChunkSize | None, None]:
    reader_gen = iter(hepmc_file)
    # raise NotImplementedError("Need to implement this! Helpful for rivet, etc")
    intermediate_data = IntermediateFields()
    i_event = 0
    # NOTE: If we used a reader, we could check reader.failed(). However, when just iterating,
    #       it appears that we don't have that capability.
    try:
        while True:
            # Loop over the requested number of events, storing as we go. Since it's one
            # event at a time, there's nothing better we can do.
            # NOTE: Start at 1 since we want to match the chunk_size counting.
            for i_event, event in enumerate(reader_gen, start=1):
                # Extract properties, status, PID, etc.
                intermediate_data.add_particles(event.numpy.particles)
                if i_event == chunk_size:
                    # Okay, we've gotten to our desired chunk size.
                    # Need an ak.concatenate
                    _result = yield intermediate_data.send()

                    # Need to break to get to the next chunk (ie. outer while loop)
                    break

            # Update the chunk size as needed.
            if _result is not None:
                _result = sources.validate_chunk_size(
                    chunk_size=_result,
                    source_default_chunk_size=source_default_chunk_size,
                )
    except StopIteration:
        # Return whatever is available.
        yield intermediate_data.send()
