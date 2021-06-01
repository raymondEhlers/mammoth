
from functools import partial
from pathlib import Path

import pytest

from mammoth.framework import sources

def test_uproot_source() -> None:
    uproot_source = partial(
        sources.UprootSource,
        tree_name="tree",
    )

    ...

def test_thermal_embedding() -> None:
    chunk_size = 500
    pythia_source = sources.ChunkSource(
        chunk_size=chunk_size,
        sources=sources.PythiaSource(
            config="test.cmnd",
            #seed=...,
            chunk_size=chunk_size,
        ),
    )

    thermal_source = sources.ChunkSource(
        chunk_size=chunk_size,
        sources=sources.ThermalBackgroundExponential(
            chunk_size=chunk_size,
            n_particles_per_event_mean=2500,
            n_particles_per_event_sigma=500,
            pt_exponential_scale=0.4
        ),
        repeat=True
    )

    # Now, just zip them together, effectively.
    combined_source = sources.MultipleSources(
        sources={"signal": pythia_source, "background": thermal_source},
        source_index_identifiers={"signal": 0, "background": 100_000},
    )

    combined_source.data()

def test_full_embedding() -> None:
    chunk_size = 500
    pythia_source = sources.ChunkSource(
        chunk_size=chunk_size,
        sources=sources.chunked_uproot_source(
            filename=Path("."),
            tree_name="tree",
            chunk_size=chunk_size,
        ),
    )

    PbPb_source = sources.ChunkSource(
        chunk_size=chunk_size,
        sources=sources.UprootSource(
            filename=Path("."),
            tree_name="tree",
        ),
        repeat=True
    )

    # Now, just zip them together, effectively.
    combined_source = sources.MultipleSources(
        sources={"signal": pythia_source, "background": PbPb_source},
        source_index_identifiers={"signal": 0, "background": 100_000},
    )

    combined_source.data()
