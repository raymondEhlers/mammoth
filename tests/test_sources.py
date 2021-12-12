
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
    # Signal
    pythia_source = sources.ChunkSource(
        chunk_size=chunk_size,
        sources=sources.PythiaSource(
            config="test.cmnd",
            #seed=...,
            chunk_size=chunk_size,
        ),
    )
    # Background
    thermal_source = sources.ThermalModelExponential(
        # Chunk sizee will be set when combining the sources.
        chunk_size=-1,
        thermal_model_parameters=sources.THERMAL_MODEL_SETTINGS["central"],
    )

    # Now, just zip them together, effectively.
    combined_source = sources.MultipleSources(
        fixed_size_sources={"signal": pythia_source},
        chunked_sources={"background": thermal_source},
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
