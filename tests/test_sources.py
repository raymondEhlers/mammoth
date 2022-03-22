
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
    pythia_source = sources.MultiSource(
        sources=sources.PythiaSource(
            config="test.cmnd",
            #seed=...,
        ),
    )
    # Background
    thermal_source = sources.ThermalModelExponential(
        thermal_model_parameters=sources.THERMAL_MODEL_SETTINGS["5020_central"],
    )

    # Now, just zip them together, effectively.
    combined_source = sources.CombineSources(
        constrained_size_source={"signal": pythia_source},
        unconstrained_size_sources={"background": thermal_source},
        source_index_identifiers={"signal": 0, "background": 100_000},
    )

    combined_source.gen_data(chunk_size=chunk_size)


def test_full_embedding() -> None:
    chunk_size = 500
    pythia_source = sources.MultiSource(
        sources=sources.define_multiple_sources_from_single_root_file(
            filename=Path("."),
            tree_name="tree",
            chunk_size=chunk_size,
        ),
    )

    PbPb_source = sources.MultiSource(
        sources=sources.UprootSource(
            filename=Path("."),
            tree_name="tree",
        ),
        repeat=True
    )

    # Now, just zip them together, effectively.
    combined_source = sources.CombineSources(
        sources={"signal": pythia_source, "background": PbPb_source},
        source_index_identifiers={"signal": 0, "background": 100_000},
    )

    combined_source.gen_data(chunk_size=chunk_size)
