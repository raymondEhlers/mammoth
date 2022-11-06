
from pathlib import Path
from typing import Any, Optional, Sequence

import pytest  # noqa: F401

from mammoth.framework import sources

_here = Path(__file__).parent
_track_skim_base_path = _here / "track_skim_validation"

def test_uproot_source() -> None:
    uproot_source = sources.UprootSource(
        # Take as an arbitrary example file
        filename=_track_skim_base_path / "reference" / "AnalysisResults__pythia__jet_R020.root",
        tree_name="AliAnalysisTaskTrackSkim_*_tree",
    )

    uproot_source.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE)
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
            # Take as an arbitrary example file
            filename=_track_skim_base_path / "reference" / "AnalysisResults__pythia__jet_R020.root",
            # Apparently the wild card doesn't work here because we need to grab the number of entries, =
            # so we just specify the name directly.
            tree_name="AliAnalysisTaskTrackSkim_pythia_tree",
            chunk_size=chunk_size,
        ),
    )

    PbPb_source = sources.MultiSource(
        sources=sources.UprootSource(
            filename=_track_skim_base_path / "reference" / "AnalysisResults__embed_pythia-PbPb__jet_R020.root",
            tree_name="AliAnalysisTaskTrackSkim_*_tree",
        ),
        repeat=True
    )

    # Now, just zip them together, effectively.
    combined_source = sources.CombineSources(
        constrained_size_source={"background": PbPb_source},
        unconstrained_size_sources={"signal": pythia_source},
        source_index_identifiers={"signal": 0, "background": 100_000},
    )

    combined_source.gen_data(chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_size, yielded_data_sizes", [(2000, None), (1000, None)])
def test_chunk_generation_from_existing_data(caplog: Any, chunk_size: int, yielded_data_sizes: Optional[Sequence[int]]) -> None:
    """Test chunk size generation when using an existing data input

    Usually, this would be via an uproot source. Here, I'm using the track skim for some extra convenience,
    since we should already have those files available.
    """
    from mammoth.framework.io import track_skim
    pythia_source = track_skim.FileSource(
        filename=_track_skim_base_path / "reference" / "AnalysisResults__pythia__jet_R020.root",
        collision_system="pythia"
    )
    # We need the full size to figure out the expect values.
    # NOTE: This is inefficient, but it's not the end of the world. We could always set it manually if becomes a problem
    # NOTE: It's 6088
    full_file_size = len(next(pythia_source.gen_data()))

    if yielded_data_sizes is None:
        yielded_data_sizes = list(range(0, full_file_size, chunk_size))
        if yielded_data_sizes[-1] != full_file_size:
            yielded_data_sizes.append(full_file_size)

    gen = pythia_source.gen_data(chunk_size=chunk_size)

    for i, data in enumerate(gen, start=1):
        assert len(data) == (yielded_data_sizes[i] - yielded_data_sizes[i - 1])

    # For the last iteration, we want to check whether it's matching the chunk size as appropriate
    if full_file_size % chunk_size == 0:
        assert len(data) == chunk_size
    else:
        assert len(data) < chunk_size

def test_multi_source_chunk_sizes() -> None:
    ...