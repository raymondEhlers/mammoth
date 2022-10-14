
from pathlib import Path

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
