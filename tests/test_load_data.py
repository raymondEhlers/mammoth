
import logging
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest

logger = logging.getLogger(__name__)

_here = Path(__file__).parent
_track_skim_base_path = _here / "track_skim_validation"


@pytest.mark.parametrize("chunk_size", [2000])
@pytest.mark.parametrize("background_is_constrained_source", [False, True])
@pytest.mark.parametrize("use_additional_files", [False, True])
def test_embedding_load_data_source_fixed_size_chunks(caplog: Any, chunk_size: int, background_is_constrained_source: bool, use_additional_files: bool) -> None:  # noqa: ARG001
    """Test the MultiSource with fixed size chunks.

    Usually, this would be via an uproot source. Here, I'm using the track skim for some extra convenience,
    since we should already have those files available.
    """
    from mammoth.framework import load_data
    from mammoth.framework.io import track_skim

    _n_repeat_pythia = 1
    _n_repeat_background = 1
    if use_additional_files:
        if background_is_constrained_source:
            _n_repeat_background = 3
        else:
            _n_repeat_pythia = 3

    _, iter_arrays = load_data.embedding(
        signal_input=[_track_skim_base_path / "reference" / "AnalysisResults__pythia__jet_R020.root"] * _n_repeat_pythia,
        signal_source=partial(track_skim.FileSource, collision_system="pythia"),
        background_input=[_track_skim_base_path / "reference" / "AnalysisResults__PbPb__jet_R020.root"] * _n_repeat_background,
        background_source=partial(track_skim.FileSource, collision_system="PbPb"),
        background_is_constrained_source=background_is_constrained_source,
        chunk_size=chunk_size,
    )

    # We need the full size to figure out the expect values.
    if background_is_constrained_source:  # noqa: SIM108
        _constrained_collision_system = "PbPb"
    else:
        _constrained_collision_system = "pythia"
    full_size_source = track_skim.FileSource(
        filename=_track_skim_base_path / "reference" / f"AnalysisResults__{_constrained_collision_system}__jet_R020.root",
        collision_system=_constrained_collision_system
    )
    # NOTE: This is inefficient, but it's not the end of the world. We could always set it manually if becomes a problem
    full_file_size = len(next(full_size_source.gen_data())) * (_n_repeat_background if background_is_constrained_source else _n_repeat_pythia)

    # Determine the expected chunk sizes
    yielded_data_sizes = [chunk_size for _ in range(int(np.floor(full_file_size / chunk_size)))]
    yielded_data_sizes.append(full_file_size % chunk_size)

    for _, (data, expected_size) in enumerate(zip(iter_arrays, yielded_data_sizes)):
        assert len(data) == expected_size

    # For the last iteration, we want to check whether it's matching the chunk size as appropriate
    if full_file_size % chunk_size == 0:
        assert len(data) == chunk_size
    else:
        assert len(data) < chunk_size
