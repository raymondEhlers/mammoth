
import itertools
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest  # noqa: F401

from mammoth.framework import sources

logger = logging.getLogger(__name__)

_here = Path(__file__).parent
_track_skim_base_path = _here / "track_skim_validation"

def test_uproot_source() -> None:
    uproot_source = sources.UprootSource(
        # Take as an arbitrary example file
        filename=_track_skim_base_path / "reference" / "AnalysisResults__pythia__jet_R020.root",
        tree_name="AliAnalysisTaskTrackSkim_*_tree",
    )

    arrays = uproot_source.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE)

    # NOTE: We only need to iterate once because we explicitly requested the entire chunk size.
    assert len(next(arrays)) > 0


def test_manual_thermal_model_embedding() -> None:
    chunk_size = 500
    # Signal
    pythia_source = sources.UprootSource(
        # Take as an arbitrary example file
        filename=_track_skim_base_path / "reference" / "AnalysisResults__pythia__jet_R020.root",
        tree_name="AliAnalysisTaskTrackSkim_*_tree",
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

    arrays_iter = combined_source.gen_data(chunk_size=chunk_size)

    # By iterating through the entire combined source, we can check that the chunk sizes are propagated correctly.
    for arrays in arrays_iter:
        assert len(arrays) > 0


def test_manual_data_embedding() -> None:
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

    arrays_iter = combined_source.gen_data(chunk_size=chunk_size)

    # By iterating through the entire combined source, we can check that the chunk sizes are propagated correctly.
    for arrays in arrays_iter:
        assert len(arrays) > 0


@pytest.mark.parametrize("chunk_size", [2000, 1000])
def test_chunk_generation_from_existing_data_with_fixed_chunk_size(
    caplog: Any, chunk_size: int
) -> None:
    """Test chunk size generation when using an existing data input

    Usually, this would be via an uproot source. Here, I'm using the track skim for some extra convenience,
    since we should already have those files available.
    """
    # Setup
    # Logging
    caplog.set_level(logging.DEBUG, logger="mammoth.framework.sources")
    # Input source
    from mammoth.framework.io import track_skim
    # NOTE: It's important that we use the root file here, as this will implicitly use an UprootSource with
    #       chunk generation from existing data.
    pythia_source = track_skim.FileSource(
        filename=_track_skim_base_path / "reference" / "AnalysisResults__pythia__jet_R020.root",
        collision_system="pythia"
    )
    # We need the full size to figure out the expect values.
    # NOTE: This is inefficient, but it's not the end of the world. We could always set it manually if becomes a problem
    # NOTE: For reference, it's 11358
    full_file_size = len(next(pythia_source.gen_data()))

    # Determine the expected chunk sizes
    yielded_data_sizes = [chunk_size for _ in range(int(np.floor(full_file_size / chunk_size)))]
    yielded_data_sizes.append(full_file_size % chunk_size)

    # Finally, actually access the data and check the chunk sizes.
    gen = pythia_source.gen_data(chunk_size=chunk_size)

    for i, (data, expected_size) in enumerate(zip(gen, yielded_data_sizes)):
        assert len(data) == expected_size

    # For the last iteration, we want to check whether it's matching the chunk size as appropriate
    if full_file_size % chunk_size == 0:
        assert len(data) == chunk_size
    else:
        assert len(data) < chunk_size


@pytest.mark.parametrize("chunk_size",
                         [
                             [2000] * 10,
                             [2000, 1000, 2500, 303, 10000],
                             [11000, 358, 200],
                         ])
def test_chunk_generation_from_existing_data_with_variable_chunk_size(
    caplog: Any, chunk_size: Sequence[int]
) -> None:
    """Test chunk size generation when using an existing data input for variable chunk sizes.

    Usually, this would be via an uproot source. Here, I'm using the track skim for some extra convenience,
    since we should already have those files available.
    """
    # Setup
    # Logging
    caplog.set_level(logging.DEBUG, logger="mammoth.framework.sources")
    # Input source
    from mammoth.framework.io import track_skim
    # NOTE: It's important that we use the root file here, as this will implicitly use an UprootSource with
    #       chunk generation from existing data.
    pythia_source = track_skim.FileSource(
        filename=_track_skim_base_path / "reference" / "AnalysisResults__pythia__jet_R020.root",
        collision_system="pythia"
    )
    # We need the full size to figure out the expect values.
    # NOTE: This is inefficient, but it's not the end of the world. We could always set it manually if becomes a problem
    # NOTE: For reference, it's 11358
    full_file_size = len(next(pythia_source.gen_data()))

    gen = pythia_source.gen_data(chunk_size=chunk_size[0])

    total_number_of_events = 0
    expecting_stop_iteration = False
    stopped_iteration = False
    try:
        # NOTE: We iterate over the chunk sizes because we can't directly iterate over a generator
        #       that we send values to (in order to change the chunk size).
        for i, current_chunk_size in enumerate(chunk_size):
            # Need to send None initially, and then we can update chunk sizes as we iterate
            data = gen.send(current_chunk_size if i > 0 else None)
            # If we've found a case where we don't have enough data, it means that
            assert expecting_stop_iteration is False

            total_number_of_events += len(data)
            # If the data size doesn't match the current chunk size, it means that we're out of data.
            if len(data) != current_chunk_size:
                assert len(data) < current_chunk_size
                expecting_stop_iteration = True
            else:
                assert len(data) == current_chunk_size
    except StopIteration:
        ...
        stopped_iteration = True

    # Useful to keep track of when debugging
    logger.info(f"{stopped_iteration}")

    # Ensure that we actually got all of the data.
    # (This is contingent on defining enough chunks in the parametrization, so it requires a bit of care).
    assert total_number_of_events == full_file_size


@pytest.mark.parametrize("number_of_repeated_files", [1, 3])
@pytest.mark.parametrize("chunk_size", [2000, 1000])
def test_multi_source_source_fixed_size_chunks(caplog: Any, chunk_size: int, number_of_repeated_files: int) -> None:
    """Test the MultiSource with fixed size chunks.

    Usually, this would be via an uproot source. Here, I'm using the track skim for some extra convenience,
    since we should already have those files available.
    """
    # Setup
    # Logging
    caplog.set_level(logging.DEBUG, logger="mammoth.framework.sources")
    # Input source
    from mammoth.framework.io import track_skim
    pythia_source = sources.MultiSource(
        sources=[
            track_skim.FileSource(
                filename=_track_skim_base_path / "reference" / "AnalysisResults__pythia__jet_R020.root",
                collision_system="pythia",
            )
            for _ in range(number_of_repeated_files)
        ]
    )

    # We need to know the full size to figure out the expected values.
    pythia_source_ref = track_skim.FileSource(
        filename=_track_skim_base_path / "reference" / "AnalysisResults__pythia__jet_R020.root",
        collision_system="pythia"
    )
    # NOTE: This is inefficient, but it's not the end of the world. We could always set it manually if becomes a problem
    # NOTE: For reference, it's 11358
    full_file_size = len(next(pythia_source_ref.gen_data())) * number_of_repeated_files

    # Determine the expected chunk sizes
    yielded_data_sizes = [chunk_size for _ in range(int(np.floor(full_file_size / chunk_size)))]
    yielded_data_sizes.append(full_file_size % chunk_size)

    # Finally, actually access the data and check the chunk sizes.
    gen = pythia_source.gen_data(chunk_size=chunk_size)

    total_data_size = 0
    for i, (data, expected_chunk_size) in enumerate(zip(gen, yielded_data_sizes)):
        assert len(data) == expected_chunk_size
        total_data_size += len(data)

    # For the last iteration, we want to check whether it's matching the chunk size as appropriate
    if full_file_size % chunk_size == 0:
        assert len(data) == chunk_size
    else:
        assert len(data) < chunk_size


@pytest.mark.parametrize("number_of_repeated_files", [1, 3])
@pytest.mark.parametrize("chunk_size",
                         [
                             [10000] * 10,
                             [2000, 1000, 2500, 303, 10000],
                             [11000, 358, 200],
                         ])
def test_multi_source_source_variable_size_chunks(
    caplog: Any, chunk_size: Sequence[int], number_of_repeated_files: int
) -> None:
    """Test chunk size generation when using an existing data input for variable chunk sizes.

    Usually, this would be via an uproot source. Here, I'm using the track skim for some extra convenience,
    since we should already have those files available.
    """
    # Setup
    # Logging
    caplog.set_level(logging.DEBUG, logger="mammoth.framework.sources")
    # Input source
    from mammoth.framework.io import track_skim
    pythia_source = sources.MultiSource(
        sources=[
            track_skim.FileSource(
                filename=_track_skim_base_path / "reference" / "AnalysisResults__pythia__jet_R020.root",
                collision_system="pythia",
            )
            for _ in range(number_of_repeated_files)
        ]
    )

    # NOTE: We don't want our chunks to run out, so we'll repeat them if they're not enough
    chunk_iter = itertools.chain.from_iterable(itertools.repeat(chunk_size))

    # We need the full size to figure out the expect values.
    # NOTE: This is inefficient, but it's not the end of the world. We could always set it manually if becomes a problem
    # NOTE: For reference, it's 11358
    pythia_source_ref = track_skim.FileSource(
        filename=_track_skim_base_path / "reference" / "AnalysisResults__pythia__jet_R020.root",
        collision_system="pythia"
    )
    full_file_size = len(next(pythia_source_ref.gen_data())) * number_of_repeated_files

    # Finally, actually access the data and check the chunk sizes.
    gen = pythia_source.gen_data(chunk_size=chunk_size[0])

    total_number_of_events = 0
    expecting_stop_iteration = False
    # Just debugging information
    stopped_iteration = False
    finished_chunk_iterator = False
    try:
        # NOTE: We iterate over the chunk sizes because we can't directly iterate over a generator
        #       that we send values to (in order to change the chunk size).
        for i, current_chunk_size in enumerate(chunk_iter):
            # Need to send None initially, and then we can update chunk sizes as we iterate
            data = gen.send(current_chunk_size if i > 0 else None)
            # If we've found a case where we don't have enough data, it means that
            assert expecting_stop_iteration is False

            total_number_of_events += len(data)
            # If the data size doesn't match the current chunk size, it means that we're out of data.
            if len(data) != current_chunk_size:
                assert len(data) < current_chunk_size
                expecting_stop_iteration = True
            else:
                assert len(data) == current_chunk_size
        finished_chunk_iterator = True
    except StopIteration:
        ...
        stopped_iteration = True

    # Useful to keep track of when debugging
    logger.info(f"{stopped_iteration=}, {finished_chunk_iterator=}")

    # Ensure that we actually got all of the data.
    # (This is contingent on defining enough chunks in the parametrization, so it requires a bit of care).
    assert total_number_of_events == full_file_size
