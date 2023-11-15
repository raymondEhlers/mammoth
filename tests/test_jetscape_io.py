""" Tests for v2 parser.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest

from mammoth.framework.io import _jetscape_parser
from mammoth.framework.io import jetscape


logger = logging.getLogger(__name__)

here = Path(__file__).parent
# Remap column names that have been renamed.
_v1_particle_property_columns = [
    "particle_ID",
    "status",
    "E",
    "px",
    "py",
    "pz",
    "eta",
    "phi",
]
_rename_columns = {
    "hydro_event_id": "event_ID",
}

def array_looks_good(reference_arrays: ak.Array, arrays: ak.Array, header_version: int) -> bool:
    # There are more fields in v2 than in the reference arrays (v1), so only take those
    # that are present in reference for comparison.
    # NOTE: We have to compare the fields one-by-one because the shapes of the fields
    #       are different, and apparently don't broadcast nicely with `__eq__`
    event_level_fields = [
        p for p in ak.fields(reference_arrays) if p not in _v1_particle_property_columns
    ]
    # Event level properties
    for field in event_level_fields:
        new_field = _rename_columns.get(field, field)
        assert ak.all(reference_arrays[field] == arrays[new_field])
    # Particle level properties
    particle_level_fields = [
        p for p in ak.fields(reference_arrays) if p in _v1_particle_property_columns
    ]
    for field in particle_level_fields:
        new_field = _rename_columns.get(field, field)
        assert ak.all(reference_arrays[field] == arrays["particles"][new_field])

    # Check for cross section if header v2
    if header_version == 2:
        assert "cross_section" in ak.fields(arrays)
        assert "cross_section_error" in ak.fields(arrays)

    # check for vertex if header v3
    if header_version == 3:
        assert "vertex_x" in ak.fields(arrays["particles"])
        assert "vertex_y" in ak.fields(arrays["particles"])
        assert "vertex_z" in ak.fields(arrays["particles"])

    return True


@pytest.mark.parametrize(
    "header_version",
    [1, 2],
    ids=["Header v1", "Header v2"]
)
@pytest.mark.parametrize(
    "events_per_chunk",
    [
        5, 16, 50, 5000,
    ], ids=["Multiple, divisible: 5", "Multiple, indivisible: 16", "Equal: 50", "Larger: 5000"]
)
def test_parsing(caplog: Any, header_version: int, events_per_chunk: int) -> None:
    # Setup
    caplog.set_level(logging.INFO)

    input_filename = here / "jetscape_parser" / f"final_state_hadrons_header_v{header_version}.dat"
    if not input_filename.exists():
        pytest.skip(reason="Missing input files - please download the files with the script")

    for i, arrays in enumerate(_jetscape_parser.read(filename=input_filename, events_per_chunk=events_per_chunk, parser="pandas")):
        # Get the reference array
        # Create the reference arrays by checking out the parser v1 (e477e0277fa560f9aba82310c02da8177e61c9e4), setting
        # the chunk size in skim_ascii, and then calling:
        # $ python jetscape_analysis/analysis/reader/skim_ascii.py -i tests/parsing/final_state_hadrons_header_v1.dat -o tests/parsing/events_per_chunk_50/parser_v1_header_v1/test.parquet
        # NOTE: The final state hadron files won't exist when you check out that branch, so
        #       it's best to copy them for your existing branch.
        reference_arrays = ak.from_parquet(
            Path(f"{here}/jetscape_parser/events_per_chunk_{events_per_chunk}/parser_v1_header_v1/test_{i:02}.parquet")
        )

        assert array_looks_good(reference_arrays=reference_arrays, arrays=arrays, header_version=header_version,)


@pytest.mark.parametrize(
    "header_version",
    [1, 2],
    ids=["Header v1", "Header v2"]
)
@pytest.mark.parametrize(
    "events_per_chunk",
    [
        5, 16, 50, 5000,
    ], ids=["Multiple, divisible: 5", "Multiple, indivisible: 16", "Equal: 50", "Larger: 5000"]
)
def test_parsing_with_parquet(caplog: Any, header_version: int, events_per_chunk: int, tmp_path: Path) -> None:
    """Parse to parquet, read back, and compare."""
    # Setup
    caplog.set_level(logging.INFO)

    input_filename = here / "jetscape_parser" / f"final_state_hadrons_header_v{header_version}.dat"
    if not input_filename.exists():
        pytest.skip(reason="Missing input files - please download the files with the script")

    # Convert to chunks in a temp directory.
    base_output_filename = tmp_path / "test.parquet"
    _jetscape_parser.parse_to_parquet(
        base_output_filename=base_output_filename,
        store_only_necessary_columns=True,
        input_filename=input_filename,
        events_per_chunk=events_per_chunk
    )

    output_filenames = tmp_path.glob("*.parquet")

    for i, output_filename in enumerate(sorted(output_filenames)):
        arrays = ak.from_parquet(output_filename)

        # Create the reference arrays by checking out the parser v1 (e477e0277fa560f9aba82310c02da8177e61c9e4), setting
        # the chunk size in skim_ascii, and then calling:
        # $ python jetscape_analysis/analysis/reader/skim_ascii.py -i tests/parsing/final_state_hadrons_header_v1.dat -o tests/parsing/events_per_chunk_50/parser_v1_header_v1/test.parquet
        # NOTE: The final state hadron files won't exist when you check out that branch, so
        #       it's best to copy them for your existing branch.
        reference_arrays = ak.from_parquet(
            Path(f"{here}/jetscape_parser/events_per_chunk_{events_per_chunk}/parser_v1_header_v1/test_{i:02}.parquet")
        )

        assert array_looks_good(reference_arrays=reference_arrays, arrays=arrays, header_version=header_version,)

@pytest.mark.parametrize(
    "events_per_chunk",
    [
        [5, 5, 5, 5, 10, 10, 10],
        [5, 5, 5, 5, 10, 10, 10, 5],
        [5, 5, 5, 5, 10, 10, 9, 5],
    ], ids=["Fits into input data", "Fits into input data, but asks for extra", "Doesn't fit into input data"]
)
def test_parsing_with_changing_chunk_size(caplog: Any, events_per_chunk: list[int]) -> None:
    # Setup
    caplog.set_level(logging.INFO)
    header_version = 2

    # Retrieve input
    N_EVENTS_IN_FILE = 50
    input_filename = here / "jetscape_parser" / f"final_state_hadrons_header_v{header_version}.dat"
    if not input_filename.exists():
        pytest.skip(reason="Missing input files - please download the files with the script")

    event_generator = _jetscape_parser.read(filename=input_filename, events_per_chunk=events_per_chunk[0], parser="pandas")
    expected_length_iter = iter(events_per_chunk)
    # NOTE: The length of the file is 50 events, so if the length ever matches 50, we would expect it to be done.
    cumulative_sum = np.cumsum(events_per_chunk)
    expect_agreement = np.any([v for v in cumulative_sum if v == N_EVENTS_IN_FILE])
    expected_disagreement_index = np.where(cumulative_sum > N_EVENTS_IN_FILE)[0]
    if len(expected_disagreement_index) > 0:
        expected_disagreement_index = expected_disagreement_index[0]
    i = 0
    try:
        arrays = next(event_generator)
        current_expected_length = next(expected_length_iter)
        while True:
            print(f"{current_expected_length=}")
            if expect_agreement or expected_disagreement_index != i:
                assert len(arrays) == current_expected_length
            else:
                with pytest.raises(AssertionError):
                    assert len(arrays) == current_expected_length

            # Update for next round
            # NOTE: If the iterator is exhausted, it will stop here
            current_expected_length = next(expected_length_iter)
            arrays = event_generator.send(current_expected_length)
            i += 1
    except StopIteration:
        ...

    expect_exhaustion_of_lengths = (cumulative_sum[-1] == N_EVENTS_IN_FILE)
    if expect_exhaustion_of_lengths:
        assert i == (len(events_per_chunk) - 1)


@pytest.mark.parametrize(
    "legacy_skim", [False, True]
)
def test_jetscape_file_source_parquet(caplog: Any, legacy_skim: bool, tmp_path: Path) -> None:
    """ Parse handling parquet inputs, including legacy skims. """
    # Setup
    caplog.set_level(logging.INFO)
    header_version = 2

    # Retrieve input
    N_EVENTS_IN_FILE = 50
    input_filename = here / "jetscape_parser" / f"final_state_hadrons_header_v{header_version}.dat"
    if not input_filename.exists():
        pytest.skip(reason="Missing input files - please download the files with the script")

    events_per_chunk = [N_EVENTS_IN_FILE]

    # Need to recreate a legacy skim...
    base_output_filename = tmp_path / "test.parquet"
    if legacy_skim:
        # If we go back to the original parser, it will give the legacy skim format.
        _jetscape_parser.parse_to_parquet(
            base_output_filename=base_output_filename,
            store_only_necessary_columns=True,
            input_filename=input_filename,
            events_per_chunk=events_per_chunk[0],
        )
        # NOTE: The filename was changed because we passed some value for events_per_chunk, so we need to find
        #       the new filename.
        output_filenames = list(tmp_path.glob("*.parquet"))
        if len(output_filenames) != 1:
            raise ValueError(f"Found wrong number of output filenames: {output_filenames}")
        base_output_filename = output_filenames[0]
    else:
        # This is a bit of a roundtrip test, but good enough for now. This will transform the output as expected,
        # and then we can read it back in.
        jetscape_source = jetscape.FileSource(filename=input_filename, metadata={"legacy_skim": legacy_skim})
        arrays = next(jetscape_source.gen_data(chunk_size=events_per_chunk[0]))
        ak.to_parquet(
            array=arrays,
            destination=str(base_output_filename),
            compression="zstd",
            parquet_dictionary_encoding=True,
            parquet_byte_stream_split=True,
        )

    # And make sure we process this file...
    # There's only one chunk, so we can handle it with one filename.
    input_filename = base_output_filename

    jetscape_source = jetscape.FileSource(filename=input_filename, metadata={"legacy_skim": legacy_skim})
    event_generator = jetscape_source.gen_data(chunk_size=events_per_chunk[0])
    expected_length_iter = iter(events_per_chunk)
    # NOTE: The length of the file is 50 events, so if the length ever matches 50, we would expect it to be done.
    cumulative_sum = np.cumsum(events_per_chunk)
    expect_agreement = np.any([v for v in cumulative_sum if v == N_EVENTS_IN_FILE])
    expected_disagreement_index = np.where(cumulative_sum > N_EVENTS_IN_FILE)[0]
    if len(expected_disagreement_index) > 0:
        expected_disagreement_index = expected_disagreement_index[0]
    i = 0
    try:
        arrays = next(event_generator)
        current_expected_length = next(expected_length_iter)
        while True:
            print(f"{current_expected_length=}")
            if expect_agreement or expected_disagreement_index != i:
                assert len(arrays) == current_expected_length
            else:
                with pytest.raises(AssertionError):
                    assert len(arrays) == current_expected_length

            # Update for next round
            # NOTE: If the iterator is exhausted, it will stop here
            current_expected_length = next(expected_length_iter)
            arrays = event_generator.send(current_expected_length)
            i += 1
    except StopIteration:
        ...

    expect_exhaustion_of_lengths = (cumulative_sum[-1] == N_EVENTS_IN_FILE)
    if expect_exhaustion_of_lengths:
        assert i == (len(events_per_chunk) - 1)
