"""Running EEC analysis on track skim

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Any, Sequence

import awkward as ak
import hist

from mammoth.eec import analysis_alice
from mammoth.framework import load_data, sources
from mammoth.framework import task as framework_task
from mammoth.framework.analysis import conventions as analysis_conventions
from mammoth.framework.io import track_skim

logger = logging.getLogger(__name__)


def eec_embed_thermal_model_analysis(  # noqa: C901
    collision_system: str,
    signal_input: Path | Sequence[Path],
    trigger_pt_ranges: dict[str, tuple[float, float]],
    min_track_pt: dict[str, float],
    momentum_weight_exponent: int | float,
    det_level_artificial_tracking_efficiency: float,
    thermal_model_parameters: sources.ThermalModelParameters,
    output_filename: Path,
    scale_factor: float,
    chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.SINGLE_FILE,
    output_trigger_skim: bool = False,
    validation_mode: bool = False,
) -> framework_task.Output:
    # Validation
    signal_input_filenames = []
    if not isinstance(signal_input, collections.abc.Iterable):
        signal_input_filenames = [signal_input]
    else:
        signal_input_filenames = list(signal_input)

    # Setup
    _parameters: dict[str, Any] = {
        "collision_system": collision_system,
        "signal_input_filenames": str([str(_filename) for _filename in signal_input_filenames]),
    }
    if chunk_size is not sources.ChunkSizeSentinel.FULL_SOURCE:
        _parameters["chunk_size"] = chunk_size
    _description = analysis_conventions.description_from_parameters(parameters=_parameters)

    # Try to bail out early to avoid reprocessing if possible.
    # This would only work if is was previously processed with one chunk, but it doesn't hurt to try
    res = analysis_conventions.check_for_skim_output_file(output_filename=output_filename, description=_description)
    if res[0]:
        return framework_task.Output(*res, collision_system, {})

    # Setup iteration over the input files
    # If we don't use a processing chunk size, it should all be done in one chunk by default.
    # However, the memory usage often gets too large, so this allows us to control the overall memory
    # size by breaking it up into chunks, such that we only generate the thermal model chunk
    # that's currently needed for processing
    try:
        source_index_identifiers, iter_arrays = load_data.embedding_thermal_model(
            signal_input=signal_input_filenames,
            signal_source=track_skim.FileSource.create_deferred_source(collision_system="pythia"),
            thermal_model_parameters=thermal_model_parameters,
            chunk_size=chunk_size,
        )
    except sources.NoDataAvailableError as e:
        # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
        # Remember that this depends heavily on the jet pt cuts!
        output_filename.with_suffix(".empty").touch()
        return framework_task.Output(True, f"Done - no data available (reason: {e}), so not trying to skim for {_description}", collision_system, {})

    # Validate that the arrays are in an a format that we can iterate over
    if isinstance(iter_arrays, ak.Array):
        iter_arrays = iter([iter_arrays])
    assert not isinstance(iter_arrays, ak.Array), "Check configuration. This should be an iterable, not an ak.Array!"

    _nonstandard_results = []
    hists: dict[str, hist.Hist] = {}
    for i_chunk, arrays in enumerate(iter_arrays):
        # Setup
        # We need to identify the chunk in the output name
        # NOTE: To be consistent with expectations for a single chunk, the output name should only append the suffix
        #       if it's more than the first chunk
        if i_chunk > 0:
            _output_filename = (
                output_filename.parent / f"{output_filename.stem}_chunk_{i_chunk:03}{output_filename.suffix}"
            )
        else:
            _output_filename = output_filename

        # Try to bail out as early to avoid reprocessing if possible.
        res = analysis_conventions.check_for_skim_output_file(output_filename=_output_filename, description=_description)
        if res[0]:
            _nonstandard_results.append(res)
            logger.info(f"Skipping already processed chunk {i_chunk}: {res}")
            continue

        try:
            analysis_output = analysis_alice.analysis_embedding(
                source_index_identifiers=source_index_identifiers,
                arrays=arrays,
                trigger_pt_ranges=trigger_pt_ranges,
                min_track_pt=min_track_pt,
                momentum_weight_exponent=momentum_weight_exponent,
                scale_factor=scale_factor,
                det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
                output_trigger_skim=output_trigger_skim,
                validation_mode=validation_mode,
            )
        except sources.NoDataAvailableError as e:
            # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
            # Remember that this depends heavily on the jet pt cuts!
            _output_filename.with_suffix(".empty").touch()
            _message = (
                True,
                f"Chunk {i_chunk}: Done - no data available (reason: {e}), so not trying to skim for {_description}",
            )
            _nonstandard_results.append(_message)
            logger.info(_message)
            continue

        trigger_skim: dict[str, ak.Array] | None = None
        if output_trigger_skim:
            assert not isinstance(analysis_output, dict)
            analysis_hists, trigger_skim = analysis_output
        else:
            assert isinstance(analysis_output, dict)
            analysis_hists = analysis_output

        # Merge the output hists
        if analysis_hists:
            from mammoth import job_utils

            job_utils.merge_results(hists, analysis_hists)

        if trigger_skim:
            for skim_name, skim_array in trigger_skim.items():
                _skim_output_filename = _output_filename.parent / skim_name / _output_filename.name
                _skim_output_filename.parent.mkdir(parents=True, exist_ok=True)
                if ak.num(skim_array, axis=0) == 0:
                    # Skip the skim if it's empty
                    _skim_output_filename.with_suffix(".empty").touch()
                else:
                    # Write the skim
                    ak.to_parquet(
                        array=skim_array,
                        #array={k: v for k, v in trigger_skim.items() if k}
                        # NOTE: Record is needed because otherwise awkward can't figure out how to write it.
                        #array=ak.Record(trigger_skim),
                        destination=str(_skim_output_filename),
                        compression="zstd",
                        # Optimize for columns with anything other than floats
                        parquet_dictionary_encoding=True,
                        # Optimize for columns with floats
                        parquet_byte_stream_split=True,
                    )

        # Cleanup (may not be necessary, but it doesn't hurt)
        del arrays
        del analysis_hists
        del analysis_output
        if trigger_skim:
            del trigger_skim

    return framework_task.Output(
        True,
        f"success for {_description}"
        + (f". Additional non-standard results: {_nonstandard_results}" if _nonstandard_results else ""),
        collision_system,
        hists,
    )
