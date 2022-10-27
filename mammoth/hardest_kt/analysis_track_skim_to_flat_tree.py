""" Adapt from the track skim to the existing code base.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import collections
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import awkward as ak
import numpy as np
from mammoth import helpers
from mammoth.framework import load_data, sources
from mammoth.framework.io import track_skim
from mammoth.framework.analysis import objects as analysis_objects
from mammoth.hardest_kt import analysis_alice

from mammoth.hardest_kt import skim_to_flat_tree


logger = logging.getLogger(__name__)

def _convert_analyzed_jets_to_all_jets_for_skim(
    jets: ak.Array,
    convert_data_format_prefixes: Mapping[str, str],
) -> Dict[str, ak.Array]:
    # Need the unsubtracted leading track pt for hybrid
    additional_columns_per_prefix = {}
    for prefix_to_check in convert_data_format_prefixes:
        if prefix_to_check in ak.fields(jets) and "unsubtracted_leading_track_pt" in ak.fields(jets[prefix_to_check]):
            # Store the unsubtracted track pt.
            # It is expected to be under "leading_track_pt" even though it's unsubtracted
            additional_columns_per_prefix[prefix_to_check] = {
                "leading_track_pt": jets[prefix_to_check, "unsubtracted_leading_track_pt"],
            }

    return {
        convert_data_format_prefixes[k]: ak.zip(
            {
                "jet_pt": jets[k].pt,
                "jet_constituents": ak.zip(
                    {
                        "pt": jets[k].constituents.pt,
                        "eta": jets[k].constituents.eta,
                        "phi": jets[k].constituents.phi,
                        "id": jets[k].constituents.index,
                    },
                    with_name="JetConstituent",
                ),
                "jet_splittings": ak.Array(
                    jets[k, "reclustering", "jet_splittings"],
                    with_name="JetSplitting",
                ),
                "subjets": ak.zip(
                    {
                        "part_of_iterative_splitting": jets[
                            k, "reclustering", "subjets", "part_of_iterative_splitting"
                        ],
                        "parent_splitting_index": jets[k, "reclustering", "subjets", "splitting_node_index"],
                        "constituent_indices": jets[k, "reclustering", "subjets", "constituent_indices"],
                    },
                    with_name="Subjet",
                    # We want to apply the behavior for each jet, and then for each subjet
                    # in the jet, so we use a depth limit of 2.
                    depth_limit=2,
                ),
                **additional_columns_per_prefix.get(k, {}),
            },
            depth_limit=1,
        )
        for k in convert_data_format_prefixes
    }


def _hardest_kt_data_skim(
    jets: ak.Array,
    input_filename: Path,
    collision_system: str,
    jet_R: float,
    iterative_splittings: bool,
    convert_data_format_prefixes: Mapping[str, str],
    output_filename: Path,
    scale_factors: Optional[Mapping[int, float]] = None,
    pt_hat_bin: Optional[int] = -1,
) -> None:
    """Implementation of the hardest kt data skim.

    Supports pp, pythia, PbPb, and embedded pythia. The data and jet finding needs to be
    handled in a separate function.
    """
    # Now, adapt into the expected format.
    all_jets = _convert_analyzed_jets_to_all_jets_for_skim(
        jets=jets, convert_data_format_prefixes=convert_data_format_prefixes,
    )

    #ak.to_parquet(all_jets, input_filename.parent / Path("intermediate.parquet"))

    prefixes = {"data": "data"}
    if collision_system == "pythia":
        assert pt_hat_bin is not None
        # Store externally provided pt hard bin
        all_jets["pt_hard_bin"] = np.ones(len(all_jets["data"]["jet_pt"])) * pt_hat_bin
        # Add the second prefix for true jets
        prefixes["true"] = "true"

    skim_to_flat_tree.calculate_data_skim_impl(
        all_jets=all_jets,
        input_filename=input_filename,
        collision_system=collision_system,
        iterative_splittings=iterative_splittings,
        prefixes=prefixes,
        jet_R=jet_R,
        output_filename=output_filename,
        scale_factors=scale_factors,
    )


def hardest_kt_data_skim(
    input_filename: Path,
    collision_system: str,
    jet_R: float,
    min_jet_pt: Mapping[str, float],
    iterative_splittings: bool,
    output_filename: Path,
    convert_data_format_prefixes: Mapping[str, str],
    # Data specific
    loading_data_rename_prefix: Optional[Mapping[str, str]] = None,
    # Pythia specific
    pt_hat_bin: Optional[int] = -1,
    scale_factors: Optional[Mapping[int, float]] = None,
    # Validation
    validation_mode: bool = False,
    background_subtraction: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, str]:
    # Validation
    if loading_data_rename_prefix is None:
        loading_data_rename_prefix = {"data": "data"}

    # Setup
    _description = _description_from_parameters(
        parameters={
            "collision_system": collision_system, "R": jet_R,
            "input_filename": str(input_filename),
        }
    )
    # Try to bail out as early to avoid reprocessing if possible.
    res = _check_for_output_file(output_filename=output_filename, description=_description)
    if res[0]:
        return res

    try:
        # NOTE: Although the later condition on pythia is technically true, the data skim appears to expects both
        #       the det level and part level to be available, so there's not a ton of value in using analysis_data
        #       with pythia (as of Feb 2022) since it will then fail during the data skim. But since we already
        #       implemented it, we leave it in place - perhaps it can be fixed later (or maybe just needs the right
        #       combination of options passed).
        if collision_system in ["pp", "PbPb"] or (collision_system in ["pythia"] and "data" in loading_data_rename_prefix):
            jets = analysis_alice.analysis_data(
                collision_system=collision_system,
                arrays=load_data.data(
                    data_input=input_filename,
                    data_source=track_skim.FileSource.create_deferred_source(collision_system=collision_system),
                    collision_system=collision_system,
                    rename_prefix=loading_data_rename_prefix,
                ),
                jet_R=jet_R,
                min_jet_pt=min_jet_pt,
                validation_mode=validation_mode,
                background_subtraction_settings=background_subtraction,
            )
        elif collision_system in ["pythia"]:
            # Although we could in principle analyze the MC loading only particle or detector level alone,
            # it's more consistent to analyze it with the data quality conditions applied on both part
            # and det level.
            # (ie. we want to analyze in exactly the same as would provided by the substructure analysis task)
            jets = analysis_alice.analysis_MC(
                arrays=load_data.data(
                    data_input=input_filename,
                    data_source=track_skim.FileSource.create_deferred_source(collision_system=collision_system),
                    collision_system=collision_system,
                    rename_prefix=loading_data_rename_prefix,
                ),
                jet_R=jet_R,
                min_jet_pt=min_jet_pt,
                validation_mode=validation_mode,
            )
        else:
            raise NotImplementedError(f"Not yet implemented for {collision_system}...")
    except sources.NoDataAvailableError as e:
        # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
        # Remember that this depends heavily on the jet pt cuts!
        output_filename.with_suffix(".empty").touch()
        return (True, f"Done - no data available (reason: {e}), so not trying to skim for {_description}")

    # NOTE: We need to know how many jets there are, so we arbitrarily take the first field. The jets are flattened,
    #       so they're as good as any others.
    _there_are_jets_left = len(jets[ak.fields(jets)[0]])

    # There were no jets. Note that with a specially crafted empty file
    if not _there_are_jets_left:
        # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
        # Remember that this depends heavily on the jet pt cuts!
        output_filename.with_suffix(".empty").touch()
        return (True, f"Done - no jets to analyze, so not trying to skim for {_description}")

    _hardest_kt_data_skim(
        jets=jets,
        input_filename=input_filename,
        collision_system=collision_system,
        jet_R=jet_R,
        iterative_splittings=iterative_splittings,
        convert_data_format_prefixes=convert_data_format_prefixes,
        output_filename=output_filename,
        pt_hat_bin=pt_hat_bin,
        scale_factors=scale_factors,
    )

    return (True, f"success for {collision_system}, R={jet_R}, {input_filename}")


def _hardest_kt_embedding_skim(
    jets: ak.Array,
    input_filename: Path,
    jet_R: float,
    iterative_splittings: bool,
    scale_factor: float,
    convert_data_format_prefixes: Mapping[str, str],
    output_filename: Path,
) -> None:
    # Now, adapt into the expected format.
    all_jets = _convert_analyzed_jets_to_all_jets_for_skim(
        jets=jets, convert_data_format_prefixes=convert_data_format_prefixes,
    )

    # Define the prefixes for analysis. This should be fairly uniform for the track skim,
    # so we hard code it for now.
    # NOTE: If this becomes an issue, we can just make it an argument.
    prefixes = {
        "hybrid": "hybrid",
        #"part_level": "part_level",
        "true": "true",
        "det_level": "det_level",
    }

    skim_to_flat_tree.calculate_embedding_skim_impl(
        all_jets=all_jets,
        input_filename=input_filename,
        iterative_splittings=iterative_splittings,
        prefixes=prefixes,
        scale_factor=scale_factor,
        jet_R=jet_R,
        output_filename=output_filename,
    )


def _description_from_parameters(parameters: Mapping[str, Any]) -> str:
    return ", ".join([f"{k}={v}" for k, v in parameters.items()])


def _check_for_output_file(output_filename: Path, description: str) -> Tuple[bool, str]:
    # Try to bail out as early to avoid reprocessing if possible.
    # First, check for the empty filename
    empty_filename = output_filename.with_suffix(".empty")
    if empty_filename.exists():
        # It will be empty, so there's nothing to check. Just return
        return (True, f"Done - no jets to analyze for {description}")

    # Next, check the contents of the output file
    if output_filename.exists():
        import uproot

        try:
            with uproot.open(output_filename) as f:
                # If the tree exists, can be read, and has more than 0 entries, we should be good
                if f["tree"].num_entries > 0:
                    # Return immediately to indicate that we're done.
                    return (True, f"already processed for {description}")
        except Exception:
            # If it fails for some reason, give up - we want to try again
            pass

    return (False, "")


def hardest_kt_embed_thermal_model_skim(
    collision_system: str,
    signal_input: Union[Path, Sequence[Path]],
    convert_data_format_prefixes: Mapping[str, str],
    jet_R: float,
    min_jet_pt: Mapping[str, float],
    iterative_splittings: bool,
    background_subtraction: Mapping[str, Any],
    det_level_artificial_tracking_efficiency: float,
    thermal_model_parameters: sources.ThermalModelParameters,
    output_filename: Path,
    scale_factor: float,
    chunk_size: sources.T_ChunkSize = sources.ChunkSizeSentinel.SINGLE_FILE,
    validation_mode: bool = False,
) -> Tuple[bool, str]:
    # Validation
    signal_input_filenames = []
    if not isinstance(signal_input, collections.abc.Iterable):
        signal_input_filenames = [signal_input]
    else:
        signal_input_filenames = list(signal_input)

    _parameters = {
        "collision_system": collision_system, "R": jet_R,
        "signal_input_filenames": str([str(_filename) for _filename in signal_input_filenames]),
    }
    if chunk_size is not sources.ChunkSizeSentinel.FULL_SOURCE:
        _parameters["chunk_size"] = chunk_size
    _description = _description_from_parameters(parameters=_parameters)

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
        empty_filename = output_filename.with_suffix(".empty")
        empty_filename.touch()
        return (True, f"Done - no data available (reason: {e}), so not trying to skim for {_description}")

    # Cross check
    assert not isinstance(iter_arrays, ak.Array), "Check configuration. This should be an iterable, not an ak.Array!"

    for i_chunk, arrays in enumerate(iter_arrays):
        # Setup. We need to identify the chunk
        _output_filename = output_filename.parent / f"{output_filename.stem}_chunk_{i_chunk:03}{output_filename.suffix}"

        # Try to bail out as early to avoid reprocessing if possible.
        res = _check_for_output_file(output_filename=_output_filename, description=_description)
        if res[0]:
            logger.info(f"Skipping chunk {i_chunk}: {res}")
            continue

        try:
            jets = analysis_alice.analysis_embedding(
                source_index_identifiers=source_index_identifiers,
                arrays=arrays,
                jet_R=jet_R,
                min_jet_pt=min_jet_pt,
                background_subtraction_settings=background_subtraction,
                det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
                validation_mode=validation_mode,
            )
        except sources.NoDataAvailableError as e:
            # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
            # Remember that this depends heavily on the jet pt cuts!
            _output_filename.with_suffix(".empty").touch()
            logger.info((True, f"Chunk {i_chunk}: Done - no data available (reason: {e}), so not trying to skim for {_description}"))
            continue

        # NOTE: We need to know how many jets there are, so we arbitrarily take the first field. The jets are flattened,
        #       so they're as good as any others.
        _there_are_jets_left = len(jets[ak.fields(jets)[0]])

        # There were no jets. Note that with a specially crafted empty file
        if not _there_are_jets_left:
            # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
            # Remember that this depends heavily on the jet pt cuts!
            _output_filename.with_suffix(".empty").touch()
            logger.info((True, f"Chunk {i_chunk}: Done - no jets to recluster, so not trying to skim for {_description}"))
            continue

        _input_filename = signal_input_filenames[0].parent / f"{signal_input_filenames[0].stem}_chunk_{i_chunk:03}{signal_input_filenames[0].suffix}"
        _hardest_kt_embedding_skim(
            jets=jets,
            # NOTE: This argument is only for logging messages. Since the PbPb is the constraining factor,
            #       we focus on processing those files.
            input_filename=_input_filename,
            jet_R=jet_R,
            iterative_splittings=iterative_splittings,
            scale_factor=scale_factor,
            convert_data_format_prefixes=convert_data_format_prefixes,
            output_filename=_output_filename,
        )

    return (True, f"success for {_description}")


def hardest_kt_embedding_skim(
    collision_system: str,
    signal_input: Union[Path, Sequence[Path]],
    background_input: Union[Path, Sequence[Path]],
    convert_data_format_prefixes: Mapping[str, str],
    jet_R: float,
    min_jet_pt: Mapping[str, float],
    iterative_splittings: bool,
    background_subtraction: Mapping[str, Any],
    det_level_artificial_tracking_efficiency: float,
    output_filename: Path,
    scale_factor: float,
    validation_mode: bool = False,
    background_is_constrained_source: bool = True,
) -> Tuple[bool, str]:
    # Validation
    signal_input_filenames = []
    if not isinstance(signal_input, collections.abc.Iterable):
        signal_input_filenames = [signal_input]
    else:
        signal_input_filenames = list(signal_input)
    background_input_filenames = []
    if not isinstance(background_input, collections.abc.Iterable):
        background_input_filenames = [background_input]
    else:
        background_input_filenames = list(background_input)
    # Try to bail out early to avoid reprocessing if possible.
    _description = _description_from_parameters(
        parameters={
            "collision_system": collision_system, "R": jet_R,
            "signal_input_filenames": str([str(_filename) for _filename in signal_input_filenames]),
            "background_input_filename": str([str(_filename) for _filename in background_input_filenames]),
        }
    )
    res = _check_for_output_file(output_filename=output_filename, description=_description)
    if res[0]:
        return res

    try:
        source_index_identifiers, arrays = load_data.embedding(
            signal_input=signal_input_filenames,
            signal_source=track_skim.FileSource.create_deferred_source(collision_system="pythia"),
            background_input=background_input_filenames,
            background_source=track_skim.FileSource.create_deferred_source(collision_system="PbPb"),
            background_is_constrained_source=background_is_constrained_source,
        )
    except sources.NoDataAvailableError as e:
        # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
        # Remember that this depends heavily on the jet pt cuts!
        empty_filename = output_filename.with_suffix(".empty")
        empty_filename.touch()
        return (True, f"Done - no data available (reason: {e}), so not trying to skim for {_description}")

    jets = analysis_alice.analysis_embedding(
        source_index_identifiers=source_index_identifiers,
        arrays=arrays,
        jet_R=jet_R,
        min_jet_pt=min_jet_pt,
        background_subtraction_settings=background_subtraction,
        det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
        validation_mode=validation_mode,
    )

    # NOTE: We need to know how many jets there are, so we arbitrarily take the first field. The jets are flattened,
    #       so they're as good as any others.
    _there_are_jets_left = len(jets[ak.fields(jets)[0]])

    # There were no jets. Note that with a specially crafted empty file
    if not _there_are_jets_left:
        # Just create the empty filename and return. This will prevent trying to re-run with no jets in the future.
        # Remember that this depends heavily on the jet pt cuts!
        output_filename.with_suffix(".empty").touch()
        return (True, f"Done - no jets to analyze, so not trying to skim for {_description}")

    _hardest_kt_embedding_skim(
        jets=jets,
        # NOTE: This argument is only for logging messages. We want the filename to be the one which
        #       is the constrained source.
        input_filename=background_input_filenames[0] if background_is_constrained_source else signal_input_filenames[0],
        jet_R=jet_R,
        iterative_splittings=iterative_splittings,
        scale_factor=scale_factor,
        convert_data_format_prefixes=convert_data_format_prefixes,
        output_filename=output_filename,
    )

    return (True, f"success for {_description}")


if __name__ == "__main__":
    helpers.setup_logging(level=logging.INFO)
    #logging.getLogger("mammoth.framework.jet_finding").setLevel(logging.INFO)
    #logging.getLogger("mammoth._ext").setLevel(logging.DEBUG)

    _min_jet_pt = {
        "pp": {"data": 5.},
        "pythia": {"det_level": 20.},
        "PbPb": {"data": 20.},
        "embed_thermal_model": {"hybrid": 20.},
        "embedPythia": {"hybrid": 20.},
    }
    # For validation, we use R = 0.4 jets
    jet_R = 0.2
    #for collision_system in ["pp", "pythia", "PbPb"]:
    for collision_system in ["pp", "pythia", "PbPb"]:
        logger.info(f"Analyzing \"{collision_system}\"")
        base_path = Path(f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}")

        scale_factors = None
        pt_hat_bin = -1
        if collision_system == "pythia":
            # NOTE: Using external information here to set this up. Normally, we would set
            #       this via a configuration file
            scale_factors = analysis_objects.read_extracted_scale_factors(
                path=Path(f"trains/{collision_system}/LHC18b8_pythia_R04_1/scale_factors.yaml")
            )
            pt_hat_bin = 12

        result = hardest_kt_data_skim(
            #input_filename=Path("/software/rehlers/dev/substructure/trains/PbPb/645/run_by_run/LHC18q/295612/AnalysisResults.18q.002.root"),
            input_filename=base_path / "AnalysisResults_track_skim.parquet",
            collision_system=collision_system,
            jet_R=jet_R,
            min_jet_pt=_min_jet_pt[collision_system],
            iterative_splittings=True,
            loading_data_rename_prefix={"data": "data"} if collision_system != "pythia" else {},
            convert_data_format_prefixes={"data": "data"} if collision_system != "pythia" else {"det_level": "data", "part_level": "true"},
            output_filename=base_path / "skim" / "skim_output.root",
            scale_factors=scale_factors,
            pt_hat_bin=pt_hat_bin,
            validation_mode=True,
        )
        logger.info(f"Result: {result}")

    ###############
    # Thermal model
    ###############
    # In general, we're probably testing with this period, so good enough to hard code it here
    scale_factors = analysis_objects.read_extracted_scale_factors(
        path=Path("trains/pythia/LHC20g4_AOD_2640/scale_factors.yaml")
    )

    base_path = Path("/software/rehlers/dev/substructure/trains/pythia/641")
    #signal_input = base_path / "run_by_run/LHC20g4/295612/11/AnalysisResults.20g4.016.root"
    #signal_input = base_path / "run_by_run/LHC20g4/297544/19/AnalysisResults.20g4.005.root"
    signal_input = base_path / "run_by_run/LHC20g4/295819/12/AnalysisResults.20g4.016.root"
    #signal_input = base_path / "run_by_run/LHC20g4/297588/4/AnalysisResults.20g4.001.root"
    pt_hat_bin = 12
    hardest_kt_embed_thermal_model_skim(
        collision_system="embed_thermal_model",
        signal_input=[signal_input],
        jet_R=jet_R,
        min_jet_pt=_min_jet_pt["embed_thermal_model"],
        iterative_splittings=True,
        output_filename=base_path / "skim" / "test" / "thermal_model_skim_output.root",
        thermal_model_parameters=sources.THERMAL_MODEL_SETTINGS["5020_central"],
        convert_data_format_prefixes={"hybrid": "hybrid", "det_level": "det_level", "part_level": "true"},
        scale_factor=scale_factors[pt_hat_bin],
        background_subtraction={"r_max": 0.25},
        det_level_artificial_tracking_efficiency=0.98,
        chunk_size=1000,
        validation_mode=False,
    )

    ###########
    # Embedding
    ###########
    standalone_tests = False

    # In general, we're probably testing with this period, so good enough to hard code it here
    scale_factors = analysis_objects.read_extracted_scale_factors(
        path=Path("trains/pythia/LHC20g4_AOD_2640/scale_factors.yaml")
    )

    # Mammoth validation needs something like
    base_path = Path("/software/rehlers/dev/mammoth/projects/framework/embedPythia")
    #signal_path = base_path / "AnalysisResults_pythia_track_skim.parquet"
    #background_path = base_path / "AnalysisResults_PbPb_track_skim.parquet"
    signal_path = base_path / "track_skim" / "pythia" / "AnalysisResults.root"
    background_path = base_path / "track_skim" / "PbPb" / "AnalysisResults.root"
    output_filename = base_path / "skim" / "skim_output.root"
    pt_hat_bin = 12
    if standalone_tests:
        # But we can also run standalone tests on the skim train output
        base_path = Path("/software/rehlers/dev/substructure/trains/PbPb/645")
        #signal_path = Path("/software/rehlers/dev/substructure/trains/pythia/2640") / "run_by_run/LHC20g4/296191/12/AnalysisResults.20g4.001.root"
        #background_path = Path("/software/rehlers/dev/substructure/trains/PbPb/645") / "run_by_run/LHC18q/295612/AnalysisResults.18q.001.root"
        signal_path = Path("/software/rehlers/dev/substructure/trains/pythia/2640") / "run_by_run/LHC20g4/295788/15/AnalysisResults.20g4.005.root"
        background_path = Path("/software/rehlers/dev/substructure/trains/PbPb/645") / "run_by_run/LHC18q/295788/AnalysisResults.18q.076.root"
        output_filename = base_path / "skim" / "test" / "embedding_skim_output.root"
        pt_hat_bin = 15

    result = hardest_kt_embedding_skim(
        collision_system="embedPythia",
        signal_input=[signal_path, signal_path, signal_path],
        background_input=background_path,
        jet_R=jet_R,
        min_jet_pt=_min_jet_pt["embedPythia"],
        iterative_splittings=True,
        output_filename=output_filename,
        convert_data_format_prefixes={"hybrid": "hybrid", "det_level": "det_level", "part_level": "true"},
        scale_factor=scale_factors[pt_hat_bin],
        background_subtraction={"r_max": 0.25},
        det_level_artificial_tracking_efficiency=1.0,
        validation_mode=True,
    )
    logger.info(f"Result: {result}")
