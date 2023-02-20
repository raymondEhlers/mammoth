"""Run mammoth skimming and analysis tasks via parsl

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import enum
import logging
import secrets
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import attrs
import IPython
from mammoth import helpers, job_utils
from mammoth.alice import job_utils as alice_job_utils
from mammoth.framework import sources, production
from mammoth.framework.analysis import objects as analysis_objects
from mammoth.framework.analysis import jets as analysis_jets
from mammoth.job_utils import python_app
from pachyderm import yaml
from parsl.data_provider.files import File


logger = logging.getLogger(__name__)


class SplittingsSelection(enum.Enum):
    recursive = 0
    iterative = 1

    def __str__(self) -> str:
        return f"{self.name}_splittings"

@attrs.frozen()
class HardestKtProductionSpecialization:
    def customize_identifier(self, analysis_settings: Mapping[str, Any]) -> str:
        name = ""
        # Selection of splittings
        splittings_selection_value = SplittingsSelection[analysis_settings["splittings_selection"]]
        name += f"_{str(splittings_selection_value)}"
        return name

    def tasks_to_execute(self, collision_system: str) -> List[str]:
        _tasks = []

        # Skim task
        _base_name = "calculate_{label}_skim"
        _label_map = {
            "pp": "data",
            "pythia": "data",
            "PbPb": "data",
            "embedPythia": "embed_pythia",
            "embed_pythia": "embed_pythia",
            "embed_thermal_model": "embed_thermal_model",
        }
        _tasks.append(_base_name.format(label=_label_map[collision_system]))
        return _tasks


def safe_output_filename_from_relative_path(filename: Path, output_dir: Path) -> str:
    """Safe and identifiable name for naming output files based on the relative path.

    Converts: "2111/run_by_run/LHC17p_CENT_woSDD/282341/AnalysisResults.17p.001.root"
           -> "2111__run_by_run__LHC17p_CENT_woSDD__282341__AnalysisResults_17p_001"

    Returns:
        Filename that is safe for using as the output filename.
    """
    # NOTE: We use the grandparent of the output dir because the input filename is going to be a different train
    #       than our output. For the case of embedding trains, we might not even share the collision system.
    #       So by going to the grandparent (ie `trains`), we end up with a shared path
    reference_dir = output_dir.parent.parent
    # `relative_to` requires that both filenames are the same type (either absolute or relative)
    # `reference_dir` is usually relative, so we may need to resolve it to ensure that the comparison will work.
    if filename.is_absolute():
        # NOTE: We can't use `resolve()` because it will resolve symlinks, which we probably don't want it to do
        #       since we usually symlink the `train` directory.
        # NOTE: `pathlib.Path.absolute()` would be perfect here, but it requires 3.11
        reference_dir = Path.cwd() / reference_dir
    return str(filename.relative_to(reference_dir).with_suffix("")).replace("/", "__").replace(".", "_")


@python_app
def _extract_scale_factors_from_hists(
    list_name: str,
    job_framework: job_utils.JobFramework,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
) -> analysis_objects.ScaleFactor:
    """
    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same beyond switching to uproot.
    """
    from pathlib import Path

    from mammoth.framework.analysis import objects as analysis_objects
    from mammoth.alice import scale_factors as sf

    res = analysis_objects.ScaleFactor.from_hists(
        *sf.scale_factor_uproot(filenames=[Path(i.filepath) for i in inputs], list_name=list_name)
    )
    return res


def setup_extract_scale_factors(
    prod: production.ProductionSettings,
    job_framework: job_utils.JobFramework,
) -> Dict[int, Future[analysis_objects.ScaleFactor]]:
    """Extract scale factors from embedding or pythia hists.

    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same.

    Note:
        This is surprisingly fast.
    """
    # Setup
    scale_factors: Dict[int, Future[analysis_objects.ScaleFactor]] = {}
    logger.info("Determining input files for extracting scale factors.")
    input_files_per_pt_hat_bin = prod.input_files_per_pt_hat()

    dataset_key = "signal_dataset" if "signal_dataset" in prod.config["metadata"] else "dataset"
    for pt_hat_bin, input_files in input_files_per_pt_hat_bin.items():
        logger.debug(f"pt_hat_bin: {pt_hat_bin}, filenames: {input_files}")
        if input_files:
            scale_factors[pt_hat_bin] = _extract_scale_factors_from_hists(
                inputs=[File(str(fname)) for fname in input_files],
                list_name=prod.config["metadata"][dataset_key]["list_name"],
                job_framework=job_framework,
            )

    return scale_factors


@python_app
def _write_scale_factors_to_yaml(
    scale_factors: Mapping[int, analysis_objects.ScaleFactor],
    job_framework: job_utils.JobFramework,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> bool:
    """
    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same.
    """
    from pathlib import Path

    from pachyderm import yaml

    from mammoth.framework.analysis import objects as analysis_objects

    # Write them to YAML for later.
    y = yaml.yaml(classes_to_register=[analysis_objects.ScaleFactor])
    output_dir = Path(outputs[0].filepath)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(output_dir, "w") as f:
        y.dump(scale_factors, f)

    return True


def setup_write_scale_factors(
    prod: production.ProductionSettings,
    scale_factors: Mapping[int, Future[analysis_objects.ScaleFactor]],
    job_framework: job_utils.JobFramework,
) -> Future[bool]:
    """
    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same.
    """
    """Write scale factors to YAML and to trees if necessary."""
    # First, we write to YAML.
    # We want to do this regardless of potentially writing the scale factor trees.
    logger.info("Writing scale factors to YAML. Jobs are executing, so this will take a minute...")
    output_filename = prod.scale_factors_filename
    parsl_output_file = File(str(output_filename))
    # NOTE: I'm guessing passing this is a problem because it's a class that's imported in an app, and then
    #       we're trying to pass the result into another app. I think we can go one direction or the other,
    #       but not both. So we just take the result.

    yaml_result = _write_scale_factors_to_yaml(
        # NOTE: We have to lie a bit about the typing here because passing a dask delayed object
        #       directly to another function will cause it to depend on it, which will fill in the
        #       value once it's executed. Parsl still have a future, so it needs to be handled separately.
        #       (Ideally, parsl would handle this correctly and provide the concrete value, but it doesn't seem
        #       seem to be able to do so...)
        scale_factors={  # type: ignore[arg-type]
            k: v.result() for k, v in scale_factors.items()
        } if job_framework == job_utils.JobFramework.parsl else scale_factors,
        job_framework=job_framework,
        outputs=[parsl_output_file],
    )

    return yaml_result


@python_app
def _extract_pt_hat_spectra(
    scale_factors: Mapping[int, float],
    offsets: Mapping[int, int],
    list_name: str,
    yaml_exists: bool,
    job_framework: job_utils.JobFramework,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
) -> bool:
    """
    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same.

    Args:
        scale_factors: pt_hat_bin to scale factor.
        offsets: pt_hat_bin to index where files in that pt hard bin start.
    """
    from pathlib import Path

    from mammoth.alice import scale_factors as sf

    # Convert back from parsl inputs
    offsets_values = list(offsets.values())
    filenames = {
        pt_hat_bin: [
            Path(f.filepath) for f in inputs[sum(offsets_values[:i]) : sum(offsets_values[: i + 1])]  # noqa: E203
        ]
        for i, pt_hat_bin in enumerate(offsets)
    }

    res = sf.pt_hat_spectra_from_hists(
        filenames=filenames,
        scale_factors=scale_factors,
        list_name=list_name,
        output_filename=Path(outputs[0].filepath),
    )
    return res


def setup_check_pt_hat_spectra(
    prod: production.ProductionSettings,
    yaml_exists: bool,
    job_framework: job_utils.JobFramework,
) -> Future[bool]:
    """
    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same.
    """
    logger.info("Checking pt hat spectra")
    # Input files
    input_files_per_pt_hat_bin = prod.input_files_per_pt_hat()

    # Must read the scale factors from file to get the properly scaled values.
    # NOTE: This means that we need to ensure externally that it's available.
    #       This is kind of hacking around the DAG, but it's convenient given the
    #       gymnastics that we need to do to support both parsl and dask
    scale_factors = prod.scale_factors()

    # Convert inputs to Parsl files.
    # Needs to be a list, so flatten them, and then unflatten in the App.
    parsl_files = []
    offsets = {}
    for pt_hat_bin, list_of_files in input_files_per_pt_hat_bin.items():
        converted_filenames = [File(str(f)) for f in list_of_files]
        offsets[pt_hat_bin] = len(converted_filenames)
        parsl_files.extend(converted_filenames)

    dataset_key = "signal_dataset" if "signal_dataset" in prod.config["metadata"] else "dataset"
    # We want to store it in the same directory as the scale factors, so it's easiest to just grab that filename.
    output_filename = prod.scale_factors_filename.parent / "pt_hat_spectra.yaml"

    results = _extract_pt_hat_spectra(
        scale_factors=scale_factors,
        offsets=offsets,
        list_name=prod.config["metadata"][dataset_key]["list_name"],
        yaml_exists=yaml_exists,
        job_framework=job_framework,
        inputs=parsl_files,
        outputs=[File(str(output_filename))],
    )

    return results


def steer_extract_scale_factors(
    prod: production.ProductionSettings,
    job_framework: job_utils.JobFramework,
) -> List[Future[Any]]:
    # Validation
    if not prod.has_scale_factors:
        raise ValueError(f"Invalid collision system for extracting scale factors: {prod.collision_system}")

    # Attempt to bail out early if it's already been extracted
    scale_factors_filename = prod.scale_factors_filename
    if scale_factors_filename.exists():
        stored_scale_factors = prod.scale_factors()
        # We check if it's non-zero to avoid the case where it's accidentally empty
        if stored_scale_factors:
            logger.info("Scale factors already exist. Skipping extracting them again!")
            return []
    logger.info("Extracting scale factors...")

    # First, we need to extract the scale factors and keep track of the results
    all_results: List[Future[Any]] = []
    scale_factors = setup_extract_scale_factors(prod=prod, job_framework=job_framework)
    all_results.extend(list(scale_factors.values()))

    # Then, we need to write them
    all_results.append(
        setup_write_scale_factors(
            prod=prod,
            scale_factors=scale_factors,
            job_framework=job_framework
        )
    )
    # Store the result in a way that we can query later.
    if job_framework == job_utils.JobFramework.parsl:
        writing_yaml_success: bool = all_results[-1].result()
    elif job_framework == job_utils.JobFramework.dask_delayed:
        # We're lying about the type here because it's convenient
        writing_yaml_success = all_results[-1].compute()  # type: ignore[attr-defined]
    else:
        # We're lying about the type here because it's convenient
        writing_yaml_success = all_results[-1]  # type: ignore[assignment]

    if writing_yaml_success is not True:
        logger.warning("Some issue with the scale factor extraction! Check on them!")
        IPython.start_ipython(user_ns={**locals(), **globals()})
        # We want to stop here, so help ourselves out by raising the exception.
        raise ValueError("Some issue with the scale factor extraction!")
    else:
        # And then create the spectra (and plot them) to cross check the extraction
        all_results.append(
            setup_check_pt_hat_spectra(
                prod=prod,
                yaml_exists=writing_yaml_success,
                job_framework=job_framework,
            )
        )

    # NOTE: We don't want to scale_factor futures because they contain the actual scale factors.
    #       We'll just return the futures associated with writing to the file and the pt hat spectra cross check.
    # NOTE: If the case of dask, we only want to return the pt hat spectra cross check. Otherwise, it will run
    #       the writing to file task again
    return all_results[-1 if job_framework == job_utils.JobFramework.dask_delayed else -2:]


@python_app
def _run_data_skim(
    collision_system: str,
    jet_R: float,
    min_jet_pt: Mapping[str, float],
    iterative_splittings: bool,
    background_subtraction: Mapping[str, Any],
    skim_type: str,
    loading_data_rename_prefix: Mapping[str, str],
    convert_data_format_prefixes: Mapping[str, str],
    pt_hat_bin: int,
    scale_factors: Optional[Mapping[int, float]],
    det_level_artificial_tracking_efficiency: float | analysis_jets.PtDependentTrackingEfficiencyParameters | None,
    job_framework: job_utils.JobFramework,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> Tuple[bool, str]:
    import traceback
    from pathlib import Path

    from mammoth.hardest_kt import analysis_track_skim_to_flat_tree

    try:
        result = analysis_track_skim_to_flat_tree.hardest_kt_data_skim(
            input_filename=Path(inputs[0].filepath),
            collision_system=collision_system,
            jet_R=jet_R,
            min_jet_pt=min_jet_pt,
            iterative_splittings=iterative_splittings,
            skim_type=skim_type,
            background_subtraction=background_subtraction,
            loading_data_rename_prefix=loading_data_rename_prefix,
            convert_data_format_prefixes=convert_data_format_prefixes,
            scale_factors=scale_factors,
            pt_hat_bin=pt_hat_bin,
            det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
            output_filename=Path(outputs[0].filepath),
        )
    except Exception:
        result = (
            False,
            f"failure for {collision_system}, R={jet_R}, {inputs[0].filepath} with: \n{traceback.format_exc()}",
        )
    return result


def setup_calculate_data_skim(
    prod: production.ProductionSettings,
    job_framework: job_utils.JobFramework,
    debug_mode: bool,
) -> List[Future[Tuple[bool, str]]]:
    """Create futures to produce hardest kt data skim"""
    # Setup input and output
    # Need to handle pt hat bin productions differently than standard productions
    # since we need to keep track of the pt hat bin
    if "n_pt_hat_bins" in prod.config["metadata"]["dataset"]:
        input_files: Dict[int, List[Path]] = prod.input_files_per_pt_hat()
    else:
        input_files = {-1: prod.input_files()}
    output_dir = prod.output_dir / "skim"
    output_dir.mkdir(parents=True, exist_ok=True)

    # If we want to debug some particular files, we can directly set them here
    if debug_mode:
        input_files = {10: [Path("trains/pythia/2619/run_by_run/LHC18b8_cent_woSDD/282008/10/AnalysisResults.18b8_cent_woSDD.003.root")]}
        #input_files = {-1: [Path("trains/pp/2111/run_by_run/LHC17p_CENT_woSDD/282341/AnalysisResults.17p.586.root")]}
        #input_files = {-1: [Path("trains/PbPb/645/run_by_run/LHC18r/297595/AnalysisResults.18r.551.root")]}

    # Setup for analysis and dataset settings
    _metadata_config = prod.config["metadata"]
    _analysis_config = prod.config["settings"]
    # Splitting selection (iterative vs recursive)
    splittings_selection = SplittingsSelection[_analysis_config["splittings_selection"]]
    # Scale factors
    scale_factors = None
    if prod.has_scale_factors:
        scale_factors = prod.scale_factors()
    # Det level artificial tracking efficiency (pythia only)
    det_level_artificial_tracking_efficiency = None
    if prod.collision_system == "pythia":
        # Artificial tracking efficiency (including the option for pt dependent tracking eff)
        # NOTE: This depends on period, so it's better to do it here!
        det_level_artificial_tracking_efficiency = _analysis_config.get("det_level_artificial_tracking_efficiency", None)
        # Pt dependent for tracking efficiency uncertainty
        if _analysis_config.get("apply_pt_dependent_tracking_efficiency_uncertainty", False):
            # NOTE: Careful - this needs to be added as 1-value. (ie. 1-.97=0.03 -> for .98 flat, we get .95)
            det_level_artificial_tracking_efficiency = analysis_jets.PtDependentTrackingEfficiencyParameters.from_file(
                # NOTE: We select "anchor_period" and "0_100" here because we know we're analyzing pythia
                period=_metadata_config["dataset"]["anchor_period"],
                event_activity="0_100",
                # NOTE: There should be the possibility to apply this on top of the .98, for example.
                baseline_tracking_efficiency_shift=det_level_artificial_tracking_efficiency,
            )

    results = []
    _file_counter = 0
    for pt_hat_bin, input_filenames in input_files.items():
        for input_filename in input_filenames:
            if _file_counter % 500 == 0:
                logger.info(f"Adding {input_filename} for analysis")

            # For debugging
            if debug_mode and _file_counter > 1:
                break

            # Setup file I/O
            # Converts: "2111/run_by_run/LHC17p_CENT_woSDD/282341/AnalysisResults.17p.001.root"
            #        -> "2111__run_by_run__LHC17p_CENT_woSDD__282341__AnalysisResults_17p_001"
            output_identifier = safe_output_filename_from_relative_path(
                filename=input_filename, output_dir=prod.output_dir
            )
            output_filename = output_dir / f"{output_identifier}_{str(splittings_selection)}.root"
            # And create the tasks
            results.append(
                _run_data_skim(
                    collision_system=prod.collision_system,
                    jet_R=_analysis_config["jet_R"],
                    min_jet_pt=_analysis_config["min_jet_pt"],
                    iterative_splittings=splittings_selection == SplittingsSelection.iterative,
                    background_subtraction=_analysis_config.get("background_subtraction", {}),
                    skim_type=_metadata_config["dataset"]["skim_type"],
                    loading_data_rename_prefix=_metadata_config["loading_data_rename_prefix"],
                    convert_data_format_prefixes=_metadata_config["convert_data_format_prefixes"],
                    pt_hat_bin=pt_hat_bin,
                    scale_factors=scale_factors,
                    det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
                    job_framework=job_framework,
                    inputs=[File(str(input_filename))],
                    outputs=[File(str(output_filename))],
                )
            )

            _file_counter += 1

    return results


@python_app
def _run_embedding_skim(
    collision_system: str,
    jet_R: float,
    min_jet_pt: Mapping[str, float],
    iterative_splittings: bool,
    background_subtraction: Mapping[str, Any],
    det_level_artificial_tracking_efficiency: float,
    convert_data_format_prefixes: Mapping[str, str],
    chunk_size: sources.T_ChunkSize,
    scale_factor: float,
    background_is_constrained_source: bool,
    n_signal_input_files: int,
    job_framework: job_utils.JobFramework,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> Tuple[bool, str]:
    import traceback
    from pathlib import Path

    from mammoth.hardest_kt import analysis_track_skim_to_flat_tree

    try:
        result = analysis_track_skim_to_flat_tree.hardest_kt_embedding_skim(
            collision_system=collision_system,
            signal_input=[Path(_input_file.filepath) for _input_file in inputs[:n_signal_input_files]],
            background_input=[Path(_input_file.filepath) for _input_file in inputs[n_signal_input_files:]],
            convert_data_format_prefixes=convert_data_format_prefixes,
            jet_R=jet_R,
            min_jet_pt=min_jet_pt,
            iterative_splittings=iterative_splittings,
            background_subtraction=background_subtraction,
            det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
            output_filename=Path(outputs[0].filepath),
            chunk_size=chunk_size,
            scale_factor=scale_factor,
            background_is_constrained_source=background_is_constrained_source,
        )
    except Exception:
        result = (
            False,
            f"failure for {collision_system}, R={jet_R}, signal={[_f.filepath for _f in inputs[:n_signal_input_files]]}, background={[_f.filepath for _f in inputs[n_signal_input_files:]]} with: \n{traceback.format_exc()}",
        )
    return result


def _extract_info_from_signal_file_list(
    signal_input_files_per_pt_hat: Mapping[int, Sequence[Path]]
) -> Tuple[List[int], List[Tuple[int, Path]]]:
    """Helper to extract the pt hat bins and flatten the input list."""
    # And since we would sample the pt hat bins, it's better to keep track of them directly.
    pt_hat_bins = list(signal_input_files_per_pt_hat)
    # Or alternatively, we sample the pythia files directly. In this case, the PYTHIA files are sampled
    # evenly, while the pt hat bins are not (we will sample the higher pt hat bins more because there
    # are more statistics).
    # To do this sampling, we need to flatten out the list of signal input files.
    # We also store the pt hat bin since we need that for the grabbing the right scale factor.
    signal_input_files_flat = [
        (_pt_hat_bin, _signal_path)
        for _pt_hat_bin, _signal_paths in signal_input_files_per_pt_hat.items()
        for _signal_path in _signal_paths
    ]

    return pt_hat_bins, signal_input_files_flat


def _determine_unconstrained_signal_input_files(
    signal_input_files_per_pt_hat: Mapping[int, Sequence[Path]],
    signal_input_files_flat: Sequence[Tuple[int, Path]],
    pt_hat_bins: Sequence[int],
    signal_input_config: Mapping[str, Any],
) -> Tuple[int, List[Path]]:
    """Determine the signal input files for the unconstrained case.

    We refactored this out since the logic is a bit complex to be inline.
    """
    # Sample the pt hat equally, or directly sample the signal_input_files
    _sample_each_pt_hat_bin_equally = signal_input_config["sample_each_pt_hat_bin_equally"]
    _n_files_to_use_per_task = signal_input_config["n_files_to_use_per_task"]

    # Randomly select (in some manner) an input file to match up with the background input file.
    # NOTE: The signal input file will repeat if there are more background events.
    #       So far, this doesn't seem to be terribly common, but even if it was, it
    #       would be perfectly fine as long as it doesn't happen too often.
    signal_input = []
    if _sample_each_pt_hat_bin_equally:
        # Each pt hat bin will be equally likely, and then we select the file from
        # those which are available.
        # NOTE: This doesn't mean that the embedded statistics will still be the same in the end.
        #       For example, if I compare a low and high pt hat bin, there are just going to be
        #       more accepted jets in the high pt hat sample.
        pt_hat_bin = secrets.choice(pt_hat_bins)
        signal_input = [
            secrets.choice(signal_input_files_per_pt_hat[pt_hat_bin]) for _ in range(_n_files_to_use_per_task)
        ]
    else:
        # Directly sample the files. This probes the generator stats because
        # the number of files is directly proportional to the generated statistics.
        pt_hat_bin, _signal_input_filename = secrets.choice(signal_input_files_flat)
        signal_input = [_signal_input_filename]
        # Since we want to keep the same pt hat bin, use the pt hat ban to randomly select additional files
        signal_input.extend(
            [
                secrets.choice(signal_input_files_per_pt_hat[pt_hat_bin])
                # -1 since we already have a filename
                for _ in range(_n_files_to_use_per_task - 1)
            ]
        )
    return pt_hat_bin, signal_input


def _select_files_for_source(
    input_files: Sequence[Path],
    selected_input_file: Path,
    n_files_to_use: int,
) -> List[Path]:
    """Select n files from a list of available files without replacement."""
    _input = [selected_input_file]

    _possible_additional_files = set(
        [
            secrets.choice(input_files)
            # -1 since we already have a filename
            # +5 since we'll remove any filenames if they're repeated
            # NOTE: +5 is arbitrary, but should be sufficient. We could do more, but it would be a waste of cycles.
            #       In any case, We'll double check below.
            for _ in range(n_files_to_use - 1 + 5)
        ]
    )
    # Remove the existing file, and then add to the list
    _possible_additional_files.discard(selected_input_file)
    _input.extend(list(_possible_additional_files)[: n_files_to_use - 1])

    # Validate that we didn't somehow end up with too few files
    # This really shouldn't happen outside of exceptional cases
    if len(_input) != n_files_to_use:
        raise ValueError(
            "You somehow don't have enough input files."
            f" Requested: {n_files_to_use}, but only have {len(_input)} files available."
            " Check your input configuration!"
        )

    return _input


def _determine_embed_pythia_input_files(
    signal_input_files_per_pt_hat: Mapping[int, Sequence[Path]],
    background_input_files: Sequence[Path],
    background_is_constrained_source: bool,
    input_handling_config: Mapping[str, Any],
) -> Iterable[Tuple[int, Sequence[Path], Sequence[Path]]]:
    """Determine the input files for embedding with pythia."""
    # Configuration setup
    signal_input_config = input_handling_config["signal_parameters"]
    background_input_config = input_handling_config["background_parameters"]

    # Some convenient quantities for working with signal inputs
    pt_hat_bins, signal_input_files_flat = _extract_info_from_signal_file_list(
        signal_input_files_per_pt_hat=signal_input_files_per_pt_hat
    )

    if background_is_constrained_source:
        for background_file in background_input_files:
            # Determine the constrained input (background)
            # Start with the file that we iterated with
            background_input = _select_files_for_source(
                input_files=background_input_files,
                selected_input_file=background_file,
                n_files_to_use=background_input_config["constrained_source"]["n_files_to_use_per_task"],
            )

            # Determine the unconstrained input (signal)
            pt_hat_bin, signal_input = _determine_unconstrained_signal_input_files(
                signal_input_files_per_pt_hat=signal_input_files_per_pt_hat,
                signal_input_files_flat=signal_input_files_flat,
                pt_hat_bins=pt_hat_bins,
                signal_input_config=signal_input_config["unconstrained_source"],
            )

            yield pt_hat_bin, signal_input, background_input
    else:
        for pt_hat_bin, signal_file in signal_input_files_flat:
            # Determine the constrained input (signal)
            # Start with the file that we iterated with
            signal_input = _select_files_for_source(
                input_files=signal_input_files_per_pt_hat[pt_hat_bin],
                selected_input_file=signal_file,
                n_files_to_use=signal_input_config["constrained_source"]["n_files_to_use_per_task"],
            )

            # Determine unconstrained source (background)
            background_input = [
                secrets.choice(background_input_files)
                for _ in range(background_input_config["unconstrained_source"]["n_files_to_use_per_task"])
            ]

            yield pt_hat_bin, signal_input, background_input


def setup_calculate_embed_pythia_skim(  # noqa: C901
    prod: production.ProductionSettings,
    job_framework: job_utils.JobFramework,
    debug_mode: bool,
) -> List[Future[Tuple[bool, str]]]:
    """Create futures to produce hardest kt embedded pythia skim"""
    # Setup input and output
    output_dir = prod.output_dir / "skim"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files (signal and background)
    background_input_files = prod.input_files()
    # We store the signal input files in a few different formats to enable sampling different ways.
    # We can sample pt hat bins equally by sampling the pt hat bin, and then taking a random file
    # from that bin. In this case, the pythia files _are not_ sampled equally.
    signal_input_files_per_pt_hat = prod.input_files_per_pt_hat()

    # If we want to debug some particular files, we can directly set them here
    if debug_mode:
        background_input_files = [Path("trains/PbPb/645/run_by_run/LHC18q/296270/AnalysisResults.18q.179.root")]
        # signal_input_files_per_pt_hat = {1: [Path("trains/pythia/2640/run_by_run/LHC20g4/297132/1/AnalysisResults.20g4.013.root")]}
        # signal_input_files_per_pt_hat = {12: [Path("trains/pythia/2640/run_by_run/LHC20g4/297132/12/AnalysisResults.20g4.013.root")]}
        # signal_input_files_per_pt_hat = {3: [
        #    #Path("trains/pythia/2640/run_by_run/LHC20g4/295819/3/AnalysisResults.20g4.006.root"),
        #    Path("trains/pythia/2640/run_by_run/LHC20g4/297317/3/AnalysisResults.20g4.013.root"),
        #    #Path("trains/pythia/2640/run_by_run/LHC20g4/296935/3/AnalysisResults.20g4.009.root"),
        # ]}
        signal_input_files_per_pt_hat = {7: [
            Path('trains/pythia/2640/run_by_run/LHC20g4/296550/7/AnalysisResults.20g4.014.root'),
            Path('trains/pythia/2640/run_by_run/LHC20g4/296244/7/AnalysisResults.20g4.001.root'),
            Path('trains/pythia/2640/run_by_run/LHC20g4/297379/7/AnalysisResults.20g4.002.root'),
        ]}
        # signal_input_files_per_pt_hat = {11: [
        #     Path('trains/pythia/2640/run_by_run/LHC20g4/296191/11/AnalysisResults.20g4.007.root'),
        #     Path('trains/pythia/2640/run_by_run/LHC20g4/297132/11/AnalysisResults.20g4.008.root'),
        #     Path('trains/pythia/2640/run_by_run/LHC20g4/295612/11/AnalysisResults.20g4.008.root'),
        # ]}
        # background_input_files = [Path('trains/PbPb/645/run_by_run/LHC18q/296270/AnalysisResults.18q.607.root')]
        # signal_input_files_per_pt_hat = {10: [
        #     Path('trains/pythia/2640/run_by_run/LHC20g4/295612/10/AnalysisResults.20g4.007.root'),
        #     Path('trains/pythia/2640/run_by_run/LHC20g4/297544/10/AnalysisResults.20g4.010.root'),
        #     Path('trains/pythia/2640/run_by_run/LHC20g4/296935/10/AnalysisResults.20g4.013.root'),
        # ]}
        # background_input_files = [Path("trains/PbPb/645/run_by_run/LHC18r/297595/AnalysisResults.18r.551.root")]
        # signal_input_files_per_pt_hat = {
        #     12: [
        #         Path("trains/pythia/2640/run_by_run/LHC20g4/296690/12/AnalysisResults.20g4.008.root"),
        #         Path("trains/pythia/2640/run_by_run/LHC20g4/295819/12/AnalysisResults.20g4.009.root"),
        #         Path("trains/pythia/2640/run_by_run/LHC20g4/297479/12/AnalysisResults.20g4.009.root"),
        #     ]
        # }
        background_input_files = [Path("trains/PbPb/645/run_by_run/LHC18q/296304/AnalysisResults.18q.333.root")]
        signal_input_files_per_pt_hat = {
            1: [
                Path("trains/pythia/2640/run_by_run/LHC20g4/295612/1/AnalysisResults.20g4.006.root"),
            ]
        }

    # Setup for dataset and input
    _metadata_config: Dict[str, Any] = prod.config["metadata"]
    _input_handling_config: Dict[str, Any] = _metadata_config["input_handling"]
    _background_is_constrained_source: bool = not (_metadata_config["input_constrained_source"].lower() == "signal")

    # Analysis settings
    _analysis_config: Dict[str, Any] = prod.config["settings"]
    # Chunk size
    _chunk_size = _analysis_config.get("chunk_size", sources.ChunkSizeSentinel.FULL_SOURCE)
    logger.info(f"Processing chunk size for {_chunk_size}")
    # Splitting selection (iterative vs recursive)
    splittings_selection = SplittingsSelection[_analysis_config["splittings_selection"]]
    # Artificial tracking efficiency (including the option for pt dependent tracking eff)
    # NOTE: This depends on centrality and period, so it's better to do it here!
    det_level_artificial_tracking_efficiency = _analysis_config["det_level_artificial_tracking_efficiency"]
    # Pt dependent for tracking efficiency uncertainty
    if _analysis_config["apply_pt_dependent_tracking_efficiency_uncertainty"]:
        _event_activity_name_to_centrality_values = {
            "central": "0_10",
            "semi_central": "30_50",
        }
        # NOTE: Careful - this needs to be added as 1-value. (ie. 1-.97=0.03 -> for .98 flat, we get .95)
        det_level_artificial_tracking_efficiency = analysis_jets.PtDependentTrackingEfficiencyParameters.from_file(
            period=_metadata_config["dataset"]["period"],
            event_activity=_event_activity_name_to_centrality_values[_analysis_config["event_activity"]],
            # NOTE: There should be the possibility to apply this on top of the .98, for example.
            baseline_tracking_efficiency_shift=det_level_artificial_tracking_efficiency,
        )

    # Scale factors
    scale_factors = None
    if prod.has_scale_factors:
        scale_factors = prod.scale_factors()
    else:
        raise ValueError("Check the embedding config - you need a signal dataset.")

    # Cross check
    # NOTE: We usually need to skip this during debug mode because we may not have all pt hat bins in the input,
    #       so it will fail trivially.
    if not debug_mode:
        pt_hat_bins, _ = _extract_info_from_signal_file_list(
            signal_input_files_per_pt_hat=signal_input_files_per_pt_hat
        )
        if set(scale_factors) != set(pt_hat_bins):
            raise ValueError(
                f"Mismatch between the pt hat bins in the scale factors ({set(scale_factors)}) and the pt hat bins ({set(pt_hat_bins)})"
            )

    logger.info(
        f"Configuring embed pythia with {'background' if _background_is_constrained_source else 'signal'} as the constrained source."
    )

    results = []
    _embedding_file_pairs = {}
    # Keep track of output identifiers. If there is already an existing identifier, then we can try again to avoid overwriting it.
    _output_identifiers = []
    input_generator = _determine_embed_pythia_input_files(
        signal_input_files_per_pt_hat=signal_input_files_per_pt_hat,
        background_input_files=background_input_files,
        background_is_constrained_source=_background_is_constrained_source,
        input_handling_config=_input_handling_config,
    )
    for _file_counter, (pt_hat_bin, signal_input, background_input) in enumerate(input_generator):
        if _file_counter % 500 == 0:
            logger.info(
                f"Adding {(background_input if _background_is_constrained_source else signal_input)} for analysis"
            )

        # For debugging
        if debug_mode and _file_counter > 1:
            break

        # Setup file I/O
        # We want to identify as: "{signal_identifier}__embedded_into__{background_identifier}"
        # Take the first signal and first background filenames as the main identifier to the path.
        # Otherwise, the filename could become indefinitely long... (apparently there are file length limits in unix...)
        output_identifier = safe_output_filename_from_relative_path(
            filename=signal_input[0], output_dir=prod.output_dir
        )
        output_identifier += "__embedded_into__"
        output_identifier += safe_output_filename_from_relative_path(
            filename=background_input[0], output_dir=prod.output_dir
        )
        # Finally, add the splittings selection
        output_identifier += f"_{str(splittings_selection)}"

        # Ensure that we don't use an output identifier twice.
        # If we've already used it, we add a counter to it
        _modifiable_output_identifier = output_identifier
        _output_identifier_counter = 0
        _output_identifier_stored = False
        while not _output_identifier_stored:
            if _modifiable_output_identifier in _output_identifiers:
                # If the identifier is in the list, try to add some counter to it.
                _output_identifier_counter += 1
                _modifiable_output_identifier = output_identifier + f"__{_output_identifier_counter:03}"
            else:
                output_identifier = _modifiable_output_identifier
                _output_identifiers.append(output_identifier)
                _output_identifier_stored = True

        # logger.info(f"output_identifier: {output_identifier}")
        output_filename = output_dir / f"{output_identifier}.root"

        # Store the file pairs for our records
        # The output identifier contains the first signal filename, as well as the background filename.
        # We use it here rather than _just_ the background filename because we may embed into data multiple times
        _embedding_file_pairs[output_identifier] = [str(_filename) for _filename in signal_input] + [
            str(_filename) for _filename in background_input
        ]

        # And create the tasks
        results.append(
            _run_embedding_skim(
                collision_system=prod.collision_system,
                jet_R=_analysis_config["jet_R"],
                min_jet_pt=_analysis_config["min_jet_pt"],
                iterative_splittings=splittings_selection == SplittingsSelection.iterative,
                background_subtraction=_analysis_config["background_subtraction"],
                det_level_artificial_tracking_efficiency=_analysis_config["det_level_artificial_tracking_efficiency"],
                convert_data_format_prefixes=_metadata_config["convert_data_format_prefixes"],
                chunk_size=_chunk_size,
                scale_factor=scale_factors[pt_hat_bin],
                background_is_constrained_source=_background_is_constrained_source,
                n_signal_input_files=len(signal_input),
                job_framework=job_framework,
                inputs=[
                    *[File(str(_filename)) for _filename in signal_input],
                    *[File(str(_filename)) for _filename in background_input],
                ],
                outputs=[File(str(output_filename))],
            )
        )

    # And write the file pairs, again for our records
    y = yaml.yaml()
    embedding_file_pairs_filename = prod.output_dir / "embedding_file_pairs.yaml"
    _existing_embedding_file_pairs = {}
    if embedding_file_pairs_filename.exists():
        with open(embedding_file_pairs_filename, "r") as f:
            _existing_embedding_file_pairs = y.load(f)
    # Add back in the existing file pairs if we've read them
    if _existing_embedding_file_pairs:
        _embedding_file_pairs.update(_existing_embedding_file_pairs)
    # And then (re)write the file pairs
    with open(embedding_file_pairs_filename, "w") as f:
        y.dump(_embedding_file_pairs, f)

    return results


@python_app
def _run_embed_thermal_model_skim(
    collision_system: str,
    jet_R: float,
    min_jet_pt: Mapping[str, float],
    iterative_splittings: bool,
    background_subtraction: Mapping[str, Any],
    det_level_artificial_tracking_efficiency: float,
    thermal_model_parameters: sources.ThermalModelParameters,
    chunk_size: int,
    convert_data_format_prefixes: Mapping[str, str],
    scale_factor: float,
    job_framework: job_utils.JobFramework,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> Tuple[bool, str]:
    import traceback
    from pathlib import Path

    from mammoth.hardest_kt import analysis_track_skim_to_flat_tree

    try:
        result = analysis_track_skim_to_flat_tree.hardest_kt_embed_thermal_model_skim(
            collision_system=collision_system,
            signal_input=[Path(_input_file.filepath) for _input_file in inputs],
            convert_data_format_prefixes=convert_data_format_prefixes,
            jet_R=jet_R,
            min_jet_pt=min_jet_pt,
            iterative_splittings=iterative_splittings,
            background_subtraction=background_subtraction,
            det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
            thermal_model_parameters=thermal_model_parameters,
            chunk_size=chunk_size,
            output_filename=Path(outputs[0].filepath),
            scale_factor=scale_factor,
        )
    except Exception:
        result = (
            False,
            f"failure for {collision_system}, R={jet_R}, signal={[_f.filepath for _f in inputs]} with: \n{traceback.format_exc()}",
        )
    return result


def setup_calculate_embed_thermal_model_skim(
    prod: production.ProductionSettings,
    job_framework: job_utils.JobFramework,
    debug_mode: bool,
) -> List[Future[Tuple[bool, str]]]:
    """Create futures to produce hardest kt embedded pythia skim"""
    # Setup input and output
    # Need to handle pt hat bin productions differently than standard productions
    # since we need to keep track of the pt hat bin
    if "n_pt_hat_bins" in prod.config["metadata"]["dataset"]:
        input_files: Dict[int, List[Path]] = prod.input_files_per_pt_hat()
    else:
        input_files = {-1: prod.input_files()}
        raise RuntimeError("Need pt hat production for embedding into a thermal model...")
    output_dir = prod.output_dir / "skim"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup for analysis and dataset settings
    _metadata_config = prod.config["metadata"]
    _analysis_config = prod.config["settings"]
    # Splitting selection (iterative vs recursive)
    splittings_selection = SplittingsSelection[_analysis_config["splittings_selection"]]
    thermal_model_parameters = sources.THERMAL_MODEL_SETTINGS[
        f"{_metadata_config['dataset']['sqrt_s']}_{_analysis_config['event_activity']}"
    ]
    chunk_size = _analysis_config["chunk_size"]
    logger.info(f"Processing chunk size for {chunk_size}")
    # Scale factors
    scale_factors = None
    if prod.has_scale_factors:
        scale_factors = prod.scale_factors()
    else:
        raise ValueError("Check the thermal model config - you need a signal dataset.")

    # Cross check
    if set(scale_factors) != set(list(input_files)):
        raise ValueError(
            f"Mismatch between the pt hat bins in the scale factors ({set(scale_factors)}) and the pt hat bins ({set(list(input_files))})"
        )

    results = []
    _file_counter = 0
    # Reversed because the higher pt hard bins are of more importance to get done sooner.
    for pt_hat_bin, input_filenames in reversed(input_files.items()):
        for input_filename in input_filenames:
            if _file_counter % 500 == 0 or debug_mode:
                logger.info(f"Adding {input_filename} for analysis")

            # For debugging
            if debug_mode and _file_counter > 1:
                break

            # Setup file I/O
            # Converts: "2111/run_by_run/LHC17p_CENT_woSDD/282341/AnalysisResults.17p.001.root"
            #        -> "2111__run_by_run__LHC17p_CENT_woSDD__282341__AnalysisResults_17p_001"
            output_identifier = safe_output_filename_from_relative_path(
                filename=input_filename, output_dir=prod.output_dir
            )
            output_filename = output_dir / f"{output_identifier}_{str(splittings_selection)}.root"
            # And create the tasks
            results.append(
                _run_embed_thermal_model_skim(
                    collision_system=prod.collision_system,
                    jet_R=_analysis_config["jet_R"],
                    min_jet_pt=_analysis_config["min_jet_pt"],
                    iterative_splittings=splittings_selection == SplittingsSelection.iterative,
                    background_subtraction=_analysis_config["background_subtraction"],
                    det_level_artificial_tracking_efficiency=_analysis_config[
                        "det_level_artificial_tracking_efficiency"
                    ],
                    thermal_model_parameters=thermal_model_parameters,
                    chunk_size=chunk_size,
                    convert_data_format_prefixes=_metadata_config["convert_data_format_prefixes"],
                    scale_factor=scale_factors[pt_hat_bin],
                    job_framework=job_framework,
                    inputs=[
                        File(str(input_filename)),
                    ],
                    outputs=[File(str(output_filename))],
                )
            )

            _file_counter += 1

    return results


def setup_job_framework(
    job_framework: job_utils.JobFramework,
    productions: Sequence[production.ProductionSettings],
    task_config: job_utils.TaskConfig,
    facility: job_utils.FACILITIES,
    walltime: str,
    target_n_tasks_to_run_simultaneously: int,
    log_level: int,
    conda_environment_name: Optional[str] = None,
) -> Tuple[job_utils.parsl.DataFlowKernel, job_utils.parsl.Config] | Tuple[job_utils.dask.distributed.Client, job_utils.dask.distributed.SpecCluster]:
    # First, need to figure out if we need additional environments such as ROOT
    _additional_worker_init_script = alice_job_utils.determine_additional_worker_init(
        productions=productions,
        conda_environment_name=conda_environment_name,
        tasks_requiring_root=[],
        tasks_requiring_aliphysics=[],
    )
    return job_utils.setup_job_framework(
        job_framework=job_framework,
        task_config=task_config,
        facility=facility,
        walltime=walltime,
        target_n_tasks_to_run_simultaneously=target_n_tasks_to_run_simultaneously,
        log_level=log_level,
        additional_worker_init_script=_additional_worker_init_script,
    )


def define_productions() -> List[production.ProductionSettings]:
    # We want to provide the opportunity to run multiple productions at once.
    # We'll do so by defining each production below and then iterating over them below
    productions = []

    # Create and store production information
    _here = Path(__file__).parent
    config_filename = Path(_here.parent / "alice" / "config" / "track_skim_config.yaml")
    productions.extend(
        [
            # Debug
            # production.ProductionSettings.read_config(
            #     collision_system="embed_pythia", number=3,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            # Production
            # production.ProductionSettings.read_config(
            #     collision_system="embed_pythia", number=65,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            # production.ProductionSettings.read_config(
            #     collision_system="embed_pythia", number=66,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            # Debug
            # production.ProductionSettings.read_config(
            #     collision_system="PbPb", number=3,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            # Production
            production.ProductionSettings.read_config(
                collision_system="embed_thermal_model", number=61,
                specialization=HardestKtProductionSpecialization(),
                track_skim_config_filename=config_filename,
            ),
            # production.ProductionSettings.read_config(
            #     collision_system="embed_thermal_model", number=62,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            # production.ProductionSettings.read_config(
            #     collision_system="PbPb",
            #     number=68,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            # production.ProductionSettings.read_config(
            #     collision_system="PbPb",
            #     number=71,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            # production.ProductionSettings.read_config(
            #     collision_system="PbPb",
            #     number=72,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            # production.ProductionSettings.read_config(
            #     collision_system="embed_pythia", number=68,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            # production.ProductionSettings.read_config(
            #     collision_system="embed_pythia", number=69,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            # production.ProductionSettings.read_config(
            #     collision_system="embed_pythia", number=70,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            # production.ProductionSettings.read_config(
            #     collision_system="embed_pythia", number=72,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            # production.ProductionSettings.read_config(
            #     collision_system="embed_pythia", number=73,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
            # production.ProductionSettings.read_config(
            #     collision_system="embed_pythia", number=74,
            #     specialization=HardestKtProductionSpecialization(),
            #     track_skim_config_filename=config_filename,
            # ),
        ]
    )

    # Write out the production settings
    for production_settings in productions:
        production_settings.store_production_parameters()

    return productions


def _hours_in_walltime(walltime: str) -> int:
    return int(walltime.split(":")[0])


def setup_and_submit_tasks(
    productions: Sequence[production.ProductionSettings],
    task_config: job_utils.TaskConfig,
    job_framework: job_utils.JobFramework,
    debug_mode: bool,
    job_executor: job_utils.parsl.DataFlowKernel | job_utils.dask.distributed.Client,
) -> List[Future[Any]]:
    all_results: List[Future[Any]] = []
    for prod in productions:
        tasks_to_execute = prod.tasks_to_execute
        logger.info(f"Tasks to execute: {tasks_to_execute} for production \"{prod.collision_system}\" #{prod.formatted_number}")

        # Setup tasks
        system_results = []
        if "extract_scale_factors" in tasks_to_execute:
            # NOTE: This will block on the result since it needs to be done before anything can proceed
            system_results.extend(
                steer_extract_scale_factors(
                    prod=prod,
                    job_framework=job_framework,
                )
            )
        if "calculate_data_skim" in tasks_to_execute:
            system_results.extend(
                setup_calculate_data_skim(
                    prod=prod,
                    job_framework=job_framework,
                    debug_mode=debug_mode,
                )
            )
        if "calculate_embed_thermal_model_skim" in tasks_to_execute:
            system_results.extend(
                setup_calculate_embed_thermal_model_skim(
                    prod=prod,
                    job_framework=job_framework,
                    debug_mode=debug_mode,
                )
            )
        if "calculate_embed_pythia_skim" in tasks_to_execute:
            system_results.extend(
                setup_calculate_embed_pythia_skim(
                    prod=prod,
                    job_framework=job_framework,
                    debug_mode=debug_mode,
                )
            )

        all_results.extend(system_results)
        logger.info(f"Accumulated {len(system_results)} futures for {prod.collision_system}")

    logger.info(f"Accumulated {len(all_results)} total futures")

    if job_framework == job_utils.JobFramework.dask_delayed:
        assert isinstance(job_executor, job_utils.dask.distributed.Client)
        all_results = job_executor.compute(  # type: ignore[no-untyped-call]
            all_results,
            # Distributed assumes functions are pure, but usually mine are not (ie. they create files)
            pure=False,
            resources={"n_cores": task_config.n_cores_per_task}
        )

    return all_results


def process_futures(
    productions: Sequence[production.ProductionSettings],
    all_results: Sequence[Future[Any]],
    job_framework: job_utils.JobFramework,
) -> None:
    # Process the futures, showing processing progress
    # Since it returns the results, we can actually use this to accumulate results.
    if job_framework == job_utils.JobFramework.dask_delayed:
        gen_results: Iterable[Any] = job_utils.dask.distributed.as_completed(all_results)  # type: ignore[no-untyped-call]
    else:
        gen_results = job_utils.provide_results_as_completed(all_results, running_with_parsl=True)

    # In order to support writing histograms from multiple systems, we need to index the output histograms
    # by the collision system + centrality.
    output_hists: Dict[str, Dict[Any, Any]] = {_p.collision_system: {} for _p in productions}
    with helpers.progress_bar() as progress:
        track_results = progress.add_task(total=len(all_results), description="Processing results...")
        # for a in all_results:
        for result in gen_results:
            # r = a.result()
            logger.info(f"result: {result[:2]}")
            if result[0] and len(result) == 4 and isinstance(result[3], dict):
                k = result[2]
                logger.info(f"Found result for key {k}")
                output_hists[k] = job_utils.merge_results(output_hists[k], result[3])
            logger.info(f"output_hists: {output_hists}")
            progress.update(track_results, advance=1)

    # Save hists to uproot (if needed)
    for system, hists in output_hists.items():
        if hists:
            import uproot

            split_system_name = system.split("_")
            # Either "pp" or "PbPb"
            collision_system = split_system_name[0]
            # Additional label for centrality when appropriate
            # NOTE: If the list is of length 1, it will be empty
            file_label = "_".join(split_system_name[1:])
            if file_label:
                file_label = f"_{file_label}"

            output_hist_filename = Path("output") / collision_system / f"hardest_kt_{file_label}.root"
            output_hist_filename.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing output_hists to {output_hist_filename} for system {system}")
            with uproot.recreate(output_hist_filename) as f:
                helpers.write_hists_to_file(hists=hists, f=f)

    # By embedded here, we can inspect results, etc in the meantime.
    # NOTE: This may be commented out sometimes when I have long running processes and wil
    #       probably forget to close it.
    IPython.start_ipython(user_ns={**locals(), **globals()})

    # In case we close IPython early, wait for all apps to complete
    # Also allows for a summary at the end.
    # By taking only the first two, it just tells use the status and a quick message.
    # Otherwise, we can overwhelm with trying to print large objects
    res = [r.result()[:2] for r in all_results]
    logger.info(res)


def run(job_framework: job_utils.JobFramework) -> List[Future[Any]]:
    # Job execution parameters
    productions = define_productions()
    task_name = "hardest_kt_mammoth"

    # Job execution configuration
    conda_environment_name = ""
    task_config = job_utils.TaskConfig(name=task_name, n_cores_per_task=1)
    # target_n_tasks_to_run_simultaneously = 120
    # target_n_tasks_to_run_simultaneously = 110
    target_n_tasks_to_run_simultaneously = 60
    log_level = logging.INFO
    walltime = "24:00:00"
    debug_mode = False
    if debug_mode:
        # Usually, we want to run in the short queue
        target_n_tasks_to_run_simultaneously = 2
        walltime = "1:59:00"
    facility: job_utils.FACILITIES = "ORNL_b587_long" if _hours_in_walltime(walltime) >= 2 else "ORNL_b587_short"

    # Keep the job executor just to keep it alive
    job_executor, _job_framework_config = setup_job_framework(
        job_framework=job_framework,
        productions=productions,
        task_config=task_config,
        facility=facility,
        walltime=walltime,
        target_n_tasks_to_run_simultaneously=target_n_tasks_to_run_simultaneously,
        log_level=log_level,
        conda_environment_name=conda_environment_name,
    )
    all_results = setup_and_submit_tasks(
        productions=productions,
        task_config=task_config,
        job_framework=job_framework,
        debug_mode=debug_mode,
        job_executor=job_executor
    )

    process_futures(productions=productions, all_results=all_results, job_framework=job_framework)

    return all_results


if __name__ == "__main__":
    run(job_framework=job_utils.JobFramework.parsl)
