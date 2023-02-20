"""Comparison between standard analysis and track skim

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBNL/UCB
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import attr
import pytest

from mammoth.framework import sources
from mammoth.framework.io import track_skim as io_track_skim
from mammoth.framework.analysis import jet_substructure as analysis_jet_substructure, objects as analysis_objects
from mammoth.hardest_kt import analysis_track_skim_to_flat_tree, run_macro, skim_to_flat_tree, substructure_comparison_tools


logger = logging.getLogger(__name__)

_here = Path(__file__).parent
_track_skim_base_path = _here / "track_skim_validation"

# ALICE data files to use for the validation
# NOTE: These files are too large to store in the git repo.
#       I've stored them multiple places to attempt to ensure that they're not lost entirely.
#       Since the AliEn path is provided, they can always be retrieved (in principle).
# NOTE: All were stored in archives (see the pachyderm dataset def for the particular archive names),
#       and then were extracted for simplicity.
_collision_system_to_aod_files = {
    "pp": [
        # 17p
        _track_skim_base_path / "input/alice/data/2017/LHC17p/000282343/pass1_FAST/AOD234/0001/root_archive.zip#AliAOD.root",
        # Default to using less statistics to keep the runtime down, but can run more if it's warranted
        # _track_skim_base_path / "input/alice/data/2017/LHC17p/000282343/pass1_FAST/AOD234/0002/root_archive.zip#AliAOD.root",
    ],
    # Strictly speaking, this should be LHC18b8 to correctly correspond to LHC17pq, but for these
    # purposes, it's fine.
    "pythia": [
        # LHC20g4
        _track_skim_base_path / "input/alice/sim/2020/LHC20g4/12/296191/AOD/001/aod_archive.zip#AliAOD.root",
    ],
    "PbPb": [
        # LHC18q
        _track_skim_base_path / "input/alice/data/2018/LHC18q/000296550/pass3/AOD252/AOD/001/aod_archive.zip#AliAOD.root",
    ],
    "embed_pythia": [
        # LHC18q
        _track_skim_base_path / "input/alice/data/2018/LHC18q/000296550/pass3/AOD252/AOD/001/aod_archive.zip#AliAOD.root",
    ],
    "embed_pythia-pythia": [
        # LHC20g4
        _track_skim_base_path / "input/alice/sim/2020/LHC20g4/12/296191/AOD/001/aod_archive.zip#AliAOD.root",
    ],
}
# For convenience
_collision_system_to_aod_files["embed_pythia-PbPb"] = _collision_system_to_aod_files["embed_pythia"]

@attr.define
class AnalysisParameters:
    """Centralize some general track skim validation parameters"""
    reference_analysis_prefixes: Dict[str, str]
    track_skim_loading_data_rename_prefix: Dict[str, str]
    track_skim_convert_data_format_prefixes: Dict[str, str]
    comparison_prefixes: List[str]
    min_jet_pt_by_R_and_prefix: Dict[float, Dict[str, float]]
    pt_hat_bin: Optional[int] = None

# Stores the parameters together. Organizing them this way somehow seems to make sense.
# Hard coding them is generally less than ideal, but they're supposed to be fixed
# parameters for the validation, so it makes sense in this context.
_all_analysis_parameters = {
    "pp": AnalysisParameters(
        reference_analysis_prefixes={
            "data": "data",
        },
        track_skim_loading_data_rename_prefix={"data": "data"},
        track_skim_convert_data_format_prefixes={"data": "data"},
        comparison_prefixes=["data"],
        min_jet_pt_by_R_and_prefix={
            0.2: {"data": 5.0},
            0.4: {"data": 5.0},
        }
    ),
    "pythia": AnalysisParameters(
        reference_analysis_prefixes={"data": "data", "true": "matched"},
        track_skim_loading_data_rename_prefix={},
        track_skim_convert_data_format_prefixes={"det_level": "data", "part_level": "true"},
        comparison_prefixes=["data", "true"],
        min_jet_pt_by_R_and_prefix={
            0.2: {"det_level": 20.0},
            0.4: {"det_level": 20.0},
        },
        pt_hat_bin=12,
    ),
    "PbPb": AnalysisParameters(
        reference_analysis_prefixes={
            "data": "data",
        },
        track_skim_loading_data_rename_prefix={"data": "data"},
        track_skim_convert_data_format_prefixes={"data": "data"},
        comparison_prefixes=["data"],
        min_jet_pt_by_R_and_prefix={
            0.2: {"data": 10.0},
            0.4: {"data": 20.0},
        },
    ),
    "embed_pythia": AnalysisParameters(
        reference_analysis_prefixes={"hybrid": "data", "true": "matched", "det_level": "detLevel"},
        # NOTE: This field is not meaningful for embedding
        track_skim_loading_data_rename_prefix={},
        track_skim_convert_data_format_prefixes={"hybrid": "hybrid", "det_level": "det_level", "part_level": "true"},
        comparison_prefixes=["hybrid", "det_level", "true"],
        min_jet_pt_by_R_and_prefix={
            0.2: {"hybrid": 10.0},
            0.4: {"hybrid": 20.0},
        },
        pt_hat_bin=12,
    ),
}


@attr.define
class TrackSkimValidationFilenames:
    """Helper to generate relevant filenames"""
    base_path: Path
    filename_type: str
    collision_system: str
    jet_R: float
    iterative_splittings: bool

    def _label(self, extra_collision_system_label: str = "") -> str:
        collision_system = self.collision_system
        if extra_collision_system_label:
            collision_system = f"{collision_system}-{extra_collision_system_label}"
        return f"{collision_system}__jet_R{round(self.jet_R*100):03}"

    def analysis_output(self, extra_collision_system_label: str = "") -> Path:
        return (
            self.base_path / self.filename_type / f"AnalysisResults__{self._label(extra_collision_system_label)}.root"
        )

    def parquet_output(self, extra_collision_system_label: str = "") -> Path:
        return (
            self.base_path
            / self.filename_type
            / f"AnalysisResults__{self._label(extra_collision_system_label)}.parquet"
        )

    def skim(self, extra_collision_system_label: str = "") -> Path:
        iterative_splittings_label = "iterative" if self.iterative_splittings else "recursive"
        return (
            self.base_path
            / self.filename_type
            / f"skim__{self._label(extra_collision_system_label)}__{iterative_splittings_label}_splittings.root"
        )


def _check_for_alice_input_files(input_files: Sequence[Path]) -> List[bool]:
    """Check for whether input ALICE data files exist.

    Supports "#" in filenames for compressed archives.
    """
    missing_files = []
    for input_file in input_files:
        file_to_check = input_file
        input_file_str = str(input_file)
        if "#" in input_file_str:
            file_to_check = Path(input_file_str[:input_file_str.find("#")])
        missing_files.append(not file_to_check.exists())

    return missing_files


def _aliphysics_to_analysis_results(  # noqa: C901
    collision_system: str, jet_R: float, input_files: Sequence[Path], validation_mode: bool = True, filename_to_rename_output_to: Optional[Path] = None,
    allow_multiple_executions_of_run_macro: bool = True,
    write_logs_to_file: bool = False,
) -> Path:
    """Helper to execute run macro

    NOTE:
        If ROOT is executed multiple times in one process, it will segfault. I suppose it's
        probably because it tries to load identical run macros multiple times. But in general,
        running multiple times is always going to be risky. So to avoid this, we add an option
        to execute in a subprocess, which avoids this issue. This option is the default.

    Args:
        collision_system: Collision system
        jet_R: Jet R
        input_files: Input files to process
        validation_mode: If True, enable validation mode
        filename_to_rename_output_to: Full path we should rename `AnalysisResults.root` to.
        allow_multiple_executions_of_run_macro: Allow ROOT to be executed more than once.
            See note above. Default: True
        write_logs_to_file: If True, write the subprocess logs to file. This can be useful
            if the logs are so long that you can't manage to scroll all the back through them.
            However, that also means that the logs are rather large, so better not to always
            write them. Default: False
    """
    # 0th, a warning / reminder. It's a little obnoxious to print this every time, but
    # it seems to be easy to forget, so I would rather be super explicit to try to help
    # my future self
    logger.warning(
        "You need a specialize AliPhysics build which fixes the fastjet area"
        " random seed! Without it, this won't work! The branch is available here:"
        " https://github.com/raymondEhlers/AliPhysics/tree/trackSkimValidation"
    )
    # First, validate input files
    # They might be missing since they're too large to store in the repo
    missing_files = _check_for_alice_input_files(input_files=input_files)
    if any(missing_files):
        raise RuntimeError(
            "Cannot generate AliPhysics reference due to missing inputs files."
            f" Missing: {[f for f, missing in zip(input_files, missing_files) if missing]}"
        )

    # Need this to be available to use the run macro
    # Strictly speaking, we need AliPHysics, so we then try to grab a value from AliPhysics
    ROOT = pytest.importorskip("ROOT")  # noqa: F841
    try:
        ROOT.AliAnalysisTaskSE
    except AttributeError:
        pytest.skip(
            "Need AliPhysics for generating reference file, but it appears that you only have ROOT."
            " Please check your configuration to ensure that AliPhysics is available"
        )

    # Select a large enough number that we'll exhaust any given input files
    n_events = 500_000
    # Further parameters
    optional_kwargs: Dict[str, Any] = {}
    if collision_system == "embed_pythia":
        _analysis_parameters = _all_analysis_parameters[collision_system]
        missing_files = _check_for_alice_input_files(input_files=_collision_system_to_aod_files["embed_pythia-pythia"])
        if any(missing_files):
            raise RuntimeError(
                "Cannot generate AliPhysics reference due to missing embedding inputs files."
                f" Missing: {[f for f, missing in zip(input_files, missing_files) if missing]}"
            )
        optional_kwargs.update(
            {
                "embed_input_files": _collision_system_to_aod_files["embed_pythia-pythia"],
                # NOTE: This implicitly encodes the period. In practice, it only matters for the event selection,
                #       but it shouldn't be forgot if later changes are made.
                "embedding_helper_config_filename": _track_skim_base_path / "input" / "embeddingHelper_LHC18_LHC20g4_kSemiCentral.yaml",
                "embedding_pt_hat_bin": _analysis_parameters.pt_hat_bin,
            }
        )

    if allow_multiple_executions_of_run_macro:
        # See the note in the docstring for why we bother with this.
        # It lets us work around ROOT issues
        import subprocess
        args = [
            "python3",
            "-m", "mammoth.hardest_kt.run_macro",
            "-c", f"{collision_system}",
            "-R", f"{jet_R}",
            "--validation-mode",
            "--input-files", f"{' '.join([str(_f) for _f in input_files])}",
            "--n-events", f"{n_events}",
        ]
        if "embed_input_files" in optional_kwargs:
            args.extend([
                "--embed-input-files", f"{' '.join([str(_f) for _f in optional_kwargs['embed_input_files']])}"
            ])
        if "embedding_helper_config_filename" in optional_kwargs:
            args.extend([
                "--embedding-helper-config-filename", str(optional_kwargs["embedding_helper_config_filename"])
            ])
        if "embedding_pt_hat_bin" in optional_kwargs:
            args.extend([
                "--embedding-pt-hat-bin", str(optional_kwargs["embedding_pt_hat_bin"])
            ])
        try:
            subprocess_result = subprocess.run(args, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.info(f"stdout: {e.stdout.decode()}")
            logger.info(f"stderr: {e.stderr.decode()}")
            raise RuntimeError("Failed to execute run macro in subprocess") from e
        # Include as debug info in case we need to compare the reclustering, etc
        logger.info(f"stdout: {subprocess_result.stdout.decode()}")
        logger.info(f"stderr: {subprocess_result.stderr.decode()}")
        # Further possible help by writing the logs to file (if enabled)
        if write_logs_to_file and filename_to_rename_output_to:
            with open(filename_to_rename_output_to.with_suffix(".stdout"), "w") as f_stdout:
                f_stdout.write(subprocess_result.stdout.decode())
            with open(filename_to_rename_output_to.with_suffix(".stderr"), "w") as f_stderr:
                f_stderr.write(subprocess_result.stderr.decode())
    else:
        run_macro.run(
            analysis_mode=collision_system,
            jet_R=jet_R,
            validation_mode=validation_mode,
            input_files=input_files,
            n_events=n_events,
            **optional_kwargs
        )
    # Next, we need to rename the output
    output_file = Path("AnalysisResults.root")
    if filename_to_rename_output_to:
        output_file.rename(
            filename_to_rename_output_to
        )

    return output_file


def _reference_aliphysics_tree_name(collision_system: str, jet_R: float, dyg_task: bool = True) -> str:
    """Determine the AliPhysics reference task tree name"""
    _jet_labels = {
        "pp": "Jet",
        "pythia": "Jet",
        "PbPb": "Jet",
        "embed_pythia": "hybridLevelJets",
    }
    _tags = {
        "pp": "_RawTree_Data_NoSub_Incl",
        "pythia": "Tree_PythiaDef_NoSub_Incl",
        # "PbPb": "ConstSub_RawTree_Data_ConstSub_Incl",
        "PbPb": "ConstSub_RawTree_Data_EventSub_Incl",
        "embed_pythia": "ConstSub_RawTree_EventSub_Incl",
    }
    return f"AliAnalysisTaskJetDynamicalGrooming_{_jet_labels[collision_system]}_AKTChargedR{round(jet_R*100):03}_tracks_pT0150_E_scheme{_tags[collision_system]}"


def _get_scale_factors_for_test() -> Dict[int, float]:
    # NOTE: This assumes that the validation uses LHC20g4_AOD. However, we do this consistently,
    #       so that's a reasonable assumption.
    # NOTE: These scale factors need to be determined externally. These were extracted separately
    #       from the main analysis. In principle, the scale factors for this validation just need to
    #       be consistent, but since we have them available, we may as well use them.
    scale_factors = analysis_objects.read_extracted_scale_factors(
        path=_track_skim_base_path / "input" / "LHC20g4_AOD_2640_scale_factors.yaml"
    )
    return scale_factors


@attr.define
class ConvertTreeToParquetArguments:
    """Trivial class to help organize arguments for converting AliPhysics output trees to parquet files."""

    prefixes: List[str]
    branches: List[str]
    prefix_branches: List[str]


def _dyg_analysis_results_to_parquet(filename: Path, collision_system: str, jet_R: float) -> Path:
    """Convert DyG AnalysisResults to parquet"""
    # Essentially porting some parsl functionality, so trying to keep this as simple as possible
    # Shared between all outputs
    _prefix_branches = [
        "{prefix}.fJetPt",
        "{prefix}.fJetConstituents.fPt",
        "{prefix}.fJetConstituents.fEta",
        "{prefix}.fJetConstituents.fPhi",
        "{prefix}.fJetConstituents.fID",
        "{prefix}.fJetSplittings.fKt",
        "{prefix}.fJetSplittings.fDeltaR",
        "{prefix}.fJetSplittings.fZ",
        "{prefix}.fJetSplittings.fParentIndex",
        "{prefix}.fSubjets.fPartOfIterativeSplitting",
        "{prefix}.fSubjets.fSplittingNodeIndex",
        "{prefix}.fSubjets.fConstituentIndices",
    ]
    _args = {
        "pp": ConvertTreeToParquetArguments(
            prefixes=list(_all_analysis_parameters[collision_system].reference_analysis_prefixes.values()),
            branches=[],
            prefix_branches=_prefix_branches,
        ),
        "pythia": ConvertTreeToParquetArguments(
            prefixes=list(_all_analysis_parameters[collision_system].reference_analysis_prefixes.values()),
            branches=["pt_hard", "pt_hard_bin"],
            prefix_branches=_prefix_branches,
        ),
        "PbPb": ConvertTreeToParquetArguments(
            prefixes=list(_all_analysis_parameters[collision_system].reference_analysis_prefixes.values()),
            branches=[],
            prefix_branches=_prefix_branches,
        ),
        "embed_pythia": ConvertTreeToParquetArguments(
            prefixes=list(_all_analysis_parameters[collision_system].reference_analysis_prefixes.values()),
            branches=[],
            prefix_branches=_prefix_branches,
        ),
    }
    arguments = _args[collision_system]

    _, output_filename = analysis_jet_substructure.convert_tree_to_parquet(
        filename=filename,
        tree_name=_reference_aliphysics_tree_name(collision_system=collision_system, jet_R=jet_R),
        prefixes=arguments.prefixes,
        branches=arguments.branches,
        prefix_branches=arguments.prefix_branches,
    )
    return output_filename

def _generate_track_skim_task_parquet_outputs_for_embedding(
    jet_R: float,
    reference_filenames: TrackSkimValidationFilenames,
    track_skim_filenames: TrackSkimValidationFilenames,
    collision_system_to_generate: str
) -> None:
    if not reference_filenames.analysis_output(extra_collision_system_label=collision_system_to_generate).exists():
        # Here, we're running pythia, and it should be treated as such.
        # We just label it as "embed_pythia-pythia" to denote that it should be used for embedding
        # (same as we label the PbPb as "embed_pythia" when we run the embedding run macro, even though
        # the track skim just sees the PbPb).
        _aliphysics_to_analysis_results(
            collision_system=collision_system_to_generate,
            jet_R=jet_R,
            input_files=_collision_system_to_aod_files[f"embed_pythia-{collision_system_to_generate}"],
            filename_to_rename_output_to=reference_filenames.analysis_output(extra_collision_system_label=collision_system_to_generate),
        )
        # And then extract the corresponding parquet
    if not track_skim_filenames.parquet_output(extra_collision_system_label=collision_system_to_generate).exists():
        _track_skim_to_parquet(
            input_filename=reference_filenames.analysis_output(extra_collision_system_label=collision_system_to_generate),
            output_filename=track_skim_filenames.parquet_output(extra_collision_system_label=collision_system_to_generate),
            collision_system=collision_system_to_generate,
        )


def _track_skim_to_parquet(input_filename: Path, output_filename: Path, collision_system: str) -> None:
    """Convert track skim task output to parquet"""
    source = io_track_skim.FileSource(
        filename=input_filename,
        collision_system=collision_system,
    )
    arrays = next(source.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE))
    io_track_skim.write_to_parquet(
        arrays=arrays,
        filename=output_filename,
    )


@pytest.mark.parametrize("jet_R", [0.2, 0.4])
@pytest.mark.parametrize("collision_system", ["pp", "pythia", "PbPb", "embed_pythia"])
def test_track_skim_validation(  # noqa: C901
    caplog: Any, jet_R: float, collision_system: str, iterative_splittings: bool = True,
    write_aliphysics_reference_logs_to_file: bool = False,
) -> None:
    # NOTE: There's some inefficiency since we store the same track skim info with the
    #       R = 0.2 and R = 0.4 outputs. However, it's much simpler conceptually, so we
    #       just accept it
    # Setup
    caplog.set_level(logging.INFO)
    # But allow us to debug mammoth more precisely
    caplog.set_level(logging.DEBUG, logger="mammoth")

    reference_filenames = TrackSkimValidationFilenames(
        base_path=_track_skim_base_path,
        filename_type="reference",
        collision_system=collision_system,
        jet_R=jet_R,
        iterative_splittings=iterative_splittings,
    )

    """
    For the AliPhysics reference:

    1. Run AliPhysics run macro
    2. Convert DyG output to parquet
    3. Convert DyG parquet to flat tree

    NOTE: When the run macro is executed, it will also run the track skim task
          (except for the embedding, which has to run separately - see below!)
    NOTE: You need a specialize AliPhysics build which fixes the fastjet area
          random seed! Without it, this won't work! The branch is available here:
          https://github.com/raymondEhlers/AliPhysics/tree/trackSkimValidation .
          The patch files are also available in `projects/track_skim_validation/AliPhysics_patches`.
          (they are identical, but I wanted a way to store them with the rest of
          the relevant code, so I generated the patch files from the branch).
    """
    generate_aliphysics_results = False
    convert_aliphysics_to_parquet = False
    skim_aliphysics_parquet = False
    # Try to run the minimal number of preparation steps for the AliPhysics reference
    # Ideally, these files are already available, but we may need to regenerate them
    # from time to time.
    if not reference_filenames.skim().exists():
        if not reference_filenames.parquet_output().exists():
            if not reference_filenames.analysis_output().exists():
                logger.info(f"{reference_filenames.analysis_output()}")
                generate_aliphysics_results = True
            else:
                convert_aliphysics_to_parquet = True
        else:
            skim_aliphysics_parquet = True
    logger.info(f"{generate_aliphysics_results=}, {convert_aliphysics_to_parquet=}, {skim_aliphysics_parquet=}")

    # Now, actually run the tasks
    # Step 1
    if generate_aliphysics_results:
        _aliphysics_to_analysis_results(
            collision_system=collision_system,
            jet_R=jet_R,
            input_files=_collision_system_to_aod_files[collision_system],
            filename_to_rename_output_to=reference_filenames.analysis_output(),
            write_logs_to_file=write_aliphysics_reference_logs_to_file,
        )

    # Step 2
    if convert_aliphysics_to_parquet or generate_aliphysics_results:
        # We need to generate the parquet if we've just executed the run macro
        _dyg_analysis_results_to_parquet(
            filename=reference_filenames.analysis_output(), collision_system=collision_system, jet_R=jet_R
        )

    # Step 3
    if skim_aliphysics_parquet or convert_aliphysics_to_parquet or generate_aliphysics_results:
        scale_factors = _get_scale_factors_for_test()
        if collision_system != "embed_pythia":
            res = skim_to_flat_tree.calculate_data_skim(
                input_filename=reference_filenames.parquet_output(),
                collision_system=collision_system,
                iterative_splittings=iterative_splittings,
                prefixes=_all_analysis_parameters[collision_system].reference_analysis_prefixes,
                jet_R=jet_R,
                output_filename=reference_filenames.skim(),
                scale_factors=scale_factors,
            )
        else:
            res = skim_to_flat_tree.calculate_embedding_skim(
                input_filename=reference_filenames.parquet_output(),
                iterative_splittings=iterative_splittings,
                prefixes=_all_analysis_parameters[collision_system].reference_analysis_prefixes,
                scale_factors=scale_factors,
                # We store a mocked up `config.yaml` so we don't have to change the interface further
                train_directory=_track_skim_base_path / "reference",
                jet_R=jet_R,
                output_filename=reference_filenames.skim(),
            )
        if not res[0]:
            raise ValueError(f"Failed to generate reference for {collision_system}, {jet_R}")

    """
    For mammoth:
    1. Convert track skim to parquet
       - For pp, pythia, and PbPb, we already have the appropriate track skim outputs from the AliPhysics run above
       - However, we don't run the track skim task when embedding since the track skim is designed to only
         store data from one source, it could only record PbPb or pythia separately. Plus, if the event selection
         of the PbPb used in the embedding was somehow different, then our validation (which uses the PbPb extracted
         separately) could be wrong. All of which is to say, we should extract the PbPb and pythia separately
         from the execution of the embedding run macro.
          - In principle, we could take the existing pythia and PbPb processed in the standalone pythia and PbPb
            cases. However, we don't do this because they may or may not be the right period (eg. the pythia shouldn't
            be the same period).
        - In summary, we have to have six runs of the run macros in total:
            - pp, pythia, PbPb: Run macro produces DyG, track_skim outputs
            - embed_pythia: Run macro produces DyG outputs
            - Separate PbPb, pythia for embed_pythia: Run macro produces track skim outputs (labeled as
              embed_pythia-pythia and embed_pythia-PbPb)
        - From each of those outputs, we then convert as appropriate to parquet
    2. Convert track skim output to parquet
    3. Analyze parquet track skim to generate flat tree
    """
    track_skim_filenames = TrackSkimValidationFilenames(
        base_path=_track_skim_base_path,
        filename_type="track_skim",
        collision_system=collision_system,
        jet_R=jet_R,
        iterative_splittings=iterative_splittings,
    )

    # Here, we want to execute for two possible cases:
    # 1. The relevant AliPhysics run macro was already executed
    # 2. If any relevant analysis outputs are missing.  This conditions is complicated because
    #    they're different from embedding vs the other cases, so we end up splitting these into
    #    two separate conditions, which means there are three total
    if (
        generate_aliphysics_results
        or (
            collision_system == "embed_pythia"
            and (
                not track_skim_filenames.analysis_output(extra_collision_system_label="pythia").exists()
                or not track_skim_filenames.analysis_output(extra_collision_system_label="PbPb").exists()
            )
        )
        or (collision_system != "embed_pythia" and not track_skim_filenames.parquet_output().exists())
    ):
        if collision_system != "embed_pythia":
            # Step 2, which depends on the AliPhysics run macro for the reference running first
            # Convert track skim to parquet
            _track_skim_to_parquet(
                input_filename=reference_filenames.analysis_output(),
                output_filename=track_skim_filenames.parquet_output(),
                collision_system=collision_system,
            )
        else:
            # Step 1 + 2
            # Generate outputs for pythia
            _generate_track_skim_task_parquet_outputs_for_embedding(
                jet_R=jet_R,
                reference_filenames=reference_filenames,
                track_skim_filenames=track_skim_filenames,
                collision_system_to_generate="pythia"
            )
            # And then for PbPb
            _generate_track_skim_task_parquet_outputs_for_embedding(
                jet_R=jet_R,
                reference_filenames=reference_filenames,
                track_skim_filenames=track_skim_filenames,
                collision_system_to_generate="PbPb"
            )

    # Now we can finally analyze the track_skim
    # We always want to run this, since this is what we're validating
    # Need to grab relevant analysis parameters
    _run_macro_default_analysis_parameters = run_macro.default_analysis_parameters[collision_system]
    _analysis_parameters = _all_analysis_parameters[collision_system]
    # Validate min jet pt
    _min_jet_pt_from_run_macro = _run_macro_default_analysis_parameters.grooming_jet_pt_threshold[jet_R]
    _values_align = [_min_jet_pt_from_run_macro == v for v in _analysis_parameters.min_jet_pt_by_R_and_prefix[jet_R].values()]
    if not all(_values_align):
        raise RuntimeError(
            "Misalignment between min pt cuts!"
            f" min jet pt from run macro: {_min_jet_pt_from_run_macro}"
            f", min jet pt dict: {_analysis_parameters.min_jet_pt_by_R_and_prefix[jet_R]}"
        )

    scale_factors = _get_scale_factors_for_test()
    # The skim task will skip the calculation if the output file already exists.
    # However, that's exactly what we want to create, so we intentionally remove it here
    # to ensure that the test actually runs.
    track_skim_filenames.skim().unlink(missing_ok=True)
    if collision_system != "embed_pythia":
        result = analysis_track_skim_to_flat_tree.hardest_kt_data_skim(
            input_filename=track_skim_filenames.parquet_output(),
            collision_system=collision_system,
            jet_R=jet_R,
            min_jet_pt=_analysis_parameters.min_jet_pt_by_R_and_prefix[jet_R],
            iterative_splittings=iterative_splittings,
            loading_data_rename_prefix=_analysis_parameters.track_skim_loading_data_rename_prefix,
            convert_data_format_prefixes=_analysis_parameters.track_skim_convert_data_format_prefixes,
            output_filename=track_skim_filenames.skim(),
            scale_factors=scale_factors,
            pt_hat_bin=_analysis_parameters.pt_hat_bin,
            validation_mode=True,
        )
    else:
        # Help out typing...
        assert _analysis_parameters.pt_hat_bin is not None

        signal_filename = track_skim_filenames.parquet_output(extra_collision_system_label="pythia")
        background_filename = track_skim_filenames.parquet_output(extra_collision_system_label="PbPb")
        result = analysis_track_skim_to_flat_tree.hardest_kt_embedding_skim(
            collision_system=collision_system,
            # Repeat the signal file to ensure that we have enough events to exhaust the background
            signal_input=[signal_filename, signal_filename, signal_filename],
            background_input=background_filename,
            jet_R=jet_R,
            min_jet_pt=_analysis_parameters.min_jet_pt_by_R_and_prefix[jet_R],
            iterative_splittings=iterative_splittings,
            output_filename=track_skim_filenames.skim(),
            convert_data_format_prefixes=_analysis_parameters.track_skim_convert_data_format_prefixes,
            scale_factor=scale_factors[_analysis_parameters.pt_hat_bin],
            background_subtraction={"r_max": 0.25},
            det_level_artificial_tracking_efficiency=1.0,
            validation_mode=True,
        )
    if not result[0]:
        raise ValueError(f"Skim failed for {collision_system}, {jet_R}")

    comparison_result = substructure_comparison_tools.compare_flat_substructure(
        collision_system=collision_system,
        jet_R=jet_R,
        prefixes=_analysis_parameters.comparison_prefixes,
        standard_filename=reference_filenames.skim(),
        track_skim_filename=track_skim_filenames.skim(),
        base_output_dir=_track_skim_base_path / "plot",
        track_skim_validation_mode=True,
        assert_false_on_failed_comparison_for_debugging_during_testing=True,
    )
    assert comparison_result, "Validation failed during comparison"

