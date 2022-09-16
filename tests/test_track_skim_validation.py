"""Comparison between standard analysis and track skim

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBNL/UCB
"""

import logging
import pprint
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import attr
import awkward as ak
import boost_histogram as bh
import mammoth.helpers
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
import pytest
import uproot
from pachyderm import binned_data, plot as pb

from mammoth.framework import sources
from mammoth.framework.io import track_skim as io_track_skim
from mammoth.framework.analysis import jet_substructure as analysis_jet_substructure, objects as analysis_objects
from mammoth.hardest_kt import analysis_track_skim_to_flat_tree, run_macro, skim_to_flat_tree

pachyderm.plot.configure()


logger = logging.getLogger(__name__)

_here = Path(__file__).parent
_track_skim_base_path = _here / "track_skim_validation"


def _aliphysics_to_analysis_results(
    collision_system: str, collision_system_label: str, jet_R: float, input_files: Sequence[Path]
) -> Path:
    # First, validate input files
    # They might be missing since they're too large to store in the repo
    missing_files = [not f.exists() for f in input_files]
    if missing_files:
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

    optional_kwargs = {}
    if collision_system == "embed_pythia":
        optional_kwargs.update(
            {
                "embed_input_files": _collision_system_to_aod_files["embed_pythia-pythia"],
            }
        )
    run_macro.run(
        analysis_mode=collision_system, jet_R=jet_R, validation_mode=True, input_files=input_files, **optional_kwargs
    )
    # Next, we need to rename the output
    output_file = Path("AnalysisResults.root")
    output_file.rename(
        _track_skim_base_path
        / "reference"
        / f"AnalysisResults_{collision_system_label}_jet_R_{round(jet_R*100):03}.root"
    )

    return output_file


def _reference_aliphysics_tree_name(collision_system: str, jet_R: float, dyg_task: bool = True) -> str:
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


def _analysis_results_to_parquet(filename: Path, collision_system: str, jet_R: float) -> Path:
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


def _track_skim_to_parquet(input_filename: Path, output_filename: Path, collision_system: str) -> None:
    source = io_track_skim.FileSource(
        filename=input_filename,
        collision_system=collision_system,
    )
    arrays = next(source.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE))
    io_track_skim.write_to_parquet(
        arrays=arrays,
        filename=output_filename,
        collision_system=collision_system,
    )


# NOTE: These files are too large to store in the git repo.
#       I've stored them multiple places to attempt to ensure that they're not lost entirely.
#       Since the AliEn path is provided, they can always be retrieved (in principle).
# NOTE: All were stored in archives (see the pachyderm dataset def for the particular archive names),
#       and then were extracted for simplicity.
_collision_system_to_aod_files = {
    "pp": [
        _track_skim_base_path / "input/alice/data/2017/LHC17p/000282343/pass1_FAST/AOD234/0001/AliAOD.root",
        _track_skim_base_path / "input/alice/data/2017/LHC17p/000282343/pass1_FAST/AOD234/0002/AliAOD.root",
        _track_skim_base_path / "input/alice/data/2017/LHC17p/000282343/pass1_FAST/AOD234/0003/AliAOD.root",
    ],
    # Strictly speaking, this should be LHC18b8 to correctly correspond to LHC17pq, but for these
    # purposes, it's fine.
    "pythia": [
        _track_skim_base_path / "input/alice/sim/2020/LHC20g4/12/296191/AOD/001/AliAOD.root",
    ],
    "PbPb": [
        _track_skim_base_path / "input/alice/data/2018/LHC18q/000296550/pass3/AOD252/AOD/001/AliAOD.root",
    ],
    "embed_pythia": [
        _track_skim_base_path / "input/alice/data/2018/LHC18q/000296550/pass3/AOD252/AOD/001/AliAOD.root",
    ],
    "embed_pythia-pythia": [
        _track_skim_base_path / "input/alice/sim/2020/LHC20g4/12/296191/AOD/001/aod_archive.zip#AliAOD.root",
    ],
}


@attr.define
class AnalysisParameters:
    reference_analysis_prefixes: Dict[str, str]
    track_skim_loading_data_rename_prefix: Dict[str, str]
    track_skim_convert_data_format_prefixes: Dict[str, str]
    comparison_prefixes: List[str]
    min_jet_pt_by_prefix: Dict[str, float]
    pt_hat_bin: Optional[int] = None


_all_analysis_parameters = {
    "pp": AnalysisParameters(
        reference_analysis_prefixes={
            "data": "data",
        },
        track_skim_loading_data_rename_prefix={"data": "data"},
        track_skim_convert_data_format_prefixes={"data": "data"},
        comparison_prefixes=["data"],
        min_jet_pt_by_prefix={"data": 5.0},
    ),
    "pythia": AnalysisParameters(
        reference_analysis_prefixes={"data": "data", "true": "matched"},
        track_skim_loading_data_rename_prefix={},
        track_skim_convert_data_format_prefixes={"det_level": "data", "part_level": "true"},
        comparison_prefixes=["data", "true"],
        min_jet_pt_by_prefix={"det_level": 20.0},
        pt_hat_bin=12,
    ),
    "PbPb": AnalysisParameters(
        reference_analysis_prefixes={
            "data": "data",
        },
        track_skim_loading_data_rename_prefix={"data": "data"},
        track_skim_convert_data_format_prefixes={"data": "data"},
        comparison_prefixes=["data"],
        min_jet_pt_by_prefix={"data": 20.0},
    ),
    "embed_pythia": AnalysisParameters(
        reference_analysis_prefixes={"hybrid": "data", "true": "matched", "det_level": "detLevel"},
        # NOTE: This field is not meaningful for embedding
        track_skim_loading_data_rename_prefix={},
        track_skim_convert_data_format_prefixes={"hybrid": "hybrid", "det_level": "det_level", "part_level": "true"},
        comparison_prefixes=["hybrid", "det_level", "true"],
        min_jet_pt_by_prefix={"hybrid": 20.0},
        pt_hat_bin=12,
    ),
}


@attr.define
class TrackSkimValidationFilenames:
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


# TODO: Re-enable 0.2
# TODO: Refactor...
# @pytest.mark.parametrize("jet_R", [0.2, 0.4])
@pytest.mark.parametrize("jet_R", [0.4])
@pytest.mark.parametrize("collision_system", ["pp", "pythia", "PbPb", "embed_pythia"])
def test_track_skim_validation(
    caplog: Any, jet_R: float, collision_system: str, iterative_splittings: bool = True
) -> None:
    # NOTE: There's some inefficiency since we store the same track skim info with the
    #       R = 0.2 and R = 0.4 outputs. However, it's much simpler conceptually, so we
    #       just accept it
    # TODO: The track skim doesn't run in the embedding, so we need to potentially have to
    #       generate those files separately :-(
    # TODO: Make another pass through the comments to figure out what can be updated, refactored, etc
    # Setup
    caplog.set_level(logging.INFO)
    # But keep numba quieter...
    caplog.set_level(logging.INFO, logger="numba")

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
          (except for the embedding - see below!)
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
            collision_system_label=collision_system,
            jet_R=jet_R,
            input_files=_collision_system_to_aod_files[collision_system],
        )

    # Step 2
    if convert_aliphysics_to_parquet or generate_aliphysics_results:
        # We need to generate the parquet if we've just executed the run macro
        _analysis_results_to_parquet(
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
    2. Analyze parquet track skim to generate flat tree
    """
    track_skim_filenames = TrackSkimValidationFilenames(
        base_path=_track_skim_base_path,
        filename_type="track_skim",
        collision_system=collision_system,
        jet_R=jet_R,
        iterative_splittings=iterative_splittings,
    )
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
            # Convert track skim to parquet
            _track_skim_to_parquet(
                input_filename=reference_filenames.analysis_output(),
                output_filename=track_skim_filenames.parquet_output(),
                collision_system=collision_system,
            )
        else:
            # Attempt to execute the run macro if needed for this addition file
            if not reference_filenames.analysis_output(extra_collision_system_label="pythia").exists():
                logger.info(f'{reference_filenames.analysis_output(extra_collision_system_label="pythia")=}')
                # Here, we're running pythia, and it should be treated as such.
                # We just label it as "embed_pythia-pythia" to denote that it should be used for embedding
                # (same as we label the PbPb as "embed_pythia" when we run the embedding run macro, even though
                # the track skim just sees the PbPb).
                _aliphysics_to_analysis_results(
                    collision_system="pythia",
                    collision_system_label="embed_pythia-pythia",
                    jet_R=jet_R,
                    input_files=_collision_system_to_aod_files["embed_pythia-pythia"],
                )
                # And then extract the corresponding parquet
            if not track_skim_filenames.parquet_output(extra_collision_system_label="pythia").exists():
                logger.info("Parquet pythia")
                _track_skim_to_parquet(
                    input_filename=reference_filenames.analysis_output(extra_collision_system_label="pythia"),
                    output_filename=track_skim_filenames.parquet_output(extra_collision_system_label="pythia"),
                    collision_system="pythia",
                )
            # Attempt to execute the run macro if needed for this addition file
            if not reference_filenames.analysis_output(extra_collision_system_label="PbPb").exists():
                # We need to do the same for the background, but instead we will treat it as PbPb
                _aliphysics_to_analysis_results(
                    collision_system="PbPb",
                    collision_system_label="embed_pythia-PbPb",
                    jet_R=jet_R,
                    # We want the embed_pythia files, which are the (background) PbPb files
                    input_files=_collision_system_to_aod_files["embed_pythia"],
                )
            if not track_skim_filenames.parquet_output(extra_collision_system_label="PbPb").exists():
                # And then extract the corresponding parquet
                _track_skim_to_parquet(
                    input_filename=reference_filenames.analysis_output(extra_collision_system_label="PbPb"),
                    output_filename=track_skim_filenames.parquet_output(extra_collision_system_label="PbPb"),
                    collision_system="PbPb",
                )

    import warnings

    warnings.filterwarnings("error")

    # Now we can finally analyze the track_skim
    # We always want to run this, since this is what we're validating
    # Need to grab relevant analysis parameters
    _run_macro_default_analysis_parameters = run_macro.default_analysis_parameters[collision_system]
    _analysis_parameters = _all_analysis_parameters[collision_system]
    # Validate min jet pt
    _min_jet_pt_from_run_macro = _run_macro_default_analysis_parameters.grooming_jet_pt_threshold
    _values_align = [_min_jet_pt_from_run_macro == v for v in _analysis_parameters.min_jet_pt_by_prefix.values()]
    if not all(_values_align):
        raise RuntimeError(
            "Misalignment between min pt cuts!"
            f" min jet pt from run macro: {_min_jet_pt_from_run_macro}"
            f", min jet pt dict: {_analysis_parameters.min_jet_pt_by_prefix}"
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
            min_jet_pt=_analysis_parameters.min_jet_pt_by_prefix,
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
            min_jet_pt=_analysis_parameters.min_jet_pt_by_prefix,
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

    compare_flat_substructure(
        collision_system=collision_system,
        prefixes=_analysis_parameters.comparison_prefixes,
        standard_filename=reference_filenames.skim(),
        track_skim_filename=track_skim_filenames.skim(),
        base_output_dir=_track_skim_base_path / "plot",
    )
    # assert False


@attr.s
class Input:
    name: str = attr.ib()
    arrays: ak.Array = attr.ib()
    attribute: str = attr.ib()


def arrays_to_hist(
    arrays: ak.Array, attribute: str, axis: bh.axis.Regular = bh.axis.Regular(30, 0, 150)
) -> binned_data.BinnedData:
    bh_hist = bh.Histogram(axis, storage=bh.storage.Weight())
    bh_hist.fill(ak.flatten(arrays[attribute], axis=None))

    return binned_data.BinnedData.from_existing_data(bh_hist)


def plot_attribute_compare(
    other: Input,
    mine: Input,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    axis: bh.axis.Regular = bh.axis.Regular(30, 0, 150),
    normalize: bool = False,
) -> None:
    # Plot
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    other_hist = arrays_to_hist(arrays=other.arrays, attribute=other.attribute, axis=axis)
    mine_hist = arrays_to_hist(arrays=mine.arrays, attribute=mine.attribute, axis=axis)
    # Normalize
    if normalize:
        other_hist /= np.sum(other_hist.values)
        mine_hist /= np.sum(mine_hist.values)

    ax.errorbar(
        other_hist.axes[0].bin_centers,
        other_hist.values,
        xerr=other_hist.axes[0].bin_widths / 2,
        yerr=other_hist.errors,
        label=other.name,
        linestyle="",
        alpha=0.8,
    )
    ax.errorbar(
        mine_hist.axes[0].bin_centers,
        mine_hist.values,
        xerr=mine_hist.axes[0].bin_widths / 2,
        yerr=mine_hist.errors,
        label=mine.name,
        linestyle="",
        alpha=0.8,
    )

    ratio = mine_hist / other_hist
    ax_ratio.errorbar(
        ratio.axes[0].bin_centers, ratio.values, xerr=ratio.axes[0].bin_widths / 2, yerr=ratio.errors, linestyle=""
    )
    logger.info(f"ratio sum: {np.sum(ratio.values)}")
    logger.info(f"other: {np.sum(other_hist.values)}")
    logger.info(f"mine: {np.sum(mine_hist.values)}")

    # Apply the PlotConfig
    plot_config.apply(fig=fig, axes=[ax, ax_ratio])

    # filename = f"{plot_config.name}_{jet_pt_bin}{grooming_methods_filename_label}_{identifiers}_iterative_splittings"
    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def compare_flat_substructure(
    collision_system: str,
    prefixes: Sequence[str],
    standard_filename: Path,
    track_skim_filename: Path,
    base_output_dir: Path = Path("comparison/track_skim"),
) -> None:
    # standard_tree_name = "AliAnalysisTaskJetHardestKt_Jet_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_Data_ConstSub_Incl"
    # if collision_system == "pp":
    #    standard_tree_name = "AliAnalysisTaskJetHardestKt_Jet_AKTChargedR040_tracks_pT0150_E_scheme_RawTree_Data_NoSub_Incl"
    # if collision_system == "pythia":
    #    standard_tree_name = "AliAnalysisTaskJetHardestKt_Jet_AKTChargedR040_tracks_pT0150_E_schemeTree_PythiaDef_NoSub_Incl"
    standard_tree_name = "tree"
    standard = uproot.open(standard_filename)[standard_tree_name].arrays()
    track_skim = uproot.open(track_skim_filename)["tree"].arrays()
    logger.info(f"standard.type: {standard.type}")
    logger.info(f"track_skim.type: {track_skim.type}")

    # For whatever reason, the sorting of the jets is inconsistent for embedding compared to all other datasets.
    # So we just apply a mask here to swap the one event where we have two jets.
    # NOTE: AliPhysics is actually the one that gets the sorting wrong here...
    # NOTE: This is a super specialized thing, but better to do it here instead of messing around with
    #       the actual mammoth analysis code.
    if collision_system == "embed_pythia":
        # NOTE: I derived this mask by hand. It swaps index -2 and -3 (== swapping index 15 and 16)
        #       It can be double checked by looking at the jet pt. The precision makes
        #       it quite obvious which should go with which.
        reorder_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15, 17]
        track_skim = track_skim[reorder_mask]
        # NOTE: ***********************************************************************************
        #       The canonical file actually did go through the steps of making the order in mammoth
        #       match AliPhysics by turning off sorting. But since we're more likely to be
        #       testing in the future with new files to do validation, it's better that we apply
        #       this remapping here.
        #       ***********************************************************************************

    output_dir = base_output_dir / collision_system
    output_dir.mkdir(parents=True, exist_ok=True)

    all_success = True
    for prefix in prefixes:
        logger.info(f"Comparing prefix '{prefix}'")

        text = f"{collision_system.replace('_', ' ')}: {prefix.replace('_', ' ')}"
        plot_attribute_compare(
            other=Input(arrays=standard, attribute=f"{prefix}_jet_pt", name="Standard"),
            mine=Input(arrays=track_skim, attribute=f"{prefix}_jet_pt", name="Track skim"),
            plot_config=pb.PlotConfig(
                name=f"{prefix}_jet_pt",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label="Prob.",
                                log=True,
                                font_size=22,
                            ),
                        ],
                        text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                        legend=pb.LegendConfig(location="center right", anchor=(0.985, 0.52), font_size=22),
                    ),
                    # Data ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "x",
                                label=r"$p_{\text{T,ch jet}}$ (GeV/$c$)",
                                font_size=22,
                            ),
                            pb.AxisConfig(
                                "y",
                                label=r"Track skim/Standard",
                                range=(0.6, 1.4),
                                font_size=22,
                            ),
                        ],
                    ),
                ],
                figure=pb.Figure(edge_padding=dict(left=0.13, bottom=0.115)),
            ),
            output_dir=output_dir,
            axis=bh.axis.Regular(50, 0, 100),
            normalize=True,
        )
        standard_jet_pt = standard[f"{prefix}_jet_pt"]
        track_skim_jet_pt = track_skim[f"{prefix}_jet_pt"]

        # Sometimes it's useful to start at this, but sometimes it's just overwhelming, so uncomment as necessary
        # logger.info(f"standard_jet_pt: {standard_jet_pt.to_list()}")
        # logger.info(f"track_skim_jet_pt: {track_skim_jet_pt.to_list()}")

        try:
            all_close_jet_pt = np.allclose(ak.to_numpy(standard_jet_pt), ak.to_numpy(track_skim_jet_pt))

            logger.info(f"jet_pt all close? {all_close_jet_pt}")
            # import IPython; IPython.embed()
            if not all_close_jet_pt:
                logger.info("jet pt")
                _arr = ak.zip({"s": standard_jet_pt, "t": track_skim_jet_pt})
                logger.info(pprint.pformat(_arr.to_list()))
                is_not_close_jet_pt = np.where(
                    ~np.isclose(ak.to_numpy(standard_jet_pt), ak.to_numpy(track_skim_jet_pt))
                )
                logger.info(f"Indices where not close: {is_not_close_jet_pt}")
                all_success = False
        except ValueError as e:
            logger.exception(e)
            all_success = False

        for grooming_method in ["dynamical_kt", "soft_drop_z_cut_02"]:
            logger.info(f'Plotting method "{grooming_method}"')
            plot_attribute_compare(
                other=Input(arrays=standard, attribute=f"{grooming_method}_{prefix}_kt", name="Standard"),
                mine=Input(arrays=track_skim, attribute=f"{grooming_method}_{prefix}_kt", name="Track skim"),
                plot_config=pb.PlotConfig(
                    name=f"{grooming_method}_{prefix}_kt",
                    panels=[
                        # Main panel
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label="Prob.",
                                    log=True,
                                    font_size=22,
                                ),
                            ],
                            text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                            legend=pb.LegendConfig(location="center right", anchor=(0.985, 0.52), font_size=22),
                        ),
                        # Data ratio
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "x",
                                    label=r"$k_{\text{T,g}}$ (GeV/$c$)",
                                    font_size=22,
                                ),
                                pb.AxisConfig(
                                    "y",
                                    label=r"Track skim/Standard",
                                    range=(0.6, 1.4),
                                    font_size=22,
                                ),
                            ],
                        ),
                    ],
                    figure=pb.Figure(edge_padding=dict(left=0.13, bottom=0.115)),
                ),
                normalize=True,
                axis=bh.axis.Regular(50, 0, 10),
                output_dir=output_dir,
            )

            standard_kt = standard[f"{grooming_method}_{prefix}_kt"]
            track_skim_kt = track_skim[f"{grooming_method}_{prefix}_kt"]

            # Sometimes it's useful to start at this, but sometimes it's just overwhelming, so uncomment as necessary
            # logger.info(f"standard_kt: {standard_kt.to_list()}")
            # logger.info(f"track_skim_kt: {track_skim_kt.to_list()}")

            try:
                all_close_kt = np.allclose(ak.to_numpy(standard_kt), ak.to_numpy(track_skim_kt), rtol=1e-4)
                logger.info(f"kt all close? {all_close_kt}")
                if not all_close_kt:
                    logger.info("kt")
                    _arr = ak.zip({"s": standard_kt, "t": track_skim_kt})
                    logger.info(pprint.pformat(_arr.to_list()))
                    is_not_close_kt = np.where(~np.isclose(ak.to_numpy(standard_kt), ak.to_numpy(track_skim_kt)))
                    logger.info(f"Indices where not close: {is_not_close_kt}")
                    all_success = False
            except ValueError as e:
                logger.exception(e)
                all_success = False

            plot_attribute_compare(
                other=Input(arrays=standard, attribute=f"{grooming_method}_{prefix}_delta_R", name="Standard"),
                mine=Input(arrays=track_skim, attribute=f"{grooming_method}_{prefix}_delta_R", name="Track skim"),
                plot_config=pb.PlotConfig(
                    name=f"{grooming_method}_{prefix}_delta_R",
                    panels=[
                        # Main panel
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label="Prob.",
                                    log=True,
                                    font_size=22,
                                ),
                            ],
                            text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                            legend=pb.LegendConfig(location="upper left", font_size=22),
                        ),
                        # Data ratio
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "x",
                                    label=r"$R_{\text{g}}$",
                                    font_size=22,
                                ),
                                pb.AxisConfig(
                                    "y",
                                    label=r"Track skim/Standard",
                                    range=(0.6, 1.4),
                                    font_size=22,
                                ),
                            ],
                        ),
                    ],
                    figure=pb.Figure(edge_padding=dict(left=0.13, bottom=0.115)),
                ),
                output_dir=output_dir,
                axis=bh.axis.Regular(50, 0, 0.6),
                normalize=True,
            )
            standard_rg = standard[f"{grooming_method}_{prefix}_delta_R"]
            track_skim_rg = track_skim[f"{grooming_method}_{prefix}_delta_R"]

            # Sometimes it's useful to start at this, but sometimes it's just overwhelming, so uncomment as necessary
            # logger.info(f"standard_zg: {standard_zg.to_list()}")
            # logger.info(f"track_skim_zg: {track_skim_zg.to_list()}")

            try:
                all_close_rg = np.allclose(ak.to_numpy(standard_rg), ak.to_numpy(track_skim_rg), rtol=1e-4)
                logger.info(f"Rg all close? {all_close_rg}")
                if not all_close_rg:
                    logger.info("delta_R")
                    _arr = ak.zip({"s": standard_rg, "t": track_skim_rg})
                    logger.info(pprint.pformat(_arr.to_list()))
                    is_not_close_rg = np.where(~np.isclose(ak.to_numpy(standard_rg), ak.to_numpy(track_skim_rg)))
                    logger.info(f"Indices where not close: {is_not_close_rg}")
                    all_success = False
            except ValueError as e:
                logger.exception(e)
                all_success = False

            # import IPython; IPython.embed()

            # logger.info(f"standard_rg: {standard_rg.to_list()}")
            # logger.info(f"track_skim_rg: {track_skim_rg.to_list()}")

            plot_attribute_compare(
                other=Input(arrays=standard, attribute=f"{grooming_method}_{prefix}_z", name="Standard"),
                mine=Input(arrays=track_skim, attribute=f"{grooming_method}_{prefix}_z", name="Track skim"),
                plot_config=pb.PlotConfig(
                    name=f"{grooming_method}_{prefix}_z",
                    panels=[
                        # Main panel
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label="Prob.",
                                    log=True,
                                    font_size=22,
                                ),
                            ],
                            text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                            legend=pb.LegendConfig(location="upper left", font_size=22),
                        ),
                        # Data ratio
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "x",
                                    label=r"$z_{\text{g}}$",
                                    font_size=22,
                                ),
                                pb.AxisConfig(
                                    "y",
                                    label=r"Track skim/Standard",
                                    range=(0.6, 1.4),
                                    font_size=22,
                                ),
                            ],
                        ),
                    ],
                    figure=pb.Figure(edge_padding=dict(left=0.13, bottom=0.115)),
                ),
                normalize=True,
                axis=bh.axis.Regular(50, 0, 0.5),
                output_dir=output_dir,
            )

            standard_zg = standard[f"{grooming_method}_{prefix}_z"]
            track_skim_zg = track_skim[f"{grooming_method}_{prefix}_z"]

            # Sometimes it's useful to start at this, but sometimes it's just overwhelming, so uncomment as necessary
            # logger.info(f"standard_zg: {standard_zg.to_list()}")
            # logger.info(f"track_skim_zg: {track_skim_zg.to_list()}")

            try:
                all_close_zg = np.allclose(ak.to_numpy(standard_zg), ak.to_numpy(track_skim_zg))
                logger.info(f"zg all close? {all_close_zg}")
                if not all_close_zg:
                    logger.info("z")
                    _arr = ak.zip({"s": standard_zg, "t": track_skim_zg})
                    logger.info(pprint.pformat(_arr.to_list()))
                    all_success = False
            except ValueError as e:
                logger.exception(e)
                all_success = False

    assert all_success is True


def run(collision_system: str, prefixes: Optional[Sequence[str]] = None) -> None:
    if prefixes is None:
        prefixes = ["data"]
    mammoth.helpers.setup_logging()
    logger.info(f"Running {collision_system} with prefixes {prefixes}")
    path_to_mammoth = Path(mammoth.helpers.__file__).parent.parent
    standard_base_filename = "AnalysisResults"
    if collision_system == "pythia":
        standard_base_filename += ".12"
    compare_flat_substructure(
        collision_system=collision_system,
        prefixes=prefixes,
        standard_filename=path_to_mammoth
        / f"projects/framework/{collision_system}/1/skim/{standard_base_filename}.repaired.00_iterative_splittings.root",
        track_skim_filename=path_to_mammoth / f"projects/framework/{collision_system}/1/skim/skim_output.root",
    )
