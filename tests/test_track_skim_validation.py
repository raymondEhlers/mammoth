"""Comparison between standard analysis and track skim

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBNL/UCB
"""

import logging
import pprint
from pathlib import Path
from typing import Any, List, Optional, Sequence

import attr
import awkward as ak
import boost_histogram as bh
import mammoth.helpers
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
import pytest
import uproot
from pachyderm import binned_data

from jet_substructure.analysis import plot_base as pb

from mammoth.framework import sources
from mammoth.framework.io import track_skim as io_track_skim
from mammoth.framework.analysis import jet_substructure as analysis_jet_substructure, objects as analysis_objects
from mammoth.hardest_kt import skim_to_flat_tree


pachyderm.plot.configure()


logger = logging.getLogger(__name__)

_here = Path(__file__).parent
_track_skim_base_path = _here / "track_skim_validation"

def _aliphysics_to_analysis_results(collision_system: str, jet_R: float) -> Path:
    # First, call the run macro
    from mammoth.hardest_kt import run_macro

    # Need this to be available to use the run macro
    # Strictly speaking, we need AliPHysics, so we then try to grab a value from AliPhysics
    ROOT = pytest.importorskip("ROOT")  # noqa: F841
    try:
        ROOT.AliAnalysisTaskSE
    except AttributeError:
        pytest.skip(
            "Need AliPhysics for generating reference file, but it appears that you only have ROOT."
            " Please check your configuration to ensure that AliPhysics is availablek"
        )

    optional_kwargs = {}
    if collision_system == "embed_pythia":
        optional_kwargs.update({
            "embed_input_files": _collision_system_to_aod_files["embed_pythia__pythia"],
        })
    run_macro.run(
        analysis_mode=collision_system, jet_R=jet_R, validation_mode=True,
        input_files=_collision_system_to_aod_files[collision_system],
        **optional_kwargs
    )
    # Next, we need to rename the output
    output_file = Path("AnalysisResults.root")
    output_file.rename(_track_skim_base_path / "reference" / f"AnalysisResults_{collision_system}_jet_R_{round(jet_R*100):03}.root")

    return output_file


def _reference_aliphyiscs_tree_name(collision_system: str, jet_R: float, dyg_task: bool = True) -> str:
    _jet_labels = {
        "pp": "Jet",
        "pythia": "Jet",
        "PbPb": "Jet",
        "embed_pythia": "hybridLevelJets",
    }
    _tags = {
        "pp": "_RawTree_Data_NoSub_Incl",
        "pythia": "_responseTree_PythiaDef_NoSub_Incl",
        #"PbPb": "ConstSub_RawTree_Data_ConstSub_Incl",
        "PbPb": "ConstSub_RawTree_Data_EventSub_Incl",
        "embed_pythia": "ConstSub_RawTree_EventSub_Incl",
    }
    return f"AliAnalysisTaskJetDynamicalGrooming_{_jet_labels[collision_system]}_AKTChargedR{round(jet_R*100):03}_tracks_pT0150_E_scheme{_tags[collision_system]}"


@attr.define
class ConvertTreeToParquetArguments:
    """Trivial class to help organize arguments for converting AliPhysics output trees to parquet files."""
    prefixes: List[str]
    branches: List[str]
    prefix_branches: List[str]


def _analysis_results_to_parquet(filename: Path, collision_system: str, jet_R: float) -> Path:
    # Esentially porting some parsl functionality, so trying to keep this as simple as possible
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
            prefixes=list(_collision_system_prefixes[collision_system].values()),
            branches=[],
            prefix_branches=_prefix_branches,
        ),
        "pythia": ConvertTreeToParquetArguments(
            prefixes=list(_collision_system_prefixes[collision_system].values()),
            branches=["ptHard", "ptHardBin"],
            prefix_branches=_prefix_branches,
        ),
        "PbPb": ConvertTreeToParquetArguments(
            prefixes=list(_collision_system_prefixes[collision_system].values()),
            branches=[],
            prefix_branches=_prefix_branches,
        ),
        "embed_pythia": ConvertTreeToParquetArguments(
            prefixes=list(_collision_system_prefixes[collision_system].values()),
            branches=[],
            prefix_branches=_prefix_branches,
        ),
    }
    arguments = _args[collision_system]

    _, output_filename = analysis_jet_substructure.convert_tree_to_parquet(
        filename=filename,
        tree_name=_reference_aliphyiscs_tree_name(collision_system=collision_system, jet_R=jet_R),
        prefixes=arguments.prefixes,
        branches=arguments.branches,
        prefix_branches=arguments.prefix_branches,
    )
    return output_filename


def track_skim_to_flat_tree(filename: Path) -> ak.Array:
    ...

# NOTE: These files are too large to store in the git repo.
#       I've stored them multiple places to attempt to ensure that they're not lost entirely.
#       Since the AliEn path is provided, they can always be retireved (in principle).
# NOTE: All were stored in archives (see the pachyderm dataset def for the particular archive names),
#       and then were extracted for simplicity.
_collision_system_to_aod_files = {
    "pp": [
        _track_skim_base_path / "input/alice/data/2017/LHC17p/000282343/pass1_FAST/AOD234/0001/AliAOD.root",
        _track_skim_base_path / "input/alice/data/2017/LHC17p/000282343/pass1_FAST/AOD234/0002/AliAOD.root",
        _track_skim_base_path / "input/alice/data/2017/LHC17p/000282343/pass1_FAST/AOD234/0003/AliAOD.root",
    ],
    "pythia": [
        _track_skim_base_path / "input/alice/sim/2020/LHC20g4/12/296191/AOD/001/AliAOD.root",
    ],
    "PbPb": [
        _track_skim_base_path / "input/alice/data/2018/LHC18q/000296550/pass3/AOD252/AOD/001/AliAOD.root",
    ],
    "embed_pythia": [
        _track_skim_base_path / "input/alice/data/2018/LHC18q/000296550/pass3/AOD252/AOD/001/AliAOD.root",
    ],
    "embed_pythia__pythia": [
        _track_skim_base_path / "input/alice/sim/2020/LHC20g4/12/296191/AOD/001/aod_archive.zip#AliAOD.root",
    ],
}

_collision_system_prefixes = {
    "pp": {
        "data": "data",
    },
    "pythia": {
        "data": "data",
        "true": "matched",
    },
    "PbPb": {
        "data": "data",
    },
    "embed_pythia": {
        "hybrid": "data",
        "true": "matched",
        "det_level": "detLevel",
    },
}

@attr.define
class TrackSkimValidationFilenames:
    base_path: Path
    filename_type: str
    collision_system: str
    jet_R: float
    iterative_splittings: bool

    @property
    def _label(self) -> str:
        iterative_splittings_label = "iterative" if self.iterative_splittings else "recursive"
        return f"{self.collision_system}_jet_R_{round(self.jet_R*100):03}_{iterative_splittings_label}_splittings"

    @property
    def analysis_output(self) -> Path:
        return self.base_path / self.filename_type / f"AnalysisResults_{self._label}.root"

    @property
    def parquet_output(self) -> Path:
        return self.base_path / self.filename_type / f"AnalysisResults_{self._label}.parquet"

    @property
    def skim(self) -> Path:
        return self.base_path / self.filename_type / f"skim_{self._label}.root"


@pytest.mark.parametrize("jet_R", [0.2, 0.4])
@pytest.mark.parametrize("collision_system", ["pp", "pythia", "PbPb", "embed_pythia"])
def test_track_skim_validation(caplog: Any, jet_R: float, collision_system: str, iterative_splittings: bool = True) -> None:
    # Setup
    caplog.set_level(logging.DEBUG)

    reference_filenames = TrackSkimValidationFilenames(
        base_path=_track_skim_base_path,
        filename_type="reference",
        collision_system=collision_system, jet_R=jet_R, iterative_splittings=iterative_splittings,
    )

    # For the AliPhysics reference:
    # 1. Run AliPhysics run macro
    # 2. Convert DyG outputs to parquet
    # 3. Convert DyG parquet to flat tree
    # NOTE: This isn't so trivial, since I have to port a good deal of code :-(
    generate_aliphysics_results = False
    convert_aliphysics_to_parquet = False
    skim_aliphysics_parquet = False
    # Determine which reference tasks need to be run
    if not reference_filenames.skim.exists():
        if not reference_filenames.parquet_output.exists():
            if not reference_filenames.analysis_output.exists():
                generate_aliphysics_results = True
            else:
                convert_aliphysics_to_parquet = True
        else:
            skim_aliphysics_parquet = True

    if generate_aliphysics_results:
        # First, validate input files
        # They might be missing since they're too large to store in the repo
        aod_input_files = _collision_system_to_aod_files[collision_system]
        missing_files = [not f.exists() for f in aod_input_files]
        if missing_files:
            raise RuntimeError(
                "Cannot generate AliPhysics reference due to missing inputs files."
                f" Missing: {[f for f, missing in zip(aod_input_files, missing_files) if missing]}"
            )

        # Now actually execute the run macro
        _aliphysics_to_analysis_results(collision_system=collision_system, jet_R=jet_R)

    if convert_aliphysics_to_parquet or generate_aliphysics_results:
        # We need to generate the parquet if we've just executed the run macro
        _analysis_results_to_parquet(filename=reference_filenames.analysis_output, collision_system=collision_system, jet_R=jet_R)

    # Need to post process regenerated outputs
    if skim_aliphysics_parquet or convert_aliphysics_to_parquet or generate_aliphysics_results:
        # NOTE: This assumes that the validation uses LHC20g4_AOD. However, we do this consistently,
        #       so that's a reasonable assumption.
        # NOTE: These scale factors need to be determined externally. These were extracted separately
        #       from the main analysis. In principle, the scale factors for this validation just need to
        #       be consistent, but since we have them available, we may as well use them.
        scale_factors = analysis_objects.read_extracted_scale_factors(
            path=_track_skim_base_path / "input" / "LHC20g4_AOD_2640_scale_factors.yaml"
        )
        if collision_system != "embed_pythia":
            res = skim_to_flat_tree.calculate_data_skim(
                input_filename=reference_filenames.parquet_output,
                collision_system=collision_system,
                iterative_splittings=iterative_splittings,
                prefixes=_collision_system_prefixes[collision_system],
                jet_R=jet_R,
                output_filename=reference_filenames.skim,
                scale_factors=scale_factors,
            )
        else:
            res = skim_to_flat_tree.calculate_embedding_skim(
                input_filename=reference_filenames.parquet_output,
                iterative_splittings=iterative_splittings,
                prefixes=_collision_system_prefixes[collision_system],
                scale_factors=scale_factors,
                train_directory=_track_skim_base_path / "reference" / "train_config.yaml",
                jet_R=jet_R,
                output_filename=reference_filenames.skim,
            )
        if not res[0]:
            raise ValueError(f"Failed to generate reference for {collision_system}, {jet_R}")

    # For mammoth:
    # 1. Convert track skim to parquet
    # 2. Analyze parquet track skim to generate flat tree
    track_skim_filenames = TrackSkimValidationFilenames(
        base_path=_track_skim_base_path,
        filename_type="track_skim",
        collision_system=collision_system, jet_R=jet_R, iterative_splittings=iterative_splittings,
    )
    if generate_aliphysics_results:
        # Convert track skim to parquet
        logger.info(f"Converting collision system {collision_system}")
        source = io_track_skim.FileSource(
            filename=reference_filenames.analysis_output,
            collision_system=collision_system,
        )
        arrays = next(source.gen_data(chunk_size=sources.ChunkSizeSentinel.FULL_SOURCE))
        io_track_skim.write_to_parquet(
            arrays=arrays,
            filename=track_skim_filenames.parquet_output,
            collision_system=collision_system,
        )

    # For embedding, we need to go to the separate signal and background files themselves.
    for collision_system in ["pythia", "PbPb"]:
        logger.info(f"Converting collision system {collision_system} for embedding")

        write_to_parquet(
            arrays=arrays,
            filename=Path(
                f"/software/rehlers/dev/mammoth/projects/framework/embedPythia/AnalysisResults_{collision_system}_track_skim.parquet"
            ),
            collision_system=collision_system,
        )



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
    print(f"ratio sum: {np.sum(ratio.values)}")
    print(f"other: {np.sum(other_hist.values)}")
    print(f"mine: {np.sum(mine_hist.values)}")

    # Apply the PlotConfig
    plot_config.apply(fig=fig, axes=[ax, ax_ratio])

    # filename = f"{plot_config.name}_{jet_pt_bin}{grooming_methods_filename_label}_{identifiers}_iterative_splittings"
    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def compare(collision_system: str, prefixes: Sequence[str], standard_filename: Path, track_skim_filename: Path) -> None:
    #standard_tree_name = "AliAnalysisTaskJetHardestKt_Jet_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_Data_ConstSub_Incl"
    #if collision_system == "pp":
    #    standard_tree_name = "AliAnalysisTaskJetHardestKt_Jet_AKTChargedR040_tracks_pT0150_E_scheme_RawTree_Data_NoSub_Incl"
    #if collision_system == "pythia":
    #    standard_tree_name = "AliAnalysisTaskJetHardestKt_Jet_AKTChargedR040_tracks_pT0150_E_schemeTree_PythiaDef_NoSub_Incl"
    standard_tree_name = "tree"
    standard = uproot.open(standard_filename)[standard_tree_name].arrays()
    track_skim = uproot.open(track_skim_filename)["tree"].arrays()
    print(f"standard.type: {standard.type}")
    print(f"track_skim.type: {track_skim.type}")

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

    output_dir = Path("comparison") / "trackSkim" / collision_system
    output_dir.mkdir(parents=True, exist_ok=True)

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

        # Sometimes it's useful to start at this, but sometimes it's just overwhelming, so uncomment as necessasry
        #logger.info(f"standard_jet_pt: {standard_jet_pt.to_list()}")
        #logger.info(f"track_skim_jet_pt: {track_skim_jet_pt.to_list()}")

        try:
            all_close_jet_pt = np.allclose(ak.to_numpy(standard_jet_pt), ak.to_numpy(track_skim_jet_pt))

            logger.info(f"jet_pt all close? {all_close_jet_pt}")
            #import IPython; IPython.embed()
            if not all_close_jet_pt:
                logger.info("jet pt")
                _arr = ak.zip({"s": standard_jet_pt, "t": track_skim_jet_pt})
                logger.info(pprint.pformat(_arr.to_list()))
                is_not_close_jet_pt = np.where(~np.isclose(ak.to_numpy(standard_jet_pt), ak.to_numpy(track_skim_jet_pt)))
                logger.info(f"Indicies where not close: {is_not_close_jet_pt}")
        except ValueError as e:
            logger.exception(e)

        for grooming_method in ["dynamical_kt", "soft_drop_z_cut_02"]:
            logger.info(f"Plotting method \"{grooming_method}\"")
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

            # Sometimes it's useful to start at this, but sometimes it's just overwhelming, so uncomment as necessasry
            #logger.info(f"standard_kt: {standard_kt.to_list()}")
            #logger.info(f"track_skim_kt: {track_skim_kt.to_list()}")

            try:
                all_close_kt = np.allclose(ak.to_numpy(standard_kt), ak.to_numpy(track_skim_kt), rtol=1e-4)
                logger.info(f"kt all close? {all_close_kt}")
                if not all_close_kt:
                    logger.info("kt")
                    _arr = ak.zip({"s": standard_kt, "t": track_skim_kt})
                    logger.info(pprint.pformat(_arr.to_list()))
                    is_not_close_kt = np.where(~np.isclose(ak.to_numpy(standard_kt), ak.to_numpy(track_skim_kt)))
                    logger.info(f"Indicies where not close: {is_not_close_kt}")
            except ValueError as e:
                logger.exception(e)

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

            # Sometimes it's useful to start at this, but sometimes it's just overwhelming, so uncomment as necessasry
            #logger.info(f"standard_zg: {standard_zg.to_list()}")
            #logger.info(f"track_skim_zg: {track_skim_zg.to_list()}")

            try:
                all_close_rg = np.allclose(ak.to_numpy(standard_rg), ak.to_numpy(track_skim_rg), rtol=1e-4)
                logger.info(f"Rg all close? {all_close_rg}")
                if not all_close_rg:
                    logger.info("delta_R")
                    _arr = ak.zip({"s": standard_rg, "t": track_skim_rg})
                    logger.info(pprint.pformat(_arr.to_list()))
                    is_not_close_rg = np.where(~np.isclose(ak.to_numpy(standard_rg), ak.to_numpy(track_skim_rg)))
                    logger.info(f"Indicies where not close: {is_not_close_rg}")
            except ValueError as e:
                logger.exception(e)

            #import IPython; IPython.embed()

            #logger.info(f"standard_rg: {standard_rg.to_list()}")
            #logger.info(f"track_skim_rg: {track_skim_rg.to_list()}")

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

            # Sometimes it's useful to start at this, but sometimes it's just overwhelming, so uncomment as necessasry
            #logger.info(f"standard_zg: {standard_zg.to_list()}")
            #logger.info(f"track_skim_zg: {track_skim_zg.to_list()}")

            try:
                all_close_zg = np.allclose(ak.to_numpy(standard_zg), ak.to_numpy(track_skim_zg))
                logger.info(f"zg all close? {all_close_zg}")
                if not all_close_zg:
                    logger.info("z")
                    _arr = ak.zip({"s": standard_zg, "t": track_skim_zg})
                    logger.info(pprint.pformat(_arr.to_list()))
            except ValueError as e:
                logger.exception(e)


def run(collision_system: str, prefixes: Optional[Sequence[str]] = None) -> None:
    if prefixes is None:
        prefixes = ["data"]
    mammoth.helpers.setup_logging()
    logger.info(f"Running {collision_system} with prefixes {prefixes}")
    path_to_mammoth = Path(mammoth.helpers.__file__).parent.parent
    standard_base_filename = "AnalysisResults"
    if collision_system == "pythia":
        standard_base_filename += ".12"
    compare(
        collision_system=collision_system,
        prefixes=prefixes,
        standard_filename=path_to_mammoth / f"projects/framework/{collision_system}/1/skim/{standard_base_filename}.repaired.00_iterative_splittings.root",
        track_skim_filename=path_to_mammoth / f"projects/framework/{collision_system}/1/skim/skim_output.root",
    )


if __name__ == "__main__":
    collision_system = "embed_pythia"

    _prefixes = {
        "pp": ["data"],
        "pythia": ["data", "true"],
        "PbPb": ["data"],
        "embed_pythia": ["hybrid", "det_level", "true"],
    }
    run(collision_system=collision_system, prefixes=_prefixes[collision_system])
