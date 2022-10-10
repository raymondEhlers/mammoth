#!/usr/bin/env python3
""" Run the event extractor AliPhysics task.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import argparse
import enum
import logging
import pprint
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, Union

import attr
import numpy as np


logger = logging.getLogger(__name__)

# Type helpers
# AnalysisTask = ROOT.AliAnalysisTaskSE
# AnalysisManager = ROOT.AliAnalysisManager
AnalysisTask = Any
AnalysisManager = Any


@attr.s
class CppEnum:
    value: str = attr.ib()

    def __str__(self) -> str:
        """Return the str without quotes by calling `str`.

        This ensures that ROOT will evaluate the enum correctly when evaluating the AddTask.
        """
        return str(self.value)


def _normalize_period(period: str) -> str:
    """Normalize period name to have "LHC" in all caps.

    Args:
        period: Period name to be normalized.
    Returns:
        Normalized period name.
    """
    return period[:3].upper() + period[3:]


def _is_run2_data(period: str) -> bool:
    """Determine if the period was data taken during Run 2.

    Performed by checking for the years LHC{15..18}. In principle, this should fail for MC productions
    (which is the desired behavior - the year of the MC production doesn't dictate the year of the data
    taking period).

    Args:
        period: Run period to be checked.
    Returns:
        True if the run period is in Run2.
    """
    for year in [15, 16, 17, 18]:
        if period.startswith(f"LHC{year}"):
            return True
    return False


def _is_MC(period: str) -> bool:
    """Determine if we are running over MC.

    Takes advantage of the fact that MC period names are always longer then 6 characters.
    For example, we compared "LHC17p" (data) vs "LHC18b8" (MC).

    Args:
        period: Run period.
    Returns:
        True if we are analyzing a MC run period.
    """
    return len(period) > 6


class DataType(enum.Enum):
    """ALICE data type.

    Either AOD or ESD.
    """

    AOD = 0
    ESD = 1


_T_BeamType = TypeVar("_T_BeamType", bound="BeamType")


class BeamType(enum.Enum):
    """Beam type.

    Uses the AliAnalysisTaskEmcal enum.
    """

    # NOTE: We manually assign the values from AliAnalysisTaskEmcal so we can make the ROOT dependence indirect
    pp = 0  # ROOT.AliAnalysisTaskEmcal.kpp
    pPb = 2  # ROOT.AliAnalysisTaskEmcal.kpA
    PbPb = 1  # ROOT.AliAnalysisTaskEmcal.kAA

    @classmethod
    def from_period(cls: Type[_T_BeamType], period: str) -> _T_BeamType:
        """Determine the beam type from the run period.

        Note:
            This is based on a enumerated list of PbPb and pPb run periods
            and will need to be updated as new periods are added.

        Args:
            period: Run period.
        Returns:
            The determined beam type.
        """
        # Validation
        period = _normalize_period(period)
        PbPb_run_periods = ["LHC10h", "LHC11h", "LHC15o", "LHC18q", "LHC18r"]
        pPb_run_periods = [
            "LHC12g",
            "LHC13b",
            "LHC13c",
            "LHC13d",
            "LHC13e",
            "LHC13f",
            "LHC16q",
            "LHC16r",
            "LHC16s",
            "LHC16t",
        ]
        # Initialized via str to help out mypy...
        if period in PbPb_run_periods:
            return cls["PbPb"]
        if period in pPb_run_periods:
            return cls["pPb"]
        return cls["pp"]


class AnalysisMode(enum.Enum):
    """Analysis mode."""

    pp = enum.auto()
    pythia = enum.auto()
    PbPb = enum.auto()
    embed_pythia = enum.auto()


def _run_add_task_macro(task_path: Union[str, Path], task_class_name: str, *args: Any) -> Any:
    """Run a given add task macro.

    Note:
        This is a cute idea, but it's also rather fragile. See the comments in the code for details how this
        is remotely possible.

    Args:
        task_path: Path to the AddTask macro to be executed.
        task_class_name: Name of the class that the AddTask returns.
        args: All of the arguments to be passed to the AddTask macro.

    Returns:
        The task returned by the AddTask.
    """
    # Delay import to avoid explicit dependence
    import ROOT  # pyright: ignore [reportMissingImports]
    # Validation
    task_path = Path(task_path)

    # Setup
    bool_map = {
        False: "false",
        True: "true",
    }
    task_args = ", ".join(
        [f'"{v}"' if isinstance(v, str) else bool_map[v] if isinstance(v, bool) else str(v) for v in args]
    )
    print(f"Running: {task_path}({task_args})")
    address = ROOT.gROOT.ProcessLine(f".x {task_path}({task_args})")
    # Need to convert the address into the task. Unfortunately, we can't cast the address directly into an object.
    # Instead, we use cling to perform the reinterpret_cast for us, and then we retrieve that task.
    # This is super convoluted, but it appears to work okay...
    # For a UUID to be a valid c++ variable, we need:
    #  - Prefix the name so that it starts with a letter.
    #  - To replace "-" with "_"
    cpp_temp_task_name = "temp_" + str(uuid.uuid4()).replace("-", "_")
    # Cast the task
    ROOT.gInterpreter.ProcessLine(f"auto * {cpp_temp_task_name} = reinterpret_cast<{task_class_name}*>({address});")
    # And then retrieve and return the actual task.
    return getattr(ROOT, cpp_temp_task_name)


def _add_physics_selection(is_MC: bool, beam_type: BeamType) -> AnalysisTask:
    # Delay import to avoid explicit dependence
    import ROOT  # pyright: ignore [reportMissingImports]
    # Enable pileup rejection (second argument) for pp
    # physics_selection_task = _run_add_task_macro(
    #    "$ALICE_PHYSICS/OADB/macros/AddTaskPhysicsSelection.C", "AliPhysicsSelectionTask",
    #    is_MC, beam_type == BeamType.pp
    # )
    physics_selection_task = ROOT.AliPhysicsSelectionTask.AddTaskPhysicsSelection(is_MC, beam_type == BeamType.pp)
    return physics_selection_task


def _add_mult_selection(is_run2_data: bool, physics_selection: int) -> Optional[AnalysisTask]:
    # Delay import to avoid explicit dependence
    import ROOT  # pyright: ignore [reportMissingImports]
    # Works for both pp and PbPb for the periods that it is calibrated
    # However, I seem to have trouble with pp MCs
    if is_run2_data:
        # multiplicity_selection_task = _run_add_task_macro(
        #    "$ALICE_PHYSICS/OADB/COMMON/MULTIPLICITY/macros/AddTaskMultSelection.C",
        #    "AliMultSelectionTask",
        #    False
        # )
        multiplicity_selection_task = ROOT.AliMultSelectionTask.AddTaskMultSelection(False)
        multiplicity_selection_task.SelectCollisionCandidates(physics_selection)
        return multiplicity_selection_task
    return None


def _handle_special_event_selection_for_pythia(period: str, task: Any) -> None:
    if period == "LHC20g4":
        # For this period, we need to manually configure the event cuts to use Run2 pp cuts.
        # Otherwise, it will use the PbPb cuts, which will dramatically cut into the statistics.
        # NOTE: This isn't an issue with embedding because we don't apply event selection to the external event
        eventCuts = task.GetEventCuts()
        eventCuts.SetManualMode()
        eventCuts.SetupRun2pp()
        print(f"Fixing event selection for {period} for {task.GetName()}")


def run_dynamical_grooming(  # noqa: C901
    task_name: str,
    analysis_mode: AnalysisMode,
    period: str,
    physics_selection: int,
    data_type: DataType,
    jet_R: float = 0.2,
    enable_track_skim: bool = False,
    enable_HF_tree: bool = False,
    grooming_jet_pt_threshold: float = 20,
    validation_mode: bool = True,
) -> AnalysisManager:
    """Run the event extractor.

    Args:
        task_name: Name of the analysis (ie. given to the analysis manager).
        analysis_mode: Mode of analysis.
        period: Run period.
        physics_selection: Physics selection to apply to the analysis.
        data_type: ALICE data type over which the analysis will run.
        jet_R: Jet R. Default: 0.2
        enable_track_skim: Add track skimming to analysis. Default: False.
        enable_HF_tree: Add HF tree skimming to analysis. Default: False.
        grooming_jet_pt_threshold: Jet pt threshold for grooming tasks. Default: 20
        validation_mode: If True, adjust some settings to enable validation mode.
            See the setup for the exact settings that are adjusted. Default: True

    Returns:
        The analysis manager.
    """
    # Delay import to avoid explicit dependence
    import ROOT  # pyright: ignore [reportMissingImports]  # pyright: ignore [reportMissingImports]
    # Validation
    period = _normalize_period(period)
    is_MC = _is_MC(period)
    is_run2_data = _is_run2_data(period) if not is_MC else False
    # Determine the beam type from the period.
    beam_type = BeamType.from_period(period)

    # Setup
    # pt hard binning is for pp MC, embedding
    if analysis_mode in [AnalysisMode.pythia, AnalysisMode.embed_pythia]:
        # Unfortunately, we have to pass a TArrayI because that is the only accepted type...
        pt_hard_binning = np.array(
            [0, 5, 7, 9, 12, 16, 21, 28, 36, 45, 57, 70, 85, 99, 115, 132, 150, 169, 190, 212, 235, 1000],
            dtype=np.int32,
        )
        pt_hard_binning_root = ROOT.TArrayI(len(pt_hard_binning), pt_hard_binning)
    # Centrality ranges
    # This is rather lazy, but works for these purposes.
    centrality_min, centrality_max = 0, 10
    if ROOT.AliVEvent.kSemiCentral & physics_selection:
        centrality_min, centrality_max = 30, 50
    if analysis_mode == AnalysisMode.PbPb:
        print(f"Centrality range: {centrality_min}-{centrality_max}")
    # jet R needs to be formatted as a string too
    jet_R_str = f"{round(jet_R*100):03}"
    # Setup for validation mode
    particle_level_min_pt = 0.
    if validation_mode:
        # In the past, we've run the AliPhysics analysis tasks with min of 0. However, as of Feb 2022, the track
        # skims tend to go down to 150 MeV even at track level to keep the data volume reasonable. For the sake
        # of validation, we set the AliPhysics tasks to go a min of 150 MeV, and can consider a re-skim later.
        particle_level_min_pt = 0.15
    particle_level_min_pt_str = f"{round(particle_level_min_pt*1000):04}"

    # Basic setup (analysis manager and input handler).
    analysis_manager = ROOT.AliAnalysisManager(task_name)
    if data_type == DataType.AOD:
        # Standard
        ROOT.AliAnalysisTaskEmcal.AddAODHandler()
        # Reduced
        # handler = _run_add_task_macro(
        #    "$ALICE_PHYSICS/PWG/EMCAL/macros/AddAODReducedBranchesInputHandler.C", "PWG::EMCAL::AliAODReducedBranchesInputHandler",
        # )
        # handler = ROOT.PWG.EMCAL.AliAODReducedBranchesInputHandler.AddAODReducedBranchesHandler()
        # handler.SetInactiveBranches("*")
        # handler.SetActiveBranches("header tracks vertices emcalCells MultSelection")
    else:
        ROOT.AliAnalysisTaskEmcal.AddESDHandler()

    # Physics selection task
    _add_physics_selection(is_MC, beam_type)

    # Multiplicity selection task.
    _add_mult_selection(is_run2_data=is_run2_data, physics_selection=physics_selection)

    ################
    # Debug settings
    ################
    # ROOT.AliLog.SetClassDebugLevel("PWG::EMCAL::AliAODReducedBranchesInputHandler", ROOT.AliLog.kDebug-1)
    # ROOT.AliLog.SetClassDebugLevel("AliEmcalCorrectionComponent", AliLog::kDebug+3)
    # ROOT.AliLog.SetClassDebugLevel("AliAnalysisTaskEmcalJetHCorrelations", AliLog::kDebug+1)
    # ROOT.AliLog.SetClassDebugLevel("PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming", ROOT.AliLog.kDebug-1)
    # ROOT.AliLog.SetClassDebugLevel("AliJetContainer", AliLog::kDebug+7)

    if enable_track_skim:
        if analysis_mode == AnalysisMode.pp:
            skim_task = _run_add_task_macro(
                "$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskTrackSkim.C",
                "PWGJE::EMCALJetTasks::AliAnalysisTaskTrackSkim",
                "pp",
            )
            # skim_task = ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskTrackSkim.AddTaskTrackSkim("pp")
            skim_task.SelectCollisionCandidates(physics_selection)
            skim_task.AddTrackContainer("tracks")
            skim_task.Initialize()
        elif analysis_mode == AnalysisMode.pythia:
            skim_task = _run_add_task_macro(
                "$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskTrackSkim.C",
                "PWGJE::EMCALJetTasks::AliAnalysisTaskTrackSkim",
                "pythia",
            )
            skim_task.SelectCollisionCandidates(physics_selection)
            skim_task.SetIsPythia(True)
            _handle_special_event_selection_for_pythia(period=period, task=skim_task)

            skim_task.AddTrackContainer("tracks")

            contMC = skim_task.AddMCParticleContainer("mcparticles")
            status_acc = 1 << 16
            contMC.SetMCFlag(ROOT.AliAODMCParticle.kPhysicalPrim | status_acc)
            contMC.SetCharge(ROOT.AliParticleContainer.kCharged)

            skim_task.Initialize()
        elif analysis_mode == AnalysisMode.PbPb:
            skim_task = _run_add_task_macro(
                "$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskTrackSkim.C",
                "PWGJE::EMCALJetTasks::AliAnalysisTaskTrackSkim",
                "PbPb",
            )
            skim_task.SetCentRange(centrality_min, centrality_max)
            skim_task.SetUseNewCentralityEstimation(True)
            skim_task.SelectCollisionCandidates(physics_selection)

            skim_task.AddTrackContainer("tracks")
            skim_task.Initialize()

    if enable_HF_tree:
        # PID response is a dependency
        pid_response = _run_add_task_macro(
            "$ALICE_ROOT/ANALYSIS/macros/AddTaskPIDResponse.C",
            "AliAnalysisTaskPIDResponse",
            is_MC,
        )
        if is_MC:
            # Disabled for testing, since the version of AliRoot doesn't yet contain this function.
            # As along as everything is consistent, it should be fine, I think.
            # pid_response.ResetTuneOnDataTOF()
            ...
        pid_response.SelectCollisionCandidates(physics_selection)

        # And then the skimming task itself
        # Apparently we have to explicitly call for the connection to AliEn...
        ROOT.TGrid.Connect("alien://")
        if analysis_mode == AnalysisMode.pp:
            # From LEGO train wagon: TreeCreatorHF_Tracks_pp
            skim_task = _run_add_task_macro(
                "$ALICE_PHYSICS/PWGHF/treeHF/macros/AddTaskHFTreeCreator.C",
                "AliAnalysisTaskSEHFTreeCreator",
                False,
                0,
                "HFTreeCreator_pp",
                # "alien://///alice/cern.ch/user/l/lvermunt/cuts_tree_creator/25-03-2020/pp/D0DsDplusDstarLcBplusBsLbCuts_pp.root",
                "/home/rehlers/code/alice/substructure/dynamicalGrooming/mammoth/framework/pp/D0DsDplusDstarLcBplusBsLbCuts_pp_kAny.root",
                0,
                False,
                False,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                CppEnum("AliHFTreeHandler::kNsigmaPID"),
                CppEnum("AliHFTreeHandler::kRedSingleTrackVars"),
                True,
            )
            skim_task.SelectCollisionCandidates(physics_selection)
            skim_task.AddTrackContainer("tracks")
        elif analysis_mode == AnalysisMode.pythia:
            # From LEGO train wagon: TreeCreatorHF_Tracks_pp_MC
            skim_task = _run_add_task_macro(
                "$ALICE_PHYSICS/PWGHF/treeHF/macros/AddTaskHFTreeCreator.C",
                "AliAnalysisTaskSEHFTreeCreator",
                True,
                0,
                "HFTreeCreator_pp_5TeV",
                # "alien://///alice/cern.ch/user/l/lvermunt/cuts_tree_creator/25-03-2020/pp/D0DsDplusDstarLcBplusBsLbCuts_pp.root",
                "/home/rehlers/code/alice/substructure/dynamicalGrooming/mammoth/framework/pp/D0DsDplusDstarLcBplusBsLbCuts_pp_kAny.root",
                0,
                False,
                True,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                CppEnum("AliHFTreeHandler::kNsigmaPID"),
                CppEnum("AliHFTreeHandler::kRedSingleTrackVars"),
                True,
            )

            skim_task.SelectCollisionCandidates(physics_selection)

            skim_task.AddTrackContainer("tracks")

            contMC = skim_task.AddMCParticleContainer("mcparticles")
            status_acc = 1 << 16
            contMC.SetMCFlag(ROOT.AliAODMCParticle.kPhysicalPrim | status_acc)
            contMC.SetCharge(ROOT.AliParticleContainer.kCharged)
        elif analysis_mode == AnalysisMode.PbPb:
            # From LEGO train wagon: TreeCreatorHF_Tracks_5TeV_PbPb
            skim_task = _run_add_task_macro(
                "$ALICE_PHYSICS/PWGHF/treeHF/macros/AddTaskHFTreeCreator.C",
                "AliAnalysisTaskSEHFTreeCreator",
                False,
                1,
                "HFTreeCreator_PbPb_5TeV",
                # NOTE: This means that the skimming will _only_ work for central!!
                "alien://///alice/cern.ch/user/j/jmulliga/cutfile/20200428_D0DsDplusDstarLcBplusBsLbCuts_PbPb2018_Central.root",
                0,
                False,
                False,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                CppEnum("AliHFTreeHandler::kNsigmaPID"),
                CppEnum("AliHFTreeHandler::kRedSingleTrackVars"),
                True,
            )
            if centrality_min != 0 and centrality_max != 10:
                raise RuntimeError(
                    "Using a central cut configuration, but central collisions are not selected. Please update the config or the event selection!"
                )
            skim_task.SelectCollisionCandidates(physics_selection)

            skim_task.AddTrackContainer("tracks")
        else:
            raise ValueError("Skimming task config not available for analysis mode")

    # Shared jet finding settings
    ghost_area = 0.005

    # Rho
    if beam_type == BeamType.PbPb:
        # Rho related
        # Jet finder for rho. Wagon name: "JetFinderKtTpcQG_EScheme"
        kt_jet_finder = ROOT.AliEmcalJetTask.AddTaskEmcalJet(
            "usedefault",
            "",
            ROOT.AliJetContainer.kt_algorithm,
            0.2,
            ROOT.AliJetContainer.kChargedJet,
            0.15,
            0,
            ghost_area,
            ROOT.AliJetContainer.E_scheme,
            "Jet",
            0,
            False,
            False,
        )
        kt_jet_finder.SelectCollisionCandidates(physics_selection)
        kt_jet_finder.SetUseNewCentralityEstimation(True)
        # Rho. Wagon name: AliAnalysisTaskQGRhoTpcExLJ_EScheme
        rho_task = _run_add_task_macro(
            "$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskRhoNew.C",
            "AliAnalysisTaskRho",
            "usedefault",
            "",
            "Rho",
            0.2,
            CppEnum("AliJetContainer::kTPCfid"),
            CppEnum("AliJetContainer::kChargedJet"),
            True,
            CppEnum("AliJetContainer::E_scheme"),
        )
        rho_task.SetExcludeLeadJets(2)
        rho_task.SelectCollisionCandidates(physics_selection)
        rho_task.SetUseNewCentralityEstimation(True)
        # rho_task.GetParticleContainer(0).SetMCTrackBitMap(TObject.kBitMask)
        # rho_task.GetClusterContainer(0).SetMCClusterBitMap(TObject.kBitMask)
        # rho_task.SetHistoBins(250,0,250)
        rho_task.SetNeedEmcalGeom(False)
        #rho_task.SetZvertexDiffValue(0.1)
        cont_rho = rho_task.GetJetContainer(0)
        cont_rho.SetJetRadius(0.2)
        cont_rho.SetJetAcceptanceType(ROOT.AliJetContainer.kTPCfid)
        cont_rho.SetMaxTrackPt(100)

        # Rho mass. Wagon name: "RhoMassQGTPC"
        rho_mass = _run_add_task_macro(
            "$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskRhoMass.C",
            "AliAnalysisTaskRhoMass",
            "Jet_KTChargedR020_tracks_pT0150_E_scheme",
            "tracks",
            "",
            "Rhomass",
            0.2,
            "TPC",
            0.01,
            0,
            0,
            2,
            True,
            "RhoMass",
        )
        rho_mass.SelectCollisionCandidates(physics_selection)
        rho_mass.SetUseNewCentralityEstimation(True)
        rho_mass.SetHistoBins(250, 0, 250)
        # rho_mass.SetScaleFunction(srhomfunc)
        rho_mass.SetNeedEmcalGeom(False)
        cont_rho_mass = rho_mass.GetJetContainer(0)
        cont_rho_mass.SetJetRadius(0.2)
        cont_rho_mass.SetJetAcceptanceType(ROOT.AliJetContainer.kTPCfid)
        cont_rho_mass.SetMaxTrackPt(100)
        rho_mass.SetNeedEmcalGeom(False)
        #rho_mass.SetZvertexDiffValue(0.1)

    # Standard akt jet finder. Wagon name: "JetFinderQGAKTCharged_R04_Escheme" from PbPb train.
    akt_jet_finder = ROOT.AliEmcalJetTask.AddTaskEmcalJet(
        "tracks",
        "",
        ROOT.AliJetContainer.antikt_algorithm,
        jet_R,
        ROOT.AliJetContainer.kChargedJet,
        0.15,
        0.30,
        ghost_area,
        ROOT.AliJetContainer.E_scheme,
        "Jet",
        0,
        False,
        False,
    )
    akt_jet_finder.SelectCollisionCandidates(physics_selection)
    _handle_special_event_selection_for_pythia(period=period, task=akt_jet_finder)
    akt_jet_finder.SetNeedEmcalGeom(False)
    # akt_jet_finder.SetZvertexDiffValue(0.1)

    # Add event subtractor in PbPb.
    if beam_type == BeamType.PbPb:
        akt_jet_finder.SetUseNewCentralityEstimation(True)

        # constUtil = akt_jet_finder.AddUtility(ROOT.AliEmcalJetUtilityConstSubtractor("ConstSubtractor"))
        constUtil = akt_jet_finder.AddUtility(ROOT.AliEmcalJetUtilityEventSubtractor("EventSubtractor"))
        genSub = akt_jet_finder.AddUtility(ROOT.AliEmcalJetUtilityGenSubtractor("GenSubtractor"))

        genSub.SetGenericSubtractionJetMass(True)
        genSub.SetGenericSubtractionExtraJetShapes(True)
        genSub.SetUseExternalBkg(True)
        genSub.SetRhoName("Rho")
        genSub.SetRhomName("Rhomass")
        constUtil.SetJetsSubName(f"{akt_jet_finder.GetName()}ConstSub")

        constUtil.SetParticlesSubName("tracksSubR02")
        constUtil.SetUseExternalBkg(True)
        constUtil.SetRhoName("Rho")
        constUtil.SetRhomName("Rhomass")
        constUtil.SetMaxDelR(0.25)

    # Particle level jet finder
    if analysis_mode in [AnalysisMode.pythia, AnalysisMode.embed_pythia]:
        # JetFinderAKTChargedMC_R04_Escheme
        akt_particle_level_jet_finder = ROOT.AliEmcalJetTask.AddTaskEmcalJet(
            "mcparticles",
            "",
            ROOT.AliJetContainer.antikt_algorithm,
            jet_R,
            ROOT.AliJetContainer.kChargedJet,
            particle_level_min_pt,
            0.0,
            ghost_area,
            ROOT.AliJetContainer.E_scheme,
            "Jet",
            0,
            False,
            False,
        )
        akt_particle_level_jet_finder.SelectCollisionCandidates(physics_selection)
        _handle_special_event_selection_for_pythia(period=period, task=akt_particle_level_jet_finder)

    # Pythia detector to particle level tagger
    if analysis_mode in [AnalysisMode.pythia, AnalysisMode.embed_pythia]:
        # Tagger
        # JetTaggerMCChargedR{jet_R_str}
        tagger = _run_add_task_macro(
            "$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskEmcalJetTagger.C",
            "AliAnalysisTaskEmcalJetTagger",
            f"Jet_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
            f"Jet_AKTChargedR{jet_R_str}_mcparticles_pT{particle_level_min_pt_str}_E_scheme",
            jet_R,
            "",
            "",
            "tracks",
            "",
            "TPC",
            "",
            CppEnum("AliVEvent::kMB"),
            "",
        )
        tagger.SetNCentBins(1)
        tagger.SelectCollisionCandidates(physics_selection)
        _handle_special_event_selection_for_pythia(period=period, task=tagger)
        # tagger.SetUseInternalEventSelection(kTRUE)
        # tagger.SetForceBeamType(AliAnalysisTaskEmcal::kpp)
        if is_MC:
            tagger.SetIsPythia(True)
        tagger.SetJetTaggingType(ROOT.AliAnalysisTaskEmcalJetTagger.kClosest)
        tagger.SetJetTaggingMethod(ROOT.AliAnalysisTaskEmcalJetTagger.kGeo)
        tagger.SetNumberOfPtHardBins(len(pt_hard_binning) - 1)
        tagger.SetUserPtHardBinning(pt_hard_binning_root)
        # This is what's used on the LEGO train (2461, for example). However, it's really broad, so we use something a bit
        # narrower here for testing and validation.
        # NOTE: In particular, type 3 adds extra margin around the jet acceptance. We don't have this functionality in
        #       the track skim analysis, so we disable it for now.
        #tagger.SetTypeAcceptance(3)
        #tagger.SetMaxDistance(1.0)
        tagger.SetTypeAcceptance(0)
        tagger.SetMaxDistance(1.0)

    # Dynamical grooming
    if analysis_mode == AnalysisMode.PbPb:
        # task = _run_add_task_macro(
        #    "$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskJetDynamicalGrooming.C", "PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming",
        #    f"Jet_AKTChargedR{jet_R_str}_tracks_pT0150_E_schemeConstSub",
        #    f"Jet_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme", "", "", jet_R, "Rho",
        #    "tracksSubR02", "tracks", "", "", "", "TPC", "V0M", physics_selection,
        #    CppEnum("PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming::kData"),
        #    CppEnum("PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming::kEventSub"),
        #    CppEnum("PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming::kInclusive"),
        #    0, 0, 0.6,
        #    CppEnum("PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming::kSecondOrder"),
        #    "Raw"
        # )
        dynamical_grooming = ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.AddTaskJetDynamicalGrooming(
            f"Jet_AKTChargedR{jet_R_str}_tracks_pT0150_E_schemeConstSub",
            f"Jet_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
            "",
            "",
            jet_R,
            "Rho",
            "tracksSubR02",
            "tracks",
            "",
            "",
            "",
            "TPC",
            "V0M",
            physics_selection,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kData,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kEventSub,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kInclusive,
            0,
            0,
            0.6,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kSecondOrder,
            "Raw",
        )
    elif analysis_mode == AnalysisMode.pp:
        # dynamical_grooming = ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.AddTaskJetDynamicalGrooming(
        #    f"Jet_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
        #    "", "", "", jet_R, "",
        #    "tracks", "", "", "", "", "TPC", "V0M", physics_selection,
        #    ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kData,
        #    ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kNoSub,
        #    ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kInclusive,
        #    0, 0, 0.6,
        #    ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kSecondOrder,
        #    "Raw"
        # )
        dynamical_grooming = ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.AddTaskJetDynamicalGrooming(
            f"Jet_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
            "",
            "",
            "",
            0.2,
            "",
            "tracks",
            "",
            "",
            "",
            "",
            "TPC",
            "V0M",
            physics_selection,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kData,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kNoSub,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kInclusive,
            0,
            0,
            0.6,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kSecondOrder,
            "Raw",
        )
    elif analysis_mode == AnalysisMode.pythia:
        dynamical_grooming = ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.AddTaskJetDynamicalGrooming(
            f"Jet_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
            "",
            f"Jet_AKTChargedR{jet_R_str}_mcparticles_pT{particle_level_min_pt_str}_E_scheme",
            "",
            jet_R,
            "tracks",
            "",
            "",
            "mcparticles",
            "",
            "",
            "TPC",
            "V0M",
            physics_selection,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kPythiaDef,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kNoSub,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kInclusive,
        )
    dynamical_grooming.SelectCollisionCandidates(physics_selection)
    if analysis_mode in [AnalysisMode.pythia, AnalysisMode.embed_pythia]:
        _handle_special_event_selection_for_pythia(period=period, task=dynamical_grooming)
        dynamical_grooming.SetNumberOfPtHardBins(len(pt_hard_binning) - 1)
        dynamical_grooming.SetUserPtHardBinning(pt_hard_binning_root)
    cont = dynamical_grooming.GetJetContainer(0)
    if beam_type == BeamType.PbPb:
        cont.SetRhoName("Rho")
        cont.SetRhoMassName("RhoMass")
    cont.SetJetRadius(jet_R)
    cont.SetJetAcceptanceType(ROOT.AliJetContainer.kTPCfid)
    cont.SetMaxTrackPt(100)
    cont.SetJetPtCut(0)

    print(f"beam_type: {beam_type}")
    # It only matters for embedding, but in any case, always disable double counting.
    dynamical_grooming.SetCutDoubleCounts(False)
    if beam_type == BeamType.PbPb:
        dynamical_grooming.SetCentralitySelectionOn(True)
        dynamical_grooming.SetUseNewCentralityEstimation(True)
        dynamical_grooming.SetMinCentrality(centrality_min)
        dynamical_grooming.SetMaxCentrality(centrality_max)
    else:
        # Need to disable here because it's on by default!!
        dynamical_grooming.SetCentralitySelectionOn(False)

    dynamical_grooming.SetJetPtThreshold(grooming_jet_pt_threshold)
    dynamical_grooming.SetNeedEmcalGeom(False)
    # dynamical_grooming.SetZvertexDiffValue(0.1)
    dynamical_grooming.SetStoreRecursiveJetSplittings(True)

    # dynamical_grooming.SetDoTwoTrack(kTRUE)
    # The hard cutoff isn't meaningful when storing all splittings.
    # dynamical_grooming.SetHardCutoff(0.1)

    if is_MC:
        dynamical_grooming.SetIsPythia(True)

    dynamical_grooming.Initialize()

    # Hardest kt cross check task
    if analysis_mode == AnalysisMode.PbPb:
        # task = _run_add_task_macro(
        #    "$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskJetHardestKt.C", "PWGJE::EMCALJetTasks::AliAnalysisTaskJetHardestKt",
        #    f"Jet_AKTChargedR{jet_R_str}_tracks_pT0150_E_schemeConstSub",
        #    f"Jet_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme", "", "", jet_R, "Rho",
        #    "tracksSubR02", "tracks", "", "", "", "TPC", "V0M", physics_selection,
        #    CppEnum("PWGJE::EMCALJetTasks::AliAnalysisTaskJetHardestKt::kData"),
        #    CppEnum("PWGJE::EMCALJetTasks::AliAnalysisTaskJetHardestKt::kEventSub"),
        #    CppEnum("PWGJE::EMCALJetTasks::AliAnalysisTaskJetHardestKt::kInclusive"),
        #    0, 0, 0.6,
        #    CppEnum("PWGJE::EMCALJetTasks::AliAnalysisTaskJetHardestKt::kSecondOrder"),
        #    "Raw"
        # )
        hardest_kt = ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.AddTaskJetHardestKt(
            f"Jet_AKTChargedR{jet_R_str}_tracks_pT0150_E_schemeConstSub",
            f"Jet_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
            "",
            "",
            jet_R,
            "Rho",
            "tracksSubR02",
            "tracks",
            "",
            "",
            "",
            "TPC",
            "V0M",
            physics_selection,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kData,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kEventSub,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kInclusive,
            0,
            0,
            0.6,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kSecondOrder,
            "Raw",
        )
    elif analysis_mode == AnalysisMode.pp:
        # hardest_kt = ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.AddTaskJetHardestKt(
        #    f"Jet_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
        #    "", "", "", jet_R, "",
        #    "tracks", "", "", "", "", "TPC", "V0M", physics_selection,
        #    ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kData,
        #    ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kNoSub,
        #    ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kInclusive,
        #    0, 0, 0.6,
        #    ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kSecondOrder,
        #    "Raw"
        # )
        hardest_kt = ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.AddTaskJetHardestKt(
            f"Jet_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
            "",
            "",
            "",
            jet_R,
            "",
            "tracks",
            "",
            "",
            "",
            "",
            "TPC",
            "V0M",
            physics_selection,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kData,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kNoSub,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kInclusive,
            0,
            0,
            0.6,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kSecondOrder,
            "Raw",
        )
    elif analysis_mode == AnalysisMode.pythia:
        hardest_kt = ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.AddTaskJetHardestKt(
            f"Jet_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
            "",
            f"Jet_AKTChargedR{jet_R_str}_mcparticles_pT{particle_level_min_pt_str}_E_scheme",
            "",
            jet_R,
            "tracks",
            "",
            "",
            "mcparticles",
            "",
            "",
            "TPC",
            "V0M",
            physics_selection,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kPythiaDef,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kNoSub,
            ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kInclusive,
        )
    hardest_kt.SelectCollisionCandidates(physics_selection)
    if analysis_mode in [AnalysisMode.pythia, AnalysisMode.embed_pythia]:
        _handle_special_event_selection_for_pythia(period=period, task=hardest_kt)
        hardest_kt.SetNumberOfPtHardBins(len(pt_hard_binning) - 1)
        hardest_kt.SetUserPtHardBinning(pt_hard_binning_root)
    cont = hardest_kt.GetJetContainer(0)
    if beam_type == BeamType.PbPb:
        cont.SetRhoName("Rho")
        cont.SetRhoMassName("RhoMass")
    cont.SetJetRadius(jet_R)
    cont.SetJetAcceptanceType(ROOT.AliJetContainer.kTPCfid)
    cont.SetMaxTrackPt(100)
    cont.SetJetPtCut(0)

    print(f"beam_type: {beam_type}")
    # It only matters for embedding, but in any case, always disable double counting.
    hardest_kt.SetCutDoubleCounts(False)
    if beam_type == BeamType.PbPb:
        hardest_kt.SetCentralitySelectionOn(True)
        hardest_kt.SetUseNewCentralityEstimation(True)
        hardest_kt.SetMinCentrality(centrality_min)
        hardest_kt.SetMaxCentrality(centrality_max)
    else:
        # Need to disable here because it's on by default!!
        hardest_kt.SetCentralitySelectionOn(False)

    hardest_kt.SetJetPtThreshold(grooming_jet_pt_threshold)
    hardest_kt.SetNeedEmcalGeom(False)
    # hardest_kt.SetZvertexDiffValue(0.1)

    # hardest_kt.SetDoTwoTrack(kTRUE)
    # The hard cutoff isn't meaningful when storing all splittings.
    # hardest_kt.SetHardCutoff(0.1)

    if is_MC:
        hardest_kt.SetIsPythia(True)

    # Finally, set the grooming method
    hardest_kt.SetGroomingMethod(ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kDynamicalKt)
    hardest_kt.Initialize()

    tasks = analysis_manager.GetTasks()
    for i in range(tasks.GetEntries()):
        task = tasks.At(i)
        if not task:
            continue
        if task.InheritsFrom("AliAnalysisTaskEmcal"):
            task.SetForceBeamType(beam_type.value)
            print(f"Setting beam type {beam_type.name} for task {task.GetName()}")
        if task.InheritsFrom("AliEmcalCorrectionTask"):
            # TODO: This may not work because the typing isn't correct...
            task.SetForceBeamType(beam_type.value)
            print(f"Setting beam type {beam_type.name} for task {task.GetName()}")

    # Abort if the initialization fails.
    if not analysis_manager.InitAnalysis():
        return

    analysis_manager.PrintStatus()
    analysis_manager.SetUseProgressBar(True, 250)

    # Write out the analysis manager.
    out_file = ROOT.TFile("train.root", "RECREATE")
    out_file.cd()
    analysis_manager.Write()
    out_file.Close()

    return analysis_manager


def run_dynamical_grooming_embedding(  # noqa: C901
    task_name: str, analysis_mode: AnalysisMode, period: str, physics_selection: int, data_type: DataType, jet_R: float = 0.2,
    grooming_jet_pt_threshold: float = 20,
    validation_mode: bool = True,
    embed_input_filename: Path = Path("embedding/embedding_file_list.txt"),
    embedding_helper_config_filename: Optional[Path] = None,
) -> AnalysisManager:
    """Run dynamical grooming embedding.

    Note:
        We don't add the track skim here because it doesn't really make sense. We only run the track skim
        on the true datasets.

    Args:
        task_name: Name of the analysis (ie. given to the analysis manager).
        analysis_mode: Mode of analysis.
        period: Run period.
        physics_selection: Physics selection to apply to the analysis.
        data_type: ALICE data type over which the analysis will run.
        jet_R: Jet R. Default: 0.2
        grooming_jet_pt_threshold: Jet pt threshold for grooming tasks. Default: 20
        validation_mode: If True, adjust some settings to enable validation mode.
            See the setup for the exact settings that are adjusted. Default: True

    Returns:
        The analysis manager.
    """
    # Delay import to avoid explicit dependence
    import ROOT  # pyright: ignore [reportMissingImports]
    # Validation
    period = _normalize_period(period)
    is_MC = _is_MC(period)
    is_run2_data = _is_run2_data(period) if not is_MC else False
    # Determine the beam type from the period.
    beam_type = BeamType.from_period(period)
    if not embedding_helper_config_filename:
        embedding_helper_config_filename = Path("embedding/embeddingHelper_LHC18_LHC20g4_kSemiCentral.yaml")

    # Setup
    # pt hard binning is for pp MC, embedding
    if analysis_mode in [AnalysisMode.pythia, AnalysisMode.embed_pythia]:
        # Unfortunately, we have to pass a TArrayI because that is the only accepted type...
        pt_hard_binning = np.array(
            [0, 5, 7, 9, 12, 16, 21, 28, 36, 45, 57, 70, 85, 99, 115, 132, 150, 169, 190, 212, 235, 1000],
            dtype=np.int32,
        )
        pt_hard_binning_root = ROOT.TArrayI(len(pt_hard_binning), pt_hard_binning)  # noqa: F841
    # Centrality ranges
    # This is rather lazy, but works for these purposes.
    centrality_min, centrality_max = 0, 10
    if ROOT.AliVEvent.kSemiCentral & physics_selection:
        centrality_min, centrality_max = 30, 50
    print(f"Centrality range: {centrality_min}-{centrality_max}")
    # jet R needs to be formatted as a string too
    jet_R_str = f"{round(jet_R*100):03}"
    # Setup for validation mode
    particle_level_min_pt = 0.
    if validation_mode:
        # In the past, we've run the AliPhysics analysis tasks with min of 0. However, as of Feb 2022, the track
        # skims tend to go down to 150 MeV even at track level to keep the data volume reasonable. For the sake
        # of validation, we set the AliPhysics tasks to go a min of 150 MeV, and can consider a re-skim later.
        particle_level_min_pt = 0.15
    particle_level_min_pt_str = f"{round(particle_level_min_pt*1000):04}"

    # Basic setup (analysis manager and input handler).
    analysis_manager = ROOT.AliAnalysisManager(task_name)
    if data_type == DataType.AOD:
        ROOT.AliAnalysisTaskEmcal.AddAODHandler()
    else:
        ROOT.AliAnalysisTaskEmcal.AddESDHandler()

    # Physics selection task
    _add_physics_selection(is_MC, beam_type)

    # Multiplicity selection task.
    _add_mult_selection(is_run2_data=is_run2_data, physics_selection=physics_selection)

    # Embedding helper
    embedding_helper = ROOT.AliAnalysisTaskEmcalEmbeddingHelper.AddTaskEmcalEmbeddingHelper()
    embedding_helper.SelectCollisionCandidates(physics_selection)
    # From the ConfigureWagon
    # Use to configure test of a specific pT Hard bin
    # embedding_helper.SetAutoConfigurePtHardBins(true)
    # embedding_helper.SetPtHardBin(14)

    embedding_helper.SetNPtHardBins(21)
    # embedding_helper.SetFilePattern("alien:///alice/sim/2019/LHC19f4_2/%d/")
    ##embedding_helper.SetInputFilename("*/AOD/*/aod_archive.zip")
    # embedding_helper.SetInputFilename("*/AOD/*/AliAOD.root")
    # embedding_helper.SetInputFilename("*/AOD/*/AliAOD.root")
    if validation_mode:
        embedding_helper.SetTriggerMask(ROOT.AliVEvent.kAnyINT)  # Match what we expect from the skim
    else:
        # embedding_helper.SetTriggerMask(ROOT.AliVEvent.kAny)  # Equivalent to 0
        embedding_helper.SetTriggerMask(0xFFFFFFFF)  # Equivalent to 0

    embedding_helper.SetRandomFileAccess(False)
    embedding_helper.SetRandomEventNumberAccess(False)
    embedding_helper.SetMCRejectOutliers()
    embedding_helper.SetPtHardJetPtRejectionFactor(4.0)
    # embedding_helper.SetZVertexCut(10.)

    # Set YAML configuration, including good runlist of embedded events, and internal event selection feature
    # embedding_helper.SetConfigurationPath("alien:///alice/cern.ch/user/l/lhavener/embeddingHelper_LHC18_LHC19f4_kCentral.yaml")
    # NOTE: This is fine for LHC20g4 also - it's not at all MC specific.
    #       However, it is specific to the centrality range
    embedding_helper.SetConfigurationPath(str(embedding_helper_config_filename))
    # Set the pt hard auto config identifier
    # embedding_helper.SetAutoConfigureIdentifier("20200514Raymond")

    # Ensure that the internal and external event acceptances are similar (within 4 cm)
    # Do _not_ use this option - it wastes too many embedded events and never finishes!
    # embedding_helper.SetMaxVertexDistance(4)

    embedding_helper.SetFileListFilename(str(embed_input_filename))
    embedding_helper.Initialize()

    ################
    # Debug settings
    ################
    # ROOT.AliLog.SetClassDebugLevel("AliEmcalCorrectionComponent", AliLog::kDebug+3)
    # ROOT.AliLog.SetClassDebugLevel("AliAnalysisTaskEmcalJetHCorrelations", AliLog::kDebug+1)
    # ROOT.AliLog.SetClassDebugLevel("PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming", ROOT.AliLog.kDebug-1)
    # ROOT.AliLog.SetClassDebugLevel("PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming", ROOT.AliLog.kDebug-1)
    # ROOT.AliLog.SetClassDebugLevel("AliJetContainer", AliLog::kDebug+7)

    # Shared jet finding settings
    ghost_area = 0.005

    # Rho related
    # Appears to be the same for PbPb and embed_pythia.
    # Jet finder for rho. Wagon name: "JetFinderKtCharged_R02_EschemeNew"
    kt_jet_finder = ROOT.AliEmcalJetTask.AddTaskEmcalJet(
        "usedefault",
        "",
        ROOT.AliJetContainer.kt_algorithm,
        0.2,
        ROOT.AliJetContainer.kChargedJet,
        0.15,
        0,
        ghost_area,
        ROOT.AliJetContainer.E_scheme,
        "Jet",
        0,
        False,
        False,
    )
    kt_jet_finder.SelectCollisionCandidates(physics_selection)
    kt_jet_finder.SetUseNewCentralityEstimation(True)

    # Rho. Wagon name: RhoESchemeNewSubstructure
    rho_task = _run_add_task_macro(
        "$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskRhoNew.C",
        "AliAnalysisTaskRho",
        "usedefault",
        "",
        "Rho",
        0.2,
        CppEnum("AliJetContainer::kTPCfid"),
        CppEnum("AliJetContainer::kChargedJet"),
        True,
        CppEnum("AliJetContainer::E_scheme"),
    )
    rho_task.SetExcludeLeadJets(2)
    rho_task.SelectCollisionCandidates(physics_selection)
    rho_task.SetUseNewCentralityEstimation(True)
    rho_task.SetNCentBins(5)
    rho_task.SetNeedEmcalGeom(False)
    rho_task.SetZvertexDiffValue(0.1)
    cont_rho = rho_task.GetJetContainer(0)
    cont_rho.SetJetRadius(0.2)
    cont_rho.SetJetAcceptanceType(ROOT.AliJetContainer.kTPCfid)
    cont_rho.SetMaxTrackPt(100)

    # Rho mass. Wagon name: "RhoMassEschemeNewSubstructure"
    rho_mass = _run_add_task_macro(
        "$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskRhoMass.C",
        "AliAnalysisTaskRhoMass",
        "Jet_KTChargedR020_tracks_pT0150_E_scheme",
        "tracks",
        "",
        "Rhomass",
        0.2,
        "TPC",
        0.01,
        0,
        0,
        2,
        True,
        "RhoMass",
    )
    rho_mass.SelectCollisionCandidates(physics_selection)
    rho_mass.SetUseNewCentralityEstimation(True)
    rho_mass.SetHistoBins(250, 0, 250)
    rho_mass.SetNeedEmcalGeom(False)
    cont_rho_mass = rho_mass.GetJetContainer(0)
    cont_rho_mass.SetJetRadius(0.2)
    cont_rho_mass.SetJetAcceptanceType(ROOT.AliJetContainer.kTPCfid)
    cont_rho_mass.SetMaxTrackPt(100)
    rho_mass.SetNeedEmcalGeom(False)
    rho_mass.SetZvertexDiffValue(0.1)
    # rho_mass.SetVzRange(-10,10)

    # Standard akt jet finder. Wagon name: "JetFinderAKTCharged_R04_EschemeNew_EventWise"
    akt_jet_finder = ROOT.AliEmcalJetTask.AddTaskEmcalJet(
        "tracks",
        "",
        ROOT.AliJetContainer.antikt_algorithm,
        jet_R,
        ROOT.AliJetContainer.kChargedJet,
        0.15,
        0.3,
        ghost_area,
        ROOT.AliJetContainer.E_scheme,
        "hybridLevelJets",
        0,
        False,
        False,
    )
    akt_jet_finder.SelectCollisionCandidates(physics_selection)
    akt_jet_finder.SetUseNewCentralityEstimation(True)

    # This introduces randomness, so we need to disable it during validation mode
    if not validation_mode:
        # Artificial tracking efficiency derived from the full jet R_AA paper.
        # 10-30 \approx .99
        # 0-10% \approx .98
        # Apply only to embedded particles
        akt_jet_finder.SetTrackEfficiency(0.98)
        akt_jet_finder.SetTrackEfficiencyOnlyForEmbedding(True)

    constUtil = akt_jet_finder.AddUtility(ROOT.AliEmcalJetUtilityEventSubtractor("EventSubtractor"))
    genSub = akt_jet_finder.AddUtility(ROOT.AliEmcalJetUtilityGenSubtractor("GenSubtractor"))
    genSub.SetUseExternalBkg(True)
    genSub.SetRhoName("Rho")
    genSub.SetRhomName("Rhomass")
    constUtil.SetJetsSubName(f"{akt_jet_finder.GetName()}ConstSub")

    constUtil.SetParticlesSubName("tracksSubR04")
    constUtil.SetUseExternalBkg(True)
    constUtil.SetRhoName("Rho")
    constUtil.SetRhomName("Rhomass")
    constUtil.SetMaxDelR(0.25)
    # Particle containers
    tracksDetLevel = ROOT.AliTrackContainer("tracks")
    # Get the det level tracks from the external event!
    tracksDetLevel.SetIsEmbedding(True)
    akt_jet_finder.AdoptTrackContainer(tracksDetLevel)

    # Particle level jet finder: "JetFinderAKTCharged_R04_EschemePartLevelNew"
    akt_particle_level_jet_finder = ROOT.AliEmcalJetTask.AddTaskEmcalJet(
        "mcparticles",
        "",
        ROOT.AliJetContainer.antikt_algorithm,
        jet_R,
        ROOT.AliJetContainer.kChargedJet,
        particle_level_min_pt,
        0.0,
        ghost_area,
        ROOT.AliJetContainer.E_scheme,
        "partLevelJets",
        0,
        False,
        False,
    )
    akt_particle_level_jet_finder.SelectCollisionCandidates(physics_selection)
    akt_particle_level_jet_finder.SetUseNewCentralityEstimation(True)

    # Setup the tracks properly to be retrieved from the external event
    truth_tracks = akt_particle_level_jet_finder.GetMCParticleContainer(0)
    truth_tracks.SetIsEmbedding(True)

    # Detector level jet finder: JetFinderAKTCharged_R04_EschemeDetLevelNew
    akt_detector_level_jet_finder = ROOT.AliEmcalJetTask.AddTaskEmcalJet(
        "usedefault",
        "",
        ROOT.AliJetContainer.antikt_algorithm,
        jet_R,
        ROOT.AliJetContainer.kChargedJet,
        0.15,
        0.3,
        ghost_area,
        ROOT.AliJetContainer.E_scheme,
        "detLevelJets",
        0,
        False,
        False,
    )
    akt_detector_level_jet_finder.SelectCollisionCandidates(physics_selection)
    akt_detector_level_jet_finder.SetUseNewCentralityEstimation(True)

    # Setup the tracks properly to be retrieved from the external event
    tracks = akt_detector_level_jet_finder.GetParticleContainer(0)
    tracks.SetIsEmbedding(True)

    # Taggers
    # Hybrid unsubtracted to subtracted tagger.
    # JetTaggerR04HybridSubUnSubNew_EventWise
    hybrid_sub_tagger = _run_add_task_macro(
        "$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskEmcalJetTagger.C",
        "AliAnalysisTaskEmcalJetTagger",
        f"hybridLevelJets_AKTChargedR{jet_R_str}_tracks_pT0150_E_schemeConstSub",
        f"hybridLevelJets_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
        jet_R,
        "Rho",
        "",
        "tracksSubR04",
        "",
        "TPCfid",
        "V0M",
        physics_selection,
        "",
    )

    hybrid_sub_tagger.SetJetTaggingType(ROOT.AliAnalysisTaskEmcalJetTagger.kClosest)
    hybrid_sub_tagger.SetJetTaggingMethod(ROOT.AliAnalysisTaskEmcalJetTagger.kGeo)
    hybrid_sub_tagger.SetTypeAcceptance(0)

    # Task level settings
    # Use default matching distance
    hybrid_sub_tagger.SetMaxDistance(0.3)
    # Redundant, but done for completeness
    hybrid_sub_tagger.SelectCollisionCandidates(physics_selection)
    hybrid_sub_tagger.SetUseNewCentralityEstimation(True)

    # Pythia detector to hybrid level.
    # JetTaggerR04DetHybridNew_EventWise
    detector_hybrid_tagger = _run_add_task_macro(
        "$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskEmcalJetTagger.C",
        "AliAnalysisTaskEmcalJetTagger",
        f"hybridLevelJets_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
        f"detLevelJets_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
        jet_R,
        "Rho",
        "",
        "tracks",
        "",
        "TPCfid",
        "V0M",
        physics_selection,
        "",
    )
    detector_hybrid_tagger.SetJetTaggingType(ROOT.AliAnalysisTaskEmcalJetTagger.kClosest)
    detector_hybrid_tagger.SetJetTaggingMethod(ROOT.AliAnalysisTaskEmcalJetTagger.kGeo)
    detector_hybrid_tagger.SetTypeAcceptance(0)

    # Task level settings
    # Use default matching distance
    detector_hybrid_tagger.SetMaxDistance(0.3)
    # Redundant, but done for completeness
    detector_hybrid_tagger.SelectCollisionCandidates(physics_selection)
    detector_hybrid_tagger.SetUseNewCentralityEstimation(True)
    # Reapply the max track pt cut off to maintain energy resolution and avoid fake tracks
    hybridJetCont = detector_hybrid_tagger.GetJetContainer(0)
    hybridJetCont.SetMaxTrackPt(100)
    detLevelJetCont = detector_hybrid_tagger.GetJetContainer(1)
    detLevelJetCont.SetMaxTrackPt(100)

    # Pythia detector to particle level tagger
    # JetTaggerR04DetPartNew_EventWise
    particle_detector_tagger = _run_add_task_macro(
        "$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskEmcalJetTagger.C",
        "AliAnalysisTaskEmcalJetTagger",
        f"detLevelJets_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
        f"partLevelJets_AKTChargedR{jet_R_str}_mcparticles_pT{particle_level_min_pt_str}_E_scheme",
        jet_R,
        "",
        "",
        "tracks",
        "",
        "TPCfid",
        "V0M",
        physics_selection,
        "",
    )

    particle_detector_tagger.SetJetTaggingType(ROOT.AliAnalysisTaskEmcalJetTagger.kClosest)
    particle_detector_tagger.SetJetTaggingMethod(ROOT.AliAnalysisTaskEmcalJetTagger.kGeo)
    particle_detector_tagger.SetTypeAcceptance(0)

    # Tag via geometrical matching
    # particle_detector_tagger.SetJetTaggingMethod(PWGJE::EMCALJetTasks::AliEmcalJetTaggerTask::kGeo)
    # Tag the closest jet
    # particle_detector_tagger.SetJetTaggingType(PWGJE::EMCALJetTasks::AliEmcalJetTaggerTask::kClosest)
    # Don't impose any additional acceptance cuts beyond the jet containers
    # particle_detector_tagger.SetTypeAcceptance(PWGJE::EMCALJetTasks::AliEmcalJetTaggerTask::kNoLimit)
    # Use default matching distance
    particle_detector_tagger.SetMaxDistance(0.3)
    # Redundant, but done for completeness
    particle_detector_tagger.SelectCollisionCandidates(physics_selection)
    particle_detector_tagger.SetUseNewCentralityEstimation(True)
    # Reapply the max track pt cut off to maintain energy resolution and avoid fake tracks
    detLevelJetCont = particle_detector_tagger.GetJetContainer(0)
    detLevelJetCont.SetMaxTrackPt(100)
    partLevelJetCont = particle_detector_tagger.GetJetContainer(1)
    partLevelJetCont.SetMaxTrackPt(1000)

    # Dynamical grooming
    # task = _run_add_task_macro(
    #    "$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskJetDynamicalGrooming.C", "PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming",
    #    f"Jet_AKTChargedR{jet_R_str}_tracks_pT0150_E_schemeConstSub",
    #    f"Jet_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme", "", "", jet_R, "Rho",
    #    "tracksSubR02", "tracks", "", "", "", "TPC", "V0M", physics_selection,
    #    CppEnum("PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming::kData"),
    #    CppEnum("PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming::kEventSub"),
    #    CppEnum("PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming::kInclusive"),
    #    0, 0, 0.6,
    #    CppEnum("PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming::kSecondOrder"),
    #    "Raw"
    # )
    dynamical_grooming = ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.AddTaskJetDynamicalGrooming(
        f"hybridLevelJets_AKTChargedR{jet_R_str}_tracks_pT0150_E_schemeConstSub",
        f"hybridLevelJets_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
        f"detLevelJets_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
        f"partLevelJets_AKTChargedR{jet_R_str}_mcparticles_pT{particle_level_min_pt_str}_E_scheme",
        jet_R,
        "Rho",
        "tracksSubR04",
        "tracks",
        "mcparticles",
        "",
        "mcparticles",
        "TPC",
        "V0M",
        physics_selection,
        ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kDetEmbPartPythia,
        ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kEventSub,
        ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kInclusive,
        0,
        0,
        0.6,
        ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetDynamicalGrooming.kSecondOrder,
        "Raw",
    )
    cont = dynamical_grooming.GetJetContainer(0)
    cont.SetRhoName("Rho")
    cont.SetRhoMassName("RhoMass")
    cont.SetJetRadius(jet_R)
    cont.SetJetAcceptanceType(ROOT.AliJetContainer.kTPCfid)
    cont.SetMaxTrackPt(100)

    detLevelJetCont = dynamical_grooming.GetJetContainer(2)
    detLevelJetCont.SetMaxTrackPt(100)

    partLevelJetCont = dynamical_grooming.GetJetContainer(3)
    partLevelJetCont.SetMaxTrackPt(1000)
    dynamical_grooming.SetUseNewCentralityEstimation(True)
    dynamical_grooming.SetJetPtThreshold(grooming_jet_pt_threshold)
    dynamical_grooming.SetCutDoubleCounts(False)
    dynamical_grooming.SetCheckResolution(False)
    dynamical_grooming.SelectCollisionCandidates(physics_selection)
    dynamical_grooming.SetNeedEmcalGeom(False)
    dynamical_grooming.SetMinCentrality(centrality_min)
    dynamical_grooming.SetMaxCentrality(centrality_max)
    dynamical_grooming.SetMinFractionShared(0.5)
    dynamical_grooming.SetDetLevelJetsOn(True)
    dynamical_grooming.SetStoreRecursiveJetSplittings(True)
    dynamical_grooming.Initialize()

    # Hardest kt cross check task.
    hardest_kt = ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.AddTaskJetHardestKt(
        f"hybridLevelJets_AKTChargedR{jet_R_str}_tracks_pT0150_E_schemeConstSub",
        f"hybridLevelJets_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
        f"detLevelJets_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
        f"partLevelJets_AKTChargedR{jet_R_str}_mcparticles_pT{particle_level_min_pt_str}_E_scheme",
        jet_R,
        "Rho",
        "tracksSubR04",
        "tracks",
        "mcparticles",
        "",
        "mcparticles",
        "TPC",
        "V0M",
        physics_selection,
        ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kDetEmbPartPythia,
        ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kEventSub,
        ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kInclusive,
        0,
        0,
        0.6,
        ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kSecondOrder,
        "Raw",
    )
    cont = hardest_kt.GetJetContainer(0)
    cont.SetRhoName("Rho")
    cont.SetRhoMassName("RhoMass")
    cont.SetJetRadius(jet_R)
    cont.SetJetAcceptanceType(ROOT.AliJetContainer.kTPCfid)
    cont.SetMaxTrackPt(100)

    detLevelJetCont = hardest_kt.GetJetContainer(2)
    detLevelJetCont.SetMaxTrackPt(100)

    partLevelJetCont = hardest_kt.GetJetContainer(3)
    partLevelJetCont.SetMaxTrackPt(1000)
    hardest_kt.SetUseNewCentralityEstimation(True)
    hardest_kt.SetJetPtThreshold(grooming_jet_pt_threshold)
    hardest_kt.SetCutDoubleCounts(False)
    hardest_kt.SetCheckResolution(False)
    hardest_kt.SelectCollisionCandidates(physics_selection)
    hardest_kt.SetNeedEmcalGeom(False)
    hardest_kt.SetMinCentrality(centrality_min)
    hardest_kt.SetMaxCentrality(centrality_max)
    hardest_kt.SetMinFractionShared(0.5)
    hardest_kt.SetDetLevelJetsOn(True)
    hardest_kt.SetEnableSubjetMatching(True)
    hardest_kt.SetHardCutoff(0.2)
    hardest_kt.SetGroomingMethod(ROOT.PWGJE.EMCALJetTasks.AliAnalysisTaskJetHardestKt.kDynamicalCore)
    hardest_kt.Initialize()
    print(hardest_kt.GroomingMethodName())

    # Setup L+L substructure task.
    ll_substructure = _run_add_task_macro(
        "$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskNewJetSubstructure.C",
        "AliAnalysisTaskNewJetSubstructure",
        f"hybridLevelJets_AKTChargedR{jet_R_str}_tracks_pT0150_E_schemeConstSub",
        f"hybridLevelJets_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
        f"detLevelJets_AKTChargedR{jet_R_str}_tracks_pT0150_E_scheme",
        f"partLevelJets_AKTChargedR{jet_R_str}_mcparticles_pT{particle_level_min_pt_str}_E_scheme",
        0.2,
        "Rho",
        "tracksSubR04",
        "tracks",
        "mcparticles",
        "",
        "mcparticles",
        "TPC",
        "V0M",
        physics_selection,
        "",
        "",
        "Raw",
        CppEnum("AliAnalysisTaskNewJetSubstructure::kDetEmbPartPythia"),
        CppEnum("AliAnalysisTaskNewJetSubstructure::kEventSub"),
        CppEnum("AliAnalysisTaskNewJetSubstructure::kInclusive"),
    )
    cont = ll_substructure.GetJetContainer(0)
    cont.SetRhoName("Rho")
    cont.SetRhoMassName("RhoMass")
    cont.SetJetRadius(jet_R)
    cont.SetJetAcceptanceType(ROOT.AliJetContainer.kTPCfid)
    cont.SetMaxTrackPt(100)

    detLevelJetCont = ll_substructure.GetJetContainer(2)
    detLevelJetCont.SetMaxTrackPt(100)

    partLevelJetCont = ll_substructure.GetJetContainer(3)
    partLevelJetCont.SetMaxTrackPt(1000)
    ll_substructure.SetUseNewCentralityEstimation(True)
    ll_substructure.SetJetPtThreshold(grooming_jet_pt_threshold)
    ll_substructure.SetCutDoubleCounts(False)
    ll_substructure.SetCheckResolution(True)
    ll_substructure.SelectCollisionCandidates(physics_selection)
    ll_substructure.SetNeedEmcalGeom(True)
    ll_substructure.SetHardCutoff(0.2)
    ll_substructure.SetMinCentrality(centrality_min)
    ll_substructure.SetMaxCentrality(centrality_max)
    ll_substructure.SetPowerAlgorithm(0)
    ll_substructure.SetMinFractionShared(0.5)
    ll_substructure.SetDetLevelJetsOn(True)

    tasks = analysis_manager.GetTasks()
    for i in range(tasks.GetEntries()):
        task = tasks.At(i)
        if not task:
            continue
        if task.InheritsFrom("AliAnalysisTaskEmcal"):
            task.SetForceBeamType(beam_type.value)
            print(f"Setting beam type {beam_type.name} for task {task.GetName()}")
        if task.InheritsFrom("AliEmcalCorrectionTask"):
            # TODO: This may not work because the typing isn't correct...
            task.SetForceBeamType(beam_type.value)
            print(f"Setting beam type {beam_type.name} for task {task.GetName()}")

    # Abort if the initialization fails.
    if not analysis_manager.InitAnalysis():
        return

    analysis_manager.PrintStatus()
    analysis_manager.SetUseProgressBar(True, 250)

    # Write out the analysis manager.
    out_file = ROOT.TFile("train.root", "RECREATE")
    out_file.cd()
    analysis_manager.Write()
    out_file.Close()

    return analysis_manager


def start_analysis_manager(
    analysis_manager: AnalysisManager, mode: str, n_events: int, input_files: Sequence[Path]
) -> None:
    """Start the given analysis manager.

    Note:
        The only mode current supported is local.

    Args:
        analysis_manager: Analysis manager to start.
        mode: Execution mode for the analysis manager.
        n_events: Number of events to run.
        input_files: Input files to be run over.

    Returns:
        None. The analysis manager is executed.
    """
    # Delay import to avoid explicit dependence
    import ROOT  # pyright: ignore [reportMissingImports]

    if mode == "local":
        # The progress bar has to be disabled for the debug level to be set. For reasons....
        # analysis_manager.SetUseProgressBar(False, 250)
        # analysis_manager.SetDebugLevel(10);
        print("Starting Analysis...")
        # Create chian from input files
        chain = ROOT.TChain("aodTree")
        for filename in input_files:
            chain.AddFile(str(filename))

        friend_tree = ROOT.TChain("aodTree", "AliAOD.VertexingHF.root")
        for filename in input_files:
            # Add HF vertexing for skimming. Needs aod_archive.zip or root_archive.zip
            temp_filename = Path(str(Path(filename)).replace("AliAOD.root", "AliAOD.VertexingHF.root"))
            print(f"temp_filename: {temp_filename}")
            friend_tree.AddFile(str(temp_filename))
        # Add friends with scale factors
        chain.AddFriend(friend_tree)

        # ROOT.AliLog.SetClassDebugLevel("AliAnalysisTaskEmcalEmbeddingHelper", ROOT.AliLog.kDebug+4)
        # ROOT.AliLog.SetClassDebugLevel("AliAnalysisTaskEmcal", 10)
        # ROOT.AliLog.SetClassDebugLevel("AliAnalysisTaskEmcal", ROOT.AliLog.kDebug+10)
        # ROOT.AliLog.SetClassDebugLevel("PWGJE::EMCALJetTasks::AliAnalysisTaskJetDynamicalGrooming", ROOT.AliLog.kDebug+5)
        # Start the analysis
        analysis_manager.StartAnalysis("local", chain, n_events)
    elif mode == "grid":
        raise RuntimeError("Not implemented yet!")


#def _run_analysis(
#    analysis_mode: AnalysisMode,
#    period_name: str,
#    physics_selection: int,
#    input_files: Sequence[Path],
#    grooming_jet_pt_threshold: float,
#    jet_R: float,
#    validation_mode: bool,
#) -> None:
#    # Setup and validation
#    task_name = "DynamicalGrooming"
#    period = _normalize_period(period_name)
#    data_type = DataType.AOD
#    ROOT.AliTrackContainer.SetDefTrackCutsPeriod(period)
#
#    analysis_manager = run_dynamical_grooming(
#        task_name=task_name,
#        analysis_mode=analysis_mode,
#        period=period,
#        physics_selection=physics_selection,
#        data_type=data_type,
#        enable_track_skim=True,
#        grooming_jet_pt_threshold=grooming_jet_pt_threshold,
#        jet_R=jet_R,
#        validation_mode=validation_mode,
#    )
#
#    start_analysis_manager(analysis_manager=analysis_manager, mode="local", n_events=1000, input_files=input_files)
#
#
#def _run_embedding_analysis(
#    analysis_mode: AnalysisMode,
#    period_name: str,
#    physics_selection: int,
#    input_files: Sequence[Path],
#    grooming_jet_pt_threshold: float,
#    jet_R: float,
#    validation_mode: bool,
#    embed_input_files: Optional[Sequence[Path]] = None
#) -> None:
#    # Setup and validation
#    task_name = "DynamicalGrooming"
#    period = _normalize_period(period_name)
#    data_type = DataType.AOD
#    ROOT.AliTrackContainer.SetDefTrackCutsPeriod(period)
#
#    analysis_manager = run_dynamical_grooming_embedding(
#        task_name=task_name,
#        analysis_mode=analysis_mode,
#        period=period,
#        physics_selection=physics_selection,
#        data_type=data_type,
#        jet_R=jet_R,
#        grooming_jet_pt_threshold=grooming_jet_pt_threshold,
#        validation_mode=validation_mode,
#    )
#
#    start_analysis_manager(analysis_manager=analysis_manager, mode="local", n_events=1000, input_files=input_files)


@attr.define
class AnalysisModeParameters:
    analysis_mode: AnalysisMode
    period_name: str
    _physics_selection: str
    # As a function of jet_R
    grooming_jet_pt_threshold: Dict[float, float]
    input_files: List[Path]
    embed_input_files: List[Path] = attr.Factory(list)

    @property
    def physics_selection(self) -> Any:
        import ROOT  # pyright: ignore [reportMissingImports]
        return getattr(ROOT.AliVEvent, self._physics_selection)


default_analysis_parameters = {
    "pp": AnalysisModeParameters(
        analysis_mode=AnalysisMode.pp,
        period_name="LHC17q",
        physics_selection="kAnyINT",
        grooming_jet_pt_threshold={
            0.2: 5,
            0.4: 5,
        },
        input_files=[
            Path(
                "/alf/data/rehlers/data/alice/data/2017/LHC17p/000282343/pass1_FAST/AOD234/0001/root_archive.zip#AliAOD.root"
            ),
            Path(
                "/alf/data/rehlers/data/alice/data/2017/LHC17p/000282343/pass1_FAST/AOD234/0002/root_archive.zip#AliAOD.root"
            ),
            Path(
                "/alf/data/rehlers/data/alice/data/2017/LHC17p/000282343/pass1_FAST/AOD234/0003/root_archive.zip#AliAOD.root"
            ),
        ],
    ),
    "pythia": AnalysisModeParameters(
        analysis_mode=AnalysisMode.pythia,
        period_name="LHC20g4",
        # physics_selection=0,
        physics_selection="kAnyINT",
        grooming_jet_pt_threshold={
            0.2: 10,
            0.4: 20,
        },
        input_files=[
            Path(
                "/alf/data/rehlers/data/alice/sim/2020/LHC20g4/12/296191/AOD/001/aod_archive.zip#AliAOD.root"
            ),
            #Path(
            #    "/alf/data/rehlers/data/alice/sim/2020/LHC20g4/12/296191/AOD/002/aod_archive.zip#AliAOD.root"
            #),
            #Path(
            #    "/alf/data/rehlers/data/alice/sim/2020/LHC20g4/12/296191/AOD/003/aod_archive.zip#AliAOD.root"
            #),

            # Path("/opt/scott/data/rehlers/alice/datasets/data/2016/LHC16j5/4/246945/AOD200/0003/AliAOD.root"),
            # Path("/opt/scott/data/rehlers/alice/datasets/data/2016/LHC16j5/4/246945/AOD200/0002/AliAOD.root"),
            # Path("/opt/scott/data/rehlers/alice/datasets/data/2016/LHC16j5/4/246945/AOD200/0001/AliAOD.root"),
            # Path("/opt/scott/data/rehlers/alice/datasets/data/2016/LHC16j5/5/246945/AOD200/0003/AliAOD.root"),
            # Path("/opt/scott/data/rehlers/alice/datasets/data/2016/LHC16j5/5/246945/AOD200/0005/AliAOD.root"),
            # Path("/opt/scott/data/rehlers/alice/datasets/data/2016/LHC16j5/5/246945/AOD200/0002/AliAOD.root"),
            # Path("/opt/scott/data/rehlers/alice/datasets/data/2016/LHC16j5/5/246945/AOD200/0001/AliAOD.root"),
            # Path("/opt/scott/data/rehlers/alice/datasets/data/2016/LHC16j5/5/246945/AOD200/0006/AliAOD.root"),
            # Path("/Users/re239/code/alice/data/LHC16j5/4/246945/AOD200/0003/AliAOD.root"),
            # Path("/Users/re239/code/alice/data/LHC16j5/4/246945/AOD200/0002/AliAOD.root"),
            # Path("/Users/re239/code/alice/data/LHC16j5/4/246945/AOD200/0001/AliAOD.root"),
            # Path("/Users/re239/code/alice/data/LHC16j5/5/246945/AOD200/0003/AliAOD.root"),
            # Path("/Users/re239/code/alice/data/LHC16j5/5/246945/AOD200/0005/AliAOD.root"),
            # Path("/Users/re239/code/alice/data/LHC16j5/5/246945/AOD200/0002/AliAOD.root"),
            # Path("/Users/re239/code/alice/data/LHC16j5/5/246945/AOD200/0001/AliAOD.root"),
            # Path("/Users/re239/code/alice/data/LHC16j5/5/246945/AOD200/0006/AliAOD.root"),
        ],
    ),
    "PbPb": AnalysisModeParameters(
        analysis_mode=AnalysisMode.PbPb,
        period_name="LHC18q",
        # NOTE: For some reason, kAnyINT will include some events where the centrality seems to be uncalibrated.
        #       It's unclear why this occurs, but since we're only interested in semi-central at the moment, it
        #       doesn't matter.
        physics_selection="kSemiCentral",
        # physics_selection=ROOT.AliVEvent.kSemiCentral | ROOT.AliVEvent.kINT7,
        # physics_selection=ROOT.AliVEvent.kCentral,
        grooming_jet_pt_threshold={
            0.2: 10,
            0.4: 20,
        },
        input_files=[
            Path(
                "/alf/data/rehlers/data/alice/data/2018/LHC18q/000296550/pass3/AOD252/AOD/001/aod_archive.zip#AliAOD.root"
            ),
            # Path(
            #     "/alf/data/rehlers/data/alice/data/2018/LHC18q/000296550/pass3/AOD252/AOD/002/aod_archive.zip#AliAOD.root"
            # ),
            # Path(
            #     "/alf/data/rehlers/data/alice/data/2018/LHC18r/000297595/pass3/AOD252/AOD/001/aod_archive.zip#AliAOD.root"
            # ),
            # Path(
            #     "/alf/data/rehlers/data/alice/data/2018/LHC18r/000297595/pass3/AOD252/AOD/002/aod_archive.zip#AliAOD.root"
            # ),
        ],
    ),
    "embed_pythia": AnalysisModeParameters(
        analysis_mode=AnalysisMode.embed_pythia,
        period_name="LHC18q",
        # NOTE: For some reason, kAnyINT will include some events where the centrality seems to be uncalibrated.
        #       It's unclear why this occurs, but since we're only interested in semi-central at the moment, it
        #       doesn't matter.
        physics_selection="kSemiCentral",
        grooming_jet_pt_threshold={
            0.2: 10,
            0.4: 20,
        },
        input_files=[
            Path(
                "/alf/data/rehlers/data/alice/data/2018/LHC18q/000296550/pass3/AOD252/AOD/001/aod_archive.zip#AliAOD.root"
            ),
            #Path(
            #    "/alf/data/rehlers/data/alice/data/2018/LHC18q/000296550/pass3/AOD252/AOD/002/aod_archive.zip#AliAOD.root"
            #),
            # Path("/opt/scott/data/rehlers/alice/datasets/data/2018/LHC18q/000296550/pass3/AOD252/AOD/003/aod_archive.zip#AliAOD.root"),
            # Path("/opt/scott/data/rehlers/alice/datasets/data/2018/LHC18q/000296550/pass3/AOD252/AOD/004/aod_archive.zip#AliAOD.root"),
        ],
        embed_input_files=[
            Path("/alf/data/rehlers/data/alice/sim/2020/LHC20g4/12/296191/AOD/001/aod_archive.zip#AliAOD.root"),
        ],
    ),
}

def run(
    analysis_mode: str,
    jet_R: float,
    validation_mode: bool,
    input_files: Optional[Sequence[Path]] = None,
    embed_input_files: Optional[Sequence[Path]] = None,
    embedding_helper_config_filename: Optional[Path] = None,
    n_events: int = 1000,
) -> None:
    # Let the user know that ROOT is required if it's not available
    try:
        import ROOT  # pyright: ignore [reportMissingImports]
    except ImportError:
        raise RuntimeError("AliPhysics + ROOT is required to use the run macro. Please check that ROOT is available!")
    # And immediately configure it to run in batch mode
    ROOT.gROOT.SetBatch(True)

    # Validation
    analysis_parameters = default_analysis_parameters[analysis_mode]
    if not input_files:
        input_files = analysis_parameters.input_files
    if not embed_input_files:
        embed_input_files = analysis_parameters.embed_input_files
    validated_analysis_mode = AnalysisMode[analysis_mode]

    # Setup
    task_name = "DynamicalGrooming"
    period = _normalize_period(analysis_parameters.period_name)
    data_type = DataType.AOD
    ROOT.AliTrackContainer.SetDefTrackCutsPeriod(period)

    if validated_analysis_mode == AnalysisMode.embed_pythia:
        # If there is an explicit file list, create a temporary file containing
        # the filenames os we can pass that into the run macro.
        embed_input_filename = Path("embedding/embedding_file_list.txt")
        f_temp = None
        if embed_input_files:
            f_temp = tempfile.NamedTemporaryFile("w+")
            embed_input_filename = Path(f_temp.name)
            # Write the filenames, one per line
            [f_temp.write(str(p)) for p in embed_input_files]
            # Apparently other processes opening this file open it at the same seek point.
            # Or at least it does in the case. So we need to seek to the beginning for it to be read
            f_temp.seek(0)

        analysis_manager = run_dynamical_grooming_embedding(
            task_name=task_name,
            analysis_mode=validated_analysis_mode,
            period=period,
            physics_selection=analysis_parameters.physics_selection,
            data_type=data_type,
            jet_R=jet_R,
            grooming_jet_pt_threshold=analysis_parameters.grooming_jet_pt_threshold[jet_R],
            validation_mode=validation_mode,
            embed_input_filename=embed_input_filename,
            embedding_helper_config_filename=embedding_helper_config_filename,
        )
        # Cleanup. We don't use a with statement here because we might want to pass an existing filename
        if f_temp:
            f_temp.close()
    else:
        analysis_manager = run_dynamical_grooming(
            task_name=task_name,
            analysis_mode=validated_analysis_mode,
            period=period,
            physics_selection=analysis_parameters.physics_selection,
            data_type=data_type,
            enable_track_skim=True,
            grooming_jet_pt_threshold=analysis_parameters.grooming_jet_pt_threshold[jet_R],
            jet_R=jet_R,
            validation_mode=validation_mode,
        )

    # Actually run the analysis
    start_analysis_manager(
        analysis_manager=analysis_manager,
        mode="local",
        n_events=n_events,
        input_files=input_files
    )

def entry_point() -> None:
    parser = argparse.ArgumentParser(description="Execute the run macro")

    parser.add_argument(
        "-a",
        "--analysis-mode",
        "-c",
        "--collision-system",
        action="store",
        dest="analysis_mode",
        type=str,
        required=True,
        help="Collision system",
    )
    parser.add_argument(
        "-r",
        "-R",
        "--jet-R",
        action="store",
        type=float,
        required=True,
        help="Jet R",
    )
    parser.add_argument(
        "--validation-mode",
        action="store_true",
        required=True,
        help="Run with validation mode",
    )
    parser.add_argument(
        "-f",
        "--input-files",
        action="store",
        required=False,
        nargs="+",
        default=None,
        help="Input files. Optional (will use default if not specified)",
    )
    parser.add_argument(
        "-e",
        "--embed-input-files",
        action="store",
        required=False,
        nargs="+",
        default=None,
        help="Embedded input files. Optional (will use default if not specified)",
    )
    parser.add_argument(
        "--embedding-helper-config-filename",
        action="store",
        required=False,
        type=Path,
        default=None,
        help="Path to the embedding helper config",
    )
    parser.add_argument(
        "-n",
        "--n-events",
        action="store",
        type=int,
        required=True,
        default=1000,
        help="Number of events to run",
    )
    args = parser.parse_args()

    # Validation
    if args.input_files:
        args.input_files = [Path(p) for p in args.input_files]
    if args.embed_input_files:
        args.embed_input_files = [Path(p) for p in args.embed_input_files]

    # For user convenience
    pprint.pprint(f"Settings: {args}")

    # And then we're actually ready to go
    run(
        analysis_mode=args.analysis_mode,
        jet_R=args.jet_R,
        validation_mode=args.validation_mode,
        input_files=args.input_files,
        embed_input_files=args.embed_input_files,
        embedding_helper_config_filename=args.embedding_helper_config_filename,
        n_events=args.n_events
    )


if __name__ == "__main__":
    entry_point()
    #run(analysis_mode="embed_pythia", jet_R=0.4, validation_mode=True)

    #analysis_mode = AnalysisMode.embed_pythia
    #jet_R = 0.4
    #validation_mode = True
    #if analysis_mode == AnalysisMode.PbPb:
    #    _run_analysis(
    #        analysis_mode=analysis_mode,
    #        period_name="LHC18q",
    #        # NOTE: For some reason, kAnyINT will include some events where the centrality seems to be uncalibrated.
    #        #       It's unclear why this occurs, but since we're only interested in semi-central at the moment, it
    #        #       doesn't matter.
    #        physics_selection=ROOT.AliVEvent.kSemiCentral,
    #        # physics_selection=ROOT.AliVEvent.kSemiCentral | ROOT.AliVEvent.kINT7,
    #        # physics_selection=ROOT.AliVEvent.kCentral,
    #        grooming_jet_pt_threshold=20,
    #        jet_R=jet_R,
    #        validation_mode=validation_mode,
    #        input_files=[
    #            Path(
    #                "/alf/data/rehlers/data/alice/data/2018/LHC18q/000296550/pass3/AOD252/AOD/001/aod_archive.zip#AliAOD.root"
    #            ),
    #            # Path(
    #            #     "/alf/data/rehlers/data/alice/data/2018/LHC18q/000296550/pass3/AOD252/AOD/002/aod_archive.zip#AliAOD.root"
    #            # ),
    #            # Path(
    #            #     "/alf/data/rehlers/data/alice/data/2018/LHC18r/000297595/pass3/AOD252/AOD/001/aod_archive.zip#AliAOD.root"
    #            # ),
    #            # Path(
    #            #     "/alf/data/rehlers/data/alice/data/2018/LHC18r/000297595/pass3/AOD252/AOD/002/aod_archive.zip#AliAOD.root"
    #            # ),
    #        ],
    #    )
    #if analysis_mode == AnalysisMode.pp:
    #    _run_analysis(
    #        analysis_mode=analysis_mode,
    #        period_name="LHC17q",
    #        physics_selection=ROOT.AliVEvent.kAnyINT,
    #        grooming_jet_pt_threshold=5,
    #        jet_R=jet_R,
    #        validation_mode=validation_mode,
    #        input_files=[
    #            Path(
    #                "/alf/data/rehlers/data/alice/data/2017/LHC17p/000282343/pass1_FAST/AOD234/0001/root_archive.zip#AliAOD.root"
    #            ),
    #            Path(
    #                "/alf/data/rehlers/data/alice/data/2017/LHC17p/000282343/pass1_FAST/AOD234/0002/root_archive.zip#AliAOD.root"
    #            ),
    #            Path(
    #                "/alf/data/rehlers/data/alice/data/2017/LHC17p/000282343/pass1_FAST/AOD234/0003/root_archive.zip#AliAOD.root"
    #            ),
    #        ],
    #    )
    #if analysis_mode == AnalysisMode.pythia:
    #    _run_analysis(
    #        analysis_mode=analysis_mode,
    #        period_name="LHC20g4",
    #        # physics_selection=0,
    #        physics_selection=ROOT.AliVEvent.kAnyINT,
    #        grooming_jet_pt_threshold=20,
    #        jet_R=jet_R,
    #        validation_mode=validation_mode,
    #        input_files=[
    #            Path(
    #                "/alf/data/rehlers/data/alice/sim/2020/LHC20g4/12/296191/AOD/001/aod_archive.zip#AliAOD.root"
    #            ),
    #            #Path(
    #            #    "/alf/data/rehlers/data/alice/sim/2020/LHC20g4/12/296191/AOD/002/aod_archive.zip#AliAOD.root"
    #            #),
    #            #Path(
    #            #    "/alf/data/rehlers/data/alice/sim/2020/LHC20g4/12/296191/AOD/003/aod_archive.zip#AliAOD.root"
    #            #),

    #            # Path("/opt/scott/data/rehlers/alice/datasets/data/2016/LHC16j5/4/246945/AOD200/0003/AliAOD.root"),
    #            # Path("/opt/scott/data/rehlers/alice/datasets/data/2016/LHC16j5/4/246945/AOD200/0002/AliAOD.root"),
    #            # Path("/opt/scott/data/rehlers/alice/datasets/data/2016/LHC16j5/4/246945/AOD200/0001/AliAOD.root"),
    #            # Path("/opt/scott/data/rehlers/alice/datasets/data/2016/LHC16j5/5/246945/AOD200/0003/AliAOD.root"),
    #            # Path("/opt/scott/data/rehlers/alice/datasets/data/2016/LHC16j5/5/246945/AOD200/0005/AliAOD.root"),
    #            # Path("/opt/scott/data/rehlers/alice/datasets/data/2016/LHC16j5/5/246945/AOD200/0002/AliAOD.root"),
    #            # Path("/opt/scott/data/rehlers/alice/datasets/data/2016/LHC16j5/5/246945/AOD200/0001/AliAOD.root"),
    #            # Path("/opt/scott/data/rehlers/alice/datasets/data/2016/LHC16j5/5/246945/AOD200/0006/AliAOD.root"),
    #            # Path("/Users/re239/code/alice/data/LHC16j5/4/246945/AOD200/0003/AliAOD.root"),
    #            # Path("/Users/re239/code/alice/data/LHC16j5/4/246945/AOD200/0002/AliAOD.root"),
    #            # Path("/Users/re239/code/alice/data/LHC16j5/4/246945/AOD200/0001/AliAOD.root"),
    #            # Path("/Users/re239/code/alice/data/LHC16j5/5/246945/AOD200/0003/AliAOD.root"),
    #            # Path("/Users/re239/code/alice/data/LHC16j5/5/246945/AOD200/0005/AliAOD.root"),
    #            # Path("/Users/re239/code/alice/data/LHC16j5/5/246945/AOD200/0002/AliAOD.root"),
    #            # Path("/Users/re239/code/alice/data/LHC16j5/5/246945/AOD200/0001/AliAOD.root"),
    #            # Path("/Users/re239/code/alice/data/LHC16j5/5/246945/AOD200/0006/AliAOD.root"),
    #        ],
    #    )
    #if analysis_mode == AnalysisMode.embed_pythia:
    #    # This is different enough that we'll use a different entry point.
    #    _run_embedding_analysis(
    #        analysis_mode=analysis_mode,
    #        period_name="LHC18q",
    #        # NOTE: For some reason, kAnyINT will include some events where the centrality seems to be uncalibrated.
    #        #       It's unclear why this occurs, but since we're only interested in semi-central at the moment, it
    #        #       doesn't matter.
    #        physics_selection=ROOT.AliVEvent.kSemiCentral,
    #        grooming_jet_pt_threshold=20,
    #        jet_R=jet_R,
    #        validation_mode=validation_mode,
    #        input_files=[
    #            Path(
    #                "/alf/data/rehlers/data/alice/data/2018/LHC18q/000296550/pass3/AOD252/AOD/001/aod_archive.zip#AliAOD.root"
    #            ),
    #            #Path(
    #            #    "/alf/data/rehlers/data/alice/data/2018/LHC18q/000296550/pass3/AOD252/AOD/002/aod_archive.zip#AliAOD.root"
    #            #),
    #            # Path("/opt/scott/data/rehlers/alice/datasets/data/2018/LHC18q/000296550/pass3/AOD252/AOD/003/aod_archive.zip#AliAOD.root"),
    #            # Path("/opt/scott/data/rehlers/alice/datasets/data/2018/LHC18q/000296550/pass3/AOD252/AOD/004/aod_archive.zip#AliAOD.root"),
    #            # Path("/Volumes/Elements/alice/data/data/2018/LHC18q/000296550/pass1/AOD/001/AliAOD.root"),
    #            # Path("/Volumes/Elements/alice/data/data/2018/LHC18q/000296550/pass1/AOD/002/AliAOD.root"),
    #            # Path("/Volumes/Elements/alice/data/data/2018/LHC18q/000296550/pass1/AOD/003/AliAOD.root"),
    #            # Path("/Volumes/Elements/alice/data/data/2018/LHC18q/000296550/pass1/AOD/004/AliAOD.root"),
    #        ],
    #    )
