"""Run ALICE analysis for substructure for pp, PbPb, MC, and embedding

Note that the embedding analysis supports analyzing embedding data as well as into the thermal model.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import vector

from mammoth import helpers
from mammoth.framework import task as framework_task
from mammoth.framework.analysis import tracking as analysis_tracking
from mammoth.reclustered_substructure import analyze_chunk, groomed_substructure_skim_to_flat_tree

logger = logging.getLogger(__name__)
vector.register_awkward()


def customize_analysis_metadata(
    task_settings: framework_task.Settings,  # noqa: ARG001
    **analysis_arguments: Any,
) -> framework_task.Metadata:
    """Customize the analysis metadata for the analysis.

    Nothing special required here as of June 2023.
    """
    return {"R": analysis_arguments["jet_R"]}


def convert_analyzed_jets_to_all_jets_for_skim(
    jets: ak.Array,
    track_skim_to_flat_skim_level_names: Mapping[str, str],
) -> dict[str, ak.Array]:
    """Converts analyzed jets from a track skim to the all_jets dict for skimming to a flat tree.

    Args:
        jets: Analyzed jets from the track skim.
        input_to_output_level_names: Mapping from the track skim format prefix to
            the desired level (ie. prefix) in the all_jets dict format.
    Returns:
        The all_jets dict for skimming to a flat tree.
    """
    # Need the unsubtracted leading track pt for hybrid
    additional_columns_per_prefix = {}
    for level_to_check in track_skim_to_flat_skim_level_names:
        if level_to_check in ak.fields(jets) and "unsubtracted_leading_track_pt" in ak.fields(jets[level_to_check]):
            # Store the unsubtracted track pt.
            # It is expected to be under "leading_track_pt" even though it's unsubtracted
            additional_columns_per_prefix[level_to_check] = {
                "leading_track_pt": jets[level_to_check, "unsubtracted_leading_track_pt"],
            }

    return {
        track_skim_to_flat_skim_level_names[k]: ak.zip(
            {
                "jet_pt": jets[k].pt,
                "jet_constituents": ak.zip(
                    {
                        "pt": jets[k].constituents.pt,
                        "eta": jets[k].constituents.eta,
                        "phi": jets[k].constituents.phi,
                        "id": jets[k].constituents.identifier,
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
        for k in track_skim_to_flat_skim_level_names
    }


def _structured_skim_to_flat_skim_for_one_and_two_track_collections(
    jets: ak.Array,
    collision_system: str,
    jet_R: float,
    iterative_splittings: bool,
    track_skim_to_flat_skim_level_names: Mapping[str, str],
    scale_factors: Mapping[int, float] | None = None,
    pt_hat_bin: int | None = -1,
    selected_grooming_methods: list[str] | None = None,
) -> groomed_substructure_skim_to_flat_tree.T_GroomingResults:
    """Convert the structured skim output to a flat skim using grooming methods.

    Supports pp, pythia, PbPb, and embedded pythia. The data and jet finding needs to be
    handled in a separate function.
    """
    # Now, adapt into the expected format.
    all_jets = convert_analyzed_jets_to_all_jets_for_skim(
        jets=jets,
        track_skim_to_flat_skim_level_names=track_skim_to_flat_skim_level_names,
    )

    # If I want to write out an the intermediate step, I can uncomment this here.
    # NOTE: It's not really systematically organized (ie. each run will overwrite),
    #       so it's only good for debugging, but it # can still be useful.
    # ak.to_parquet(all_jets, input_filename.parent / Path("intermediate.parquet"))

    prefixes = {"data": "data"}
    if collision_system in ["pythia", "pp_MC", "PbPb_MC"]:
        assert pt_hat_bin is not None
        # Store externally provided pt hard bin
        all_jets["pt_hard_bin"] = np.ones(len(all_jets["data"]["jet_pt"])) * pt_hat_bin
        # Add the second prefix for true jets
        # NOTE: We only want to do this if we're already stated that we want to
        #       want to analyze both the part and det level jets.
        if len(track_skim_to_flat_skim_level_names) > 1:
            prefixes["true"] = "true"

    return groomed_substructure_skim_to_flat_tree.calculate_one_or_two_input_level_skim_impl(
        all_jets=all_jets,
        collision_system=collision_system,
        prefixes=prefixes,
        iterative_splittings=iterative_splittings,
        jet_R=jet_R,
        scale_factors=scale_factors,
        selected_grooming_methods=selected_grooming_methods,
    )


def analyze_chunk_two_input_level(
    *,
    collision_system: str,
    arrays: ak.Array,
    input_metadata: framework_task.InputMetadata,
    # Analysis arguments
    pt_hat_bin: int,
    scale_factors: dict[int, float],
    track_skim_to_flat_skim_level_names: Mapping[str, str],
    jet_R: float,
    min_jet_pt: dict[str, float],
    iterative_splittings: bool,
    det_level_artificial_tracking_efficiency: float | analysis_tracking.PtDependentTrackingEfficiencyParameters,
    reclustering_settings: Mapping[str, Any] | None = None,
    selected_grooming_methods: list[str] | None = None,
    # Default analysis arguments
    validation_mode: bool = False,
    return_skim: bool = False,  # noqa: ARG001
    # NOTE: kwargs are required because we pass the config as the analysis arguments,
    #       and it contains additional values.
    **kwargs: Any,  # noqa: ARG001
) -> framework_task.AnalysisOutput:
    """Analysis of MC with one or two track collections (part and det level).

    This implements the Analysis interface.
    """
    # Cross check. This shouldn't be allowed by the type arguments, but it will help catch older
    # configurations where this isn't specified.
    assert det_level_artificial_tracking_efficiency is not None

    # Perform jet finding and reclustering
    if len(track_skim_to_flat_skim_level_names) == 1:
        msg = "Shouldn't be able to access this functionality anymore."
        raise RuntimeError(msg)
        ## Treat as a one track collection (will be either part or det level)
        ## Validation
        ## We almost certainly don't want an artificial tracking efficiency here
        # assert isinstance(
        #    det_level_artificial_tracking_efficiency, float | np.number
        # ), f"Det level tracking efficiency should almost certainly be a float. Passed: {det_level_artificial_tracking_efficiency}"
        # assert np.isclose(
        #    det_level_artificial_tracking_efficiency, 1.0
        # ), f"Det level tracking efficiency should almost certainly be 1.0. Passed: {det_level_artificial_tracking_efficiency}"

        # jets = reclustered_substructure.analyze_track_skim_and_recluster_data(
        #    collision_system=collision_system,
        #    arrays=arrays,
        #    jet_R=jet_R,
        #    min_jet_pt=min_jet_pt,
        #    reclustering_settings=reclustering_settings,
        #    validation_mode=validation_mode,
        # )

    # Two track collections (part and det level)
    jets = analyze_chunk.analyze_track_skim_and_recluster_MC(
        arrays=arrays,
        input_metadata=input_metadata,
        jet_R=jet_R,
        min_jet_pt=min_jet_pt,
        reclustering_settings=reclustering_settings,
        det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
        validation_mode=validation_mode,
    )

    # NOTE: We need to know how many jets there are, so we arbitrarily take the first field. The jets are flattened,
    #       so they're as good as any others.
    _there_are_jets_left = len(jets[ak.fields(jets)[0]]) > 0
    # There were no jets. Note that with a specially crafted output
    if not _there_are_jets_left:
        # Let the analyzer know. This will likely lead to an empty filename to prevent re-running with no jets in the future.
        # Remember that this depends heavily on the jet pt cuts!
        _msg = "Done - no jets left to analyze, so not trying to run flat skim"
        raise framework_task.NoUsefulAnalysisOutputError(_msg)

    jets = _structured_skim_to_flat_skim_for_one_and_two_track_collections(
        jets=jets,
        collision_system=collision_system,
        jet_R=jet_R,
        iterative_splittings=iterative_splittings,
        track_skim_to_flat_skim_level_names=track_skim_to_flat_skim_level_names,
        scale_factors=scale_factors,
        pt_hat_bin=pt_hat_bin,
        selected_grooming_methods=selected_grooming_methods,
    )

    # NOTE: We don't look at return_skim here because the skim is the only possible output for this analysis.
    #       If we didn't return it, then all of the processing is useless.
    return framework_task.AnalysisOutput(
        skim=jets,
    )



def analyze_chunk_one_input_level(
    *,
    collision_system: str,
    arrays: ak.Array,
    input_metadata: framework_task.InputMetadata,
    # Analysis arguments
    track_skim_to_flat_skim_level_names: Mapping[str, str],
    jet_R: float,
    min_jet_pt: dict[str, float],
    iterative_splittings: bool,
    particle_column_name: str = "data",
    background_subtraction_settings: Mapping[str, Any] | None = None,
    reclustering_settings: Mapping[str, Any] | None = None,
    selected_grooming_methods: list[str] | None = None,
    # Default analysis arguments
    validation_mode: bool = False,
    return_skim: bool = False,  # noqa: ARG001
    # NOTE: kwargs are required because we pass the config as the analysis arguments,
    #       and it contains additional values.
    **kwargs: Any,  # noqa: ARG001
) -> framework_task.AnalysisOutput:
    """Analysis of data with one track collections.

    This implements the Analysis interface.
    """
    jets = analyze_chunk.analyze_track_skim_and_recluster_data(
        collision_system=collision_system,
        arrays=arrays,
        input_metadata=input_metadata,
        jet_R=jet_R,
        min_jet_pt=min_jet_pt,
        background_subtraction_settings=background_subtraction_settings,
        reclustering_settings=reclustering_settings,
        particle_column_name=particle_column_name,
        validation_mode=validation_mode,
    )

    # NOTE: We need to know how many jets there are, so we arbitrarily take the first field. The jets are flattened,
    #       so they're as good as any others.
    _there_are_jets_left = len(jets[ak.fields(jets)[0]]) > 0
    # There were no jets. Note that with a specially crafted output
    if not _there_are_jets_left:
        # Let the analyzer know. This will likely lead to an empty filename to prevent re-running with no jets in the future.
        # Remember that this depends heavily on the jet pt cuts!
        _msg = "Done - no jets left to analyze, so not trying to run flat skim"
        raise framework_task.NoUsefulAnalysisOutputError(_msg)

    jets = _structured_skim_to_flat_skim_for_one_and_two_track_collections(
        jets=jets,
        collision_system=collision_system,
        jet_R=jet_R,
        iterative_splittings=iterative_splittings,
        track_skim_to_flat_skim_level_names=track_skim_to_flat_skim_level_names,
        selected_grooming_methods=selected_grooming_methods,
    )

    # NOTE: We don't look at return_skim here because the skim is the only possible output for this analysis.
    #       If we didn't return it, then all of the processing is useless.
    return framework_task.AnalysisOutput(
        skim=jets,
    )


def _structured_skim_to_flat_skim_for_three_track_collections(
    jets: ak.Array,
    jet_R: float,
    iterative_splittings: bool,
    track_skim_to_flat_skim_level_names: Mapping[str, str],
    scale_factor: float,
    selected_grooming_methods: list[str] | None = None,
) -> groomed_substructure_skim_to_flat_tree.T_GroomingResults:
    """Convert the structured skim output to a flat skim using grooming methods.

    Supports embedded pythia with three track collections. The data and jet finding needs to be
    handled in a separate function.
    """
    # Now, adapt into the expected format.
    all_jets = convert_analyzed_jets_to_all_jets_for_skim(
        jets=jets,
        track_skim_to_flat_skim_level_names=track_skim_to_flat_skim_level_names,
    )

    # Define the prefixes for analysis of the all_jets structure to the flat skim.
    # We take care of modifying the labeling above, so we can hard code it here.
    prefixes = {
        "hybrid": "hybrid",
        "true": "true",
        "det_level": "det_level",
    }

    return groomed_substructure_skim_to_flat_tree.calculate_three_input_level_skim_impl(
        all_jets=all_jets,
        prefixes=prefixes,
        iterative_splittings=iterative_splittings,
        jet_R=jet_R,
        scale_factor=scale_factor,
        selected_grooming_methods=selected_grooming_methods,
    )


def analyze_chunk_three_input_level(
    *,
    source_index_identifiers: Mapping[str, int],
    arrays: ak.Array,
    input_metadata: framework_task.InputMetadata,
    # Analysis arguments
    scale_factor: float,
    track_skim_to_flat_skim_level_names: Mapping[str, str],
    jet_R: float,
    min_jet_pt: dict[str, float],
    iterative_splittings: bool,
    det_level_artificial_tracking_efficiency: float | analysis_tracking.PtDependentTrackingEfficiencyParameters,
    background_subtraction_settings: Mapping[str, Any] | None = None,
    reclustering_settings: Mapping[str, Any] | None = None,
    selected_grooming_methods: list[str] | None = None,
    # Default analysis arguments
    validation_mode: bool = False,
    return_skim: bool = False,  # noqa: ARG001
    # NOTE: kwargs are required because we pass the config as the analysis arguments,
    #       and it contains additional values.
    **kwargs: Any,  # noqa: ARG001
) -> framework_task.AnalysisOutput:
    """Analysis of embedding with three track collections (part, det, and hybrid level).

    This implements the Analysis interface.
    """
    jets = analyze_chunk.analyze_track_skim_and_recluster_embedding(
        source_index_identifiers=source_index_identifiers,
        arrays=arrays,
        input_metadata=input_metadata,
        jet_R=jet_R,
        min_jet_pt=min_jet_pt,
        background_subtraction_settings=background_subtraction_settings,
        det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
        reclustering_settings=reclustering_settings,
        validation_mode=validation_mode,
    )

    # NOTE: We need to know how many jets there are, so we arbitrarily take the first field. The jets are flattened,
    #       so they're as good as any others.
    _there_are_jets_left = len(jets[ak.fields(jets)[0]]) > 0
    # There were no jets. Note that with a specially crafted output
    if not _there_are_jets_left:
        # Let the analyzer know. This will likely lead to an empty filename to prevent re-running with no jets in the future.
        # Remember that this depends heavily on the jet pt cuts!
        _msg = "Done - no jets left to analyze, so not trying to run flat skim"
        raise framework_task.NoUsefulAnalysisOutputError(_msg)

    jets = _structured_skim_to_flat_skim_for_three_track_collections(
        jets=jets,
        jet_R=jet_R,
        iterative_splittings=iterative_splittings,
        track_skim_to_flat_skim_level_names=track_skim_to_flat_skim_level_names,
        scale_factor=scale_factor,
        selected_grooming_methods=selected_grooming_methods,
    )

    # NOTE: We don't look at return_skim here because the skim is the only possible output for this analysis.
    #       If we didn't return it, then all of the processing is useless.
    return framework_task.AnalysisOutput(
        skim=jets,
    )


def run_some_standalone_tests() -> None:
    #########################################
    # Explicitly for testing + experiments...
    #########################################

    # Delayed import since we only need these for the experiments
    from mammoth.framework import load_data
    from mammoth.framework.io import track_skim

    # Some tests:
    #######
    # Data:
    ######
    # pp: needs min_jet_pt = 5 to have any jets
    # collision_system = "pp"
    # pythia: Can test both "part_level" and "det_level" in the rename map.
    # collision_system = "pp_MC"
    # PbPb:
    # collision_system = "PbPb"
    # for collision_system in ["pp", "pp_MC", "PbPb"]:
    # for collision_system in ["pp"]:
    #    logger.info(f'Analyzing "{collision_system}"')
    #    jets = analysis_data(
    #        collision_system=collision_system,
    #        arrays=load_data.data(
    #            data_input=Path(
    #                f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults_track_skim.parquet"
    #            ),
    #            data_source=partial(track_skim.FileSource, collision_system=collision_system),
    #            collision_system=collision_system,
    #            rename_levels={"data": "data"} if collision_system != "pp_MC" else {"data": "det_level"},
    #        ),
    #        jet_R=0.4,
    #        iterative_splittings=True,
    #        track_skim_to_flat_skim_level_names={"data": "data"},
    #        min_jet_pt={"data": 5.0 if collision_system == "pp" else 20.0},
    #    )

    # import IPython; IPython.embed()
    ######
    # MC
    ######
    collision_system = "pp_MC"
    jets = analyze_chunk_two_input_level(  # noqa: F841
        arrays=load_data.data(
            # data_input=[Path("trains/")],
            data_input=[Path("trains/pythia/2640/run_by_run/LHC20g4/296191/1/AnalysisResults.20g4.008.root")],
            data_source=partial(track_skim.FileSource, collision_system=collision_system),
            collision_system=collision_system,
            rename_levels={},
        ),
        collision_system=collision_system,
        input_metadata={},
        # Analysis arguments
        pt_hat_bin=1,
        scale_factors={i: 1 for i in range(1, 21)},
        track_skim_to_flat_skim_level_names={
            "det_level": "data",
            "part_level": "true",
        },
        jet_R=0.2,
        min_jet_pt={
            "det_level": 5,
        },
        iterative_splittings=True,
        det_level_artificial_tracking_efficiency=1.0,
        selected_grooming_methods=["soft_drop_z_cut_02"],
    )
    import IPython

    IPython.embed()  # type: ignore[no-untyped-call]

    ###########
    # Embedding
    ###########
    # jets = analysis_embedding(
    #     *load_embedding(
    #         signal_filename=Path("/software/rehlers/dev/mammoth/projects/framework/pythia/AnalysisResults.parquet"),
    #         background_filename=Path("/software/rehlers/dev/mammoth/projects/framework/PbPb/AnalysisResults.parquet"),
    #     ),
    #     jet_R=0.4,
    #     min_jet_pt={
    #         "hybrid_level": 20,
    #         "det_level": 1,
    #         "part_level": 1,
    #     },
    # )
    ###############
    # Thermal model
    ###############
    # jets = analysis_embedding(
    #     *load_embed_thermal_model(
    #         signal_filename=Path("/software/rehlers/dev/substructure/trains/pythia/641/run_by_run/LHC20g4/295612/1/AnalysisResults.20g4.001.root"),
    #         thermal_model_parameters=sources.THERMAL_MODEL_SETTINGS["central"],
    #     ),
    #     jet_R=0.2,
    #     min_jet_pt={
    #         "hybrid_level": 20,
    #         #"det_level": 1,
    #         #"part_level": 1,
    #     },
    #     r_max=0.25,
    # )


if __name__ == "__main__":
    helpers.setup_logging(level=logging.INFO)

    run_some_standalone_tests()
