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
from mammoth.alice import helpers as alice_helpers
from mammoth.framework import jet_finding, load_data
from mammoth.framework.analysis import jets as analysis_jets
from mammoth.framework.analysis import tracking as analysis_tracking
from mammoth.framework.io import track_skim

logger = logging.getLogger(__name__)
vector.register_awkward()


def analysis_MC(
    arrays: ak.Array,
    jet_R: float,
    min_jet_pt: Mapping[str, float],
    det_level_artificial_tracking_efficiency: float | analysis_tracking.PtDependentTrackingEfficiencyParameters = 1.0,
    validation_mode: bool = False,
) -> ak.Array:
    logger.info("Start analyzing")
    # Event selection
    arrays = alice_helpers.standard_event_selection(arrays=arrays)

    # Track cuts
    arrays = alice_helpers.standard_track_selection(
        arrays=arrays, require_at_least_one_particle_in_each_collection_per_event=True
    )

    # Jet finding
    logger.info("Find jets")
    # First, setup what is needed for validation mode if enabled
    area_kwargs: dict[str, Any] = {}
    if validation_mode:
        area_kwargs["random_seed"] = jet_finding.VALIDATION_MODE_RANDOM_SEED

    # Calculate the relevant masks for det level particles to potentially apply an
    # artificial tracking inefficiency
    det_level_mask = analysis_tracking.det_level_particles_mask_for_jet_finding(
        arrays=arrays,
        det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
        validation_mode=validation_mode,
    )
    # Require that events have at least one particle after any possible masking.
    # If not, the entire array will be thrown out during jet finding, so better to
    # remove them now and be able to analyze the rest of the array. We're not missing
    # anything meaningful by doing this because we can't analyze such a case anyway
    # (and it might only be useful for an efficiency of losing events due to tracking,
    # which should be exceptionally rare).
    _events_with_at_least_one_particle = (ak.num(arrays["part_level"]) > 0) & (
        ak.num(arrays["det_level"][det_level_mask]) > 0
    )
    arrays = arrays[_events_with_at_least_one_particle]
    # NOTE: We need to apply it to the det level mask as well because we
    #       may be dropping some events, which then needs to be reflected in
    det_level_mask = det_level_mask[_events_with_at_least_one_particle]

    jets = ak.zip(
        {
            "part_level": jet_finding.find_jets(
                particles=arrays["part_level"],
                jet_finding_settings=jet_finding.JetFindingSettings(
                    R=jet_R,
                    algorithm="anti-kt",
                    # NOTE: We only want the minimum pt to apply to the detector level.
                    #       Otherwise, we'll bias our particle level jets.
                    #       However, we keep a small pt cut to avoid mismatches to low pt garbage
                    pt_range=jet_finding.pt_range(pt_min=min_jet_pt.get("part_level", 1.0)),
                    eta_range=jet_finding.eta_range(
                        jet_R=jet_R,
                        # We will require the det level jets to be in the fiducial acceptance, which means
                        # that the part level jets don't need to be within the fiducial acceptance - just
                        # within the overall acceptance.
                        # NOTE: This is rounding a little bit since we still have an eta cut at particle
                        #       level which will then cut into the particle level jets. However, this is
                        #       what we've done in the past, so we continue it here.
                        fiducial_acceptance=False,
                    ),
                    area_settings=jet_finding.AreaPP(**area_kwargs),
                ),
            ),
            "det_level": jet_finding.find_jets(
                particles=arrays["det_level"][det_level_mask],
                jet_finding_settings=jet_finding.JetFindingSettings(
                    R=jet_R,
                    algorithm="anti-kt",
                    pt_range=jet_finding.pt_range(pt_min=min_jet_pt["det_level"]),
                    eta_range=jet_finding.eta_range(jet_R=jet_R, fiducial_acceptance=True),
                    area_settings=jet_finding.AreaPP(**area_kwargs),
                ),
            ),
        },
        depth_limit=1,
    )
    logger.info(f"Found det_level n jets: {np.count_nonzero(np.asarray(ak.flatten(jets['det_level'].px, axis=None)))}")

    # Apply jet level cuts.
    jets, _qa_hists = alice_helpers.standard_jet_selection(
        jets=jets,
        jet_R=jet_R,
        collision_system="pp_MC",
        substructure_constituent_requirements=True,
    )
    logger.info(f"all jet cuts n accepted: {np.count_nonzero(np.asarray(ak.flatten(jets['det_level'].px, axis=None)))}")

    # Jet matching
    # NOTE: There is small departure from the AliPhysics approach here because we apply the jet cuts
    #       _before_ the matching, while AliPhysics does it after. However, I think removing really
    #       soft jets before matching makes more sense - it can avoid a mismatch to something really
    #       soft that just happens to be closer.
    logger.info("Matching jets")
    jets = analysis_jets.jet_matching_MC(
        jets=jets,
        # NOTE: This is larger than the matching distance that I would usually use (where we usually use 0.3 =
        #       in embedding), but this is apparently what we use in pythia. So just go with it.
        part_level_det_level_max_matching_distance=1.0,
    )

    # Reclustering
    logger.info("Reclustering jets...")
    for level in ["part_level", "det_level"]:
        logger.info(f"Reclustering {level}")
        # We only do the area calculation for data.
        reclustering_kwargs = {}
        if level != "part_level":
            reclustering_kwargs["area_settings"] = jet_finding.AreaSubstructure(**area_kwargs)
        jets[level, "reclustering"] = jet_finding.recluster_jets(
            jets=jets[level],
            jet_finding_settings=jet_finding.ReclusteringJetFindingSettings(**reclustering_kwargs),  # type: ignore[arg-type]
            store_recursive_splittings=True,
        )
    logger.info("Done with reclustering")

    logger.info(f"n events: {len(jets)}")
    logger.info(f"n jets accepted: {np.count_nonzero(np.asarray(ak.flatten(jets['det_level'].px, axis=None)))}")

    # Next step for using existing skimming:
    # Flatten from events -> jets
    # NOTE: Apparently it's takes issues with flattening the jets directly, so we have to do it
    #       separately for the different collections and then zip them together. This should keep
    #       matching together as appropriate.
    jets = ak.zip(
        {k: ak.flatten(v, axis=1) for k, v in zip(ak.fields(jets), ak.unzip(jets), strict=True)},
        depth_limit=1,
    )

    # Now, the final transformation into a form that can be used to skim into a flat tree.
    return jets  # noqa: RET504


def analysis_data(
    collision_system: str,
    arrays: ak.Array,
    jet_R: float,
    min_jet_pt: Mapping[str, float],
    particle_column_name: str = "data",
    validation_mode: bool = False,
    background_subtraction: Mapping[str, Any] | None = None,
) -> ak.Array:
    # Validation
    if background_subtraction is None:
        background_subtraction = {}

    logger.info("Start analyzing")
    # Event selection
    arrays = alice_helpers.standard_event_selection(arrays=arrays)

    # Track cuts
    arrays = alice_helpers.standard_track_selection(
        arrays=arrays,
        require_at_least_one_particle_in_each_collection_per_event=False,
        selected_particle_column_name=particle_column_name,
    )

    # Jet finding
    logger.info("Find jets")
    # First, setup what is needed for validation mode if enabled
    area_kwargs = {}
    if validation_mode:
        area_kwargs["random_seed"] = jet_finding.VALIDATION_MODE_RANDOM_SEED

    area_settings = jet_finding.AreaPP(**area_kwargs)  # type: ignore[arg-type]
    additional_kwargs: dict[str, Any] = {}
    if collision_system in ["PbPb", "embedPythia", "embed_pythia", "embed_thermal_model"]:
        area_settings = jet_finding.AreaAA(**area_kwargs)  # type: ignore[arg-type]
        additional_kwargs["background_subtraction"] = jet_finding.BackgroundSubtraction(
            type=jet_finding.BackgroundSubtractionType.event_wise_constituent_subtraction,
            estimator=jet_finding.JetMedianBackgroundEstimator(
                jet_finding_settings=jet_finding.JetMedianJetFindingSettings(
                    area_settings=jet_finding.AreaAA(**area_kwargs)  # type: ignore[arg-type]
                )
            ),
            subtractor=jet_finding.ConstituentSubtractor(
                r_max=background_subtraction["r_max"],
            ),
        )

    logger.warning(f"For particle column '{particle_column_name}', additional_kwargs: {additional_kwargs}")
    jets = ak.zip(
        {
            particle_column_name: jet_finding.find_jets(
                particles=arrays[particle_column_name],
                jet_finding_settings=jet_finding.JetFindingSettings(
                    R=jet_R,
                    algorithm="anti-kt",
                    pt_range=jet_finding.pt_range(pt_min=min_jet_pt.get(particle_column_name, 1.0)),
                    eta_range=jet_finding.eta_range(jet_R=jet_R, fiducial_acceptance=True),
                    area_settings=area_settings,
                ),
                **additional_kwargs,
            ),
        },
        depth_limit=1,
    )
    logger.warning(
        f"Found n jets: {np.count_nonzero(np.asarray(ak.flatten(jets[particle_column_name].px, axis=None)))}"
    )

    # Apply jet level cuts.
    jets, _qa_hists = alice_helpers.standard_jet_selection(
        jets=jets,
        jet_R=jet_R,
        collision_system=collision_system,
        substructure_constituent_requirements=True,
        selected_particle_column_name=particle_column_name,
    )

    # Reclustering
    # If we're out of jets, reclustering will fail. So if we're out of jets, then skip this step
    # NOTE: We need to flatten since we could just have empty events.
    # NOTE: Further, we need to access some variable to avoid flattening into a record, so we select px arbitrarily.
    # NOTE: We have to use pt because of awkward #2207 (https://github.com/scikit-hep/awkward/issues/2207)
    _there_are_jets_left = len(ak.flatten(jets[particle_column_name].pt, axis=None)) > 0
    # Now, we actually run the reclustering if possible
    if not _there_are_jets_left:
        logger.warning("No jets left for reclustering. Skipping reclustering...")
    else:
        logger.info(f"Reclustering {particle_column_name} jets...")
        jets[particle_column_name, "reclustering"] = jet_finding.recluster_jets(
            jets=jets[particle_column_name],
            jet_finding_settings=jet_finding.ReclusteringJetFindingSettings(
                # We perform the area calculation here since we're dealing with data, as is done in the AliPhysics DyG task
                area_settings=jet_finding.AreaSubstructure(**area_kwargs)  # type: ignore[arg-type]
            ),
            store_recursive_splittings=True,
        )
        logger.info("Done with reclustering")

    logger.warning(f"n events: {len(jets)}")

    # Next step for using existing skimming:
    # Flatten from events -> jets
    # NOTE: Apparently it's takes issues with flattening the jets directly, so we have to do it
    #       separately for the different collections and then zip them together. This should keep
    #       matching together as appropriate.
    jets = ak.zip(
        {k: ak.flatten(v, axis=1) for k, v in zip(ak.fields(jets), ak.unzip(jets), strict=True)},
        depth_limit=1,
    )

    logger.warning(f"n jets: {len(jets)}")

    # Now, the final transformation into a form that can be used to skim into a flat tree.
    return jets


def analysis_embedding(
    source_index_identifiers: Mapping[str, int],
    arrays: ak.Array,
    jet_R: float,
    min_jet_pt: Mapping[str, float],
    background_subtraction: Mapping[str, Any],
    validation_mode: bool = False,
    shared_momentum_fraction_min: float = 0.5,
    det_level_artificial_tracking_efficiency: float | analysis_tracking.PtDependentTrackingEfficiencyParameters = 1.0,
) -> ak.Array:
    # Event selection
    # This would apply to the signal events, because this is what we propagate from the embedding transform
    arrays = alice_helpers.standard_event_selection(arrays=arrays)

    # Track cuts
    arrays = alice_helpers.standard_track_selection(
        arrays=arrays, require_at_least_one_particle_in_each_collection_per_event=True
    )

    # Jet finding
    logger.info("Find jets")
    # First, setup what is needed for validation mode if enabled
    area_kwargs = {}
    if validation_mode:
        area_kwargs["random_seed"] = jet_finding.VALIDATION_MODE_RANDOM_SEED

    # Calculate the relevant masks for hybrid level particles:
    # 1. We may need to mask the hybrid level particles to apply an artificial tracking inefficiency
    # 2. We usually calculate rho only using the PbPb particles (ie. not including the embedded det_level),
    #    so we need to select only them.
    hybrid_level_mask, background_only_particles_mask = analysis_tracking.hybrid_level_particles_mask_for_jet_finding(
        arrays=arrays,
        det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
        source_index_identifiers=source_index_identifiers,
        validation_mode=validation_mode,
    )

    # Finally setup and run the jet finders
    jets = ak.zip(
        {
            "part_level": jet_finding.find_jets(
                particles=arrays["part_level"],
                jet_finding_settings=jet_finding.JetFindingSettings(
                    R=jet_R,
                    algorithm="anti-kt",
                    # NOTE: We only want the minimum pt to apply to the detector level.
                    #       Otherwise, we'll bias our particle level jets.
                    pt_range=jet_finding.pt_range(pt_min=min_jet_pt.get("part_level", 1.0)),
                    # NOTE: We only want fiducial acceptance at the "data" level (ie. hybrid)
                    eta_range=jet_finding.eta_range(jet_R=jet_R, fiducial_acceptance=False),
                    area_settings=jet_finding.AreaPP(**area_kwargs),  # type: ignore[arg-type]
                ),
            ),
            "det_level": jet_finding.find_jets(
                particles=arrays["det_level"],
                jet_finding_settings=jet_finding.JetFindingSettings(
                    R=jet_R,
                    algorithm="anti-kt",
                    # NOTE: We still keep this pt cut low, but not down to 0.15. We're trying to
                    #       balance avoiding bias while avoiding mismatching with really soft jets
                    pt_range=jet_finding.pt_range(pt_min=min_jet_pt.get("det_level", 1.0)),
                    # NOTE: We only want fiducial acceptance at the "data" level (ie. hybrid)
                    eta_range=jet_finding.eta_range(jet_R=jet_R, fiducial_acceptance=False),
                    area_settings=jet_finding.AreaPP(**area_kwargs),  # type: ignore[arg-type]
                ),
            ),
            "hybrid_level": jet_finding.find_jets(
                particles=arrays["hybrid_level"][hybrid_level_mask],
                jet_finding_settings=jet_finding.JetFindingSettings(
                    R=jet_R,
                    algorithm="anti-kt",
                    pt_range=jet_finding.pt_range(pt_min=min_jet_pt["hybrid_level"]),
                    eta_range=jet_finding.eta_range(jet_R=jet_R, fiducial_acceptance=True),
                    area_settings=jet_finding.AreaAA(**area_kwargs),  # type: ignore[arg-type]
                ),
                background_particles=arrays["hybrid_level"][background_only_particles_mask],
                background_subtraction=jet_finding.BackgroundSubtraction(
                    type=jet_finding.BackgroundSubtractionType.event_wise_constituent_subtraction,
                    estimator=jet_finding.JetMedianBackgroundEstimator(
                        jet_finding_settings=jet_finding.JetMedianJetFindingSettings(
                            area_settings=jet_finding.AreaAA(**area_kwargs)  # type: ignore[arg-type]
                        )
                    ),
                    subtractor=jet_finding.ConstituentSubtractor(
                        r_max=background_subtraction["r_max"],
                    ),
                ),
            ),
        },
        depth_limit=1,
    )

    # Apply jet level cuts.
    jets, _qa_hists = alice_helpers.standard_jet_selection(
        jets=jets,
        jet_R=jet_R,
        collision_system="embed_pythia",
        substructure_constituent_requirements=True,
    )

    # Jet matching
    # NOTE: There is small departure from the AliPhysics approach here because we apply the jet cuts
    #       _before_ the matching, while AliPhysics does it after. However, I think removing really
    #       soft jets before matching makes more sense - it can avoid a mismatch to something really
    #       soft that just happens to be closer.
    logger.info("Matching jets")
    jets = analysis_jets.jet_matching_embedding(
        jets=jets,
        # Values match those used in previous substructure analyses
        det_level_hybrid_max_matching_distance=0.3,
        part_level_det_level_max_matching_distance=0.3,
    )

    # Reclustering
    # If we're out of jets, reclustering will fail. So if we're out of jets, then skip this step
    # NOTE: We need to select a level to use for checking jets. We select the hybrid arbitrarily
    #       (it shouldn't matter overly much because we've required jet matching at this point)
    # NOTE: We need to flatten since we could just have empty events.
    # NOTE: Further, we need to access some variable to avoid flattening into a record, so we select px arbitrarily.
    # NOTE: We have to use pt because of awkward #2207 (https://github.com/scikit-hep/awkward/issues/2207)
    _there_are_jets_left = len(ak.flatten(jets["hybrid_level"].pt, axis=None)) > 0
    # Now, we actually run the reclustering if possible
    if not _there_are_jets_left:
        logger.warning("No jets left for reclustering. Skipping reclustering...")
    else:
        logger.info("Reclustering jets...")
        for level in ["hybrid_level", "det_level", "part_level"]:
            logger.info(f"Reclustering {level}")
            # We only do the area calculation for data.
            reclustering_kwargs = {}
            if level != "part_level":
                reclustering_kwargs["area_settings"] = jet_finding.AreaSubstructure(**area_kwargs)  # type: ignore[arg-type]
            jets[level, "reclustering"] = jet_finding.recluster_jets(
                jets=jets[level],
                jet_finding_settings=jet_finding.ReclusteringJetFindingSettings(**reclustering_kwargs),  # type: ignore[arg-type]
                store_recursive_splittings=True,
            )

        logger.info("Done with reclustering")

    logger.warning(f"n events: {len(jets)}")

    # Next step for using existing skimming:
    # Flatten from events -> jets
    # NOTE: Apparently it's takes issues with flattening the jets directly, so we have to do it
    #       separately for the different collections and then zip them together. This should keep
    #       matching together as appropriate.
    jets = ak.zip(
        {k: ak.flatten(v, axis=1) for k, v in zip(ak.fields(jets), ak.unzip(jets), strict=True)},
        depth_limit=1,
    )

    # I'm not sure if these work when there are no jets, so better to skip any calculations if there are no jets.
    if _there_are_jets_left:
        # Shared momentum fraction
        jets["det_level", "shared_momentum_fraction"] = jet_finding.shared_momentum_fraction_for_flat_array(
            generator_like_jet_pts=jets["det_level"].pt,
            generator_like_jet_constituents=jets["det_level"].constituents,
            measured_like_jet_constituents=jets["hybrid_level"].constituents,
        )

        # Require a shared momentum fraction (default >= 0.5)
        shared_momentum_fraction_mask = jets["det_level", "shared_momentum_fraction"] >= shared_momentum_fraction_min
        n_jets_removed = len(jets) - np.count_nonzero(shared_momentum_fraction_mask)
        logger.info(
            f"Removing {n_jets_removed} events out of {len(jets)} total jets ({round(n_jets_removed / len(jets) * 100, 2)}%) due to shared momentum fraction"
        )
        jets = jets[shared_momentum_fraction_mask]

    # Now, the final transformation into a form that can be used to skim into a flat tree.
    return jets


def run_some_standalone_tests() -> None:
    # Some tests:
    #######
    # Data:
    ######
    # pp: needs min_jet_pt = 5 to have any jets
    # collision_system = "pp"
    # pythia: Can test both "part_level" and "det_level" in the rename map.
    # collision_system = "pythia"
    # PbPb:
    # collision_system = "PbPb"
    # for collision_system in ["pp", "pythia", "PbPb"]:
    for collision_system in ["pp"]:
        logger.info(f'Analyzing "{collision_system}"')
        jets = analysis_data(  # noqa: F841
            collision_system=collision_system,
            arrays=load_data.data(
                data_input=Path(
                    f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults_track_skim.parquet"
                ),
                data_source=partial(track_skim.FileSource, collision_system=collision_system),
                collision_system=collision_system,
                rename_levels={"data": "data"} if collision_system != "pythia" else {"data": "det_level"},
            ),
            jet_R=0.4,
            min_jet_pt={"data": 5.0 if collision_system == "pp" else 20.0},
        )

        # import IPython; IPython.embed()
    ######
    # MC
    ######
    # jets = analysis_MC(
    #     arrays=load_MC(filename=Path("/software/rehlers/dev/mammoth/projects/framework/pythia/AnalysisResults.parquet")),
    #     jet_R=0.4,
    #     min_jet_pt={
    #         "det_level": 20,
    #     },
    # )
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

    # run_some_standalone_tests()

    ###########################
    # Explicitly for testing...
    ###########################
    # collision_system = "PbPb"
    # logger.info(f'Analyzing "{collision_system}"')
    # jets = analysis_data(
    #    collision_system=collision_system,
    #    arrays=load_data.data(
    #        data_input=Path(
    #            "trains/PbPb/645/run_by_run/LHC18q/296270/AnalysisResults.18q.580.root"
    #        ),
    #        data_source=partial(track_skim.FileSource, collision_system=collision_system),
    #        collision_system=collision_system,
    #        rename_levels={"data": "data"} if collision_system != "pythia" else {"data": "det_level"},
    #    ),
    #    jet_R=0.2,
    #    min_jet_pt={"data": 20.0 if collision_system == "pp" else 20.0},
    #    background_subtraction={"r_max": 0.1},
    # )

    source_index_identifiers, iter_arrays = load_data.embedding(
        signal_input=[Path("trains/pythia/2640/run_by_run/LHC20g4/296191/1/AnalysisResults.20g4.008.root")],
        signal_source=partial(track_skim.FileSource, collision_system="pythia"),
        background_input=[
            Path("trains/PbPb/645/run_by_run/LHC18r/296799/AnalysisResults.18r.179.root"),
            Path("trains/PbPb/645/run_by_run/LHC18r/296894/AnalysisResults.18r.337.root"),
        ],
        background_source=partial(track_skim.FileSource, collision_system="PbPb"),
        background_is_constrained_source=False,
        chunk_size=2500,
        post_embedding_rename_levels={},
    )

    for i_chunk, arrays in enumerate(iter_arrays):
        logger.info(f"Processing chunk: {i_chunk}")
        jets = analysis_embedding(
            source_index_identifiers=source_index_identifiers,
            arrays=arrays,
            jet_R=0.2,
            min_jet_pt={
                "hybrid_level": 20,
            },
            background_subtraction={"r_max": 0.1},
            det_level_artificial_tracking_efficiency=0.99,
        )

    import IPython

    IPython.embed()  # type: ignore[no-untyped-call]
