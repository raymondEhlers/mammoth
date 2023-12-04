"""Run ALICE analysis for energy-energy correlators for pp, PbPb, MC, and embedding

Note that the embedding analysis supports analyzing embedding data as well as into the thermal model.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import awkward as ak
import hist
import numpy as np
import vector

from mammoth import helpers
from mammoth.alice import helpers as alice_helpers
from mammoth.framework import jet_finding, load_data
from mammoth.framework import task as framework_task
from mammoth.framework.analysis import array_helpers as analysis_array_helpers
from mammoth.framework.analysis import tracking as analysis_tracking
from mammoth.framework.io import output_utils, track_skim

logger = logging.getLogger(__name__)
vector.register_awkward()


def customize_analysis_metadata(
    task_settings: framework_task.Settings,  # noqa: ARG001
    **analysis_arguments: Any,  # noqa: ARG001
) -> framework_task.Metadata:
    """Customize the analysis metadata for the analysis.

    Nothing special required here as of June 2023.
    """
    return {}


def _chunks_for_combinatorics(combinatorics_chunk_size: int, array_length: int) -> Iterator[tuple[int, int]]:
    """ Yield the start and stop indices for calculating the combinatorics in chunks.

    This is intended to keep memory usage lower when calculating and filling hists.
    To be seen (June 2023) how well it actually works...

    Args:
        combinatorics_chunk_size: The size of the chunk to use for calculating the combinatorics.
        array_length: The length of the array to be chunked.

    Returns:
        The start and stop indices for the chunks.
    """
    # Validation + short circuit
    if combinatorics_chunk_size<= 0:
        # IF we don't want to chunk, then just return the full range.
        return None, None

    start = 0
    continue_iterating = True
    while continue_iterating:
        end = start + combinatorics_chunk_size
        # Ensure that we never ask for more entries than are in the file.
        if start + combinatorics_chunk_size > array_length:
            end = array_length
            continue_iterating = False
        # Store the start and stop for convenience.
        yield start, end
        # Move up to the next iteration.
        start = end


def _setup_embedding_hists(trigger_pt_ranges: dict[str, tuple[float, float]]) -> dict[str, hist.Hist]:
    """Setup the histograms for the embedding analysis."""
    hists = {}

    # Spectra
    for level in ["part_level", "det_level", "hybrid"]:
        # Inclusive spectra
        hists[f"{level}_inclusive_trigger_spectra"] = hist.Hist(
            hist.axis.Regular(200, 0, 100, label="trigger_pt"), storage=hist.storage.Weight()
        )
        # Select spectra
        hists[f"{level}_trigger_spectra"] = hist.Hist(
            hist.axis.Regular(50, 0, 50, label="trigger_pt"), storage=hist.storage.Weight()
        )

    # EECs
    for level in ["part_level", "det_level", "hybrid"]:
        for trigger_name, trigger_range_tuple in trigger_pt_ranges.items():
            trigger_pt_bin_args = (round((trigger_range_tuple[1] - trigger_range_tuple[0]) * 4), *trigger_range_tuple)
            hists[f"{level}_{trigger_name}_eec"] = hist.Hist(
                *[
                    hist.axis.Regular(200, 1e-4, 1.5, label="R_L"),
                    hist.axis.Regular(*trigger_pt_bin_args, label="trigger_pt"),
                ],
                storage=hist.storage.Weight()
            )
            hists[f"{level}_{trigger_name}_eec_unweighted"] = hist.Hist(
                *[
                    hist.axis.Regular(200, 1e-4, 1.5, label="ln(R_L)", transform=hist.axis.transform.log),
                    hist.axis.Regular(*trigger_pt_bin_args, label="trigger_pt"),
                ],
                storage=hist.storage.Weight()
            )
            #hists[f"{level}_{trigger_name}_eec_log"] = hist.Hist(
            #    *[
            #        hist.axis.Regular(200, 1e-4, 1.5, label="ln(R_L)", transform=hist.axis.transform.log),
            #        hist.axis.Regular(*trigger_pt_bin_args, label="trigger_pt"),
            #    ],
            #    storage=hist.storage.Weight()
            #)
            if level == "hybrid":
                hists[f"{level}_{trigger_name}_eec_bg_only"] = hist.Hist(
                    *[
                        hist.axis.Regular(200, 1e-4, 1.5, label="R_L"),
                        hist.axis.Regular(*trigger_pt_bin_args, label="trigger_pt"),
                    ],
                    storage=hist.storage.Weight()
                )
                hists[f"{level}_{trigger_name}_eec_unweighted_bg_only"] = hist.Hist(
                    *[
                        hist.axis.Regular(200, 1e-4, 1.5, label="R_L"),
                        hist.axis.Regular(*trigger_pt_bin_args, label="trigger_pt"),
                    ],
                    storage=hist.storage.Weight()
                )
                #hists[f"{level}_{trigger_name}_eec_log_bg_only"] = hist.Hist(
                #    *[
                #        hist.axis.Regular(200, 1e-4, 1.5, label="ln(R_L)", transform=hist.axis.transform.log),
                #        hist.axis.Regular(*trigger_pt_bin_args, label="trigger_pt"),
                #    ],
                #    storage=hist.storage.Weight()
                #)

    return hists


def _calculate_weight_for_plotting(
    left: ak.Array,
    right: ak.Array,
    trigger_pt_event_wise: ak.Array,
    momentum_weight_exponent: int | float,
    left_right_mask: ak.Array | None = None
) -> ak.Array:
    """Calculate the weight for plotting the EECs.

    Args:
        left: Left side of combination of particles.
        right: Right side of combination of particles.
        trigger_pt_event_wise: The trigger pt for the event.
        momentum_weight_exponent: The exponent to use for the momentum weighting.
        left_right_mask: A mask to apply to the combined left and right arrays.
    Returns:
        The weight for plotting the EECs.
    """
    if momentum_weight_exponent == 1:
        left_right = (left.pt * right.pt)
        trigger_pt = trigger_pt_event_wise ** 2
    else:
        w = momentum_weight_exponent
        left_right = (left.pt ** w) * (right.pt ** w)
        trigger_pt = trigger_pt_event_wise ** (2*w)

    if left_right_mask is not None:
        left_right = left_right[left_right_mask]
    return ak.flatten(
        left_right / trigger_pt
    )


def analysis_embedding(
    *,
    collision_system: str,  # noqa: ARG001
    source_index_identifiers: dict[str, int],
    arrays: ak.Array,
    # Analysis arguments
    trigger_pt_ranges: dict[str, tuple[float, float]],
    min_track_pt: dict[str, float],
    momentum_weight_exponent: int | float,
    combinatorics_chunk_size: int,
    scale_factor: float,
    det_level_artificial_tracking_efficiency: float | analysis_tracking.PtDependentTrackingEfficiencyParameters,
    use_jet_trigger: bool,
    # Default analysis arguments
    validation_mode: bool = False,
    return_skim: bool = False,
    # NOTE: kwargs are required because we pass the config as the analysis arguments,
    #       and it contains additional values.
    **kwargs: Any,  # noqa: ARG001
) -> framework_task.AnalysisOutput:
    # Validation
    if return_skim and combinatorics_chunk_size < 0:
        logger.info(f"Requested to return the skim, but the combination chunk size is {combinatorics_chunk_size} (> 0), which won't work. So we disable it.")

    # Setup
    hists = _setup_embedding_hists(trigger_pt_ranges=trigger_pt_ranges)
    trigger_skim_output: dict[str, ak.Array] = {}

    # Event selection
    # This would apply to the signal events, because this is what we propagate from the embedding transform
    arrays = alice_helpers.standard_event_selection(arrays=arrays)

    # Track cuts
    arrays = alice_helpers.standard_track_selection(
        arrays=arrays, require_at_least_one_particle_in_each_collection_per_event=True
    )

    # Apply artificial tracking efficiency
    # 1. We may need to mask the hybrid level particles to apply an artificial tracking inefficiency
    # 2. We usually calculate rho only using the PbPb particles (ie. not including the embedded det_level),
    #    so we need to select only them.
    # NOTE: Technically, we aren't doing jet finding here, but the convention is still the same.
    #       I suppose this would call for a refactor, but I don't want to deal with that right now.
    hybrid_level_mask, _ = analysis_tracking.hybrid_level_particles_mask_for_jet_finding(
        arrays=arrays,
        det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
        source_index_identifiers=source_index_identifiers,
        validation_mode=validation_mode
    )
    arrays["hybrid"] = arrays["hybrid"][hybrid_level_mask]

    # Find trigger(s)
    logger.info("Finding trigger(s)")
    triggers_dict: dict[str, dict[str, ak.Array]] = {}
    event_selection_mask: dict[str, dict[str, ak.Array]] = {}
    if use_jet_trigger:
        ...
    else:
        # NOTE: Use the signal fraction because we don't want any overlap between signal and reference events!
        signal_event_fraction = 0.8
        _rng = np.random.default_rng()
        is_signal_event = _rng.random(ak.num(arrays, axis=0)) < signal_event_fraction

        # Trigger QA
        for level in ["part_level", "det_level", "hybrid"]:
            hists[f"{level}_inclusive_trigger_spectra"].fill(ak.flatten(arrays[level].pt), weight=scale_factor)

        # Random choice args
        random_choice_kwargs: dict[str, Any] = {}
        if validation_mode:
            random_choice_kwargs["random_seed"] = jet_finding.VALIDATION_MODE_RANDOM_SEED[0]
        for level in ["part_level", "det_level", "hybrid"]:
            triggers_dict[level] = {}
            event_selection_mask[level] = {}
            for trigger_name, trigger_range_tuple in trigger_pt_ranges.items():
                trigger_mask = (
                    (arrays[level].pt >= trigger_range_tuple[0])
                    & (arrays[level].pt < trigger_range_tuple[1])
                )
                # Add signal mask
                event_trigger_mask = ~is_signal_event
                if trigger_name == "signal":
                    event_trigger_mask = is_signal_event

                # Apply the masks separately since one is particle-wise and one is event-wise
                triggers = arrays[level][trigger_mask]
                event_trigger_mask = event_trigger_mask & (ak.num(triggers, axis=-1) > 0)
                triggers = triggers[event_trigger_mask]

                # NOTE: As of April 2023, I don't record the overall number of events because I think
                #       we're looking for a per-trigger yield (or we will normalize the different trigger
                #       classes relative to each other, in which case, I don't think we'll care about the
                #       precise event count).

                # Randomly select if there is more than one trigger
                # NOTE: This must operator on a concrete field, so we use px as a proxy.
                select_trigger_mask = analysis_array_helpers.random_choice_jagged(arrays=triggers.px, **random_choice_kwargs)
                # NOTE: Since we've now selected down to at most one trigger per event, and no empty events,
                #       our triggers can now be represented with regular numpy arrays.
                #       By simplifying now, it can make later operations easier.
                triggers_dict[level][trigger_name] = ak.to_regular(triggers[select_trigger_mask])
                event_selection_mask[level][trigger_name] = event_trigger_mask

                # Fill in spectra
                hists[f"{level}_trigger_spectra"].fill(
                    ak.flatten(triggers_dict[level][trigger_name].pt),
                    weight=scale_factor,
                )

    recoil_direction: dict[str, dict[str, ak.Array]] = {}
    for level in ["part_level", "det_level", "hybrid"]:
        recoil_direction[level] = {}
        for trigger_name, _ in trigger_pt_ranges.items():
            for _start, _end in _chunks_for_combinatorics(combinatorics_chunk_size=combinatorics_chunk_size, array_length=len(triggers_dict[level][trigger_name])):
                # Next, go to away side in phi and define our recoil region
                recoil_vector = -1 * triggers_dict[level][trigger_name][_start:_end]
                trigger_pt_event_wise = triggers_dict[level][trigger_name][_start:_end].pt
                # Mock up a four vector so we can use the calculation functionality from vector
                recoil_direction[level][trigger_name] = vector.Array(
                    ak.zip({
                        "pt": recoil_vector.eta * 0,
                        "eta": recoil_vector.eta,
                        "phi": recoil_vector.phi,
                        "m": recoil_vector.eta * 0,
                    })
                )
                logger.warning(f"{level=}, {trigger_name=}: {_start=}, {_end=}, {len(recoil_vector)=} (Initial size: {len(triggers_dict[level][trigger_name])})")

                # For recoil region, look at delta_phi between them
                event_selected_array = arrays[level][
                    event_selection_mask[level][trigger_name]
                ][_start:_end]
                # We perform the min pt selection first to reduce the number of calculations required.
                particle_pt_mask = event_selected_array.pt > min_track_pt[level]
                event_selected_array = event_selected_array[particle_pt_mask]
                logger.info(f"{level}, {trigger_name}: About to find particles within recoil cone")
                within_hemisphere = (recoil_direction[level][trigger_name].deltaphi(event_selected_array) < np.pi/4)
                eec_particles = event_selected_array[within_hemisphere]

                if return_skim and combinatorics_chunk_size < 0:
                    # NOTE: If we're using a combination chunk size, it's going to be really inefficient to return the skim
                    #       since we would need to recombine the chunks. So we just skip it in this case.
                    trigger_skim_output[f"{level}_{trigger_name}"] = ak.zip(
                        {
                            "triggers": triggers_dict[level][trigger_name],
                            "particles": eec_particles,
                        },
                        depth_limit=1
                    )

                # NOTE: These selections could have the potential edge effects in eta, but since we have those in both
                #       the signal and reference, these should be accounted for automatically.
                #       This correspondence will be even better when we use mixed events.

                # NOTE: Memory gets bad here. Steps to address:
                #         1. Use the combinatorics chunk size to reduce the number of combinations in memory
                #         2. Implement the calculation with numba
                #         3. Implement the calculation in c++
                #       As of 2023 June 20, we're only on step 1.
                logger.info(f"{level}, {trigger_name}: About to calculate combinations")
                # TODO: Mateusz claims that we should double count here, per the theorists (ie. Kyle).
                #       This seems odd...
                left, right = ak.unzip(ak.combinations(eec_particles, 2))
                distances = left.deltaR(right)

                # One argument should be the power
                # First, need to broadcast the trigger pt
                logger.info(f"{level}, {trigger_name}: About to fill hists")
                trigger_pt, _ = ak.broadcast_arrays(
                    trigger_pt_event_wise,
                    distances,
                )
                # Save an additional set of calls to exponent if can be avoided
                weight = _calculate_weight_for_plotting(
                    left=left, right=right, trigger_pt_event_wise=trigger_pt_event_wise,
                    momentum_weight_exponent=momentum_weight_exponent,
                )
                hists[f"{level}_{trigger_name}_eec"].fill(
                    ak.flatten(distances),
                    ak.flatten(trigger_pt),
                    weight=weight * scale_factor,
                )
                hists[f"{level}_{trigger_name}_eec_unweighted"].fill(
                    ak.flatten(distances),
                    ak.flatten(trigger_pt),
                    weight=np.ones_like(weight) * scale_factor,
                )
                if level == "hybrid":
                    # We're about to recalculate the weights and trigger pt, so let's release them now
                    del weight
                    del trigger_pt
                    # Compare to background only particles
                    # Need to select distances of particles for left and right which only are background particles
                    left_mask = left["source_index"] >= source_index_identifiers["background"]
                    right_mask = right["source_index"] >= source_index_identifiers["background"]
                    background_mask = left_mask & right_mask
                    distances = distances[background_mask]

                    trigger_pt, _ = ak.broadcast_arrays(
                        trigger_pt_event_wise,
                        distances,
                    )
                    # Recalculate weight
                    weight = _calculate_weight_for_plotting(
                        left=left, right=right, trigger_pt_event_wise=trigger_pt_event_wise,
                        momentum_weight_exponent=momentum_weight_exponent,
                        left_right_mask=background_mask,
                    )

                    hists[f"{level}_{trigger_name}_eec_bg_only"].fill(
                        ak.flatten(distances),
                        ak.flatten(trigger_pt),
                        weight=weight * scale_factor,
                    )
                    hists[f"{level}_{trigger_name}_eec_unweighted_bg_only"].fill(
                        ak.flatten(distances),
                        ak.flatten(trigger_pt),
                        weight=np.ones_like(weight) * scale_factor,
                    )

                # Probably makes no difference...
                del eec_particles
                del left
                del right
                del distances
                del trigger_pt
                del weight

    #IPython.embed()  # type: ignore[no-untyped-call]

    return framework_task.AnalysisOutput(
        hists=hists,
        skim=trigger_skim_output,
    )


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
    #for collision_system in ["pp"]:
    #    logger.info(f'Analyzing "{collision_system}"')
    #    jets = analysis_data(
    #        collision_system=collision_system,
    #        arrays=load_data.data(
    #            data_input=Path(
    #                f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults_track_skim.parquet"
    #            ),
    #            data_source=partial(track_skim.FileSource, collision_system=collision_system),
    #            collision_system=collision_system,
    #            rename_prefix={"data": "data"} if collision_system != "pythia" else {"data": "det_level"},
    #        ),
    #        jet_R=0.4,
    #        min_jet_pt={"data": 5.0 if collision_system == "pp" else 20.0},
    #    )
    #
    #    # import IPython; IPython.embed()
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
    #         "hybrid": 20,
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
    #         "hybrid": 20,
    #         #"det_level": 1,
    #         #"part_level": 1,
    #     },
    #     r_max=0.25,
    # )
    ...


if __name__ == "__main__":
    helpers.setup_logging(level=logging.INFO)

    # run_some_standalone_tests()

    ###########################
    # Explicitly for testing...
    ###########################
    #collision_system = "PbPb"
    #logger.info(f'Analyzing "{collision_system}"')
    #jets = analysis_data(
    #    collision_system=collision_system,
    #    arrays=load_data.data(
    #        data_input=Path(
    #            "trains/PbPb/645/run_by_run/LHC18q/296270/AnalysisResults.18q.580.root"
    #        ),
    #        data_source=partial(track_skim.FileSource, collision_system=collision_system),
    #        collision_system=collision_system,
    #        rename_prefix={"data": "data"} if collision_system != "pythia" else {"data": "det_level"},
    #    ),
    #    jet_R=0.2,
    #    min_jet_pt={"data": 20.0 if collision_system == "pp" else 20.0},
    #    background_subtraction_settings={"r_max": 0.1},
    #)

    #source_index_identifiers, iter_arrays = load_data.embedding(
    #    #signal_input=[Path("trains/pythia/2640/run_by_run/LHC20g4/296191/1/AnalysisResults.20g4.008.root")],
    #    # NOTE: This isn't anchored, but it's convenient for testing...
    #    signal_input=[Path("trains/pythia/2619/run_by_run/LHC18b8_fast/282125/14/AnalysisResults.18b8_fast.008.root")],
    #    signal_source=partial(track_skim.FileSource, collision_system="pythia"),
    #    background_input=[
    #        Path("trains/PbPb/645/run_by_run/LHC18r/296799/AnalysisResults.18r.179.root"),
    #        Path("trains/PbPb/645/run_by_run/LHC18r/296894/AnalysisResults.18r.337.root"),
    #    ],
    #    background_source=partial(track_skim.FileSource, collision_system="PbPb"),
    #    background_is_constrained_source=False,
    #    chunk_size=2500,
    #)
    from mammoth.framework import sources
    source_index_identifiers, iter_arrays = load_data.embedding_thermal_model(
        # NOTE: This isn't anchored, but it's convenient for testing...
        #signal_input=[Path("trains/pythia/2619/run_by_run/LHC18b8_fast/282125/14/AnalysisResults.18b8_fast.008.root")],
        signal_input=[Path("trains/pythia/2640/run_by_run/LHC20g4/296415/4/AnalysisResults.20g4.011.root")],
        signal_source=partial(track_skim.FileSource, collision_system="pythia"),
        thermal_model_parameters=sources.THERMAL_MODEL_SETTINGS["5020_central"],
        #chunk_size=2500,
        chunk_size=1000,
    )

    # NOTE: Just for quick testing
    merged_hists: dict[str, hist.Hist] = {}
    # END NOTE
    for i_chunk, arrays in enumerate(iter_arrays):
        logger.info(f"Processing chunk: {i_chunk}")
        analysis_output = analysis_embedding(
            collision_system="embed_thermal_model",
            source_index_identifiers=source_index_identifiers,
            arrays=arrays,
            trigger_pt_ranges={
                "reference": (5, 7),
                "signal": (20, 50),
            },
            min_track_pt={
                "part_level": 1.,
                "det_level": 1.,
                "hybrid": 1.,
            },
            momentum_weight_exponent=1,
            combinatorics_chunk_size=500,
            scale_factor=1,
            det_level_artificial_tracking_efficiency=0.99,
            use_jet_trigger=False,
            return_skim=False,
        )
        merged_hists = output_utils.merge_results(merged_hists, analysis_output.hists)

    import IPython

    IPython.embed()  # type: ignore[no-untyped-call]

    # Just for quick testing!
    import uproot

    with uproot.recreate("test_eec_thermal_model.root") as f:
        output_utils.write_hists_to_file(hists=merged_hists, f=f)
