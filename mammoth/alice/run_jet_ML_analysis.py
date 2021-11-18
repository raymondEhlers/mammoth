"""Run jet background ML related analyses via parsl

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
import secrets
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import IPython
from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from rich.progress import Progress

from mammoth import helpers, job_utils

logger = logging.getLogger(__name__)


@python_app  # type: ignore
def run_jet_background_ML_embedding_analysis(
    system_label: str,
    background_collision_system_tag: str,
    jet_R: float,
    min_jet_pt: Mapping[str, float],
    use_standard_rho_subtraction: bool = True,
    use_constituent_subtraction: bool = False,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> AppFuture:
    import traceback
    from pathlib import Path

    from mammoth.alice import jet_ML_background

    try:
        jet_ML_background.run_embedding_analysis(
            signal_filename=Path(inputs[0].filepath),
            background_filename=Path(inputs[1].filepath),
            background_collision_system_tag=background_collision_system_tag,
            jet_R=jet_R,
            min_jet_pt=min_jet_pt,
            output_filename=Path(outputs[0].filepath),
            use_standard_rho_subtraction=use_standard_rho_subtraction,
            use_constituent_subtraction=use_constituent_subtraction,
        )
        result = True, f"success for {system_label}, R={jet_R}, {inputs[0].filepath}, {inputs[1].filepath}"
    except Exception as e:
        result = False, f"failure for {system_label}, R={jet_R}, {inputs[0].filepath} with: \n{traceback.format_exc()}"
    return result


def extract_pt_hat_bin_label_from_JEWEL_filename(filename: Path) -> str:
    """Extract pt hat bin from filename.

    Example filename: "JEWEL_NoToy_PbPb_3050_PtHard60_80_052.parquet", which would return "pt_hat_60_80"
    """
    pt_hat_bin = str(filename.name).split("_")[-3:-1]
    pt_hat_bin[0] = pt_hat_bin[0].replace("PtHard", "pt_hat_")
    return "_".join(pt_hat_bin)


def setup_jet_background_ML_embedding_analysis(
    system_label: str,
    background_collision_system_tag: str,
    jet_R_values: Sequence[float],
    min_jet_pt: Mapping[str, float],
    sample_each_pt_hat_bin_equally: bool,
    signal_input_dir: Path,
    background_input_dir: Path,
    base_output_dir: Path,
) -> List[AppFuture]:
    """Setup jet ML embedding analysis.

    Args:
        background_collision_system_tag: Tag to identify the collision system for the background.
            Usually the collision system label, but may have something else such as the centrality
        jet_R_values: Jet R values to analyze
        min_jet_pt: Minimum jet pt for jet collections at different levels
        signal_input_dir: Location of the signal input files
        background_input_dir: Location of the background files
        base_output_dir: Output directory.

    Returns:
        Futures containing the status of the embedding analysis.
    """
    # NOTE: Sort by lower bin edge of the pt hat bin
    signal_input_files = list(sorted(signal_input_dir.glob("*.parquet"), key=lambda p: int(str(p.name).split("_")[4].replace("PtHard", ""))))
    background_input_files = list(sorted(background_input_dir.glob("*/*.root")))

    # TEMP for testing
    #background_input_files = background_input_files[:2]
    # ENDTEMP

    # We want to even sample all of the pt hat bins (or at least approximately)
    # However, there are many more high pt hat files than low pt hat files (because there
    # are more statistics at high pt hat). So, we should sample the pt hat bin, and then
    # randomly select one of the files for that pt hat bin
    # NOTE: These are only used when sampling each pt hat bin equally
    signal_input_files_by_pt_hat_bin_label: Dict[str, List[Path]] = defaultdict(list)
    for signal_input in signal_input_files:
        pt_hat_bin_label = extract_pt_hat_bin_label_from_JEWEL_filename(signal_input)
        signal_input_files_by_pt_hat_bin_label[pt_hat_bin_label].append(signal_input)
    # NOTE: We convert back to list in the end because random choice is expecting a
    #       sequence, and it's more efficient to convert it only once.
    pt_hat_bin_labels = list(set(list(signal_input_files_by_pt_hat_bin_label)))

    results = []
    logger.info("Creating embedding tasks. This may make time a minute...")
    for background_input_file in background_input_files:
        # NOTE: We iterate first by jet_R because I want to avoid autocorrelations if we create
        #       a ratio as a function of R. As it's configured here, the signal file will be random,
        #       but the background will be the same. That should be enough to avoid autocorrelation issues.
        for jet_R in jet_R_values:
            # Randomly select (in some manner) an input file to match up with the background input file.
            # NOTE: The signal input file will repeat if there are more background events.
            #       So far, this doesn't seem to be terribly common, but even if it was, it
            #       would be perfectly fine as long as it doesn't happen too often.
            if sample_each_pt_hat_bin_equally:
                # Each pt hat bin will be equally likely, and then we select the file from
                # those which are available.
                # NOTE: This doesn't mean that the embedded statistics will still be the same in the end.
                #       For example, if I compare a low and high pt hat bin, there are just going to be
                #       more accepted jets in the high pt hat sample.
                pt_hat_bin_label = secrets.choice(pt_hat_bin_labels)
                signal_input_file = secrets.choice(signal_input_files_by_pt_hat_bin_label[pt_hat_bin_label])
            else:
                # Directly sample the files. This probes the generator stats because
                # the number of files is directly proportional to the generated statistics.
                signal_input_file = secrets.choice(signal_input_files)
                pt_hat_bin_label = extract_pt_hat_bin_label_from_JEWEL_filename(signal_input_file)

            output_filename = base_output_dir / f"jetR{round(jet_R * 100):03}" / pt_hat_bin_label / f"jetR{round(jet_R * 100):03}_{signal_input_file.stem}_{background_input_file.parent.stem}_{background_input_file.stem}.root"
            output_filename.parent.mkdir(exist_ok=True, parents=True)

            #logger.info(f"Adding {signal_input_file}, {background_input_file} for analysis")
            #logger.info(f"Output file: {output_filename}")
            results.append(
                run_jet_background_ML_embedding_analysis(
                    system_label=system_label,
                    background_collision_system_tag=background_collision_system_tag,
                    jet_R=jet_R,
                    min_jet_pt=min_jet_pt,
                    inputs=[
                        File(str(signal_input_file)),
                        File(str(background_input_file)),
                    ],
                    outputs=[
                        File(str(output_filename)),
                    ],
                )
            )

    return results


def run() -> None:
    # Basic configuration
    _possible_systems = ["PbPb_00_10", "PbPb_30_50"]
    _system_to_paths = {
        # Pairs of (signal inputs, background inputs)
        "PbPb_00_10": (
            Path("/alf/data/rehlers/skims/JEWEL_PbPb_no_recoil/skim/central_00_10"),
            Path("/alf/data/rehlers/substructure/trains/PbPb/7666/run_by_run/LHC15o"),
        ),
        "PbPb_30_50": (
            Path("/alf/data/rehlers/skims/JEWEL_PbPb_no_recoil/skim/semi_central_30_50"),
            Path("/alf/data/rehlers/substructure/trains/PbPb/7668/run_by_run/LHC15o"),
        ),
    }
    _background_collision_system_tag = {
        "PbPb_00_10": "PbPb_central",
        "PbPb_30_50": "PbPb_semi_central",
    }
    _base_output_dir = Path("/alf/data/rehlers/skims/JEWEL_PbPb_no_recoil/embedding/LHC15o")

    # Job execution parameters
    task_name = "jet_background_ML"
    tasks_to_execute = [
        "embedding_analysis",
    ]
    jet_R_values = [0.2, 0.4, 0.6]
    min_jet_pt = {"hybrid": 10}
    sample_each_pt_hat_bin_equally = True
    #systems_to_process = _possible_systems
    systems_to_process = _possible_systems[1:]

    # Job execution configuration
    task_config = job_utils.TaskConfig(name=task_name, n_cores_per_task=1)
    n_cores_to_allocate = 80
    walltime = "24:00:00"
    #n_cores_to_allocate = 6
    #walltime = "02:00:00"

    # Basic setup: logging and parsl.
    # NOTE: Parsl's logger setup is broken, so we have to set it up before starting logging. Otherwise,
    #       it's super verbose and a huge pain to turn off. Note that by passing on the storage messages,
    #       we don't actually lose any info.
    config, facility_config, stored_messages = job_utils.config(
        facility="ORNL_b587_vip",
        task_config=task_config,
        n_tasks=n_cores_to_allocate,
        walltime=walltime,
        enable_monitoring=True,
    )
    # Keep track of the dfk to keep parsl alive
    dfk = helpers.setup_logging_and_parsl(
        parsl_config=config,
        level=logging.INFO,
        stored_messages=stored_messages,
    )

    all_results = []
    for system in systems_to_process:
        # Setup tasks
        system_results = []
        signal_input_dir, background_input_dir = _system_to_paths[system]
        output_dir = _base_output_dir / system
        if "embedding_analysis" in tasks_to_execute:
            system_results.extend(
                setup_jet_background_ML_embedding_analysis(
                    system_label=system,
                    background_collision_system_tag=_background_collision_system_tag[system],
                    jet_R_values=jet_R_values,
                    min_jet_pt=min_jet_pt,
                    sample_each_pt_hat_bin_equally=sample_each_pt_hat_bin_equally,
                    signal_input_dir=signal_input_dir,
                    background_input_dir=background_input_dir,
                    base_output_dir=output_dir,
                )
            )
        all_results.extend(system_results)
        logger.info(f"Accumulated {len(system_results)} futures for {system}")

    logger.info(f"Accumulated {len(all_results)} total futures")

    # Process the futures, showing processing progress
    # Since it returns the results, we can actually use this to accumulate results.
    gen_results = job_utils.provide_results_as_completed(all_results, running_with_parsl=True)

    # In order to support writing histograms from multiple systems, we need to index the output histograms
    # by the collision system + centrality.
    output_hists: Dict[str, Dict[Any, Any]] = {
        k: {} for k in systems_to_process
    }
    with Progress(console=helpers.rich_console, refresh_per_second=1) as progress:
        track_results = progress.add_task(total=len(all_results), description="Processing results...")
        #for a in all_results:
        for result in gen_results:
            #r = a.result()
            #logger.info(f"result: {result[:2]}")
            if result[0] and len(result) == 4 and isinstance(result[3], dict):
                k = result[2]
                logger.info(f"Found result for key {k}. Merging...")
                output_hists[k] = job_utils.merge_results(output_hists[k], result[3])
            #logger.info(f"output_hists: {output_hists}")
            progress.update(track_results, advance=1)

    # Save hists to uproot
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

            output_hist_filename = _base_output_dir / collision_system / f"jet_background_ML_{file_label}.root"
            output_hist_filename.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing output_hists to {output_hist_filename} for system {system}")
            with uproot.recreate(output_hist_filename) as f:
                helpers.write_hists_to_file(hists=hists, f=f)

    # Add a log message here to get the time that the futures are done
    logger.info("Futures done!")

    # As far as I can tell, jobs will start executing as soon as they can, regardless of
    # asking for the result. By embedded here, we can inspect results, etc in the meantime.
    # NOTE: This may be commented out sometimes when I have long running processes and wil
    #       probably forget to close it.
    IPython.start_ipython(user_ns=locals())

    # In case we close IPython early, wait for all apps to complete
    # Also allows for a summary at the end.
    # By taking only the first two, it just tells use the status and a quick message.
    # Otherwise, we can overwhelm with trying to print large objects
    res = [r.result()[:2] for r in all_results]
    logger.info(res)


if __name__ == "__main__":
    run()
