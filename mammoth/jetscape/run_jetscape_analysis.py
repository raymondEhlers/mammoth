"""Run jetscape analyses via parsl

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import IPython
from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from rich.progress import Progress

from mammoth import helpers, job_utils

logger = logging.getLogger(__name__)


@python_app  # type: ignore
def convert_jetscape_files(
    output_filename_template: str,
    events_per_chunk: int,
    store_only_necessary_columns: bool,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> AppFuture:
    from pathlib import Path

    from mammoth.framework.normalize_data import jetscape

    try:
        jetscape.parse_to_parquet(
            input_filename=Path(inputs[0].filepath),
            base_output_filename=output_filename_template,
            store_only_necessary_columns=store_only_necessary_columns,
            events_per_chunk=events_per_chunk,
        )
        # There's no return value for the conversion, so return True by convention
        status = True, f"success for {inputs[0].filepath}"
    except ValueError as e:
        status = False, f"File {inputs[0].filepath} failed with {e}"
    return status


def setup_convert_jetscape_files(
    ascii_output_dir: Path,
    events_per_chunk: int = 50000,
    store_only_necessary_columns: bool = True,
    input_filename_template: str = "JetscapeHadronListBin{pt_hat_bin}",
    output_filename_template: str = "",
) -> List[AppFuture]:
    from mammoth.framework.normalize_data import jetscape

    # Strictly speaking, I don't think it's necessary to control this in so much detail
    # (ie. just searching the directory would be fine too), but it seems convenient to
    # have this fully under control.
    pt_hat_bins = jetscape.find_production_pt_hat_bins_in_filenames(
        ascii_output_dir=ascii_output_dir,
        filename_template=input_filename_template.format(pt_hat_bin=""),
    )

    results = []
    for pt_hat_bin in pt_hat_bins:
        logger.info(f"Processing pt hat range: {pt_hat_bin}")

        input_file = File(
            str((ascii_output_dir / input_filename_template.format(pt_hat_bin=pt_hat_bin)).with_suffix(".out"))
        )
        # Without knowning the number of events per file, we can't calculate the number of output files.
        # Fortunately, since we're not using parsl to stage the files, we just need one file as a proxy
        template_to_use_for_output = input_filename_template
        if output_filename_template:
            template_to_use_for_output = output_filename_template

        output_filename_template_for_skim = ascii_output_dir / "skim" / template_to_use_for_output.format(pt_hat_bin=pt_hat_bin)
        output_file = File(
            str(Path(f"{output_filename_template_for_skim}_00").with_suffix(".parquet"))
        )
        results.append(
            convert_jetscape_files(
                output_filename_template=output_filename_template_for_skim.with_suffix(".parquet"),
                events_per_chunk=events_per_chunk,
                store_only_necessary_columns=store_only_necessary_columns,
                inputs=[
                    input_file,
                ],
                outputs=[
                    output_file,
                ],
            )
        )

    return results


@python_app  # type: ignore
def run_RAA_analysis(
    system: str,
    jet_R_values: Sequence[float],
    min_jet_pt: float,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> AppFuture:
    import traceback
    from pathlib import Path

    from mammoth.jetscape import jet_raa

    try:
        hists = jet_raa.run(
            arrays=jet_raa.load_data(
                Path(inputs[0].filepath)
            ),
            jet_R_values=jet_R_values,
            min_jet_pt=min_jet_pt,
        )
        result = True, f"success for {system}, {inputs[0].filepath}", system, hists
    except Exception as e:
        result = False, f"failure for {system}, {inputs[0].filepath} with: \n{traceback.format_exc()}", system, {}
    return result


def setup_RAA_analysis(
    system: str,
    parquet_input_dir: Path,
    jet_R_values: Optional[Sequence[float]] = None,
    min_jet_pt: float = 5,
) -> List[AppFuture]:
    """Setup jet RAA analysis using the converted jetscape outputs.

    Args:
        system: Key describing the collision system (eg. `PbPb_00_10`)
        parquet_input_dir: Directory containing the converetd parquet files.
        jet_R_values: Jet R values to analyze. Default: [0.2, 0.4, 0.6]
        min_jet_pt: Minimum jet pt. Default: 5.

    Returns:
        Futures containing the output histograms from the analysis.
    """
    if jet_R_values is None:
        jet_R_values = [0.2, 0.4, 0.6]
    # NOTE: Sort by lower bin edge of the pt hat bin
    input_files = sorted(parquet_input_dir.glob("*.parquet"), key=lambda p: int(str(p.name).split("_")[0].replace("JetscapeHadronListBin", "")))

    # TEMP for testing
    # input_files = input_files[:2]
    # ENDTEMP

    results = []
    for input_file in input_files:
        logger.info(f"Adding {input_file} for analysis")
        results.append(
            run_RAA_analysis(
                system=system,
                jet_R_values=jet_R_values,
                min_jet_pt=min_jet_pt,
                inputs=[
                    File(str(input_file))
                ]
            )
        )

    return results



def run() -> None:
    # Basic configuration
    _possible_systems = ["pp", "PbPb_00_05", "PbPb_05_10", "PbPb_30_40", "PbPb_40_50"]
    _system_to_base_path = {
        "pp": Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/5020_PP_Colorless/"),
        "PbPb_00_05": Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_0-5_0.30_2.0_1"),
        "PbPb_05_10": Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_5-10_0.30_2.0_1"),
        "PbPb_30_40": Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_30-40_0.30_2.0_1"),
        "PbPb_40_50": Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_40-50_0.30_2.0_1"),
    }

    # Job execution parameters
    task_name = "Jetscape_RAA"
    tasks_to_execute = [
        #"convert",
        "analyze_RAA",
    ]
    systems_to_process = _possible_systems[1:]
    # Job execution configuration
    task_config = job_utils.TaskConfig(name=task_name, n_cores_per_task=1)
    n_cores_to_allocate = 64
    #n_cores_to_allocate = 21
    #n_cores_to_allocate = 2
    walltime = "24:00:00"

    # Basic setup: logging and parsl.
    # NOTE: Parsl's logger setup is broken, so we have to set it up before starting logging. Otherwise,
    #       it's super verbose and a huge pain to turn off. Note that by passing on the storage messages,
    #       we don't actually lose any info.
    config, facility_config, stored_messages = job_utils.config(
        facility="ORNL_b587_long",
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
        if "convert" in tasks_to_execute:
            system_results.extend(
                setup_convert_jetscape_files(
                    ascii_output_dir=_system_to_base_path[system],
                    events_per_chunk=5000,
                )
            )
        if "analyze_RAA" in tasks_to_execute:
            system_results.extend(
                setup_RAA_analysis(
                    system=system,
                    parquet_input_dir=_system_to_base_path[system] / "skim",
                    min_jet_pt=10,
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
            logger.info(f"result: {result[:2]}")
            if result[0] and len(result) == 4 and isinstance(result[3], dict):
                k = result[2]
                logger.info(f"Found result for key {k}")
                output_hists[k] = job_utils.merge_results(output_hists[k], result[3])
            logger.info(f"output_hists: {output_hists}")
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

            output_hist_filename = Path("output") / collision_system / f"jetscape_RAA{file_label}.root"
            output_hist_filename.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing output_hists to {output_hist_filename} for system {system}")
            with uproot.recreate(output_hist_filename) as f:
                helpers.write_hists_to_file(hists=hists, f=f)

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
