"""Run jetscape analyses via parsl

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import concurrent.futures
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import IPython
from attr import attr
import parsl
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
    jet_R_values: Sequence[float],
    min_jet_pt: float,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> AppFuture:
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
        result = True, f"success for {inputs[0].filepath}", hists
    except Exception as e:
        result = False, f"File {inputs[0].filepath} failed with {e}", {}
    return result


def setup_RAA_analysis(
    parquet_input_dir: Path,
    jet_R_values: Optional[Sequence[float]] = None,
    min_jet_pt: float = 5,
) -> List[AppFuture]:
    """Setup jet RAA analysis using the converted jetscape outputs.

    Args:
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

    # TEMP
    input_files = input_files[:2]
    # ENDTEMP

    results = []
    for input_file in input_files:
        logger.info(f"Adding {input_file} for analysis")
        results.append(
            run_RAA_analysis(
                jet_R_values=jet_R_values,
                min_jet_pt=min_jet_pt,
                inputs=[
                    File(str(input_file))
                ]
            )
        )

    return results


def _cancel(job: AppFuture) -> None:
    """
    Taken directly from: `coffea.processor.executor`
    """
    try:
        # this is not implemented with parsl AppFutures
        job.cancel()
    except NotImplementedError:
        pass


def _futures_handler(input_futures: Sequence[AppFuture], timeout: Optional[float] = None, running_with_parsl: bool = False) -> Iterable[Any]:
    """Essentially the same as concurrent.futures.as_completed
    but makes sure not to hold references to futures any longer than strictly necessary,
    which is important if the future holds a large result.

    Taken directly from: `coffea.processor.executor`
    """
    futures = set(input_futures)
    try:
        while futures:
            try:
                done, futures = concurrent.futures.wait(
                    futures,
                    timeout=timeout,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if len(done) == 0:
                    logger.warning(
                        f"No finished jobs after {timeout}s, stopping remaining {len(futures)} jobs early"
                    )
                    break
                while done:
                    try:
                        yield done.pop().result()
                    except concurrent.futures.CancelledError:
                        pass
            except KeyboardInterrupt as e:
                for job in futures:
                    _cancel(job)
                running = sum(job.running() for job in futures)
                logger.warning(
                    f"Early stop: cancelled {len(futures) - running} jobs, will wait for {running} running jobs to complete"
                )
                # parsl can't cancel, so we need to break out ourselves
                # It's most convenient to do this by just reraising the ctrl-c
                if running_with_parsl:
                    raise e
    finally:
        running = sum(job.running() for job in futures)
        if running:
            logger.warning(
                f"Cancelling {running} running jobs (likely due to an exception)"
            )
        while futures:
            _cancel(futures.pop())


from typing import Dict

def merge_result(a: Dict[Any, Any], b: Dict[Any, Any]) -> Dict[Any, Any]:
    """Merge results from jobs together

    By convention, we merge into the first argument.
    """
    # Short circuit if nothing to be done
    if not b and a:
        logger.info("Returning a since b is None")
        return a
    if not a and b:
        logger.info("Returning b since a is None")
        return b

    all_keys = set(a) | set(b)

    for k in all_keys:
        a_value = a.get(k)
        b_value = b.get(k)
        # Nothing to be done
        if a_value and b_value is None:
            logger.info(f"b_value is None for {k}. Skipping")
            continue
        # Just take the b value and move on
        if a_value is None and b_value:
            logger.info(f"a_value is None for {k}. Assigning")
            a[k] = b_value
            continue
        # At this point, both a_value and b_value should be not None
        assert a_value is not None and b_value is not None

        # Reccursve on dict
        if isinstance(a_value, dict):
            logger.info(f"Recursing on dict for {k}")
            a[k] = merge_result(a_value, b_value)
        else:
            # Otherwise, merge
            logger.info(f"Mergng for {k}")
            a[k] = a_value + b_value

    return a

from typing import BinaryIO

def _write_hists(output_hists: Dict[Any, Any], f: BinaryIO, prefix: str = "") -> bool:
    for k, v in output_hists.items():
        if isinstance(v, dict):
            _write_hists(output_hists=v, f=f, prefix=f"{prefix}_{k}")
        else:
            f[str(k)] = v  # type: ignore

    return True


def run() -> None:
    task_config = job_utils.TaskConfig(n_cores_per_task=1)
    #n_cores_to_allocate = 64
    #n_cores_to_allocate = 21
    n_cores_to_allocate = 2
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

    tasks_to_execute = [
        "analyze_RAA",
    ]

    all_results = []
    if "convert" in tasks_to_execute:
        all_results.extend(
            setup_convert_jetscape_files(
                #ascii_output_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/5020_PP_Colorless/"),
                ascii_output_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_0-5_0.30_2.0_1"),
                events_per_chunk=5000,
            )
        )
    if "analyze_RAA" in tasks_to_execute:
        all_results.extend(
            setup_RAA_analysis(
                parquet_input_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/5020_PP_Colorless/skim"),
                min_jet_pt=10,
            )
        )

    logger.warning(f"Accumulated {len(all_results)} results")
    logger.info(f"Results: {all_results}")

    # Show processing progress
    # Since it returns the outputs, we can actually use this to accumulate results.
    gen_results = _futures_handler(all_results, running_with_parsl=True)
    #gen_results = concurrent.futures.as_completed(all_results)

    output_hists: Dict[Any, Any] = {}
    with Progress(console=helpers.rich_console, refresh_per_second=1) as progress:
        track_results = progress.add_task(total=len(all_results), description="Processing results...")
        #for a in all_results:
        for result in gen_results:
            #r = a.result()
            logger.info(f"result: {result}")
            if result[0] and len(result) == 3 and isinstance(result[2], dict):
                output_hists = merge_result(output_hists, result[2])
            logger.info(f"output_hists: {output_hists}")
            progress.update(track_results, advance=1)

    # Save hists to uproot
    if output_hists:
        import uproot
        output_hist_filename = Path("output") / "pp" / "jetscape_RAA.root"
        output_hist_filename.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing output_hists to {output_hist_filename}")
        with uproot.recreate(output_hist_filename) as f:
            _write_hists(output_hists=output_hists, f=f)

    # As far as I can tell, jobs will start executing as soon as they can, regardless of
    # asking for the result. By embedded here, we can inspect results, etc in the meantime.
    # NOTE: This may be commented out sometimes when I have long running processes and wil
    #       probably forget to close it.
    IPython.start_ipython(user_ns=locals())

    # In case we close IPython early, wait for all apps to complete
    # Also allows for a summary at the end.
    res = [r.result() for r in all_results]
    logger.info(res)


if __name__ == "__main__":
    run()
