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

    #try:
    jetscape.parse_to_parquet(
        input_filename=Path(inputs[0].filepath),
        base_output_filename=output_filename_template,
        store_only_necessary_columns=store_only_necessary_columns,
        events_per_chunk=events_per_chunk,
    )
    # There's no return value for the conversion, so return True by convention
    status = True, f"success for {inputs[0].filepath}"
    #except ValueError as e:
    #    status = False, f"File {inputs[0].filepath} failed with {e}"
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
    for pt_hat_bin in pt_hat_bins[:2]:
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

    hists = jet_raa.run(
        arrays=jet_raa.load_data(
            Path(inputs[0].filepath)
        ),
        jet_R_values=jet_R_values,
        min_jet_pt=min_jet_pt,
    )

    return hists


def setup_RAA_analysis() -> List[AppFuture]:
    ...


def _cancel(job: AppFuture) -> None:
    """
    Taken directly from: `coffea.processor.executor`
    """
    try:
        # this is not implemented with parsl AppFutures
        job.cancel()
    except NotImplementedError:
        pass


def _futures_handler(input_futures: Sequence[AppFuture], timeout: Optional[float] = None) -> Iterable[Any]:
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
            except KeyboardInterrupt:
                for job in futures:
                    _cancel(job)
                running = sum(job.running() for job in futures)
                logger.warning(
                    f"Early stop: cancelled {len(futures) - running} jobs, will wait for {running} running jobs to complete"
                )
    finally:
        running = sum(job.running() for job in futures)
        if running:
            logger.warning(
                f"Cancelling {running} running jobs (likely due to an exception)"
            )
        while futures:
            _cancel(futures.pop())


#def run() -> None:
if __name__ == "__main__":
    task_config = job_utils.TaskConfig(n_cores_per_task=1)
    #n_cores_to_allocate = 64
    n_cores_to_allocate = 21
    walltime = "2:00:00"

    # Basic setup: logging and parsl.
    # NOTE: Parsl's logger setup is broken, so we have to set it up before starting logging. Otherwise,
    #       it's super verbose and a huge pain to turn off. Note that by passing on the storage messages,
    #       we don't actually lose any info.
    config, facility_config, stored_messages = job_utils.config(
        facility="ORNL_b587_short",
        task_config=task_config,
        n_tasks=n_cores_to_allocate,
        walltime=walltime,
        enable_monitoring=False,
    )
    import IPython; IPython.embed()
    rich_console = helpers.setup_logging(level=logging.WARNING, stored_messages=stored_messages, aggressively_quiet_parsl_logging=True)
    executor = parsl.load(config)
    parsl.set_stream_logger(level=logging.WARNING)
    logger.info(f"handlers root: {logging.getLogger().handlers}, local: {logger.handlers}")
    #logging.getLogger("database_manager").setLevel(logging.WARNING)
    for name, v in logging.root.manager.loggerDict.items():
        if not isinstance(v, logging.PlaceHolder):
            print(f"name: {name}, handlers: {v.handlers}")
        else:
            print(f"Placeholder: {name}")
    import IPython; IPython.embed()

    all_results = setup_convert_jetscape_files(
        #ascii_output_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/5020_PP_Colorless/"),
        ascii_output_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_30-40_0.30_2.0_1"),
        events_per_chunk=5000,
    )

    logger.warning(f"Accumulated {len(all_results)} results")

    # Show processing progress
    # Since it returns the outputs, we can actually use this to accumulate results.
    #gen_results = _futures_handler(all_results)
    #gen_results = concurrent.futures.as_completed(all_results)

    logger.warning(f"Warning post gen results")
    logger.info(f"handlers root: {logging.getLogger().handlers}, local: {logger.handlers}")

    #rich_logger = logging.getLogger("rich")

    # wattttttt? Why does it break here??

    print("About to loop")
    logger.info("Does this work???")
    print("After log, pre loop")

    #with Progress(console=helpers.rich_console, refresh_per_second=10) as progress:
    #    track_results = progress.add_task(total=len(all_results), description="Processing results...")
    #for r in gen_results:
    #for a in all_results:
    #    r = a.result()
    #    print("printing...")
    #    print(f"print: {r}")
    #    logger.info(f"log info: {r}")
    #    logger.warning(f"log warning: {r}")
    #    #rich_logger.warning(f"rich log warning: {r}")
    #    #progress.console.log(f"progress object: {r}")
    #    logger.warning(f"log warning after: {r}")
    #    #progress.update(track_results, advance=1)

    # As far as I can tell, jobs will start executing as soon as they can, regardless of
    # asking for the result. By embedded here, we can inspect results, etc in the meantime.
    # NOTE: This may be commented out sometimes when I have long running processes and wil
    #       probably forget to close it.
    #IPython.start_ipython(user_ns=locals())

    logger.info("Yo1")
    #rich_logger.info("rich log info after")
    #rich_logger.warning("rich log warning after")
    print("after rich logger...")

    # In case we close IPython early, wait for all apps to complete
    print(f"print handlers root: {logging.getLogger().handlers}, local: {logger.handlers}")
    logger.info(f"log handlers root: {logging.getLogger().handlers}, local: {logger.handlers}")
    res = [r.result() for r in all_results]
    print(f"print handlers root: {logging.getLogger().handlers}, local: {logger.handlers}")
    logger.info(f"log handlers root: {logging.getLogger().handlers}, local: {logger.handlers}")
    logger.info(res)
    print(f"print handlers root: {logging.getLogger().handlers}, local: {logger.handlers}")
    logger.info(f"log handlers root: {logging.getLogger().handlers}, local: {logger.handlers}")

    logger.info("Done")
    logger.info("Yo")

    IPython.start_ipython(user_ns=locals())


#if __name__ == "__main__":
#    run()
