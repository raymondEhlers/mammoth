"""Run jetscape analyses via parsl

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import List, Sequence

import IPython
import parsl
from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

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

    jetscape.parse_to_parquet(
        input_filename=Path(inputs[0].filepath),
        base_output_filename=output_filename_template,
        store_only_necessary_columns=store_only_necessary_columns,
        events_per_chunk=events_per_chunk,
    )
    # There's no return value for the conversion, so return True by convention
    return True


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


def run() -> None:
    task_config = job_utils.TaskConfig(n_cores_per_task=1)
    #n_cores_to_allocate = 64
    n_cores_to_allocate = 2
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
        enable_monitoring=True,
    )
    executor = parsl.load(config)
    helpers.setup_logging(stored_messages=stored_messages)

    all_results = setup_convert_jetscape_files(
        ascii_output_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/5020_PP_Colorless/"),
    )

    logger.info(f"Accumulated {len(all_results)} results")

    # As far as I can tell, jobs will start executing as soon as they can, regardless of
    # asking for the result. By embedded here, we can inspect results, etc in the meantime.
    # NOTE: This may be commented out sometimes when I have long running processes and wil
    #       probably forget to close it.
    IPython.start_ipython(user_ns=locals())

    # In case we close IPython early, wait for all apps to complete
    res = [r.result() for r in all_results]
    logger.info(res)

    logger.info("Done")


if __name__ == "__main__":
    run()
