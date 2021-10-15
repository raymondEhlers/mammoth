"""Run AnalysisSoftwareEIC analyses via ROOT

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
import itertools
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Union

import attr
import IPython
import parsl
from mammoth import helpers, job_utils
from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from rich.progress import Progress


logger = logging.getLogger(__name__)


def iterate_in_chunks(n: int, iterable: Iterable[Any]) -> Iterable[Any]:
    """Iterate in chunks of n

    From: https://stackoverflow.com/a/8998040/12907985
    """
    it = iter(iterable)
    while True:
        chunk_it = itertools.islice(it, n)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk_it)


@attr.s(frozen=True)
class Dataset:
    data: Path = attr.ib()
    geometry: Path = attr.ib()


@attr.s(frozen=True)
class DatasetSpec:
    site: str = attr.ib()
    label: str = attr.ib()

    @property
    def identifier(self) -> str:
        return ""

    def __str__(self) -> str:
        s = f"{self.site}-{self.identifier}"
        if self.label:
            s += f"-{self.label}"
        return s


@attr.s(frozen=True)
class DatasetSpecSingleParticle(DatasetSpec):
    particle: str = attr.ib()
    momentum_selection: List[float] = attr.ib()

    @property
    def identifier(self) -> str:
        return f"single{self.particle.capitalize()}-p-{self.momentum_selection[0]:g}-to-{self.momentum_selection[1]:g}"


@attr.s(frozen=True)
class DatasetSpecPythia(DatasetSpec):
    generator: str = attr.ib()
    electron_beam_energy: int = attr.ib()
    proton_beam_energy: int = attr.ib()
    _q2_selection: List[int] = attr.ib()

    @property
    def q2(self) -> str:
        if len(self._q2_selection) == 2:
            return f"q2-{self._q2_selection[0]}-to-{self._q2_selection[1]}"
        elif len(self._q2_selection) == 1:
            return f"q2-{self._q2_selection[0]}"
        return ""

    @property
    def q2_display(self) -> str:
        if len(self._q2_selection) == 2:
            return fr"{self._q2_selection[0]} < Q^{{2}} < {self._q2_selection[1]}"
        elif len(self._q2_selection) == 1:
            return fr"Q^{{2}} > {self._q2_selection[0]}"
        return ""

    @property
    def identifier(self) -> str:
        return f"{self.generator}-{self.electron_beam_energy}x{self.proton_beam_energy}-{self.q2}"


@python_app  # type: ignore
def run_ecce_afterburner(
    tree_processing_code_directory: Path,
    output_identifier: str,
    output_dir: Path,
    do_reclustering: bool = True,
    do_jet_finding: bool = True,
    has_timing: bool = True,
    is_all_silicon: bool = True,
    max_n_events: int = -1,
    verbosity: int = 0,
    do_calibration: bool = False,
    primary_track_source: int = 0,
    jet_algorithm: str = "anti-kt",
    jet_R_parameters: Sequence[float] = [0.3, 0.5, 0.8, 1.0],
    max_track_pt_in_jet: float = 30.0,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
    stdout: str = parsl.AUTO_LOGNAME,
    stderr: str = parsl.AUTO_LOGNAME,
) -> AppFuture:
    import traceback
    import tempfile
    from pathlib import Path

    from mammoth.eic import ecce_afterburner

    with tempfile.NamedTemporaryFile("w+") as f:
        f.write("\n".join([input_file.filepath for input_file in inputs[:-1]]))
        # Apparently other processes opening this file open it at the same seek point.
        # Or at least it does in the case. So we need to seek to the beginning for it to be read
        f.seek(0)

        try:
            result = ecce_afterburner.run_afterburner(
                tree_processing_code_directory=tree_processing_code_directory,
                input_file=Path(inputs[0]) if len(inputs) == 2 else Path(f.name),
                geometry_file=Path(inputs[-1]),
                output_identifier=output_identifier,
                do_reclustering=do_reclustering,
                do_jet_finding=do_jet_finding,
                has_timing=has_timing,
                is_all_silicon=is_all_silicon,
                max_n_events=max_n_events,
                verbosity=verbosity,
                do_calibration=do_calibration,
                primary_track_source=primary_track_source,
                jet_algorithm=jet_algorithm,
                jet_R_parameters=jet_R_parameters,
                max_track_pt_in_jet=max_track_pt_in_jet,
                output_dir=output_dir,
            )
        except Exception:
            result = (
                False,
                f"failure for {inputs[0]}, identifier {output_identifier} with: \n{traceback.format_exc()}",
            )
    return result


def setup_ecce_afterburner(
    dataset: Dataset,
    dataset_spec: DatasetSpec,
    jet_R_parameters: Sequence[float],
    jet_algorithm: str,
    tree_processing_code_directory: Path,
    output_dir: Path,
    n_files_per_job: int = 10,
) -> Sequence[AppFuture]:
    # Validate that the processing code is available.
    tree_processing_entry_point = tree_processing_code_directory / "treeProcessing.C"
    if not tree_processing_entry_point.exists():
        raise ValueError(f"Tree processing at {tree_processing_entry_point} doesn't appear to be available. Check your path")

    # Find all of the input files.
    input_files = sorted(dataset.data.glob("*.root"))

    # TEMP for testing
    #input_files = input_files[:4]
    input_files = input_files[:100]
    # ENDTEMP

    futures = []
    for index, input_files in enumerate(iterate_in_chunks(n_files_per_job, input_files)):
        input_files_list = list(input_files)
        logger.info(f"Adding {index}: {input_files_list}")

        output_identifier = f"{str(dataset_spec)}/{index:03}"
        futures.append(
            run_ecce_afterburner(
                tree_processing_code_directory=tree_processing_code_directory,
                output_identifier=output_identifier,
                output_dir=output_dir,
                do_jet_finding=(jet_algorithm != ""),
                jet_algorithm=jet_algorithm,
                jet_R_parameters=jet_R_parameters,
                #max_n_events=2,
                inputs=[
                    *[File(str(input_file)) for input_file in input_files_list],
                    File(str(dataset.geometry)),
                ],
                outputs=[
                    File(str(Path("treeProcessing/{output_identifier}/output_JetObservables.root"))),
                ]
            )
        )

    return futures


def run() -> None:
    # Basic setup
    afterburner_dir = Path("/software/rehlers/dev/eic/analysis_software_EIC")
    output_dir = Path("/alf/data/rehlers/eic/afterburner")
    jet_R_parameters = [0.3, 0.5, 0.8, 1.0]
    #jet_algorithm = "anti-kt"
    jet_algorithm = ""
    # Dataset selection
    datasets_to_process = [
        #DatasetSpecPythia(
        #    site="production",
        #    generator="pythia6",
        #    electron_beam_energy=18, proton_beam_energy=275,
        #    q2_selection=[1, 100],
        #    label="",
        #),
        #DatasetSpecPythia(
        #    site="production",
        #    generator="pythia8",
        #    electron_beam_energy=10, proton_beam_energy=100,
        #    q2_selection=[1, 100],
        #    label="",
        #),
        DatasetSpecPythia(
            site="production",
            generator="pythia8",
            electron_beam_energy=10, proton_beam_energy=100,
            q2_selection=[100],
            label="",
        ),
        # Single particle
        # Production
        #DatasetSpecSingleParticle(
        #    site="production",
        #    particle="electron",
        #    momentum_selection=[0.0, 20],
        #    label="",
        #),
        #DatasetSpecSingleParticle(
        #    site="production",
        #    particle="pion",
        #    momentum_selection=[0.0, 20],
        #    label="",
        #),
        # CADES
        #DatasetSpecSingleParticle(
        #    site="cades",
        #    particle="electron",
        #    momentum_selection=[0.3, 20],
        #    label="geoOption5",
        #),
        #DatasetSpecSingleParticle(
        #    site="cades",
        #    particle="pion",
        #    momentum_selection=[0.3, 20],
        #    label="geoOption5",
        #),
        #DatasetSpecSingleParticle(
        #    site="cades",
        #    particle="electron",
        #    momentum_selection=[0.3, 20],
        #    label="geoOption6",
        #),
        #DatasetSpecSingleParticle(
        #    site="cades",
        #    particle="pion",
        #    momentum_selection=[0.3, 20],
        #    label="geoOption6",
        #),
        DatasetSpecPythia(
            site="cades",
            generator="pythia8",
            electron_beam_energy=10, proton_beam_energy=100,
            q2_selection=[1, 100],
            label="geoOption5",
        ),
        DatasetSpecPythia(
            site="cades",
            generator="pythia8",
            electron_beam_energy=10, proton_beam_energy=100,
            q2_selection=[100],
            label="geoOption5",
        ),
        DatasetSpecPythia(
            site="cades",
            generator="pythia8",
            electron_beam_energy=10, proton_beam_energy=100,
            q2_selection=[1, 100],
            label="geoOption6",
        ),
        DatasetSpecPythia(
            site="cades",
            generator="pythia8",
            electron_beam_energy=10, proton_beam_energy=100,
            q2_selection=[100],
            label="geoOption6",
        ),
        # NOTE: Put last because it has the most files!
        DatasetSpecPythia(
            site="production",
            generator="pythia8",
            electron_beam_energy=10, proton_beam_energy=100,
            q2_selection=[1, 100],
            label="",
        ),
    ]

    # Job execution parameters
    task_name = "ecce_mammoth"
    tasks_to_execute = [
        "ecce_afterburner"
    ]

    # Job execution configuration
    task_config = job_utils.TaskConfig(name=task_name, n_cores_per_task=1)
    #n_cores_to_allocate = 120
    #walltime = "1:59:00"
    n_cores_to_allocate = 100
    walltime = "10:00:00"
    #walltime = "1:59:00"
    #n_cores_to_allocate = 2

    # Validation
    # Possible datasets
    _datasets = {
        # Central productions
        "production-pythia8-18x275-q2-1-to-100": Dataset(
            data=Path("/alf/data/rehlers/eic/official_prod/prop.4/prop.4.0/HFandJets/pythia8/ep-18x275-q2-1-to-100/eval_00002"),
            geometry=Path("/alf/data/rehlers/eic/official_prod/prop.4/geometry.root")
        ),
        "production-pythia8-18x275-q2-100": Dataset(
            data=Path("/alf/data/rehlers/eic/official_prod/prop.4/prop.4.0/HFandJets/pythia8/ep-18x275-q2-100/eval_00002"),
            geometry=Path("/alf/data/rehlers/eic/official_prod/prop.4/geometry.root")
        ),
        "production-pythia8-10x100-q2-1-to-100": Dataset(
            data=Path("/alf/data/rehlers/eic/official_prod/prop.4/prop.4.0/HFandJets/pythia8/ep-10x100-q2-1-to-100/eval_00002"),
            geometry=Path("/alf/data/rehlers/eic/official_prod/prop.4/geometry.root")
        ),
        "production-pythia8-10x100-q2-100": Dataset(
            data=Path("/alf/data/rehlers/eic/official_prod/prop.4/prop.4.0/HFandJets/pythia8/ep-10x100-q2-100/eval_00002"),
            geometry=Path("/alf/data/rehlers/eic/official_prod/prop.4/geometry.root")
        ),
        "production-pythia6-18x275-q2-1-to-100": Dataset(
            data=Path("/alf/data/rehlers/eic/official_prod/prop.4/prop.4.0/SIDIS/pythia6/ep-18x275-q2-1-to-100/eval_00001/"),
            geometry=Path("/alf/data/rehlers/eic/official_prod/prop.4/geometry.root")
        ),
        # Not available yet
        "production-pythia6-18x275-q2-100": Dataset(
            data=Path("/alf/data/rehlers/eic/official_prod/prop.4/prop.4.0/SIDIS/pythia6/ep-18x275-q2-100/eval_00001/"),
            geometry=Path("/alf/data/rehlers/eic/official_prod/prop.4/geometry.root")
        ),
        #"pythia6-10x100-q2-100"
        #"pythia6-10x100-q2-1-to-100": Path("/alf/data/rehlers/"),
        # Single particle
        "production-singleElectron-p-0-to-20": Dataset(
            data=Path("/alf/data/rehlers/eic/official_prod/prop.4/prop.4.0/General/particleGun/singleElectron/eval_00001"),
            geometry=Path("/alf/data/rehlers/eic/official_prod/prop.4/geometry.root")
        ),
        "production-singlePion-p-0-to-20": Dataset(
            data=Path("/alf/data/rehlers/eic/official_prod/prop.4/prop.4.0/General/particleGun/singlePion/eval_00001"),
            geometry=Path("/alf/data/rehlers/eic/official_prod/prop.4/geometry.root")
        ),
        # CADES productions
        # Option geo 5
        # Single particle electron
        "cades-singleElectron-p-0.3-to-20-geoOption5": Dataset(
            data=Path("/alf/data/rehlers/eic/cades/LYSO_1fwd_1bkd_option5/output_TTLGEO_5_SimpleElectron/"),
            geometry=Path("/alf/data/rehlers/eic/cades/LYSO_1fwd_1bkd_option5/geometry.root"),
        ),
        # Single particle pion
        "cades-singlePion-p-0.3-to-20-geoOption5": Dataset(
            data=Path("/alf/data/rehlers/eic/cades/LYSO_1fwd_1bkd_option5/output_TTLGEO_5_SimplePion/"),
            geometry=Path("/alf/data/rehlers/eic/cades/LYSO_1fwd_1bkd_option5/geometry.root"),
        ),
        # Pythia8
        "cades-pythia8-10x100-q2-1-to-100-geoOption5": Dataset(
            data=Path("/alf/data/rehlers/eic/cades/LYSO_1fwd_1bkd_option5/output_TTLGEO_5_Jets_pythia8_ep-10x100-q2-1-to-100/"),
            geometry=Path("/alf/data/rehlers/eic/cades/LYSO_1fwd_1bkd_option5/geometry.root"),
        ),
        "cades-pythia8-10x100-q2-100-geoOption5": Dataset(
            data=Path("/alf/data/rehlers/eic/cades/LYSO_1fwd_1bkd_option5/output_TTLGEO_5_Jets_pythia8_ep-10x100-q2-100/"),
            geometry=Path("/alf/data/rehlers/eic/cades/LYSO_1fwd_1bkd_option5/geometry.root"),
        ),
        # Option geo 6
        # Single particle electron
        "cades-singleElectron-p-0.3-to-20-geoOption6": Dataset(
            data=Path("/alf/data/rehlers/eic/cades/LYSO_1fwd_1bkd_option6/output_TTLGEO_6_SimpleElectron/"),
            geometry=Path("/alf/data/rehlers/eic/cades/LYSO_1fwd_1bkd_option5/geometry.root"),
        ),
        # Single particle pion
        "cades-singlePion-p-0.3-to-20-geoOption6": Dataset(
            data=Path("/alf/data/rehlers/eic/cades/LYSO_1fwd_1bkd_option6/output_TTLGEO_6_SimplePion/"),
            geometry=Path("/alf/data/rehlers/eic/cades/LYSO_1fwd_1bkd_option5/geometry.root"),
        ),
        "cades-pythia8-10x100-q2-1-to-100-geoOption6": Dataset(
            data=Path("/alf/data/rehlers/eic/cades/LYSO_1fwd_1bkd_option6/output_TTLGEO_6_Jets_pythia8_ep-10x100-q2-1-to-100/"),
            geometry=Path("/alf/data/rehlers/eic/cades/LYSO_1fwd_1bkd_option6/geometry.root"),
        ),
        "cades-pythia8-10x100-q2-100-geoOption6": Dataset(
            data=Path("/alf/data/rehlers/eic/cades/LYSO_1fwd_1bkd_option6/output_TTLGEO_6_Jets_pythia8_ep-10x100-q2-100/"),
            geometry=Path("/alf/data/rehlers/eic/cades/LYSO_1fwd_1bkd_option6/geometry.root"),
        ),
    }
    for d in datasets_to_process:
        print(str(d))
        if str(d) not in _datasets:
            raise ValueError(f"Invalid dataset name: {d}")

    # Basic setup: logging and parsl.
    # We need ROOT, fastjet, and LHAPDF for these jobs, so we need to setup an additional initialization.
    alibuild_based_software = ["ROOT/latest"]
    additional_worker_init_script = f"eval `/usr/bin/alienv -w /software/rehlers/alice/sw --no-refresh printenv {','.join(alibuild_based_software)}`"
    # LHAPDF and fastjet are included in the external dir
    additional_worker_init_script += f"; source {afterburner_dir / 'external' / 'setup_cpp.sh'}"
    # NOTE: Parsl's logger setup is broken, so we have to set it up before starting logging. Otherwise,
    #       it's super verbose and a huge pain to turn off. Note that by passing on the storage messages,
    #       we don't actually lose any info.
    config, facility_config, stored_messages = job_utils.config(
        #facility="ORNL_b587_long",
        #facility="ORNL_b587_short",
        facility="ORNL_b587_vip",
        task_config=task_config,
        n_tasks=n_cores_to_allocate,
        walltime=walltime,
        enable_monitoring=True,
        additional_worker_init_script=additional_worker_init_script,
    )
    # Keep track of the dfk to keep parsl alive
    dfk = helpers.setup_logging_and_parsl(
        parsl_config=config,
        level=logging.INFO,
        stored_messages=stored_messages,
    )
    # Further setup
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for dataset_spec in datasets_to_process:
        # Setup tasks
        dataset_results: List[AppFuture] = []
        if "ecce_afterburner" in tasks_to_execute:
            dataset_results.extend(
                setup_ecce_afterburner(
                    dataset=_datasets[str(dataset_spec)],
                    dataset_spec=dataset_spec,
                    jet_algorithm=jet_algorithm,
                    jet_R_parameters=jet_R_parameters,
                    tree_processing_code_directory=afterburner_dir / "treeAnalysis",
                    output_dir=output_dir,
                    n_files_per_job=5,
                )
            )

        all_results.extend(dataset_results)
        logger.info(f"Accumulated {len(dataset_results)} futures for {dataset_spec}")

    logger.info(f"Accumulated {len(all_results)} total futures")

    # Process the futures, showing processing progress
    # Since it returns the results, we can actually use this to accumulate results.
    gen_results = job_utils.provide_results_as_completed(all_results, running_with_parsl=True)

    # In order to support writing histograms from multiple systems, we need to index the output histograms
    # by the collision system + centrality.
    output_hists: Dict[str, Dict[Any, Any]] = {str(k): {} for k in datasets_to_process}
    with Progress(console=helpers.rich_console, refresh_per_second=1, speed_estimate_period=300) as progress:
        track_results = progress.add_task(total=len(all_results), description="Processing results...")
        # for a in all_results:
        for result in gen_results:
            # r = a.result()
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

            output_hist_filename = Path("output") / collision_system / f"hardest_kt_{file_label}.root"
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
