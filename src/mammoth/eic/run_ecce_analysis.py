"""Run AnalysisSoftwareEIC analyses via ROOT

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""
from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import attrs
import IPython
import parsl
from parsl.app.app import bash_app, python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from rich.progress import Progress

from mammoth import helpers, job_utils
from mammoth.eic.base import DatasetSpec, DatasetSpecPythia, DatasetSpecSingleParticle
from mammoth.framework.io import output_utils

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


@attrs.frozen
class Dataset:
    data: Path
    geometry: Path


@bash_app
def run_ecce_afterburner_bash(
    tree_processing_code_directory: Path,
    output_identifier: str,
    output_dir: Path,
    do_reclustering: bool = True,
    do_jet_finding: bool = True,
    do_calo_res: bool = False,
    is_single_particle_production: bool = False,
    max_n_events: int = -1,
    verbosity: int = 0,
    primary_track_source: int = 0,
    remove_tracklets: bool = False,
    track_projections_are_broken: bool = True,
    jet_algorithm: str = "anti-kt",
    jet_R_parameters: Sequence[float] = [0.3, 0.5, 0.8, 1.0],  # noqa: ARG001
    max_track_pt_in_jet: float = 30.0,  # noqa: ARG001
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],  # noqa: ARG001
    stdout: int | str = parsl.AUTO_LOGNAME,  # noqa: ARG001
    stderr: int | str = parsl.AUTO_LOGNAME,  # noqa: ARG001
) -> str:
    import uuid
    from pathlib import Path

    # Apparently NamedTemporarilyFile doesn't work here (reasons are unclear), so let's do it by hand...
    temp_filename = f"/tmp/{uuid.uuid4()}.txt"
    with Path(temp_filename).open("w") as f:
        f.write("\n".join([input_file.filepath for input_file in inputs[:-1]]))

    args = [
        f"\"{str(Path(inputs[0])) if len(inputs) == 2 else str(Path(f.name))}\"",
        f"\"{Path(inputs[-1])!s}\"",
        f"\"{output_identifier!s}\"",
        f"\"{output_dir!s}\"",
        str(max_n_events),
        str(do_reclustering).lower(),
        str(do_jet_finding).lower(),
        str(do_calo_res).lower(),
        str(is_single_particle_production).lower(),
        str(verbosity),
        str(primary_track_source),
        str(remove_tracklets).lower(),
        str(track_projections_are_broken).lower(),
        f"\"{jet_algorithm!s}\"",
    ]

    # NOTE: Includes the cleanup of the temporary file
    s = f"root -b -q '{tree_processing_code_directory}/treeProcessing.C({', '.join(args)})'; rm {temp_filename}"
    return s  # noqa: RET504


@python_app
def run_ecce_afterburner(
    tree_processing_code_directory: Path,
    output_identifier: str,
    output_dir: Path,
    do_reclustering: bool = True,
    do_jet_finding: bool = True,
    do_calo_res: bool = False,
    is_single_particle_production: bool = False,
    max_n_events: int = -1,
    verbosity: int = 0,
    primary_track_source: int = 0,
    remove_tracklets: bool = False,
    track_projections_are_broken: bool = True,
    jet_algorithm: str = "anti-kt",
    jet_R_parameters: Sequence[float] = [0.3, 0.5, 0.8, 1.0],
    max_track_pt_in_jet: float = 30.0,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],  # noqa: ARG001
    stdout: int | str = parsl.AUTO_LOGNAME,  # noqa: ARG001
    stderr: int | str = parsl.AUTO_LOGNAME,  # noqa: ARG001
) -> tuple[bool, str]:
    import tempfile
    import traceback
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
                output_dir=output_dir,
                max_n_events=max_n_events,
                do_reclustering=do_reclustering,
                do_jet_finding=do_jet_finding,
                do_calo_res=do_calo_res,
                is_single_particle_production=is_single_particle_production,
                verbosity=verbosity,
                primary_track_source=primary_track_source,
                remove_tracklets=remove_tracklets,
                track_projections_are_broken=track_projections_are_broken,
                jet_algorithm=jet_algorithm,
                jet_R_parameters=jet_R_parameters,
                max_track_pt_in_jet=max_track_pt_in_jet,
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
    primary_track_source: int,
    remove_tracklets: bool,
    tree_processing_code_directory: Path,
    output_dir: Path,
    use_bash_app: bool,
    n_files_per_job: int = 10,
) -> Sequence[AppFuture]:
    # Validate that the processing code is available.
    tree_processing_entry_point = tree_processing_code_directory / "treeProcessing.C"
    if not tree_processing_entry_point.exists():
        _msg = f"Tree processing at {tree_processing_entry_point} doesn't appear to be available. Check your path"
        raise ValueError(_msg)
    # Further validation
    if use_bash_app:  # noqa: SIM102
        # Can't set a vector via bash (as far as I know), so if we're non default, we need to notify immediately.
        if jet_R_parameters != [0.3, 0.5, 0.8, 1.0]:
            _msg = f"Cannot specify non-default values of jet_R_parameters ({jet_R_parameters}). Please update the default values in treeProcessing.C to change them."
            raise RuntimeError(_msg)

    # Find all of the input files.
    input_files = sorted(dataset.data.glob("*.root"))

    # TEMP for testing
    #input_files = input_files[1:2]
    # ENDTEMP

    # Limit stats to keep things moving...
    # This was sufficient to get all files for Q2 > 100, pythia8
    #input_files = input_files[:1000]

    futures = []
    func = run_ecce_afterburner_bash if use_bash_app else run_ecce_afterburner
    for index, _input_files in enumerate(iterate_in_chunks(n_files_per_job, input_files)):
        input_files_list = list(_input_files)
        logger.info(f"Adding {index}: {input_files_list}")

        output_identifier = f"{dataset_spec!s}/{index:03}"
        futures.append(
            func(
                tree_processing_code_directory=tree_processing_code_directory,
                output_identifier=output_identifier,
                output_dir=output_dir,
                do_jet_finding=(jet_algorithm != ""),
                do_calo_res=False,
                is_single_particle_production=isinstance(dataset, DatasetSpecSingleParticle),
                jet_algorithm=jet_algorithm,
                jet_R_parameters=jet_R_parameters,
                primary_track_source=primary_track_source,
                remove_tracklets=remove_tracklets,
                # Track projections are broken in prop.4, but work for future productions.
                track_projections_are_broken="prop.4" in str(dataset.data),
                max_n_events=-1,
                #verbosity=2,
                inputs=[
                    *[File(str(input_file)) for input_file in input_files_list],
                    File(str(dataset.geometry)),
                ],
                #outputs=[
                #    File(str(Path(output_dir / f"treeProcessing/{output_identifier}/output_JetObservables.root"))),
                #]
            )
        )

    return futures


def run() -> None:
    # Basic setup
    afterburner_dir = Path("/software/rehlers/dev/eic/analysis_software_EIC")
    output_dir = Path("/alf/data/rehlers/eic/afterburner/ReA/2022-01-12/min_p_cut_EPPS")
    #output_dir = Path("/alf/data/rehlers/eic/afterburner/ReA/test")
    jet_R_parameters = [0.3, 0.5, 0.8, 1.0]
    jet_algorithm = "anti-kt"
    #jet_algorithm = ""
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
        #DatasetSpecPythia(
        #    site="production",
        #    generator="pythia8",
        #    electron_beam_energy=10, proton_beam_energy=100,
        #    q2_selection=[100],
        #    label="",
        #),
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
        #DatasetSpecPythia(
        #    site="cades",
        #    generator="pythia8",
        #    electron_beam_energy=10, proton_beam_energy=100,
        #    q2_selection=[1, 100],
        #    label="geoOption5",
        #),
        #DatasetSpecPythia(
        #    site="cades",
        #    generator="pythia8",
        #    electron_beam_energy=10, proton_beam_energy=100,
        #    q2_selection=[100],
        #    label="geoOption5",
        #),
        #DatasetSpecPythia(
        #    site="cades",
        #    generator="pythia8",
        #    electron_beam_energy=10, proton_beam_energy=100,
        #    q2_selection=[1, 100],
        #    label="geoOption6",
        #),
        #DatasetSpecPythia(
        #    site="cades",
        #    generator="pythia8",
        #    electron_beam_energy=10, proton_beam_energy=100,
        #    q2_selection=[100],
        #    label="geoOption6",
        #),
        # Productions from Nico at CADES in December 2021.
        # These should be _the_ canonical productions
        #DatasetSpecPythia(
        #    site="cades",
        #    generator="pythia6",
        #    electron_beam_energy=10, proton_beam_energy=100,
        #    q2_selection=[100],
        #    label="",
        #),
        DatasetSpecPythia(
            site="cades",
            generator="pythia8",
            electron_beam_energy=10, proton_beam_energy=100,
            q2_selection=[100],
            label="",
        ),
    ]

    # Job execution parameters
    task_name = "ecce_mammoth"
    tasks_to_execute = [
        #"ecce_afterburner",
        "ecce_afterburner_bash",
    ]

    # Job execution configuration
    task_config = job_utils.TaskConfig(name=task_name, n_cores_per_task=1)
    #n_cores_to_allocate = 120
    #walltime = "1:59:00"
    n_cores_to_allocate = 90
    #walltime = "20:00:00"
    walltime = "06:00:00"
    #walltime = "02:00:00"
    # For testing
    #walltime = "1:59:00"
    #n_cores_to_allocate = 1

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
        # Productions from Nico at CADES in December 2021.
        # These should be _the_ canonical productions
        # Pythia6
        # NOTE: There is a second pythia6 production (FSTFIX_2), which is identical as this one, but contains additional stats
        #       However, we're not statistics starved, so we don't bother with merging them (since it would require re-indexing
        #       the files, which is often a pain)
        "cades-pythia6-10x100-q2-100": Dataset(
            data=Path("/alf/data/rehlers/eic/cades/nico_december_2021/output_TTLGEO_8_P6_HITS_Q2_FSTFIX_1_phpythia6_ep18x275_q2_100"),
            geometry=Path("/alf/data/rehlers/eic/cades/nico_december_2021/geometry.root"),
        ),
        # Pythia8
        "cades-pythia8-10x100-q2-100": Dataset(
            data=Path("/alf/data/rehlers/eic/cades/nico_december_2021/output_TTLGEO_7_HITS_EEMAPUPDATE_P8_10x100_Q2_JETS_1_Jets_pythia8_ep-10x100-q2-100"),
            geometry=Path("/alf/data/rehlers/eic/cades/nico_december_2021/geometry.root"),
        ),
    }
    for d in datasets_to_process:
        print(str(d))  # noqa: T201
        if str(d) not in _datasets:
            _msg = f"Invalid dataset name: {d}"
            raise ValueError(_msg)

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
        job_framework=job_utils.JobFramework.parsl,
        facility="ORNL_b587_long",
        #facility="ORNL_b587_short",
        #facility="ORNL_b587_vip",
        task_config=task_config,
        target_n_tasks_to_run_simultaneously=n_cores_to_allocate,
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
        dataset_results: list[AppFuture] = []
        if "ecce_afterburner" in tasks_to_execute or "ecce_afterburner_bash" in tasks_to_execute:
            dataset_results.extend(
                setup_ecce_afterburner(
                    dataset=_datasets[str(dataset_spec)],
                    dataset_spec=dataset_spec,
                    jet_algorithm=jet_algorithm,
                    jet_R_parameters=jet_R_parameters,
                    primary_track_source=0,
                    remove_tracklets=False,
                    tree_processing_code_directory=afterburner_dir / "treeAnalysis",
                    output_dir=output_dir,
                    use_bash_app=("ecce_afterburner_bash" in tasks_to_execute),
                    # Seems to work better for jets
                    #n_files_per_job=2,
                    #n_files_per_job=5,
                    n_files_per_job=12,
                    # Seems to work for single particle
                    #n_files_per_job=10,
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
    output_hists: dict[str, dict[Any, Any]] = {str(k): {} for k in datasets_to_process}
    with Progress(console=helpers.rich_console, refresh_per_second=1, speed_estimate_period=300) as progress:
        track_results = progress.add_task(total=len(all_results), description="Processing results...")
        # for a in all_results:
        for result in gen_results:
            # r = a.result()
            # NOTE: a bash app will just return an int, so there's not super interesting to be done.
            #       Just update the progress.
            if not isinstance(result, int):
                # There's more information here - let the user see it
                logger.info(f"result: {result[:2]}")
                if result[0] and len(result) == 4 and isinstance(result[3], dict):
                    k = result[2]
                    logger.info(f"Found result for key {k}")
                    output_hists[k] = output_utils.merge_results(output_hists[k], result[3])
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
                output_utils.write_hists_to_file(hists=hists, f=f)

    # Add a log message here to get the time that the futures are done
    logger.info("Futures done!")

    # As far as I can tell, jobs will start executing as soon as they can, regardless of
    # asking for the result. By embedded here, we can inspect results, etc in the meantime.
    # NOTE: This may be commented out sometimes when I have long running processes and wil
    #       probably forget to close it.
    IPython.start_ipython(user_ns=locals())  # type: ignore[no-untyped-call]

    # In case we close IPython early, wait for all apps to complete
    # Also allows for a summary at the end.
    # By taking only the first two, it just tells use the status and a quick message.
    # Otherwise, we can overwhelm with trying to print large objects
    if "ecce_afterburner_bash" in tasks_to_execute:
        # Bash only returns a single value, so we need to be careful
        res = [r.result() for r in all_results]
    else:
        res = [r.result()[:2] for r in all_results]
    logger.info(res)


if __name__ == "__main__":
    run()
