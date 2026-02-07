"""Generate standalone slurm scripts for analyzing the Hybrid model outputs at the Cambridge UK facility.

This approach is a bit hacky, but has it's convenient, so we go with it.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from mammoth.helpers import setup_logging
except ImportError:

    def setup_logging(level: int) -> None:
        # First, setup logging
        FORMAT = "%(message)s"
        # NOTE: For shutting up parsl, it's import to set the level both in the logging setup
        #       as well as in the handler.
        logging.basicConfig(
            level=level,
            format=FORMAT,
            datefmt="[%X]",
        )

        # Capture warnings into logging
        logging.captureWarnings(True)


logger = logging.getLogger(__name__)


def _handle_local_jetscape_analysis_dir(local_jetscape_analysis_dir: Path | None = None) -> str:
    """Handle propagating the local jetscape analysis directory.

    Use this with care! Loading python packages (e.g. a virtualenv) from a HPC shared filesystem with many jobs
    can be very high load. Use this for testing, but probably best to build a new container when it's all ready.

    Args:
        local_jetscape_analysis_dir: Local jetscape analysis directory to be loaded in the container.
    Returns:
        String that will mount the dir if a local dir was provided.
    """
    if local_jetscape_analysis_dir is not None:
        return f"-B {local_jetscape_analysis_dir}:/jetscapeOpt/jetscape-analysis"
    return ""


slurm_template_convert_output_to_parquet = """#!/usr/bin/env bash

#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/log_%A_%a.stdout
#SBATCH --error={log_dir}/log_%A_%a.stderr
#SBATCH --time={walltime_in_minutes}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array={start_job_index}-{end_job_index}

# Repeat commands for clarity
set -x

# Setup
hybrid_job_index=${{SLURM_ARRAY_TASK_ID}}
job_id_fixed=$(printf "%04d" $SLURM_ARRAY_TASK_ID)
# Final job output dir
job_output_dir={output_dir}
mkdir -p "$job_output_dir"
# Local job output dir on the node (we need this to be unique for safety)
local_job_output_dir={local_base_output_dir}/${{SLURM_JOB_ID}}_${{job_id_fixed}}
mkdir -p "$local_job_output_dir"

echo "hybrid_job_index=${{hybrid_job_index}}, job_id_fixed=${{job_id_fixed}}, job_output_dir=${{job_output_dir}}, local_job_output_dir=${{local_job_output_dir}}"

# Convert the output files to parquet
# NOTE: HARDCODE: This is the only Cambridge HPC specific part. RJE does not want to deal with generalizing this at the moment (Jan 2026).
time apptainer exec --cleanenv --no-home {local_jetscape_analysis_dir} -B /rds/project/rds-hCZCEbPdvZ8 -B {local_base_output_dir} {container_path} bash -c "cd /jetscapeOpt/jetscape-analysis/; COLUMNS=120 python3 -m jetscape_analysis.analysis.reader.skim_ascii -i {input_dir}/job-${{hybrid_job_index}}/HYBRID_Hadrons.out -o ${{local_job_output_dir}}/HYBRID_PbPb_${{job_id_fixed}}_final_state_hadrons.parquet -n {n_events_per_parquet_file}"

# Copy all .parquet and .root files from the local job output dir to the job output dir, maintaining directory structure
# We include the root files because we may have outputs from the post processing, and we do not want to have to update this command
# NOTE: no-g ensures that the group of my destination folder is consistent.
rsync -avm --no-g --include='*/' --include='*.parquet' --include='*.root' --exclude='*' ${{local_job_output_dir}}/ ${{job_output_dir}}/

# Cleanup the local job output dir
rm -rf "${{local_job_output_dir}}"
"""


def convert_to_parquet(
    # Input parameters from the Hybrid model run
    input_dir: Path,
    # Output parameters
    base_output_dir: Path,
    production_number: int,
    run_number: int,
    # Job parameters
    start_job_index: int,
    end_job_index: int,
    container_path: Path,
    # n_events optimized for PbPb since we already converted the pp sample.
    # NOTE(RJE): I changed this to 6000 since the PbPb production tends to be slightly more than 5000.
    #            Bumping slightly cuts the number of files in half, and we're not doing array at a time
    #            analysis, so we don't need to worry about the memory cost as much.
    n_events_per_parquet_file: int = 6000,
    job_name: str = "hybrid_bayesian",
    walltime_in_minutes: int = 60,
    local_jetscape_analysis_dir: Path | None = None,
    # Customized for the Cambridge HPC system
    partition: str = "icelake",
    account: str = "IRIS-IP012-CPU",
    local_base_output_dir: Path = Path("/local/hybrid-bayesian"),
) -> None:
    # Calculate useful parameters
    run_number_formatted = f"Run{run_number:04d}"

    # Now, define outputs as expected on the jetscape side
    output_dir = base_output_dir / f"production_{production_number}" / run_number_formatted
    # Log directory
    log_dir = output_dir.parent / "logs"

    # Ensure we're working with a reasonable directory.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Format the slurm script with the defined parameters
    # NOTE: Directories must be absolute!
    # NOTE: RJE 2025/04/08: In order to reduce load on the cluster, we use a node_work_dir (called local_base_output_dir),
    #       and then rsync files back to the storage directory. It only copies parquet and root files!
    slurm_script_simulation_and_analysis = slurm_template_convert_output_to_parquet.format(
        job_name=f"{job_name}_convert_to_parquet",
        log_dir=log_dir.resolve(),
        input_dir=input_dir.resolve(),
        output_dir=output_dir.resolve(),
        n_events_per_parquet_file=n_events_per_parquet_file,
        walltime_in_minutes=walltime_in_minutes,
        partition=partition,
        account=account,
        start_job_index=start_job_index,
        end_job_index=end_job_index,
        local_jetscape_analysis_dir=_handle_local_jetscape_analysis_dir(local_jetscape_analysis_dir),
        local_base_output_dir=local_base_output_dir.resolve(),
        container_path=container_path.resolve(),
    )
    # Follow-up: create log directory
    log_dir.mkdir(parents=True, exist_ok=True)

    # Write the slurm script to a file
    submit_slurm_path = base_output_dir / f"submit_convert_to_parquet_prod_{production_number}_run_{run_number}.slurm"
    with submit_slurm_path.open("w") as f:
        f.write(slurm_script_simulation_and_analysis)

    logger.info(f"Generated conversion slurm script at {submit_slurm_path}")


slurm_template_analysis_only = """#!/usr/bin/env bash

#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/log_%A_%a.stdout
#SBATCH --error={log_dir}/log_%A_%a.stderr
#SBATCH --time={walltime_in_minutes}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array={start_job_index}-{end_job_index}

# Repeat commands for clarity
set -x

# Setup
job_id_fixed=$(printf "%04d" $SLURM_ARRAY_TASK_ID)
# Final job output dir
job_output_dir={output_dir}
mkdir -p "$job_output_dir"
# Local job output dir
local_job_output_dir={local_base_output_dir}/${{SLURM_JOB_ID}}_${{job_id_fixed}}
mkdir -p "$local_job_output_dir"

echo "job_id_fixed=${{job_id_fixed}}, job_output_dir=${{job_output_dir}}, local_job_output_dir=${{local_job_output_dir}}"

# Copy the simulation outputs for the particular job
# NOTE: We need to handle this pattern carefully because we need to wait to expand the wildcard, but we need to evaluate
#       the variable. By defining it and then passing it, this seems to work
include_pattern="*${{job_id_fixed}}_final_state_hadrons*.parquet"
rsync -avm --include="${{include_pattern}}" --exclude='*' ${{job_output_dir}}/ ${{local_job_output_dir}}/
# Just double check that everything is copied.
echo "Local job directory contents:"
ls -la ${{local_job_output_dir}}

# Next, setup and run the post-processing
# NOTE: HARDCODE: The sqrt_s is specified here.
# First argument is the jetscape_analysis config file
# Second argument is a string containing the sqrt_s
# Third argument is the observable output dir
# Fourth argument is the histogram output dir
# After the fourth argument, we pass all of the input filenames (namely, the parquet final state hadron files).
post_processing_input_filenames=$(find ${{local_job_output_dir}} -name "*final_state_hadrons*.parquet")
echo "post_processing_input_filenames=${{post_processing_input_filenames}}"
time apptainer run --cleanenv --no-home {local_jetscape_analysis_dir} -B /rds/project/rds-hCZCEbPdvZ8 -B {local_base_output_dir} --app post-processing {container_path} /jetscapeOpt/jetscape-analysis/config/STAT_5020.yaml 5020 ${{local_job_output_dir}}/observables ${{local_job_output_dir}}/histograms --hadrons ${{post_processing_input_filenames[@]}}

# Copy all .parquet and .root files from the local job output dir to the job output dir, maintaining directory structure
# We include the root files because we may have outputs from the post processing, and we do not want to have to update this command
# NOTE: no-g ensures that the group of my destination folder is consistent.
rsync -avm --no-g --include='*/' --include='*.parquet' --include='*.root' --exclude='*' ${{local_job_output_dir}}/ ${{job_output_dir}}/

# Cleanup the local job output dir
rm -rf "${{local_job_output_dir}}"
"""


def analyze_output(
    # Input parameters from the JETSCAPE conversion
    base_input_dir: Path,
    production_number: int,
    run_number: int,
    # Job parameters
    start_job_index: int,
    end_job_index: int,
    container_path: Path,
    job_name: str = "hybrid_bayesian",
    walltime_in_minutes: int = 240,
    local_jetscape_analysis_dir: Path | None = None,
    # Customized for the Cambridge HPC system
    partition: str = "icelake",
    account: str = "IRIS-IP012-CPU",
    local_base_output_dir: Path = Path("/local/hybrid-bayesian"),
) -> None:
    # Calculate useful parameters
    run_number_formatted = f"Run{run_number:04d}"

    # Now, define outputs as expected on the jetscape side
    input_dir = base_input_dir / f"production_{production_number}" / run_number_formatted
    # Log directory
    log_dir = input_dir.parent / "logs"

    # Format the analysis slurm script with the defined parameters
    # Next, the template that runs just the analysis
    # NOTE: Directories must be absolute!
    # NOTE: RJE 2025/04/08: In order to reduce load on the HPC system, we use a node_work_dir (called local_base_output_dir),
    #       and then rsync files back to the storage directory. It only copies parquet and root files!
    slurm_script_analysis_only = slurm_template_analysis_only.format(
        job_name=f"{job_name}_analysis",
        log_dir=log_dir.resolve(),
        # We read and write from the same directory
        output_dir=input_dir.resolve(),
        walltime_in_minutes=walltime_in_minutes,
        partition=partition,
        account=account,
        start_job_index=start_job_index,
        end_job_index=end_job_index,
        local_jetscape_analysis_dir=_handle_local_jetscape_analysis_dir(local_jetscape_analysis_dir),
        local_base_output_dir=local_base_output_dir.resolve(),
        container_path=container_path.resolve(),
    )
    # Follow-up: create log directory
    log_dir.mkdir(parents=True, exist_ok=True)

    # Write the slurm script to a file
    submit_slurm_path = base_input_dir / f"submit_analysis_prod_{production_number}_run_{run_number}.slurm"
    with submit_slurm_path.open("w") as f:
        f.write(slurm_script_analysis_only)

    logger.info(f"Generated analysis slurm script at {submit_slurm_path}")


def main_entry_point() -> None:
    # Setup
    setup_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate standalone slurm scripts")
    parser.add_argument(
        "-c",
        "--container-path",
        action="store",
        type=Path,
        required=True,
        metavar="container_path",
        help="Path to the container to be used",
    )
    parser.add_argument(
        "--local-jetscape-analysis-dir",
        action="store",
        type=Path,
        required=False,
        default=None,
        metavar="local_dir",
        help="Directory where your local copy of jetscape-analysis is stored. It will be mounted in the container. (Use with care - it can cause load issues on HPC systems with many jobs. See the function for more info). Default: Not specified, which will use the jetscape-analysis repo already in the container.",
    )
    parser.add_argument(
        "--hybrid-output-dir",
        action="store",
        type=Path,
        required=False,
        default=Path("unset"),
        metavar="hybrid_output_dir",
        help="Directory where hybrid model for a given design point are stored. e.g. `/rds/project/rds-hCZCEbPdvZ8/peibols/bayesian/runs/0-5/point_5`. Default: Not specified, which will skip generating the parquet conversion script.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        action="store",
        type=Path,
        required=True,
        metavar="jetscape_output_dir",
        help="Base output directory where analyzed outputs are stored. (e.g. conversion outputs or analysis outputs).",
    )
    parser.add_argument(
        "-s",
        "--start-job-index",
        action="store",
        type=int,
        required=True,
        metavar="start_index",
        default=0,
        help="Start the job index from this value.",
    )
    parser.add_argument(
        "-e",
        "--end-job-index",
        action="store",
        type=int,
        required=True,
        metavar="end_index",
        help="End the job index at this value.",
    )
    parser.add_argument(
        "-p",
        "--production-number",
        action="store",
        type=int,
        required=True,
        metavar="production_number",
        help="Production number (on the analysis side)",
    )
    parser.add_argument(
        "-r",
        "--run-number",
        action="store",
        type=int,
        required=True,
        metavar="run_number",
        help="Run number (on the analysis side)",
    )

    args = parser.parse_args()

    if args.hybrid_output_dir != Path("unset"):
        logger.info(f"Hybrid output directory: {args.hybrid_output_dir}")
    logger.info("Analysis parameters")
    logger.info(f"\tbase output dir: {args.output_dir}")
    logger.info(f"\tProduction number: {args.production_number}")
    logger.info(f"\tRun number: {args.run_number}")
    logger.info("Job parameters")
    logger.info(f"\tIndex: {args.start_job_index}-{args.end_job_index}")
    logger.info(f"Container path: {args.container_path}")
    if args.local_jetscape_analysis_dir:
        logger.info(f"Local jetscape analysis dir: {args.local_jetscape_analysis_dir}")

    # Job script to convert outputs to parquet
    # Example:
    # base_input_dir = Path("/rds/project/rds-hCZCEbPdvZ8/peibols/bayesian/runs/0-5/point_5")
    # output_dir = Path("/rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian")
    if args.hybrid_output_dir != Path("unset"):
        # Only generate if we actually set an hybrid output dir (otherwise, it's not meaningful)
        convert_to_parquet(
            # Input parameters from the Hybrid model run
            input_dir=args.hybrid_output_dir,
            # Output parameters
            base_output_dir=args.output_dir,
            production_number=args.production_number,
            run_number=args.run_number,
            # Job parameters
            start_job_index=args.start_job_index,
            end_job_index=args.end_job_index,
            container_path=args.container_path,
            local_jetscape_analysis_dir=args.local_jetscape_analysis_dir,
        )

    # Job script to submit analysis jobs
    analyze_output(
        # Jetscape parameters
        base_input_dir=args.output_dir,
        production_number=args.production_number,
        run_number=args.run_number,
        # Job parameters
        start_job_index=args.start_job_index,
        end_job_index=args.end_job_index,
        container_path=args.container_path,
        local_jetscape_analysis_dir=args.local_jetscape_analysis_dir,
    )


if __name__ == "__main__":
    main_entry_point()
