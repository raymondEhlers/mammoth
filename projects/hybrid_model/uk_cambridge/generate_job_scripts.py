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

slurm_template_convert_output_to_parquet = """#!/usr/bin/env bash

#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/log_%A_%a.stdout
#SBATCH --error={log_dir}/log_%A_%a.stderr
#SBATCH --time={walltime_in_minutes}
#SBATCH --partition={partition}
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
local_job_output_dir={local_base_output_dir}/${{job_id_fixed}}
mkdir -p "$local_job_output_dir"

echo "hybrid_job_index=${{hybrid_job_index}}, job_id_fixed=${{job_id_fixed}}, job_output_dir=${{job_output_dir}}, local_job_output_dir=${{local_job_output_dir}}"

# Convert the output files to parquet
# NOTE: HARDCODE: This is the only Cambridge HPC specific part. RJE does not want to deal with generalizing this at the moment (Jan 2026).
time apptainer exec --cleanenv --no-home -B /rds/project/rds-hCZCEbPdvZ8 -B {local_base_output_dir} {container_path} bash -c "cd /jetscapeOpt/jetscape-analysis/; python3 -m jetscape_analysis.analysis.reader.skim_ascii -i {input_dir}/job-${{hybrid_job_index}}/HYBRID_Hadrons.out -o ${{local_job_output_dir}}/HYBRID_PbPb_${{job_id_fixed}}_final_state_hadrons.parquet -n {n_events_per_parquet_file}"

# Copy all .parquet and .root files from the local job output dir to the job output dir, maintaining directory structure
# We include the root files because we may have outputs from the post processing, and we do not want to have to update this command
rsync -avm --include='*/' --include='*.parquet' --include='*.root' --exclude='*' ${{local_job_output_dir}}/ ${{job_output_dir}}/

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
    n_events_per_parquet_file: int = 5000,
    job_name: str = "hybrid_bayesian",
    walltime_in_minutes: int = 60,
    # Customized for the Cambridge HPC system
    partition: str = "cclake",
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
        start_job_index=start_job_index,
        end_job_index=end_job_index,
        local_base_output_dir=local_base_output_dir.resolve(),
        container_path=container_path.resolve(),
    )
    # Follow-up: create log directory
    log_dir.mkdir(parents=True, exist_ok=True)

    # Write the slurm script to a file
    submit_slurm_path = (
        base_output_dir / f"submit_convert_to_parquet_prod_{production_number}_{run_number_formatted}.slurm"
    )
    with submit_slurm_path.open("w") as f:
        f.write(slurm_script_simulation_and_analysis)

    logger.info(f"Generated conversion slurm script at {submit_slurm_path}")


slurm_template_analysis_only = """#!/usr/bin/env bash

#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/log_%A_%a.stdout
#SBATCH --error={log_dir}/log_%A_%a.stderr
#SBATCH --time={walltime_in_minutes}
#SBATCH --partition={partition}
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
local_job_output_dir={local_base_output_dir}/${{job_id_fixed}}
mkdir -p "$local_job_output_dir"

echo "job_id_fixed=${{job_id_fixed}}, job_output_dir=${{job_output_dir}}, local_job_output_dir=${{local_job_output_dir}}"

# Copy the simulation outputs for the particular job
rsync -avm --include='*${{job_id_fixed}}_final_state_hadrons*.parquet' --exclude='*' ${{job_output_dir}}/ ${{local_job_output_dir}}/
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
time apptainer run --cleanenv --no-home -B time apptainer exec --cleanenv --no-home -B /rds/project/rds-hCZCEbPdvZ8 -B {local_base_output_dir} --app post-processing {container_path} /jetscapeOpt/jetscape-analysis/config/STAT_5020.yaml 5020 ${{local_job_output_dir}}/observables ${{local_job_output_dir}}/histograms ${{post_processing_input_filenames[@]}}

# Copy all .parquet and .root files from the local job output dir to the job output dir, maintaining directory structure
# We include the root files because we may have outputs from the post processing, and we do not want to have to update this command
rsync -avm --include='*/' --include='*.parquet' --include='*.root' --exclude='*' ${{local_job_output_dir}}/ ${{job_output_dir}}/

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
    # Customized for the Cambridge HPC system
    partition: str = "cclake",
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
        input_dir=input_dir.resolve(),
        walltime_in_minutes=walltime_in_minutes,
        partition=partition,
        start_job_index=start_job_index,
        end_job_index=end_job_index,
        local_base_output_dir=local_base_output_dir.resolve(),
        container_path=container_path.resolve(),
    )
    # Follow-up: create log directory
    log_dir.mkdir(parents=True, exist_ok=True)

    # Write the slurm script to a file
    submit_slurm_path = base_input_dir / f"submit_analysis_{production_number}_{run_number}.slurm"
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
        "-h",
        "--hybrid-output-dir",
        action="store",
        type=Path,
        required=False,
        default=Path("unset"),
        metavar="hybrid_output_directory",
        help="Base directory where hybrid model outputs are stored.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        action="store",
        type=Path,
        required=True,
        metavar="jetscape_output_directory",
        help="Base output directory where analyzed outputs are stored. (e.g. conversion outputs or analysis outputs).",
    )
    parser.add_argument(
        "-s",
        "--start-job-index",
        action="store",
        type=int,
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

    logger.info(f"Base hybrid output directory: {args.output_dir}")
    logger.info(f"Input configuration file: {args.input_config_file}")
    logger.info(f"Container path: {args.container_path}")

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
            base_output_dir=args.jetscape_output_dir,
            production_number=args.production_number,
            run_number=args.run_number,
            # Job parameters
            start_job_index=args.start_job_index,
            end_job_index=args.end_job_index,
            container_path=args.container_path,
        )

    # Job script to submit analysis jobs
    analyze_output(
        # Jetscape parameters
        base_input_dir=args.jetscape_output_dir,
        base_output_dir=args.jetscape_output_dir,
        production_number=args.production_number,
        # Job parameters
        start_job_index=args.start_job_index,
        end_job_index=args.end_job_index,
        container_path=args.container_path,
    )


if __name__ == "__main__":
    main_entry_point()
