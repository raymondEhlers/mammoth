#!/usr/bin/env bash

#SBATCH --job-name=analyze_hybrid_AA
#SBATCH --output=/rstorage/rehlers/hybrid-bayesian/2025-summer-sandbox-data-sample/analysis_output/logs/log_%A_%a.stdout
#SBATCH --error=/rstorage/rehlers/hybrid-bayesian/2025-summer-sandbox-data-sample/analysis_output/logs/log_%A_%a.stderr
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=1

#SBATCH --partition=quick
#SBATCH --cpus-per-task=1
#SBATCH --array=0-199

# Repeat commands for clarity
set -x

# Setup
job_id_fixed=$(printf "%04d" $SLURM_ARRAY_TASK_ID)
input_dir=/rstorage/rehlers/hybrid-bayesian/2025-summer-sandbox-data-sample/skim
# 0-5%
# NOTE: Can't combine to run as two separate e.g. apptainer commands because I need to edit the centrality selection by hand as of Nov 2025
input_files=$(find ${input_dir}/HYBRID_Hadrons_05_Lres2_kappa_0p428/ -name "*${job_id_fixed}_final_state_hadrons*.parquet")
analysis_output_dir="/rstorage/rehlers/hybrid-bayesian/2025-summer-sandbox-data-sample/analysis_output/HYBRID_Hadrons_05_Lres2_kappa_0p428"
# 5-10%
# NOTE: Can't combine to run as two separate e.g. apptainer commands because I need to edit the centrality selection by hand as of Nov 2025
#input_files=$(find ${input_dir}/HYBRID_Hadrons_510_Lres2_kappa_0p428/ -name "*${job_id_fixed}_final_state_hadrons*.parquet")
#analysis_output_dir="/rstorage/rehlers/hybrid-bayesian/2025-summer-sandbox-data-sample/analysis_output/HYBRID_Hadrons_510_Lres2_kappa_0p428"

echo "Analyzing: ${input_files}"

apptainer run --cleanenv --no-home -B /rstorage -B /software/users/rehlers/dev/jetscape/jetscape-analysis:/jetscapeOpt/jetscape-analysis --app post-processing /rstorage/jetscape/containers/local/stat_local_gcc_v5.2.sif /jetscapeOpt/jetscape-analysis/config/STAT_5020.yaml 5020 ${analysis_output_dir}/observables ${analysis_output_dir}/histograms ${input_files}
