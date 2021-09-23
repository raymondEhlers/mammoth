#!/usr/bin/env bash

# Setup for singularity
module purge
module load PE-gnu singularity/3.6.3

export ECCE_ROOT="/lustre/or-scratch/cades-birthright/$USER/ECCE"
#singularity exec -B ${ECCE_ROOT}/Singularity/cvmfs:/cvmfs -B /lustre/or-scratch:/lustre/or-scratch ${ECCE_ROOT}/Singularity/cvmfs/eic.opensciencegrid.org/singularity/rhic_sl7_ext.simg ${ECCE_ROOT}/jobSteering/run_Fun4All_G4_FullDetectorModular.sh "$@"
echo "Running run_Fun4All_G4_${1}.sh"
# We need the first argument to determine the macro. We pass the rest of them along
singularity exec -B ${ECCE_ROOT}/Singularity-new/cvmfs:/cvmfs -B /lustre/or-scratch:/lustre/or-scratch ${ECCE_ROOT}/Singularity-new/cvmfs/eic.opensciencegrid.org/singularity/rhic_sl7_ext.simg ${ECCE_ROOT}/jobSteering/run_Fun4All_G4_${1}.sh "${@:2}"
