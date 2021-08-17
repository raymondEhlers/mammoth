#!/usr/bin/env bash

# Setup for singularity
module purge
module load PE-gnu singularity/3.6.3

export ECCE_ROOT="/lustre/or-scratch/cades-birthright/rehlers/ECCE"
singularity exec -B ${ECCE_ROOT}/Singularity/cvmfs:/cvmfs -B /lustre/or-scratch:/lustre/or-scratch ${ECCE_ROOT}/Singularity/cvmfs/eic.opensciencegrid.org/singularity/rhic_sl7_ext.simg ${ECCE_ROOT}/jobSteering/run_Fun4All_G4_FullDetectorModular.sh "$@"
