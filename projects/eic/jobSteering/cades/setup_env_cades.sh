#!/usr/bin/env bash

# NOTE: To use this, you must have access to cvmfs or be inside the singularity container

export ECCE_ROOT="/lustre/or-scratch/cades-birthright/${USER}/ECCE"
export ECCE_MYINSTALL="${ECCE_ROOT}/install"

# Need to setup using the build from updatebuild.sh
# ECCE (seemingly needed for s3 access)
#source /cvmfs/eic.opensciencegrid.org/ecce/gcc-8.3/opt/fun4all/core/bin/ecce_setup.sh -n new
# Local installation
#source /cvmfs/eic.opensciencegrid.org/ecce/gcc-8.3/opt/fun4all/core/bin/setup_local.sh ${ECCE_MYINSTALL}
# eic (works better for the modular detector.
source /cvmfs/eic.opensciencegrid.org/gcc-8.3/opt/fun4all/core/bin/eic_setup.sh -n new
# Local installation
source /cvmfs/eic.opensciencegrid.org/gcc-8.3/opt/fun4all/core/bin/setup_local.sh ${ECCE_MYINSTALL}
export ROOT_INCLUDE_PATH="${ECCE_ROOT}/fun4all_eicmacros/common:${ROOT_INCLUDE_PATH}"
