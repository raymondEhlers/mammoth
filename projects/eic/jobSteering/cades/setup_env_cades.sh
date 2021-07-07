#!/usr/bin/env bash

# NOTE: To use this, you must have access to cvmfs or be inside the singularity container

export ECCE_ROOT="/lustre/or-scratch/cades-birthright/rehlers/ECCE"
export MYNISTALL="${ECCE_ROOT}/install"

# Need to setup using the build from updatebuild.sh
source /cvmfs/eic.opensciencegrid.org/ecce/gcc-8.3/opt/fun4all/core/bin/ecce_setup.sh -n new
# Local installation
source /cvmfs/eic.opensciencegrid.org/ecce/gcc-8.3/opt/fun4all/core/bin/setup_local.sh ${MYINSTALL}
export ROOT_INCLUDE_PATH="${ECCE_ROOT}/fun4all_eicmacros/common:${ROOT_INCLUDE_PATH}"
