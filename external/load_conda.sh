#!/usr/bin/env bash

envName=${1}

# Need to setup conda first in the subshell
# See https://github.com/conda/conda/issues/7980
MY_CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "${MY_CONDA_BASE}"/etc/profile.d/conda.sh

# And then load the environment
conda activate "${envName}"
