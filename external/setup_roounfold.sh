#!/usr/bin/env bash

# shellcheck disable=SC2046
currentDir=$(realpath $(dirname "$0"))

export ROOUNFOLD_ROOT="$currentDir/install/roounfold"
export PATH="${ROOUNFOLD_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROOUNFOLD_ROOT}/lib:${LD_LIBRARY_PATH}"
