#!/usr/bin/env bash

# shellcheck disable=SC2046
currentDir=$(realpath $(dirname "$0"))

export FASTJET_ROOT="$currentDir/install/fastjet"
export PATH="${FASTJET_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${FASTJET_ROOT}/lib:${LD_LIBRARY_PATH}"

# Needed for the cmake config. I prefer FASTJET_ROOT, but it's easier to just define both.
export FASTJET="${FASTJET_ROOT}"
