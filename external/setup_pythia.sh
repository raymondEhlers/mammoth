#!/usr/bin/env bash

# shellcheck disable=SC2046
currentDir=$(realpath $(dirname "$0"))

export PYTHIA_ROOT="${currentDir}/install/pythia"
export PYTHIA8="${PYTHIA_ROOT}"
export PYTHIA8DATA="${PYTHIA_ROOT}/share/Pythia8/xmldoc"
