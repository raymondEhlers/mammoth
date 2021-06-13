#!/usr/bin/env bash

export FASTJET_ROOT="$PWD/install/fastjet"
export PATH="${FASTJET_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH=${FASTJET_ROOT}/lib:${LD_LIBRARY_PATH}"
