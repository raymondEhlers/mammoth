#!/usr/bin/env bash
set +e

# NOTE: RooUnfold requires boost (for tests) and ROOT.
# NOTE: If building inside conda, it seems to find ROOT just fine.

# Setup
INSTALL_PREFIX=${PWD}/install/roounfold
ADDITIONAL_ARGS="${CMAKE_ARGS}"
CMAKE_BUILD_TYPE="RELWITHDEBINFO"
CXXSTD=17

# Download RooUnfold, using the ALICE tag
if [ ! -d RooUnfold ]; then
    git clone --depth 1 --branch alice/V02-00-01 https://github.com/alisw/RooUnfold.git
fi

# Build and install
cd RooUnfold
ROOUNFOLD_SOURCE_DIR=${PWD}
cmake -S ${ROOUNFOLD_SOURCE_DIR}                    \
      -B ${ROOUNFOLD_SOURCE_DIR}/build              \
      -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}      \
      ${CXXSTD:+-DCMAKE_CXX_STANDARD=$CXXSTD}       \
      ${CMAKE_BUILD_TYPE:+-DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"}  \
      ${CMAKE_ARGS}
cmake --build build -j 4 --target install

# Need to copy the headers manually, because RooUnfold...
mkdir -p ${INSTALL_PREFIX}/include/
rsync -av ${ROOUNFOLD_SOURCE_DIR}/include/ ${INSTALL_PREFIX}/include/
