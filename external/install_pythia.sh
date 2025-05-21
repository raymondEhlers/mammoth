#!/usr/bin/env bash

set -x

# We make enough assumptions about being in this directory that it's best that we just move there
# shellcheck disable=SC2046
currentDir=$(realpath $(dirname "$0"))
cd "${currentDir}" || exit 1

# Setup
export CXXFLAGS="-std=c++17 -O2 -pedantic -W -Wall -Wshadow -fPIC"
# Specify the pythia version.
pythia_version=8.314
prefix=$PWD/install/pythia

# Setup
# The pythia convention is to remove the period in version
pythia_version="${pythia_version/./}"

# Build in a new folder
mkdir -p pythia
cd pythia || exit 1

if [ ! -d "pythia${pythia_version}" ]; then
    echo "Downloading Pythia ${pythia_version}..."
    curl -O -J -L "https://pythia.org/download/pythia83/pythia${pythia_version}.tgz"
    tar xfz "pythia${pythia_version}.tgz"
fi

# Determine Python config
PYTHON=$(which python3)
# NOTE: using python3-config won't work as expected because it may not be shimmed by pyenv.
#       It's better to use the python executable directly to retrieve the paths.
PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIBS=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
#PYTHON_CONFIG=$(which python3-config)
#PYTHON_INCLUDE=$(python3-config --includes)
#PYTHON_LIBS=$(python3-config --ldflags)

# Setup
cd "pythia${pythia_version}" || exit 1
# Cleanup previous builds
make clean

# Generate the python bindings
# NOTE: Don't do this - it requires to run in a docker container, which is a big pain!
#       The bindings already exist in the tarball, so best to just leave them there
#cd plugins/python || exit 1
#./generate
#cd - || exit 1

# NOTE: RJE tried using just `--with-python`, but it didn't seem to work, so I decided
#       to be fully explicit to avoid further issues.
./configure --prefix="${prefix}" \
    --cxx-common="${CXXFLAGS}" \
    --with-python \
    --with-python-bin="${PYTHON}" \
    --with-python-include="${PYTHON_INCLUDE}" \
    --with-python-lib="${PYTHON_LIBS}"

make -j4
make install
