#!/usr/bin/env bash

# We make enough assumptions about being in this directory that it's best that we just move there
# shellcheck disable=SC2046
currentDir=$(realpath $(dirname "$0"))
cd "${currentDir}" || exit 1

# Based on: https://github.com/scikit-hep/pyjet/blob/master/install-fastjet.sh

# If you need to specify CC and CXX (for example, if there are multiple compilers),
# you can do so here.
# NOTE: This isn't required on the ORNL cluster anymore
#export CC=gcc-10
#export CXX=g++-10

# Setup
export CXXFLAGS="-std=c++17 -O2"
# Specify the fastjet and fjcontrib versions.
# NOTE: Validated 3.4.0 (fj) and 1.048 (contrib) in pp, pythia, and rho subtracted PbPb vs 3.3.3_1.042 in AliPhysics, giving the same results
# FastJet:
fastjet_version=3.4.3
# NOTE: Can also install a fastjet tag (experimental as of May 2025)
#fastjet_version=73a1c97478ed9ac4f517492c523c20e09bee3ec9
# FJContrib:
fjcontrib_version=1.101
prefix=$PWD/install/fastjet

# Build in a new folder
mkdir -p fastjet
cd fastjet || exit 1

if [ ! -d fastjet-${fastjet_version} ]; then
    # Contains a ".", which would indicate a specific version
    if [ "${fastjet_version}" != "${fastjet_version/./}" ]; then
        echo "Using fastjet tag: ${fastjet_version}"
        curl -O -J -L http://fastjet.fr/repo/fastjet-${fastjet_version}.tar.gz
        tar xfz fastjet-${fastjet_version}.tar.gz
    else
        echo "Checking out fastjet from gitlab: ${fastjet_version}"
        git clone --depth 1 https://gitlab.com/fastjet/fastjet.git fastjet-${fastjet_version}
        cd fastjet-${fastjet_version} || exit 1
        # Grab the desired fastjet version
        git checkout ${fastjet_version}
        # Setup siscone
        git submodule init
        git submodule update
        # And all done - go back
        cd - || exit 1
    fi
fi

if [ ! -d fjcontrib-${fjcontrib_version} ]; then
    echo "Using fj-contrib tag: ${fjcontrib_version}"
    curl -O -J -L http://fastjet.hepforge.org/contrib/downloads/fjcontrib-${fjcontrib_version}.tar.gz
    tar xfz fjcontrib-${fjcontrib_version}.tar.gz
fi

# Determine how we want to set the rpath. This depends on the operating system...
# See: https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling#recommendations
rpathOrigin="\$\$ORIGIN"
if [[ "$OSTYPE" == "darwin"* ]]; then
    rpathOrigin="@rpath"
fi

# NOTE: Cannot use "=" as it will appear to be a variable assignment in autotools (?)
# See https://stackoverflow.com/a/61381437/12907985
export LDFLAGS="-Wl,-rpath,${rpathOrigin}/fastjet/lib -Wl,-rpath,${prefix}/lib ${LDFLAGS}"

# fastjet
pwd
cd "fastjet-${fastjet_version}" || exit 1
make clean
# NOTE: Only reconfigure if we haven't configured before
if [[ ! -f "config.status" ]]; then
    # NOTE: Need to disable autoptr because we're using c++17
    # NOTE: We use limited thread safety since we 1) it supports a more safe mode of operation, especially if fastjet is used
    #       in ways that are not yet anticipated, and 2) it has a limited impact on performance, which is critical since we're
    #       not actually using threading as of May 2025
    # NOTE: Need to disable the d0runiicone since there's a bug in 3.4.3 as of May 2025. It's fixed, but there's no release yet.
    configureArgs=(
        "--prefix=${prefix}"
        "--enable-allcxxplugins"
        "--enable-all-plugins"
        "--disable-auto-ptr"
        "--enable-limited-thread-safety"
        "--disable-d0runiicone"
    )
    # Needed in the case of a cloned git repo (from fastjet INSTALL txt file). Apparently it also runs configure...?
    if [ "${fastjet_version}" == "${fastjet_version/./}" ]; then
        # NOTE: This needs GNU libtool and libtoolize. On macOS, need to install these via homebrew.
        ./autogen.sh "${configureArgs[@]}"
    else
        ./configure "${configureArgs[@]}"
    fi
else
    echo "Skipping configuration for fastjet due to existing build"
fi
make -j4
make install

# fjcontribu
cd ../fjcontrib-${fjcontrib_version} || exit 1
# We need to apply the rpath patch for fjcontrib
# However, we don't want to try to apply it if we've already done it.
# For checking, see: https://unix.stackexchange.com/a/86872
if ! patch -R -p0 -s -f --dry-run < ../../fjcontrib_ldflags_rpath.patch &> /dev/null; then
    echo "Applying patch to fjcontrib..."
    patch < ../../fjcontrib_ldflags_rpath.patch
fi

# Now on to build fjcontrib
make clean
# configure for fj-contrib ignores CXXFLAGS unless we pass them explicitly...
# Seriously...? :-(
# Figured out by look at alidist: https://github.com/alisw/alidist/blob/8e772427a4c51717f45ec9e22f39944512983b02/fastjet.sh#L63-L67
# NOTE: Their configure and Makefile is really a mess.
# NOTE: Only reconfigure if we haven't configured before
if [[ ! -f "Makefile" ]]; then
    # shellcheck disable=SC2153
    ./configure --prefix="${prefix}" --fastjet-config="${prefix}"/bin/fastjet-config \
        CXXFLAGS="$CXXFLAGS" \
        CFLAGS="${CFLAGS}" \
        CPATH="${CPATH}" \
        C_INCLUDE_PATH="$C_INCLUDE_PATH" \
        LD_FLAGS="${LD_FLAGS}"
else
    echo "Skipping configuration for fastjet-contrib due to existing build"
fi
make -j4
make install
make fragile-shared -j4
make fragile-shared-install
