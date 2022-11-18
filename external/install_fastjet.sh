#!/usr/bin/env bash

# Based on: https://github.com/scikit-hep/pyjet/blob/master/install-fastjet.sh

# If you need to specify CC and CXX (for example, if there are multiple compilers),
# you can do so here.
# NOTE: This isn't required on the ORNL cluster anymore
#export CC=gcc-10
#export CXX=g++-10

# Setup
export CXXFLAGS="-std=c++17 -O2"
# My reference AliPhysics Run2 build uses
#fastjet_version=3.3.3
#fjcontrib_version=1.042
# More update to date verision in the same set of patch releases would be:
#fastjet_version=3.3.4
#fjcontrib_version=1.045
# Update to date as of Feb 2022
# Validated these versions in pp, pythia, and rho subtracted PbPb vs 3.3.3_1.042 in AliPhysics, giving the same results
fastjet_version=3.4.0
fjcontrib_version=1.048
prefix=$PWD/install/fastjet

# Build in a new folder
mkdir -p fastjet
cd fastjet

if [ ! -d fastjet-${fastjet_version} ]; then
    curl -O -J -L http://fastjet.fr/repo/fastjet-${fastjet_version}.tar.gz
    tar xfz fastjet-${fastjet_version}.tar.gz
fi

if [ ! -d fjcontrib-${fjcontrib_version} ]; then
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
cd fastjet-${fastjet_version}
make clean
# NOTE: Need to disable autoptr because we're using c++17
./configure --prefix=$prefix --enable-allcxxplugins --enable-all-plugins --disable-auto-ptr
make -j4
make install

# fjcontribu
cd ../fjcontrib-${fjcontrib_version}
# We need to apply the rpath patch for fjcontrib
# However, we don't want to try to apply it if we've already done it.
# For checking, see: https://unix.stackexchange.com/a/86872
if [[ ! patch -R -p0 -s -f --dry-run < ../../fjcontrib_ldflags_rpath.patch &> /dev/null ]]; then
    echo "Applying patch to fjcontrib..."
    patch < ../../fjcontrib_ldflags_rpath.patch
fi

# Now on to build fjcontrib
make clean
# configure for fj-contrib ignores CXXFLAGS unless we pass them explicitly...
# Seriously...? :-(
# Figured out by look at alidist: https://github.com/alisw/alidist/blob/8e772427a4c51717f45ec9e22f39944512983b02/fastjet.sh#L63-L67
# NOTE: Their configure and Makefile is really a mess.
./configure --prefix=$prefix --fastjet-config=$prefix/bin/fastjet-config \
    CXXFLAGS="$CXXFLAGS" \
    CFLAGS="$CFLAGS" \
    CPATH="$CPATH" \
    C_INCLUDE_PATH="$C_INCLUDE_PATH" \
    LD_FLAGS="${LD_FLAGS}"
make -j4
make install
make fragile-shared -j4
make fragile-shared-install
