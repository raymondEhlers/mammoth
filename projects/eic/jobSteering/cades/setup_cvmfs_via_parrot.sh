#!/usr/bin/env bash

export PARROT_ALLOW_SWITCHING_CVMFS_REPOSITORIES=yes
export BNL_S1="http://cvmfs-s1bnl.opensciencegrid.org/cvmfs"
export PUBKEYFILE="/etc/cvmfs/keys/opensciencegrid.org/opensciencegrid.org.pub"
export PARROT_CVMFS_REPO="<default-repositories> \
  eic.opensciencegrid.org:url=${BNL_S1}/eic.opensciencegrid.org,pubkey=${PUBKEYFILE}"
export HTTP_PROXY="http://cvmfsproxy1.cades.ornl.gov:3128"

# Add parrot_run to the PATH
export PATH="/lustre/or-scratch/cades-birthright/rehlers/install/bin/:${PATH}"
