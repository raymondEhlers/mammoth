#!/usr/bin/env bash

downloadInputFiles=$1

inputFilesDate="2022-10-14"
referenceFilesDate="2022-10-14"

currentDir=$(realpath $(dirname "$0"))
# input
inputDir="${currentDir}/track_skim_validation/input"
aliceInputTarGzFilename="alice_input_files-${inputFilesDate}.tar.gz"
# reference
referenceDir="${currentDir}/track_skim_validation/reference"
referenceTarGzFilename="reference_files-${referenceFilesDate}.tar.gz"

# Helpers
bold=$(tput bold)
normal=$(tput sgr0)

echo "=> ${bold}Input files${normal}"
if [[ ! -z "${downloadInputFiles}" ]];
then
    # Arbitrarily use the first file to check if the input is available.
    if [[ ! -f ${inputDir}/alice/data/2017/LHC17p/000282343/pass1_FAST/AOD234/0001/root_archive.zip ]];
    then
        echo "--> Downloading missing ALICE input files"
        pushd ${inputDir}
        # NOTE: To get the download link, I had to share, open the share incognito, go to the details, and then copy the direct link
        curl -O -J -L https://new.cernbox.cern.ch/remote.php/dav/public-files/PwTZazAvBDoWkYz/${aliceInputTarGzFilename}
        # Extract the files in the inputDir
        tar -xf ${aliceInputTarGzFilename}
        # Cleanup
        rm ${aliceInputTarGzFilename}
        popd
    else
        echo "--> Already have ALICE input files - skipping"
    fi
else
    echo "--> Skipping downloading ALICE input files. If you need them, please request them via passing an argument."
fi

echo -e "\n=> ${bold}Reference files${normal}"
# This returns a string containing all of the files in the directory
referenceDirFiles=$(shopt -s nullglob dotglob; echo ${referenceDir}/AnalysisResults*.root)
# We want to count the number of files, so we split the string into an array
referenceDirFiles=(${referenceDirFiles//;/ })
# There should be 12 files: (pp, pythia, PbPb, embed_pythia, embed_pythia-pythia, embed_pythia-PbPb) x 2 for two jetR
if [[ ! ${#referenceDirFiles[@]} -eq 12 ]];
then
    echo "--> Downloading missing reference files"
    pushd ${referenceDir}
    echo "current dir: $(pwd)"
    # NOTE: To get the download link, I had to share, open the share incognito, go to the details, and then copy the direct link
    curl -O -J -L https://new.cernbox.cern.ch/remote.php/dav/public-files/mMAUdHDFsWAuegk/${referenceTarGzFilename}
    # Extract the files in the referenceDir
    tar -xf ${referenceTarGzFilename}
    # Cleanup
    rm ${referenceTarGzFilename}
    popd
else
    echo "--> Already have reference files - skipping!"
fi
