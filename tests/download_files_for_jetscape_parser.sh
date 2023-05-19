#!/usr/bin/env bash

inputFilesDate="2023-05-19"

currentDir=$(realpath $(dirname "$0"))
# input
inputDir="${currentDir}/jetscape_parser"
inputTarGzFilename="jetscape_parser-${inputFilesDate}.tar.gz"

# Helpers
bold=$(tput bold)
normal=$(tput sgr0)

echo "=> ${bold}Input files${normal}"
# Arbitrarily use a file to check if the input is available.
if [[ ! -f ${inputDir}/final_state_hadrons_header_v1.dat ]];
then
    echo "--> Downloading missing JETSCAPE parser files"
    # NOTE: To get the download link, I had to share, open the share incognito, go to the details, and then copy the direct link
    curl -O -J -L https://cernbox.cern.ch/remote.php/dav/public-files/cdcBzXbmsnYgMRF/${inputTarGzFilename}
    # Extract the files in the current directory since the archive contains the `jetscape_parser` directory
    tar -xf ${inputTarGzFilename}
    # Cleanup
    rm ${inputTarGzFilename}
else
    echo "--> Already have JETSCAPE input files - skipping"
fi

echo "=> Done!"