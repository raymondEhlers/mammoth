#!/usr/bin/env bash

source /lustre/or-scratch/cades-birthright/$USER/ECCE/jobSteering/setup_env_cades.sh

cd ${ECCE_ROOT}/fun4all_eicmacros/detectors/Modular

nEvents=$1
particlemomMin=$2
particlemomMax=$3
specialSetting=$4
pythia6Settings=$5
inputFile=$6
#outputFile=$7
embed_input_file=$7
skip=$8
outDir=${9}

# Determine the output filename here because we need to add the index.
# We want to normalize the output to contain leading zeros.
ifileLeadingZero=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
echo $ifileLeadingZero
outputFile="G4EICDetector_${ifileLeadingZero}.root"

echo "nEvents = $nEvents"
echo "particlemomMin = $particlemomMin"
echo "particlemomMax = $particlemomMax"
echo "specialSetting = $specialSetting"
echo "pythia6Settings = $pythia6Settings"
echo "inputFile = $inputFile"
echo "outputFile = $outputFile"
echo "embed_input_file = $embed_input_file"
echo "skip = $skip"
echo "outDir = $outDir"

root -l -b <<EOF
.x Fun4All_G4_FullDetectorModular.C($nEvents, $particlemomMin, $particlemomMax, "$specialSetting", "$pythia6Settings", "$inputFile", "$outputFile", "$embed_input_file", $skip, "$outDir")
.q
EOF
