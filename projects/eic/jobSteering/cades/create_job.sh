#!/usr/bin/env bash

date

if [ $# != 9 ]; then
     echo "This script needs 9 parameters !"
     echo "Parameters: nEvents, particlemomMin, particlemomMax, specialSetting, pythia6Settings, inputFile (usually PYTHIA config), macroName (FullDetectorModular or EICDetector), njobs, cleanPreviousSimulations"
     exit 0
fi

nEvents=$1
particlemomMin=$2
particlemomMax=$3
specialSetting=$4
pythia6Settings="$5"
#inputFile="https://www.phenix.bnl.gov/WWW/publish/phnxbld/sPHENIX/files/sPHENIX_G4Hits_sHijing_9-11fm_00000_00010.root"
inputFile="$6"
macroName="$7"
njobs="$8"
cleanPreviousSimulations="$9"
embed_input_file="https://www.phenix.bnl.gov/WWW/publish/phnxbld/sPHENIX/files/sPHENIX_G4Hits_sHijing_9-11fm_00000_00010.root"
skip=0

# Validation of macro name
if [[ "$macroName" != "EICDetector" && "$macroName" != "FullDetectorModular" ]];
then
    echo "Invalid macro name $macroName. Please check"
    exit 1
fi

uniqueID=${specialSetting}_${pythia6Settings}

# If pythia6Settings is a path rather than a setting, take just the filename without the extension
if [[ "${pythia6Settings}" == /* ]]; then
    uniqueID="${specialSetting}_$(basename ${pythia6Settings%%.*})"
fi
# Alternatively, use the inputFile if "PYTHIA" is in the special string
# This is for the EICDetector
if [[ "$macroName" == "EICDetector" && "${pythia6Settings}" == *"PYTHIA"* ]]; then
    uniqueID="${specialSetting}_$(basename ${inputFile%%.*})"
fi

# pythia6Settings is empty
if [ -z $pythia6Settings ]; then
    uniqueID=${specialSetting}
fi

echo "uniqueID: ${uniqueID}"

initDir=$PWD
outputDir="outputDir"
outDir="$outputDir/output_$uniqueID"
logDir="$outputDir/log_$uniqueID"

mkdir -p $outDir $logDir

# Determine which index to start the job index from
startingIndex=1
if [[ "${cleanPreviousSimulations}" == true ]];
then
    echo "Removing previous simulations results for \"${uniqueID}\""
    rm -rf "${outDir:?}/*" "${logDir:?}/*"
else
    # shellcheck disable=SC2012
    startingIndex=$(ls "${outDir}/" | sort | tail -n 1 | cut -d '_' -f2)
    # Force bash to treat this as an integer (it could be treated as octal if it has a leading 0)
    startingIndex=$((10#$startingIndex))
    # We want to take the next index, so we increment one further
    startingIndex=$((startingIndex + 1))
fi
echo "Starting index: ${startingIndex}"

slurmJobConfig="$initDir/$outputDir/slurmJob_${uniqueID}.sbatch"

echo "nEvents=$nEvents"
echo "particlemomMin=$particlemomMin"
echo "particlemomMax=$particlemomMax"
echo "specialSetting=\"$specialSetting\""
echo "pythia6Settings=\"$pythia6Settings\""
echo "outDir=\"$outDir\""

# We want a ceiling function so we always have enough nodes.
# Based on https://stackoverflow.com/a/12536521/12907985
nNodes=$((($njobs + 32 - 1) / 32))
# shellcheck disable=SC2034
tasksPerNode=$((($njobs + $nNodes - 1) / $nNodes))

# Now, implement steering script for slurm
#SBATCH --tasks-per-node=1
cat > "${slurmJobConfig}" <<- _EOF_
#!/usr/bin/env bash
#SBATCH -A birthright
#SBATCH -p burst
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -J eic-fun4all-sim
#SBATCH --mem=4G
#SBATCH -t 4:00:00
#SBATCH -o ${logDir}/%A-%a.stdout
#SBATCH -e ${logDir}/%A-%a.stderr

./run_job_with_singularity.sh $macroName $nEvents $particlemomMin $particlemomMax $specialSetting $pythia6Settings $inputFile $embed_input_file $skip $initDir/$outDir
_EOF_

#cat > $condorJobCfg <<- _EOF_
#Universe     = vanilla
#Notification = never
#Initialdir   = $initDir
#GetEnr       = True
#+Job_Type    = "cas"
#Executable   = Fun4All_G4_FullDetectorModular.sh
#_EOF_
#
#ifile=1
##njobs=$[$ifile+$njobs]
#while [ $ifile -le $njobs ]
#do
#     ifileLeadingZero=$(printf "%03d" $ifile)
#     echo $ifileLeadingZero
#
#     outputFile="G4EICDetector_${ifileLeadingZero}.root"
#
#cat >> $condorJobCfg <<- _EOF_
#
#arguments = $nEvents $particlemomMin $particlemomMax $specialSetting $pythia6Settings $inputFile $outputFile $embed_input_file $skip $outDir
#output = ${logDir}/log_${ifileLeadingZero}.out
#error  = ${logDir}/log_${ifileLeadingZero}.err
#log    = ${logDir}/log_${ifileLeadingZero}.olog
#Queue
#_EOF_
#
#     let "ifile+=1";
#done

#sbatch -a 1-$njobs $slurmJobConfig
# -1 for the upper edge because it's inclusive
sbatch -a ${startingIndex}-$(($startingIndex + $njobs - 1)) "${slurmJobConfig}"
