#!/usr/bin/env bash

date

if [ $# != 6 ]; then
     echo "This script needs 6 parameters !"
     echo "Six parameters: nEvents, particlemomMin, particlemomMax, specialSetting, pythia6Settings, and njobs"
     exit 0
fi

nEvents=$1
particlemomMin=$2
particlemomMax=$3
specialSetting=$4
pythia6Settings="$5"
njobs="$6"
inputFile="https://www.phenix.bnl.gov/WWW/publish/phnxbld/sPHENIX/files/sPHENIX_G4Hits_sHijing_9-11fm_00000_00010.root"
embed_input_file="https://www.phenix.bnl.gov/WWW/publish/phnxbld/sPHENIX/files/sPHENIX_G4Hits_sHijing_9-11fm_00000_00010.root"
skip=0

uniqueID=${specialSetting}_${pythia6Settings}

# If pythia6Settings is a path rather than a setting, take just the filename without the extension
if [[ "${pythia6Settings}" == /* ]]; then
     uniqueID="${specialSetting}_$(basename ${pythia6Settings%%.*})"
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
rm -rf $outDir/* $logDir/*

slurmJobConfig="$initDir/$outputDir/slurmJob_${uniqueID}.sbatch"

echo "nEvents=$nEvents"
echo "particlemomMin=$particlemomMin"
echo "particlemomMax=$particlemomMax"
echo "specialSetting=\"$specialSetting\""
echo "pythia6Settings=\"$pythia6Settings\""
echo "outDir=\"$outDir\""

# We want a ceiling function so we always have enough nodes.
# Based on https://stackoverflow.com/a/12536521/12907985
nNodes=$(((${njobs} + 32 - 1) / 32))
tasksPerNode=$(((${njobs} + ${nNodes} - 1) / ${nNodes}))

# Now, implement steering script for slurm
#SBATCH --tasks-per-node=1
cat > ${slurmJobConfig} <<- _EOF_
#!/usr/bin/env bash
#SBATCH -A birthright
#SBATCH -p burst
#SBATCH -N $nNodes
#SBATCH -n $tasksPerNode
#SBATCH -c 1
#SBATCH -J eic-fun4all-sim
#SBATCH --mem=3G
#SBATCH -t 4:00:00
#SBATCH -o ${logDir}/%A-%a.stdout
#SBATCH -e ${logDir}/%A-%a.stderr

srun ./run_job_with_singularity.sh $nEvents $particlemomMin $particlemomMax $specialSetting $pythia6Settings $inputFile $outputFile $embed_input_file $skip $initDir/$outDir
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

sbatch -a 1-$njobs $slurmJobConfig

