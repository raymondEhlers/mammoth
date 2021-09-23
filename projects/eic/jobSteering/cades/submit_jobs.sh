#!/bin/bash
date

#nEvents=$1
#particlemomMin=$2
#particlemomMax=$3
#specialSetting=$4
#pythia6Settings=$5
#njobs=$6

#./submitJob.sh 2000 -1 -1 ALLSILICON-FTTLSE2LC-ETTL-CTTLSE1-ACLGAD-TREXTOUT epMB 1000
#./submitJob.sh 2500 -1 -1 ALLSILICON-FTTLS3LC-ETTL-CTTL-ACLGAD-TREXTOUT epMB 1000
#./submitJob.sh 2500 -1 -1 ALLSILICON-FTTLSE2LC-ETTL-CTTL-ACLGAD-TREXTOUT epMB 1000
#./submitJob.sh 500 -1 -1 EVTTREE-ALLSILICON-TREXTOUT epMB 500
#./submitJob.sh 500 -1 -1 EVTTREE-ALLSILICON-TREXTOUT pTHard5 500
#./submitJob.sh 500 -1 -1 EVTTREE-ALLSILICON-FTTLS3LC-ETTL-CTTL-ACLGAD-TREXTOUT epMB 500
#./submitJob.sh 500 -1 -1 EVTTREE-ALLSILICON-FTTLS3LC-ETTL-CTTL-ACLGAD-TREXTOUT pTHard5 500
#./submitJob.sh 500 -1 -1 ALLSILICON-FTTLS3LC-ETTL-CTTL-TREXTOUT epMB 500
#./submitJob.sh 500 -1 -1 ALLSILICON-FTTLS3LC-ETTL-CTTL-ACLGAD-TREXTOUT epMB 500
#./submitJob.sh 500 -1 -1 ALLSILICON-FTTLS3LC-ETTL-CTTL-TREXTOUT pTHard5 500
#./submitJob.sh 500 -1 -1 ALLSILICON-FTTLS3LC-ETTL-CTTL-ACLGAD-TREXTOUT pTHard5 500
#./submitJob.sh 500 -1 -1 BARRELV4-FSTV3-FTTLS3LC-ETTL-CTTL-TREXTOUT pTHard5 500
#./submitJob.sh 500 -1 -1 BARRELV4-FSTV3-FTTLS3LC-ETTL-CTTL-ACLGAD-TREXTOUT pTHard5 500
#./submitJob.sh 500 -1 -1 ALLSILICON-TREXTOUT pTHard5 500
#./submitJob.sh 500 -1 -1 BARRELV4-FSTV3-TREXTOUT pTHard5 500
#./submitJob.sh 500 -1 -1 ALLSILICON-FTTLS3LVC-ETTLLC-CTTLLC-TREXTOUT pTHard5 500
#./submitJob.sh 500 -1 -1 BARRELV4-FSTV3-FTTLS3LVC-ETTLLC-CTTLLC-TREXTOUT pTHard5 500
#./submitJob.sh 500 -1 -1 ALLSILICON-FTTLSE2LC-ETTL-CTTLSE1-TREXTOUT pTHard5 500
#./submitJob.sh 500 -1 -1 ALLSILICON-FTTLS2LC-ETTL-CTTL-TREXTOUT pTHard5 500
#./submitJob.sh 500 -1 -1 ALLSILICON-FTTLSE1LC-ETTLSE1-CTTLSE1-TREXTOUT pTHard5 500


#./submitJob.sh 1000 -1 -1 ALLSILICON-FTTLS3LC-ETTL-CTTL-TREXTOUT e10p250pTHard5 250
#./submitJob.sh 1000 -1 -1 ALLSILICON-FTTLS3LC-ETTL-CTTL-TREXTOUT e10p250MB 250
#./submitJob.sh 1000 -1 -1 ALLSILICON-FTTLS3LC-ETTL-CTTL-TREXTOUT-HC2xEC2x e10p250pTHard5 250
#./submitJob.sh 2500 0 30 FHCALSTANDALONE SimplePion 200
#./submitJob.sh 2500 0 30 CALOSTANDALONE SimplePion 200
#./submitJob.sh 2500 0 30 CALOSTANDALONE SimplePhoton 200
#./submitJob.sh 1000 -1 -1 ALLSILICON-FTTLS3LC-ETTL-CTTL-TREXTOUT e18p275pTHard5 250
#./submitJob.sh 1000 -1 -1 ALLSILICON-FTTLS3LC-ETTL-CTTL-TREXTOUT e18p275MB 250
#./submitJob.sh 1000 -1 -1 ALLSILICON-FTTLS3LC-ETTL-CTTL-TREXTOUT e5p100pTHard5 250
#./submitJob.sh 1000 -1 -1 ALLSILICON-FTTLS3LC-ETTL-CTTL-TREXTOUT e5p100MB 250
#./submitJob.sh 1000 -1 -1 ALLSILICON-FTTLS3LC-ETTL-CTTL-ACLGAD-TREXTOUT e10p250pTHard5 250
#./submitJob.sh 1000 -1 -1 ALLSILICON-FTTLS3LC-ETTL-CTTL-ACLGAD-TREXTOUT e10p250MB 250
#./submitJob.sh 1000 -1 -1 ALLSILICON-TREXTOUT e10p250pTHard5 250
#./submitJob.sh 1000 -1 -1 ALLSILICON-TREXTOUT e10p250MB 250
#./submitJob.sh 1000 -1 -1 ALLSILICON-FTTLS3LC-ETTL-CTTL-ACLGAD-TREXTOUT e10p250MB 250

# Mine
#./submitJob.sh 1000 -1 -1 ALLSILICON-FTTLS3LC-ETTL-CTTL-INNERTRACKING e10p250pTHard5 10
#./submitJob.sh 1000 -1 -1 ALLSILICON-FTTLS3LC-ETTL-CTTL-ACLGAD-TREXTOUT e10p250pTHard5 250

# 3 June 2021
# Testing
#./submitJob.sh 10 -1 -1 EHCAL-ALLSILICON-FTTLS3LC-ETTL-CTTL-INNERTRACKING e10p250MB 1
# Pt hard > 5 for mid rapidity
#./submitJob.sh 10 -1 -1 ALLSILICON-ETTL-CTTL-INNERTRACKING-ASYM-FTTLS3LC e10p250pTHard5 1
# Q2 > 20 for forward
#./submitJob.sh 10 -1 -1 ALLSILICON-ETTL-CTTL-INNERTRACKING-ASYM-FTTLS3LC /gpfs/mnt/gpfs02/eic/${USER}/code/fun4all_eicmacros/detectors/Modular/config/phpythia6_ep_MinQ2_10.cfg 1
# First submission
#./submitJob.sh 1000 -1 -1 EHCAL-ALLSILICON-FTTLS3LC-ETTL-CTTL-INNERTRACKING e10p250MB 250

# Prod
# Pt hard > 5 for mid rapidity
#./submitJob.sh 1000 -1 -1 ALLSILICON-ETTL-CTTL-INNERTRACKING-ASYM-FTTLS3LC e10p250pTHard5 250
# Q2 > 20 for forward
#./submitJob.sh 1000 -1 -1 ALLSILICON-ETTL-CTTL-INNERTRACKING-ASYM-FTTLS3LC /gpfs/mnt/gpfs02/eic/${USER}/code/fun4all_eicmacros/detectors/Modular/config/phpythia6_ep_MinQ2_20.cfg 250
#./create_job.sh 1000 -1 -1 ALLSILICON-ETTL-CTTL-INNERTRACKING-ASYM-FTTLS3LC /lustre/or-scratch/cades-birthright/${USER}/mammoth/projects/eic/config/phpythia6_ep_MinQ2_20.cfg 250
# Test
#./create_job.sh 100 -1 -1 ALLSILICON-ETTL-CTTL-INNERTRACKING-ASYM-FTTLS3LC PYTHIA8 /lustre/or-scratch/cades-birthright/${USER}/mammoth/projects/eic/config/phpythia6_ep_MinQ2_10.cfg 1
#./create_job.sh 10 -1 -1 TTLGEO_5 PYTHIA8 /lustre/or-scratch/cades-birthright/${USER}/mammoth/projects/eic/config/Jets_pythia8_ep-10x100-q2-100.cfg FullDetectorModular 1
#./create_job.sh 10 -1 -1 TTLGEO_5 PYTHIA8 /lustre/or-scratch/cades-birthright/${USER}/mammoth/projects/eic/config/Jets_pythia8_ep-10x100-q2-100.cfg EICDetectorsdfsdf 1
#./create_job.sh 10 -1 -1 TTLGEO_5 PYTHIA8 /lustre/or-scratch/cades-birthright/${USER}/mammoth/projects/eic/config/Jets_pythia8_ep-10x100-q2-100.cfg EICDetector 1

# Single particle

./create_job.sh 10 -1 -1 TTLGEO_5 PYTHIA8 /lustre/or-scratch/cades-birthright/${USER}/mammoth/projects/eic/config/Jets_pythia8_ep-10x100-q2-100.cfg EICDetector 1
