#!/bin/bash
#
#SBATCH --job-name=track1_tracking
#SBATCH --partition=cpu
#SBATCH --output=track1_detection-%j.out
#SBATCH --time=23:59:59
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=yhuang7@scu.edu
#

cd /WAVE/workarea/users/sli13/AIC23_MTPC/clustering/
# scenes="scene_061 scene_062 scene_063 scene_064 scene_065" change this to 86-80



python aic_hungarian.py --scene scene_061
