#!/bin/bash
#
#SBATCH --job-name=track1_reassign
#SBATCH --partition=cpu
#SBATCH --output=validation_reassign-%j.out
#SBATCH --time=1-23:59:59
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=yhuang7@scu.edu
#

cd /WAVE/workarea/users/sli13/AIC23_MTPC/stcra/
# scenes="scene_061 scene_062 scene_063 scene_064 scene_065" 
# extend the scenes too 61-90
scenes="scene_044"

for scene in $scenes
do
    python aic_reassignment_val.py --scene "$scene"
done