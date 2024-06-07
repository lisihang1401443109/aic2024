#!/bin/bash
#
#SBATCH --job-name=track1_tracking
#SBATCH --partition=cpu
#SBATCH --output=track1_detection-%j.out
#SBATCH --time=23:59:59
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=sli13@scu.edu
#

cd /WAVE/workarea/users/sli13/AIC23_MTPC/clustering/
scenes="scene_071 scene_072 scene_073 scene_074 scene_075"
for scene in $scenes
do
    python aic_hungarian.py --scene "$scene"
done