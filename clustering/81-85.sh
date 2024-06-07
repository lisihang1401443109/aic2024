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
scenes="scene_081 scene_082 scene_083 scene_084 scene_085"
for scene in $scenes
do
    python aic_hungarian.py --scene "$scene"
done