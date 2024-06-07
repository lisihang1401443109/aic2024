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
# scenes="scene_061 scene_062 scene_063 scene_064 scene_065" change this to 76-80
scenes="scene_076 scene_077 scene_078 scene_079 scene_080"

for scene in $scenes
do
    python aic_hungarian.py --scene "$scene"
done