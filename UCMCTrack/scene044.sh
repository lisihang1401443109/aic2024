#!/bin/bash
#
#SBATCH --job-name=track1_tracking
#SBATCH --partition=hub
#SBATCH --output=track1_detection-%j.out
#SBATCH --time=23:59:59
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=yhuang7@scu.edu
#

cd /WAVE/workarea/users/sli13/AIC23_MTPC/UCMCTrack/
scenes="scene_044"
for scene in $scenes
do
    python aic_demo_validation.py --scene "$scene" --a 15 --high_score 0.8 
done