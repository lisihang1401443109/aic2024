#!/bin/bash
#
#SBATCH --job-name=inference
#SBATCH --partition=gpu
#SBATCH --output=int-%j.out
#SBATCH --time=1-23:59:59
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=yhuang7@scu.edu
#

cd /WAVE/workarea/users/sli13/AIC23_MTPC/vid_inf

scenes="scene_065 scene_066 scene_067 scene_068 scene_069 scene_070"

for scene in $scenes
do
    python inf_gen.py --scene "$scene"
done