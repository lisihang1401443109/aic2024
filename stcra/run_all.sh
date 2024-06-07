#!/bin/bash
#
#SBATCH --job-name=track1_reassign
#SBATCH --partition=cpu
#SBATCH --output=track1_detection-%j.out
#SBATCH --time=1-23:59:59
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=sli13@scu.edu
#

cd /WAVE/workarea/users/sli13/AIC23_MTPC/stcra/
# scenes="scene_061 scene_062 scene_063 scene_064 scene_065" 
# extend the scenes too 61-90
scenes="scene_061 scene_062 scene_063 scene_064 scene_065 scene_066 scene_067 scene_068 scene_069 scene_070
        scene_071 scene_072 scene_073 scene_074 scene_075 scene_076 scene_077 scene_078 scene_079 scene_080
        scene_081 scene_082 scene_083 scene_084 scene_085 scene_086 scene_087 scene_088 scene_089 scene_090"

for scene in $scenes
do
    python aic_reassignment_run.py --scene "$scene"
done