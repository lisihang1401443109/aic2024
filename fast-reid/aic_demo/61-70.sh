#!/bin/bash
#
#SBATCH --job-name=track1_reid
#SBATCH --partition=hub
#SBATCH --output=track1_reid-%j.out
#SBATCH --time=23:59:59
#SBATCH --nodelist=hub01
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=sli13@scu.edu
#

scenes="scene_061 scene_062 scene_063 scene_064 scene_065 scene_066 scene_067 scene_068 scene_069 scene_070"

for scene in $scenes
do
   python aic_demo.py --scene "$scene" 
done