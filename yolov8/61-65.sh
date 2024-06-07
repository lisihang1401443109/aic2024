#!/bin/bash
#
#SBATCH --job-name=track1_detection
#SBATCH --partition=hub
#SBATCH --output=track1_detection-%j.out
#SBATCH --time=23:59:59
#SBATCH --nodelist=hub03
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=sli13@scu.edu
#

scenes="scene_061 scene_062 scene_063 scene_064 scene_065"

for scene in $scenes
do
   python test_predictor.py -s "$scene" 
done