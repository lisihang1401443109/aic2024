#!/bin/bash
#
#SBATCH --job-name=track1_detection
#SBATCH --partition=hub
#SBATCH --output=track1_detection-%j.out
#SBATCH --time=23:59:59
#SBATCH --nodelist=hub07
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=sli13@scu.edu
#

scenes="scene_081 scene_082 scene_083 scene_084 scene_085"

for scene in $scenes
do
  python test_predictor.py -s "$scene"
done