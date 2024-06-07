#!/bin/bash
#
#SBATCH --job-name=track1_detection
#SBATCH --partition=hub
#SBATCH --output=track1_detection-%j.out
#SBATCH --time=23:59:59
#SBATCH --nodelist=hub02
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=yhuang7@scu.edu
#

scenes="scene_044"

for scene in $scenes
do
   python test_predictor.py -s "$scene" 
done