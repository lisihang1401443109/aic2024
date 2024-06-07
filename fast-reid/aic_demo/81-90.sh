#!/bin/bash
#
#SBATCH --job-name=track1_reid
#SBATCH --partition=hub
#SBATCH --output=track1_reid-%j.out
#SBATCH --time=23:59:59
#SBATCH --nodelist=hub07
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=sli13@scu.edu
#

scenes="scene_081 scene_082 scene_083 scene_084 scene_085 scene_086 scene_087 scene_088 scene_089 scene_090"

for scene in $scenes
do
  python aic_demo.py --scene "$scene"
done