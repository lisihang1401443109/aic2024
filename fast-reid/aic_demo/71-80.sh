#!/bin/bash
#
#SBATCH --job-name=track1_reid
#SBATCH --partition=hub
#SBATCH --output=track1_reid-%j.out
#SBATCH --time=23:59:59
#SBATCH --nodelist=hub03
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=sli13@scu.edu
#

scenes="scene_071 scene_072 scene_073 scene_074 scene_075 scene_076 scene_077 scene_078 scene_079 scene_080"

for scene in $scenes
do
  python aic_demo.py --scene "$scene"
done