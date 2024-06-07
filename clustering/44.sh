#!/bin/bash
#
#SBATCH --job-name=track1_tracking
#SBATCH --partition=cpu
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=2-00:00:00  

#SBATCH --output=track1_detection-%j.out
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=yhuang7@scu.edu
#

cd /WAVE/workarea/users/sli13/AIC23_MTPC/clustering/
scenes="scene_044"
for scene in $scenes
do
    python aic_hungarian_val.py --scene "$scene"
done