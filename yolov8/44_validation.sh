#!/bin/bash
#
#SBATCH --job-name=track1_detection

#SBATCH --partition=gpu
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=2-00:00:00 

#SBATCH --output=track1_detection-%j.out
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=yhuang7@scu.edu
#

scenes="scene_044"

for scene in $scenes
do
   python predictor_validation.py -s "$scene" 
done