#!/bin/bash
#
#SBATCH --job-name=track1_detection
#SBATCH --partition=gpu
#SBATCH --mem=256g
#SBATCH --gres=gpu
#SBATCH --output=track1_detection-%j.out
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=sli13@scu.edu
#

module load Anaconda3
cd /WAVE/workarea/users/sli13/AIC23_MTPC/yolov8
conda init
conda deactivate
conda activate ultralytics
python test_predictor.py -s scene_044
