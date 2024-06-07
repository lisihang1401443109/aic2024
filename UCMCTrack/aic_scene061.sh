#!/bin/bash
#
#SBATCH --job-name=track1_tracking
#SBATCH --partition=hub
#SBATCH --output=track1_detection-%j.out
#SBATCH --time=23:59:59
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=sli13@scu.edu
#

cd /WAVE/workarea/users/sli13/AIC23_MTPC/UCMCTrack/

python aic_demo.py --scene scene_061 --a 15 --high_score 0.8 
