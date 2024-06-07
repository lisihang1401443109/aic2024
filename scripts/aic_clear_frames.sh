#!/bin/bash
#
#SBATCH --job-name=prep_data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=4
#SBATCH --output=prep_data-%j.out
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sli13@scu.edu
#

module load Anaconda3
cd /WAVE/workarea/users/sli13/AIC23_MTPC/scripts/
conda init
conda deactivate
conda activate aic
python /WAVE/workarea/users/sli13/AIC23_MTPC/scripts/aic_clear_frames.py \
python /WAVE/workarea/users/sli13/AIC23_MTPC/scripts/aic_clear_frames.py
