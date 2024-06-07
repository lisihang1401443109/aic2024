module load Anaconda3
cd /WAVE/workarea/users/sli13/AIC23_MTPC/yolov8
conda init
conda deactivate
conda activate ultralytics
python test_predictor.py