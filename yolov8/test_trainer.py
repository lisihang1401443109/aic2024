from ultralytics import YOLO
from aic_trainer import AIC_trainer

from loguru import logger

model = YOLO('yolov8n.pt')

model.train(trainer = AIC_trainer, data = '/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24/train/scene_001/camera_0001', augment=False)
# model.train(trainer = AIC_trainer)