from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics import YOLO
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.utils import colorstr
from aic_dataset import AIC_dataset

from time import time
from loguru import logger

import os

def build_aic_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
    logger.debug(f'building aic_dataset -------')
    return AIC_dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        # augment=mode == "train",  # augmentation
        augment = False,
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )

class AIC_trainer(DetectionTrainer):
        
        
    def build_dataset(self, img_path, mode='train', batch=None):
        
        
        # adapt to videos, where img_path points to the folder containing videos
        if 'video.mp4' not in img_path:
            img_path = os.path.join(img_path, 'video.mp4')
            
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_aic_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
    
    
    
__name__ = 'aic_trainer'