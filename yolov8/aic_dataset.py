from copy import deepcopy
import math
import cv2
from pycocotools.coco import COCO
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.data.base import BaseDataset

from ultralytics.utils.ops import resample_segments
from ultralytics.models.yolo.detect.train import build_yolo_dataset
from ultralytics.data.augment import Compose, Format, Instances, LetterBox, classify_augmentations, classify_transforms, v8_transforms, Instances

import torch
import numpy as np
# from torchvision.io import read_video
from test_read_video import read_video_and_resize as read_video
import os
import argparse

from loguru import logger

from time import time

class AIC_dataset(BaseDataset):
    
    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        
        # print(f'{kwargs=}')
        
        self.video_path = kwargs['img_path'] if 'video.mp4' in kwargs['img_path'] else os.path.join(kwargs['img_path'], 'video.mp4')
        
        super().__init__(*args, **kwargs)
        
        self.augment = False
        
        
        # load the entire video and labels into memory
        # pts_unit = 'sec' to supress warning
        time_start = time()
        # self.video_tensor = read_video(self.video_path, pts_unit='sec' )
        self.video_tensor = read_video(self.video_path, size=(int(1080/2), int(1920/2)))
        # logger.info(f'Skipping Reading video: {self.video_path}')
        # self.video_tensor = np.zeros((20000, 1080, 1920, 3))
        
        
        logger.info(f'time taken: {time() - time_start}')
        
    
        
        
    def get_img_files(self, img_path):
        return []
        
    def get_labels(self):
        """
        Users can customize their own format here.

        Note:
            Ensure output is a dictionary with the following keys:
            ```python
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
            ```
        """
        logger.info(f'Loading labels from {self.video_path}')
        
        label_path = os.path.join(os.path.dirname(self.video_path), 'label.txt')
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"The label file {label_path} does not exist.")

        # dict: frame_id -> list[(x, y, w, h)]
        label_dict = {}
        with open(label_path, 'r') as f:
            # Assumed format: track_id, frame_id x, y, w, h, global_x, global_y
            max_frame_id = 0
            for line in f:
                frame_id, trk_id, x, y, w, h, gx, gy = line.strip().split(' ')
                # logger.debug(f'{frame_id=}')
                max_frame_id = max(max_frame_id, int(frame_id))
                bbox = np.array([float(x), float(y), float(w), float(h)])
                if frame_id not in label_dict:
                    label_dict[frame_id] = []
                label_dict[frame_id].append(bbox)
                
        logger.debug(f'length of label_dict: {len(label_dict)}')
        for frame_id, data in label_dict.items():
            label_dict[frame_id] = {'bboxes': np.array(data).reshape((-1, 4)), 
                                    'im_file': self.video_path + ':%05d' % int(frame_id),
                                    'shape': (1080, 1920),
                                    'cls': np.zeros(len(data)),
                                    'segments': np.array([]),
                                    'keypoints': None,
                                    'normalized': True,
                                    'bbox_format': 'xywh'}
        label_list = np.ndarray((max_frame_id + 1), dtype=object)
        logger.info(f'max_frame_id: {max_frame_id}')
        for i in range(max_frame_id + 1):
            if str(i) in label_dict: 
                # label_list[i] = label_dict.get(i, None)
                label_list[i] = label_dict[str(i)]
            else:
                label_list[i] = {
                    'im_file': self.video_path + ':%05d' % int(i),
                    'shape': (1080, 1920),
                    'cls': np.array([]),
                    'bboxes': np.array([]),
                    'segments': np.array([]),
                    'keypoints': None,
                    'normalized': True,
                    'bbox_format': 'xywh'
                }
        logger.debug(f'label_list_instance: {label_list[125]}')        
        
        return label_list

    
    
    # def get_image_and_label(self, index):
    #     pass
    
    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im = self.video_tensor[i]
        if not rect_mode:
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        return im, (1920, 1080), im.shape[:2]


    
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms
        
        
    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    # NOTE: suppressed because not used (yet)
    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            # (N, 1000, 2)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in ["masks", "keypoints", "bboxes", "cls", "segments", "obb"]:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch
    
    
__name__ = 'aic_dataset'