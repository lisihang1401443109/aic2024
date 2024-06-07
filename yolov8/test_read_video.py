from time import time
import torch

import numpy as np
from tqdm import tqdm
from loguru import logger

import cv2

import gc




def read_video_and_resize(video_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24/train/scene_001/camera_0001/video.mp4', size = None, debug = False, start=0, end=None, half=False):
    '''
    read the video into a tensor using cv2
    param:
        video_path: path of the video
        size: size to resize the video in hw format
    return:
        video_tensor: tensor of the video
    '''
    
    
    # if debug:
    #     logger.warning(f'debugging mode is on, will generate fake data')
    #     video_tensor = np.ones((5000, 3, size[1], size[0]), dtype=np.uint8)
    #     logger.info(f'the video tensor take {sys.getsizeof(video_tensor)/1024/1024/1024} giabytes')
    #     return torch.from_numpy(video_tensor)
    
    logger.info(f'Reading video from {video_path}')

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if end is None:
        end = n_frames
    
    length = end - start
    
    n_channels = 3
    # get the frame size from cv2
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_tensor = np.zeros((length, n_channels, size[1], size[0]), dtype=np.uint8)

    t1 = time()
    
    # set cv2 to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    
    
    
    # pbar = tqdm(range(0, n_frames, batch_size))
    pbar = tqdm(range(0, length))
    for i in pbar:
        ret, frame = cap.read()
        if ret:
            if size is None:
                video_tensor[i] = frame.transpose((2, 0, 1))
            else:
                video_tensor[i] = cv2.resize(frame, size, interpolation = cv2.INTER_LINEAR).transpose((2, 0, 1))
    
    
    
    
    t2 = time()


    # Output the size of the merged tensor
    logger.debug(f'Merged tensor size: {video_tensor.shape}')
    logger.debug(f'process took {t2-t1} seconds')


    # memory_in_bytes = sys.getsizeof(merged_video_tensor)
    memory_in_bytes = video_tensor.nbytes

    logger.debug(f'1 video takes {memory_in_bytes}, bytes memory',)
    logger.debug(f'1 video takes {memory_in_bytes/1024}, KB memory')
    logger.debug(f'1 video takes {memory_in_bytes/1024/1024}, MB memory')
    logger.debug(f'1 video takes {memory_in_bytes/1024/1024/1024} GB memory')
    logger.debug(f'size of tensor is {video_tensor.shape=}')
    
    return torch.from_numpy(video_tensor)


if __name__ == '__main__':
    print(f'starting...')
    read_video_and_resize('/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24/train/scene_001/camera_0001/video.mp4', size=(540, 960))