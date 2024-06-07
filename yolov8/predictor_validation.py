import math
from tqdm import tqdm

import os

import argparse


from ultralytics.utils import ASSETS
from ultralytics.models.yolo.pose import PosePredictor
from loguru import logger

# from fast_reid import fast_reid_interface as reID

# size = (540, 960)
# video_tensor = read_video_and_resize('/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24_for_training/camera_0001/video.mp4', size = size, debug=True)

parser = argparse.ArgumentParser()
parser.add_argument("-r" , "--reverse", help = 'reverse order', action = 'store_true')
parser.add_argument('-s', '--scene', help = 'scene')
args = parser.parse_args()

predictor_args = {
    'save_dir': './results1/',
    'vid_stride': 1,
    'verbose': False,
    'save_txt': True,
    'save': True,
    'classes': [0],
    'save_crop': False,
    'show': False,
    'save_conf': True,
}
predictor = PosePredictor(overrides = predictor_args)

# size = (540, 960)
n_frames = 23994

# fmt = "%04d"

# sources = [f'/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24_for_training/camera_{fmt % a}/video.mp4' for a in [1, 6, 11, 16]]

# video_tensor = read_video_and_resize('/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24_for_training/camera_0001/video.mp4', size = size, debug=False, start=0, end=None)

scene_path = "/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24/val/scene_044"
for camera in sorted(os.listdir(scene_path)):
    video_path = os.path.join(scene_path, camera, 'video.mp4')
    res_path = f'/WAVE/workarea/users/sli13/AIC23_MTPC/yolov8/results/scene_044/{camera}.txt'

    logger.info(f'processing {video_path}')

    results = predictor(source = video_path, model = 'yolov8m-pose.pt', stream=True)
    pbar = tqdm(range(n_frames), desc=f'processing scene_044/{camera}', position = 0 )
    for i, result in enumerate(results):
        pbar.update()
        # pass
        pass
    print(f'video {video_path} processed')