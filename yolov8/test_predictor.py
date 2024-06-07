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

v_pathes = []
# with open('processed.log', 'r') as f:
#     for line in f:
#         v_pathes.append(line.strip())

# reid = reID.ReID()

# results = predictor(source = video_tensor, model = 'yolov8m.pt', stream=True)\
for scene in os.listdir('/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24/test') if not args.scene else [args.scene]:
    for camera in sorted(os.listdir(f'/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24/test/{scene}') if not args.reverse else os.listdir(f'/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24/test/{scene}'))[::-1]:
        # print(camera)
    # for camera in ['camera_0649']:
        # if camera != 'camera_0390':
        #     continue
        video_path = f'/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24/test/{scene}/{camera}/video.mp4'
        # video_path = '/WAVE/datasets/dmlab/aicity/aic24/track1/test/scene_071/camera_0649/video.mp4'
        
        res_path = f'/WAVE/workarea/users/sli13/AIC23_MTPC/yolov8/results/{scene}/{camera}.txt'
        
        if video_path in v_pathes:
            logger.info(f'video {video_path} already processed')
            continue
        
        logger.info(f'processing {video_path}')
        
        results = predictor(source = video_path, model = 'yolov8m-pose.pt', stream=True)
        pbar = tqdm(range(n_frames), desc=f'processing {scene}/{camera}', position = 0 )
        # video_tensor = read_video_and_resize(cap, size = size, debug=True, start=batch, end=batch+batch_size)
        for i, result in enumerate(results):
            pbar.update()
            # pass
            pass
            
        print(f'video {video_path} processed')
        
        # cleanup
        # out_dir = f'/WAVE/workarea/users/sli13/AIC23_MTPC/yolov8/results/predict/labels/{scene}/'
        # with open(out_dir + f'{camera}.txt', 'w') as f:
        #     for file in tqdm(os.listdir(out_dir), desc='writing labels'):
        #         if file[11] != '_': # not a frame txt
        #             continue
        #         if camera not in file: # not for this camera
        #             continue
        #         frame = file.split('.')[0].split('_')[-1]
        #         with open(out_dir + file, 'r') as f2:
        #             for line in f2:
        #                 f.write(frame+' '+line)
        #         os.remove(os.path.join(out_dir, file))
        
        # with open('processed.log', 'a') as f:
        #     f.write(f'{video_path}\n')
            
        
            