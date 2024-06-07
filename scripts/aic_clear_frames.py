import os
import shutil
from utils import WHERE_IS_CAM
from tqdm import tqdm

DATA_FOLDER_PATH = '/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic23/'

def rmtree(folder_dir, verbose=False):
    '''
        removing frame/frames and all its contents
    '''
    if verbose:
        for file in tqdm(os.listdir(folder_dir)):
            os.remove(os.path.join(folder_dir, file))
    else:
        for file in os.listdir(folder_dir):
            os.remove(os.path.join(folder_dir, file))    

def clear_frames():
    for usage, scenes in WHERE_IS_CAM.items():
        for scene, cams in scenes.items():
            for cam in cams:
                cam_path = os.path.join(DATA_FOLDER_PATH, usage, scene, cam)
                # Check and remove 'frame' directory
                frame_dir = os.path.join(cam_path, 'frame')
                if os.path.isdir(frame_dir):
                    print(f'Deleting {frame_dir}')
                    rmtree(frame_dir, verbose=True)
                # Check and remove 'frames' directory
                frames_dir = os.path.join(cam_path, 'frames')
                if os.path.isdir(frames_dir):
                    print(f'Deleting {frames_dir}')
                    rmtree(frames_dir, verbose=True)

if __name__ == '__main__':
    clear_frames()
