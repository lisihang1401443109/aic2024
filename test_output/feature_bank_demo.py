import cv2
from aic_path_manager import AIC_PATH

aic = AIC_PATH()

from tqdm import tqdm
from loguru import logger
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--scene', type=int, default=0, help='scene index')
    parser.add_argument('--cam', type=int, default=0, help='camera index')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    scene = args.scene
    if isinstance(scene, int):
        scene = 'scene_%03d' % scene
    cam = args.cam
    if isinstance(cam, int):
        cam = 'camera_%04d' % cam
    video_path = aic[scene,cam]['video']
    det_path = aic[scene,cam]['detection']
    
    cap = cv2.VideoCapture(video_path)
    with open(det_path, 'r') as f:
        det = list(f.readlines())
        
    # make folder for the scene_cam if not exist
    if not os.path.exists(f'result/{scene}_{cam}'):
        os.makedirs(f'result/{scene}_{cam}')
    
    # detection in format [frame_id, x, y, w, h, ... ,conf]
    global_counter = 0
    last_frame = -1
    for i in tqdm(range(len(det))):
        line = det[i].strip().split(' ')
        frame_id = int(line[0])
        x, y, w, h = float(line[1]), float(line[2]), float(line[3]), float(line[4])
        conf = float(line[-1])
        
        if frame_id != last_frame:
            last_frame = frame_id
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
        
        crop = frame[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
        cv2.imwrite(f'result/{scene}_{cam}/{global_counter}.jpg', crop)
        global_counter += 1
        
    
    
    
    

if __name__ == '__main__':
    main()
    