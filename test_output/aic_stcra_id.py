from aic_path_manager import AIC_PATH
import cv2
import os
from tqdm import tqdm
aic = AIC_PATH()
from loguru import logger
import numpy as np

# default dict
from collections import defaultdict
from heapq import heapify, heappop

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--scene', type=int, default=0, help='scene index')
    parser.add_argument('--cam', type=int, default=0, help='camera index')
    # toggle flag -r for Reassign
    parser.add_argument('-reassign', action='store_true')
    parser
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    scene = args.scene
    cam = args.cam
    
    reassign = args.reassign
    
    if isinstance(scene, int):
        scene = 'scene_%03d' % scene
        
    if isinstance(cam, int):
        cam = 'camera_%04d' % cam
    
    logger.info(f'for scene {scene} camera {cam}---------')
    
    # create folder for the scene if not exist
    if not os.path.exists(f'result/{scene}/{cam}'):
        os.makedirs(f'result/{scene}/{cam}')
        
    no_stcra_path = aic[scene,cam]['cluster']
    stcra_path = aic[scene, cam]['reassign']
    video_path = aic[scene, cam]['video']
    
    logger.info(f'for {no_stcra_path=} \n {stcra_path=} ---------')
    
    # make two video outputs
    no_stcra = list(open(no_stcra_path, 'r').readlines())
    stcra = list(open(stcra_path, 'r').readlines())
    # no_stcra_bbox_dict = defaultdict([])
    # stcra_bbox_dict = defaultdict([])
    no_stcra_bbox_dict = defaultdict(list)
    stcra_bbox_dict = defaultdict(list)
    frame_with_det = set()
    
    for line in no_stcra:
        # format: frame, gid, x, y, w, h, conf, -1, -1, -1
        line = line.strip().split(',')
        frame = int(float(line[0]))
        gid = int(line[1])
        x, y, w, h = map(float, line[2:6])
        conf = float(line[6])
        
        no_stcra_bbox_dict[frame].append((gid, x, y, w, h))
        frame_with_det.add(frame)
    
    for line in stcra:
        # format: frame, gid, x, y, w, h, conf, -1, -1, -1
        line = line.strip().split(',')
        frame = int(line[0])
        gid = int(line[1])
        x, y, w, h = map(float, line[2:6])
        conf = float(line[6])
        
        stcra_bbox_dict[frame].append((gid, x, y, w, h))
        frame_with_det.add(frame)
        
    heap = list(frame_with_det)
    heapify(heap)
    
    # mkdir if not exist
    if not os.path.exists(f'result/{scene}_{cam}'):
        os.makedirs(f'result/{scene}_{cam}')
    
    cap = cv2.VideoCapture(video_path)
    
    if reassign:
        out2 = cv2.VideoWriter(f'result/{scene}_{cam}/aic_stcra_id.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
    else:
        out1 = cv2.VideoWriter(f'result/{scene}_{cam}/aic_no_stcra_id.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
    bar = tqdm(total=len(no_stcra) + len(stcra))
    while heap:
        frame_num = heappop(heap)
        
        if frame_num > 1000:
            break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        no_stcra_bboxes = no_stcra_bbox_dict[frame_num]
        stcra_bboxes = stcra_bbox_dict[frame_num]
        
        
            
        # draw stcra bboxes and label
        if reassign:
            for gid, x, y, w, h in stcra_bboxes:
                x = x + w/2
                y = y + h/2
                cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
                cv2.putText(frame, str(gid), (int(x-w/2), int(y-h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                bar.update(1)
            out2.write(frame)
        else:
            # draw no stcra bboxes and label
            for gid, x, y, w, h in no_stcra_bboxes:
                x = x + w/2
                y = y + h/2
                cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 0, 255), 2)
                cv2.putText(frame, str(gid), (int(x-w/2), int(y-h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                bar.update(1)
            out1.write(frame)
            
        
    if not reassign:
        out1.release()
    else:
        out2.release()
        
        
        
    
        
        
        
        
    
        
    
    
                
        
if __name__ == '__main__':
    main()
            
            