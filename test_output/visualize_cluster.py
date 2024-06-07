from aic_path_manager import AIC_PATH
import cv2
import os
from tqdm import tqdm
aic = AIC_PATH()
from loguru import logger

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--scene', type=int, default=0, help='scene index')
    # parser.add_argument('--cam', type=int, default=0, help='camera index')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    scene = args.scene
    # cam = args.cam
    if isinstance(scene, int):
        scene = 'scene_%03d' % scene
    
    logger.info(f'for scene {scene} ---------')
    
    # create folder for the scene if not exist
    if not os.path.exists('result/%s' % scene):
        logger.info('create folder for the scene %s' % scene)
        os.makedirs('result/%s' % scene)
        
    
    bbox_file = aic[scene]['cluster_bbox']
    cluster_map = {}
    gids = set()
    # format: global_id, cam_id, frame, x, y, w, h, conf
    with open(bbox_file, 'r') as f:
        for line in f:
            gid, cam_id, frame_id, x, y, w, h, conf = line.strip().split(' ')
            gid = int(gid)
            cam_id = int(float(cam_id))
            frame_id = int(float(frame_id))
            x, y, w, h, conf = float(x), float(y), float(w), float(h), float(conf)
            if gid not in gids:
                gids.add(gid)
            if cam_id not in cluster_map:
                cluster_map[cam_id] = {gid: [[frame_id, x, y, w, h, conf]]}
            else:
                if gid not in cluster_map[cam_id]:
                    cluster_map[cam_id][gid] = []
                cluster_map[cam_id][gid].append([frame_id, x, y, w, h, conf])
    
    logger.info('total %d clusters in scene %s' % (len(gids), scene))
    
    for gid in gids:
        if not os.path.exists('result/%s/%d' % (scene, gid)):
            os.makedirs('result/%s/%d' % (scene, gid))
        
    global_counter = 0
    # for cam_id, clusters in cluster_map.items():
    #tqdm
    for cam_id, clusters in tqdm(cluster_map.items()):
        video_path = aic[scene, 'camera_%04d' % cam_id]['video']
        cap = cv2.VideoCapture(video_path)
        for gid, bboxes in clusters.items():
            # print(f'gid: {gid}, bboxes: {bboxes}') # (bboxes)
            bboxes = sorted(bboxes, key=lambda x: x[0])
            for frame_id, x, y, w, h, conf in bboxes:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                # frame = cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
                # crop the bbox
                frame = frame[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
                cv2.imwrite('result/%s/%d/%d.png' % (scene, gid, global_counter), frame)
                global_counter += 1
                
        
if __name__ == '__main__':
    main()
            
            