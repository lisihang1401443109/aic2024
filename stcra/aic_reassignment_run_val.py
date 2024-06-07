import os

from validation_cam_utils import CamInfo
from file_utils import xywh2location
from path_utils import AIC_PATH
from loguru import logger
import argparse


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str)
    return parser.parse_args()

def main():
    aic = AIC_PATH()
    arg = make_args()
    try:
        scene = arg.scene
    except:
        raise ValueError('Please provide scene number as --scene=xxx')

    if scene is None:
        raise ValueError('Please provide scene number as --scene=xxx')

    scene_result_path = aic[scene]['result']
    scene_reassign_path = aic[scene]['cluster']
    # scene_reassign_path = 
    logger.info(f'Processing scene {scene} ...')
    logger.info(f'Result path: {scene_result_path}')
    logger.info(f'Reassign path: {scene_reassign_path}')
    # os.mkdir(scene_result_path, exist_ok=True)
    
    os.makedirs(scene_result_path, exist_ok=True)
    output_file = open(os.path.join(scene_result_path, f'{scene}.txt'), 'w')
    cameras = os.listdir(aic[scene]['reassign'])
    
    print(f'{cameras=}')
    # quit()

    for camera in cameras:
        logger.info(f'Processing camera {camera} ...')
        camera_number = camera.split('.')[0]
        # cam_file = os.path.join(scene_reassign_path, camera)
        cam_file = aic[scene, 'camera_' + camera_number]['cluster']
        logger.info(f'{cam_file} loaded')
        cam_info = CamInfo(scene, camera_number)
        with open(cam_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                values = line.split(',')
                frame = values[0]
                global_id = values[1]
                x = float(values[2])
                y = float(values[3])
                w = float(values[4])
                h = float(values[5])

                x_loc, y_loc = xywh2location(x, y, w, h)
                x_world, y_world = cam_info.image2world(x_loc, y_loc)
                i_x = int(x)
                i_y = int(y)
                i_w = int(w)
                i_h = int(h)
                output_file.write(f'{camera_number} {global_id} {frame} {i_x} {i_y} {i_w} {i_h} {x_world} {y_world}\n')

    logger.info(f'done processing {scene}')

if __name__ == '__main__':
    main()