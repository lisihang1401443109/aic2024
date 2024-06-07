# from ultralytics import YOLO
import os
import argparse

import numpy as np

from util.aic_ucmc import run_ucmc, make_args

def run_scene(args, scene):
    base_directory = '/WAVE/workarea/users/sli13/AIC23_MTPC/yolov8/results'
    for camera in os.listdir(os.path.join(base_directory, scene)):
        cam = camera[:-4] # remove .txt
        run_ucmc(args, det_path=base_directory, scene=scene, camera=cam, cam_path='/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24/val', orig_save_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/UCMCTrack/trk_results')
        



def main(args):
    scene = args.scene
    
    if isinstance(scene, int):
        scene = 'scene_%03d' % scene
        
    run_scene(args, scene)
    
    



if __name__ == '__main__':
    args = make_args()
    main(args)





