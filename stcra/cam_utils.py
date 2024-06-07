import os
from path_utils import AIC_PATH
import json
import numpy as np


aic_path = AIC_PATH()

class CamInfo(object):
    
    def __init__(self, scene, cam):
        
        if isinstance(scene, int) or str.isdigit(scene):
            scene = 'scene_%03d' % int(scene)
        if isinstance(cam, int) or str.isdigit(cam):
            cam = 'camera_%04d' % int(cam)
        self.scene = scene
        self.cam = cam
        
        info_file = aic_path[scene,cam]['video']

        info_file = info_file.replace('video.mp4', 'calibration.json')
        self.info = json.load(open(info_file, 'r'))
        
        self.homo = np.matrix(self.info['homography matrix'])
        self.proj = np.matrix(self.info['camera projection matrix'])
        self.error = self.info['reprojection_error']
    
    def image2world(self, x, y):
        n2d = np.array([x, y, 1.0])
        
        # 3d coordinate
        n3d = np.linalg.inv(self.homo) @ n2d.T
        # convert n3d from matrix to array
        n3d = np.array(n3d).flatten()
        # return n3d[0] / n3d[2], n3d[1] / n3d[2]
        x3d = n3d[0] / n3d[2]
        y3d = n3d[1] / n3d[2]
        
        
        return x3d, y3d
    
    def world2image(self, x, y):
        n3d = np.array([x, y, 1.0])
        
        # 2d coordinate
        n2d = self.homo @ n3d
        return n2d[0] / n2d[2], n2d[1] / n2d[2]
    
            
    
    
        