import numpy as np
import os
import json


def getUVError(box):
    u = 0.05*box[3]
    v = 0.05*box[3]
    if u>13:
        u = 13
    elif u<2:
        u = 2
    if v>10:
        v = 10
    elif v<2:
        v = 2
    return u,v
    


def readAICCalib(filename):
    para = json.load(open(filename, 'r'))
    mat = np.array(para['homography matrix'])
    return mat, True
