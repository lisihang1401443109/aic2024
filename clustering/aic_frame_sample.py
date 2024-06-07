import os
import cv2
import numpy as np

def frame_sample(max_len = None):
    if max_len:
        rate = int(max_len / 100)
        return list(range(0, max_len, rate))

    else:
        raise NotImplemented('please provide max_len for random sampling')
        
        
def load_cam_embeddings(scene, cam, 
                        emb_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/fast-reid/aic_results',
                        det_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/yolov8/results',
                        vid_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24/test'):
    if isinstance(cam, int) or str.isdigit(cam):
        cam = 'camera_%04d' % int(cam)
        
    if isinstance(scene, int) or str.isdigit(scene):
        scene = 'scene_%03d' % int(scene)      
    
    embedding_path = os.path.join(emb_path, scene, cam+'.npy')
    detection_path = os.path.join(det_path, scene, cam+'.txt')
    video_path = os.path.join(vid_path, scene, cam, 'video.mp4')
    n_frames = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)
    
    
    
    frame_samples = frame_sample(n_frames)  # TODO: Get your dumbass code here
    frame_samples_set = set(frame_samples) # TODO: Fk you
    
    selected = []
    
    dets = []
    lines = open(detection_path, 'r').readlines()
    embeddings = np.load(embedding_path)
    
    for i, line in enumerate(lines):
        if i in frame_samples_set:
            frame, x, y, w, h, conf = read_single_detection(line)
            
            # TODO：ensure the embeddings are copied so `embeddings` gets released 
            dets.append(Detection(frame, x, y, w, h, conf, embeddings[i][::]), cam)
            
    del embeddings #! this might cause problem
            
    return dets
        
def read_single_detection(line):
    line_list = line.strip('\n').split(' ')
    frame = int(line_list[0])
    x = float(line_list[1])
    y = float(line_list[2])
    w = float(line_list[3])
    h = float(line_list[4])
    conf = float(line_list[-1])
    return frame, x, y, w, h, conf

   
class Detection(object):
    
    def __init__(self, frame, x, y, w, h, conf, emb, cam) -> None:
        self.frame = frame
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.conf = conf
        self.emb = emb
        self.cam = cam
        
    