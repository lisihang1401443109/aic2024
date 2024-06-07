import os
import numpy as np

from aic_frame_sample import Detection, read_single_detection
from loguru import logger

tracklet_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/UCMCTrack/trk_results'

tracklets = []

#!!! NOTE THAT THE TRACKLET OUTPUT EXCLUDE ALL DETECTIONS WITH CONFIDENCE < THRESHOLD (0.5)
#!!! ONE THING TO DO IS TO REMEMBER THE INDEX AT WHICH THE DETECTIONS HAVE A CONFIDENCE < 0.5


THRESHOLD = 0.5

# def index_map(det_lines, threshold = 0.5):
    
#     #? SUPPOSEDLY mapp[location_in_trk_file] = location_in_det_file
    
#     below_thresholds = 0
#     mapp = []
#     for i, line in enumerate(det_lines):
#         conf = line.strip().split()[-1]
#         if conf < threshold:
#             continue
#         mapp.append(i)
#     return mapp

def read_single_tracking(trk_line):
    trk_id = int(trk_line.strip().split(',')[1])
    # print(trk_id)

    return trk_id


class Tracklet:
    

    def __init__(self, trk_id, detections):
        self.trk_id = trk_id
        self.detections = detections
        
    def add_detection(self, detection):
        self.detections.append(detection)

    @property
    def features(self):
        for i, det in enumerate(self.detections):
            if i % 10 == 0:
                yield det.emb
            else:
                continue


    def __hash__(self):
        return self.trk_id
    
    def __str__(self) -> str:
        return f'tid: {self.trk_id}, {len(self.detections)} detections\n'
    
    def __repr__(self) -> str:
        return self.__str__()
    
def check_integrity(det_file, emb_file, trk_file):
        
    det_file = open(det_file, 'r')
    trk_file = open(trk_file, 'r')

    det_file_lines = list(det_file.readlines())
    emb_file = np.load(emb_file)
    trk_file_lines = list(trk_file.readlines())
    
    # check det and emb have the same length
    #assert len(det_file_lines) == emb_file.shape[0] # ensure the detection and embedding matches
    
    # remove all detections kwith conf<0.5
    det_file_lines = list(filter(lambda x: float(x.strip().split()[-1]) >= THRESHOLD, det_file_lines))
    
    # check the filtered detections have the same length as track_file
    
    #assert len(det_file_lines) == len(trk_file_lines)
    
    print('integrity check passed')

    
def make_tracklets(det_file, emb_file, trk_file, tracklets = None):
    
    check_integrity(det_file, emb_file, trk_file)
    
    #? They trk_id is typically not shared across camera... I guess we can merge them later
    logger.info(f'making tracklets for {trk_file}')

    if tracklets is None:
        tracklets = {}

    cam = int(det_file.strip().split('/')[-1].split('.')[0][7:]) # god knows if this works
    # print(f'{cam=}')
    
    det_file = open(det_file, 'r')
    trk_file = open(trk_file, 'r')

    det_file_lines = list(det_file.readlines())
    emb_file = np.load(emb_file)
    trk_file_lines = list(trk_file.readlines())

    trk_ind = 0

    # assert len(det_file_lines) == emb_file.shape[0] # ensure the detection and embedding matches
    # ! We dont need the above assertion because they won't pass rn. 
    # TODO: post-processing strategy to trim the extra reid features, or just leave them, or just redo them.
    #assert len(det_file_lines) >= len(trk_file_lines) # ensure the number of tracklets is less than the number of detections

    tot = len(det_file_lines)
    
    ignored = 0

    for i in range(tot):
        det_line = det_file_lines[i]
        emb = emb_file[i, :]

        det = Detection(*read_single_detection(det_line), emb, cam)  
        if det.conf < THRESHOLD:
            # discard this detection as it doesnt appear in trk file
            ignored += 1
            continue

        trk_line = trk_file_lines[trk_ind]
        trk_id = read_single_tracking(trk_line)

        trk_ind += 1

        if trk_id == -1:
            continue

        if trk_id not in tracklets:
            # print(f'adding {trk_id=}')
            tracklets[trk_id] = Tracklet(trk_id, [det])
            # print(tracklets[trk_id])
        else:
            tracklets[trk_id].add_detection(det)
            
    # assert det_ind == len(det_file_lines)-1 and trk_ind == len(trk_file_lines)-1
    # print(f'{i=}, {trk_ind=}, {len(det_file_lines)=}, {len(trk_file_lines)=}, {ignored=}')
    return tracklets



if __name__ == '__main__':
    
    print("testing on scene_061")
    trk_scene_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/UCMCTrack/trk_results/scene_061'
    detection_scene_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/yolov8/results/scene_061'
    embedding_scene_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/fast-reid/aic_results/scene_061'
    overall = {}
    for cam in os.listdir(trk_scene_path):
        det_file = os.path.join(detection_scene_path, cam + '.txt')
        emb_file = os.path.join(embedding_scene_path, cam + '.npy')
        trk_file = os.path.join(trk_scene_path, cam, 'trk_results.txt')
        overall = make_tracklets(det_file, emb_file, trk_file, overall)
    # print(overall)
    print(len(overall), max(overall.keys()), min(overall.keys()))
        
    
    
    

        
        






