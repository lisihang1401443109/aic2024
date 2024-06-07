from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def yolobbox2bbox(x,y,w,h):
    x1,y1 = x-w/2, y-h/2
    x2,y2 = x+w/2, y+h/2
    return x1,y1,x2,y2


def nms_fast(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]
    w = boxes[:, 4]
    h = boxes[:, 5]

    x1,y1,x2,y2 = yolobbox2bbox(x1,y1,w,h)

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked
    return boxes[pick].astype("float"), pick

# def filter_embs(emb, det, all_emb, all_det):
#     threshold = 0.9
#     if(all_emb is None):
#         all_emb = emb
#         all_det = det
#     else:
#         for i in emb:
#             flag = False
#             for j in all_emb:
#                 emb_matrix = np.vstack((i,j))
#                 similarity_matrix = cosine_similarity(emb_matrix, emb_matrix)
#                 similarity = similarity_matrix[0][1]
#                 if(similarity >= threshold):          #Check if emb is similar enough to any emb in the matrix
#                     flag = True
#                     break
#             if(not flag):
#                 all_emb = np.vstack((all_emb, emb))
#                 all_det = np.vstack((all_det, det))
    
#     return all_emb, all_det

class FindRepresentativeFrames:
    
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        pass
        
        

    def filter_embs(self, emb, all_emb):
        change = False

        if(len(all_emb) == 0):
            all_emb.append(emb)
            change = True
        else:
            flag = False
            for j in all_emb:
                # emb_matrix = np.vstack((emb,j)) # 2*2046
                # similarity_matrix = cosine_similarity(emb_matrix, emb_matrix) # 2*2
                # similarity = similarity_matrix[0][1]
                similarity = cosine_similarity(emb.reshape(1, -1), j.reshape(1, -1))[0][0] # cosine_similarity(emb, j)
                if(similarity >= self.threshold):          #Check if emb is similar enough to any emb in the matrix
                    flag = True
                    break
            if(not flag):
                change = True
                all_emb.append(emb)
        return all_emb, change

    # def find_representative_frames(self, detections, embeddings, nms_thres=1, threshold):                      #Locate frames where embeddings variance > some threshold

    #     cams = np.unique(detections[:,0])    

    #     all_detection = None
    #     all_emb = None

    #     for cam in cams:
    #         # inds = detections[:,0] == cam
    #         cam_all_emb = None
    #         cam_all_det = None
    #         tmp = detections[detections[:,0] == cam]
    #         frames = max(np.unique(tmp[:, 1]))

    #         for frame in range(1, frames, 250):
    #             frame_inds = detections[:,1] == str(frame-1)
    #             frame_detections = detections[frame_inds]
    #             frame_embeddings = embeddings[frame_inds]
    #             inds = frame_detections[:,0] == cam
    #             cam_det = frame_detections[inds][:,1:].astype("float")
    #             cam_embedding = frame_embeddings[inds]
    #             cam_det,pick = nms_fast(cam_det,None,nms_thres)
    #             cam_embedding = cam_embedding[pick]
    #             # if len(cam_det) == 0:continue
    #             # inds = cam_det[:,6]>threshold #!filter out detections that are below the threshold and keeping the rest
    #             # cam_det = cam_det[inds]
    #             # cam_embedding = cam_embedding[inds]

    #             cam_all_emb, cam_all_det = filter_embs(cam_embedding, cam_det, cam_all_emb, cam_all_det)

    #         if(all_emb is None):
    #             all_emb = cam_all_emb
    #             all_detection = cam_all_det
    #         else:
    #             all_emb = np.vstack((all_emb, cam_all_emb))
    #             all_detection = np.vstack((all_detection, cam_all_det))
        
    #     return all_emb, all_detection