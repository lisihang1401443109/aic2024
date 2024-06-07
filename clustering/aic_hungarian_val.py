'''
adapted from https://github.com/lisihang1401443109/AIC23_Track1_UWIPL_ETRI/blob/8e23a69de57d2547b774399ae45792f83a2d17ef/BoT-SORT/tools/aic_hungarian_cluster.py

2023 aicity challenge

Using auto anchor generation.

input:single camera tracking results, tracklets from BoT-SORT and embedding
output:produce MCMT result at hungarian

'''
import pickle
import os
import collections
import argparse
import numpy as np
from loguru import logger
from collections import Counter
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment
import sys

from tqdm import tqdm
from aic_frame_sample import read_single_detection
from aic_load_tracklets import make_tracklets
from aic_path_manager import AIC_PATH
from aic_representative_frame import FindRepresentativeFrames

sys.path.append('.')
video_files_path = "/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24/val"
detection_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/yolov8/results'
embedding_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/fast-reid/aic_results'
root_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/clustering/results'

aic_path = AIC_PATH()

def make_parser():
    parser = argparse.ArgumentParser("clustering for synthetic data")
    # parser.add_argument("root_path", default="/WAVE/workarea/users/sli13/AIC23_MTPC/data", type=str)
    parser.add_argument("--scene", default=None, type=str)
    parser.add_argument("--anchor_only", action="store_true")
    return parser



# def collectDetections(scene):
#     detection_scene_path = os.path.join(detection_path,scene)
#     detection_cams = os.listdir(detection_scene_path)
#     detections = None

#     for indx,cam in enumerate(detection_cams):
#         dets = np.genfromtxt(os.path.join(detection_scene_path, cam), delimiter=' ', dtype=str)
#         num_cams = dets.shape(0)
#         cam_column = np.full((num_cams,1),indx+1)
#         dets = np.column_stack((cam_column, dets))
#         if not detections:
#             detections = dets
#         else:
#             detections = np.vstack((detections, dets))
#     return detections

# def collectEmbeddings(scene):
#     embedding_scene_path = os.path.join(embedding_path,scene)
#     embedding_cams = os.listdir(embedding_scene_path)
#     embeddings = None

#     for indx,cam in enumerate(embedding_cams):
#         embed = np.load(os.path.join(embedding_scene_path, cam), allow_pickle=True)
#         embed = np.array(embed)
#         num_cams = embed.shape(0)
#         cam_column = np.full((num_cams, 1), indx+1)
#         embed = np.column_stack((cam_column, embed))
#         if not embeddings:
#             embeddings = embed
#         else:
#             embeddings = np.vstack((embeddings, embed))
#     return embeddings

def collectDetctionsAndEmbeddings(scene, conf_threshold = 0.9, sample_threshold = 0.95):
    
    # ?don't really change the threshold... it needs to be the same as the one used in SCT
    # ?well, the SCT runs really fast, so just make sure to run it again
    
    # cams = list(map(lambda i: i[:-4],    [os.listdir(os.path.join(detection_path,scene))]))

    model = FindRepresentativeFrames(threshold=sample_threshold) 
    
    cams = os.listdir(os.path.join(video_files_path, scene))
    for i in cams:
        if(i=="ground_truth.txt"):
            cams.remove("ground_truth.txt")
    result_embedding = np.array([])
    result_detection = np.array([])
    print(cams)
    # for cam in tqdm(cams, desc=f'{scene}'):
    for cam in cams:
        emb_file = os.path.join(embedding_path, scene, cam+'.npy')
        det_file = os.path.join(detection_path, scene, cam+'.txt')
        
        cam_id = int(cam.split('_')[-1])
        
        cam_emb = []
        cam_det = []

        try:        
            embs = np.load(emb_file)
            dets = list(open(det_file, 'r').readlines())
        except Exception as e:
            print(e)
            continue
        
        #assert len(embs) <= len(dets)
        
        # for i in range(len(dets)): #? So right now the detection and embedding do not align perfectly, we might run reid again to fix that
        # for i in range(min(len(embs), len(dets))):
        for i in tqdm(range(len(dets)), desc=f'{scene} {cam}'):
            frame, x, y, w, h, conf = read_single_detection(dets[i])
            if conf < conf_threshold:
                continue
            cam_emb, flg = model.filter_embs(embs[i], cam_emb)
            if(flg):
                cam_det.append(np.array([cam_id, frame, x, y, w, h, conf]))
            # cam_emb.append(embs[i])
            # cam_det.append(np.array([cam_id, frame, x, y, w, h, conf]))
            
        cam_emb = np.stack(cam_emb)
        cam_det = np.stack(cam_det)
        if result_detection.shape[0] == 0:
            result_detection = cam_det
            result_embedding = cam_emb
        else:
            result_detection = np.vstack((result_detection, cam_det))
            result_embedding = np.vstack((result_embedding, cam_emb))
        
    print(result_embedding.shape, result_detection.shape)
    return result_detection, result_embedding
    

def get_people(distance_thresh, all_embs):   # *Eric: making modification to get_people() to make it work on our code
    # scenes = ['S003','S009','S014','S018','S021','S022']
    # assert scene in scenes
    # nms_thres = 1 #TODO 
    # threshold = 0.92 #TODO
    # distance_thers = [13,19.5,16,13,16,16] #TODO Find distance threshold for scene
    # # seq_idx = scenes.index(scene)
    # detections = collectDetections(scene)
    # embeddings = collectEmbeddings(scene)

    # model = FindRepresentativeFrames();
    # all_emb, all_detection = model.find_representative_frames(detections, embeddings, nms_thres, threshold)
    

    clustering = AgglomerativeClustering(distance_threshold=distance_thresh,n_clusters=None, metric='cosine', linkage='complete').fit(all_embs)
    
    return max(clustering.labels_)+1

def get_anchor(p, all_embs, all_dets):
    
    '''
    
    input: scene
    output: dictionary (keys: anchor's global id, values: a list of embeddings for that anchor)
    
    '''
    
    # if dataset == 'test':

    #     scenes = ['S003','S009','S014','S018','S021','S022']
        
    # else:
    #     raise ValueError('{} not supported dataset!'.format(dataset))
    

    # if scene in scenes:
    #     seq_idx = scenes.index(scene)
    # else:
    #     raise ValueError('scene not in {} set!'.format(dataset))

    # scene = scenes[seq_idx]
    # k = get_people(scene,dataset,threshold)

    # detections = np.genfromtxt(root_path+'/test_det/{}.txt'.format(scene), delimiter=',', dtype=str)
    # embeddings = np.load(root_path+'/test_emb/{}.npy'.format(scene),allow_pickle = True)

    # '''
    # nms
    # '''
    # embeddings = embeddings.tolist()
    # embeddings = np.array(embeddings)

    # all_dets = None
    # all_embs = None

    # for frame in threshold[seq_idx][0]:

    #     inds = detections[:,1] == str(frame-1)
    #     frame_detections = detections[inds]
    #     frame_embeddings = embeddings[inds]

    #     cams = np.unique(detections[:,0])

    #     for cam in cams:
    #         inds = frame_detections[:,0]==cam
    #         cam_det = frame_detections[inds][:,1:].astype("float")
    #         cam_embedding = frame_embeddings[inds]
    #         cam_det,pick = nms_fast(cam_det,None,nms_thres)
    #         cam_embedding = cam_embedding[pick]
    #         if len(cam_det) == 0:continue
    #         inds = cam_det[:,6]>threshold[seq_idx][1]
    #         cam_det = cam_det[inds]
    #         cam_embedding = cam_embedding[inds]
    #         if all_dets is None:
    #             all_dets = cam_det
    #             all_embs = cam_embedding
    #         else:
    #             all_dets = np.vstack((all_dets,cam_det))
    #             all_embs = np.vstack((all_embs,cam_embedding))

            #cam_record += [cam]*len(cam_det)

    clustering = AgglomerativeClustering(n_clusters=p, metric='cosine', linkage='complete').fit(all_embs)

    # print(clustering.labels_)

    anchors = collections.defaultdict(list)
    anchors_bbox = collections.defaultdict(list)

    for global_id in range(p):
        for n in range(len(all_embs)):
            if global_id == clustering.labels_[n]:
                anchors[global_id].append(all_embs[n])
                anchors_bbox[global_id].append(all_dets[n])
    
    return anchors, anchors_bbox

def get_box_dist(feat,anchors):
    '''
    input : feature, anchors                                                anc1  anc2  anc3 ...
    output : a list with distance between feature and anchor, e.g.    feat [dist1,dist2,dist3,...]
    '''
    box_dist = []

    for idx in anchors:
        dists = []
        for anchor in anchors[idx]:
            anchor /= np.linalg.norm(anchor)
            dists += [distance.cosine(feat,anchor)]
        
        #dist = min(dists) # or average..?
        # dist = sum(dists)/len(dists)
        dist = min(dists)
        
        box_dist.append(dist)    
            
    return box_dist

# filter the embeddings based on detetion confidence
# !not used 
def filter_emb(all_emb, all_det):
    # detection format: [cam_id, frame, x, y, w, h, conf]
    inds = all_det[:,6] >= 0.9
    logger.info('filtering %d boxes with confidence >= 0.9' % len(inds))
    return all_emb[inds], all_det[inds]
        

if __name__ == "__main__":
    
    args = make_parser().parse_args()
    if not args.scene:
        raise ValueError('scene not specified!')
    
    anchor_only = args.anchor_only
    
    # !!! Just for fast producing the anchors
    # anchor_only = True
    anchor_only = False
    
    scene = args.scene
    # root_path = args.root_path

    os.makedirs(os.path.join(root_path,'hungarian_cluster'),exist_ok=True)

    n = 15
    nms_thres = 1
    #nms_thres = 0.7
    #nms_thres = 0.3
    
    #dataset = 'validation'
    # dataset = 'test'
    # scenes = ['scene_061', 'scene_062', 'scene_063', 'scene_064', 'scene_065']
    # scene_path =  os.path.join(root_path,dataset)
    
    # demo_synthetic_only = True
    
    # if demo_synthetic_only:
    #     if dataset == 'test':
    #         scenes = ['S003','S009','S014','S018','S021','S022']
    #     else:
    #         scenes = ['S005','S008','S013','S017']

    # test
    # threshold = [([1,5000,10000,15000],0.9), #OK
    #          ([1,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000],0.92), #OK
    #          ([1,2500,5000,7500,10000,15000],0.96), #OK
    #          ([1,5000,10000,15000],0.9), #OK
    #          ([1,2500,5000,7500,10000,12500,15000],0.95), #OK
    #          ([1,2500,5000,7500,10000],0.9)] #OK
    
    # logger.info('clustering {} set'.format(dataset))
    # logger.info('scenes list: {}'.format(scenes))
    # logger.info('n = {}'.format(n))
    # logger.info('nms_thres = {}'.format(nms_thres))
    # logger.info('demo_synthetic_only = {}'.format(demo_synthetic_only))
    # logger.info('threshold = {}'.format(threshold))
    scenes = [scene]
    
    for scene in scenes:
        nms_thres = 1 #TODO 
        threshold = 0.95 #TODO
        distance_thresh = 0.06 #TODO Find distance threshold for scene
    # seq_idx = scenes.index(scene)
        # detections = collectDetections(scene)
        # embeddings = collectEmbeddings(scene)
        detections, embeddings = collectDetctionsAndEmbeddings(scene, conf_threshold=0.8, sample_threshold=threshold)
        logger.info(f'{embeddings.shape[0]} samples created')
        
        # TODO: fix the above to makesure detections and embeddings have trk id associated with it
        # TODO: we will exclude those with trk_id == -1 (unassigned detections), but we might make use of them later

        # model = FindRepresentativeFrames();
        # all_emb, all_detection = model.find_representative_frames(detections, embeddings, nms_thres, threshold)

        # people = get_people(distance_thresh, all_emb)
        # anchors = get_anchor(people, all_emb)
        
        #? before we do the clustering, we filter out the ones that are partial or have low confidence
        # filtered_emb, filtered_detections = filter_emb(embeddings, detections)
        filtered_emb, filtered_detections = embeddings, detections
        
        
        
        # people = get_people(distance_thresh=distance_thresh, all_embs=embeddings)
        # anchors, anchors_bbox = get_anchor(people, embeddings, detections)
        
        
        people = get_people(distance_thresh=distance_thresh, all_embs=filtered_emb)
        
        anchors, anchors_bbox = get_anchor(people, filtered_emb, filtered_detections)
        
        anchor_bbox_path = root_path+'/bbox_results/{}.txt'.format(scene)
        if not os.path.exists(root_path+'/bbox_results'):
            os.mkdir(root_path+'/bbox_results')
        
        logger.info('number of anchors {}'.format(len(anchors)))
        logger.info('number of people {}'.format(people))
        
        for anchor in anchors:
            logger.info('anchor {} : number of features {}'.format(anchor,len(anchors[anchor])))
        
        with open(anchor_bbox_path,'w') as f:
            for global_id, bbox in anchors_bbox.items():
                for cam_id, frame, x, y, w, h, conf in bbox:
                    f.write('{} {} {} {} {} {} {} {}\n'.format(global_id, cam_id, frame, x, y, w, h, conf))
                    
        if anchor_only:
            continue
        
        scene_path = aic_path[scene]['tracklet']
        cams = os.listdir(scene_path)
        cams = sorted([cam for cam in cams if not cam.endswith('png') and not cam=='dti'])
        
        for cam in cams:
       
            logger.info('processing scene {} cam {}'.format(scene,cam))
            
            # with open(root_path+'/tracklet/{}_{}.pkl'.format(scene,cam),'rb') as f:
            #     tracklets = pickle.load(f)
            path_to_scene = aic_path[scene, cam]
            tracklets = make_tracklets(path_to_scene['detection'], path_to_scene['embedding'], path_to_scene['tracklet'], tracklets = None)
    
            # TODO: So... tracklets = list[tracklet]
            # TODO: tracklet.features = iterable[feature]
            # TODO: I think that's what we need from the tracklet class...
            
    
            mapper = collections.defaultdict(list)
            global_id_mapper = collections.defaultdict(list)
            
            # for i,trk_id in tqdm(enumerate(tracklets), desc='{}:{} hungarian'.format(scene,cam), total=len(tracklets)):
            for i,trk_id in enumerate(tracklets):
                trk = tracklets[trk_id]
                for feat in trk.features:
                    box_dist = get_box_dist(feat,anchors)
                    mapper[trk_id].append(box_dist)
                mapper[trk_id].append(box_dist) # extra length to prevent bug -- len(features) < len(tracklets) 
            # sct = np.genfromtxt(root_path+'/SCT/{}_{}.txt'.format(scene,cam), delimiter=',', dtype=None)
            sct = aic_path[scene, cam]['tracklet']
            sct = np.genfromtxt(sct, delimiter=',')

            counter = collections.defaultdict(int)
            
            new_sct = root_path+'/hungarian_cluster/{}_{}.txt'.format(scene,cam)

            results = []
                
            cur_frame = -1
            cost_matrix = None

            for frame_id,trk_id,x,y,w,h,score,_,_,_ in tqdm(sct, desc='{}:{} assign'.format(scene,cam), total=len(sct)):
                if(trk_id == -1):
                    continue
                trk = tracklets[trk_id] # TODO Is the trk_id correct
                if(not trk.features): # too short trajectories, not sure if this can caused bug!
                    continue
                
                if frame_id != cur_frame and cost_matrix is None: # first frame
                    cost_matrix = []
                    frame_trk_ids = []
                    cur_frame = frame_id
                
                elif frame_id != cur_frame and cost_matrix is not None: # next frame => conduct hungarian, clear cost matrix
                    cost_matrix = np.array(cost_matrix)
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    for row,col in zip(row_ind,col_ind):
                        global_id_mapper[frame_trk_ids[row]].append(col)
                    cost_matrix = []
                    frame_trk_ids = []
                    cur_frame = frame_id
                # cost_matrix.append(mapper[trk_id][counter[trk_id]])   # row is cost for this feature to all anchors
                cost_matrix.append(mapper[trk_id][int(counter[trk_id]/10)])   # row is cost for this feature to all anchors
                frame_trk_ids.append(trk_id)

                counter[trk_id] += 1

            # for trk_id in global_id_mapper:
            #     print(trk_id,Counter(global_id_mapper[trk_id]).most_common())

            new_global_id_mapper = collections.defaultdict(list)

            for trk_id in global_id_mapper:
                ids = global_id_mapper[trk_id] # id list
                new_ids = []
                cur_ids = []
                for id in ids:
                    cur_ids.append(id)
                    if len(cur_ids) == n:
                        new_ids += [Counter(cur_ids).most_common(1)[0][0]]*n
                        cur_ids = []
                        
                if len(cur_ids)>0:
                    if len(new_ids)>0:
                        new_ids += [new_ids[-1]]*(len(cur_ids)+1)
                    else:
                        new_ids += [Counter(cur_ids).most_common(1)[0][0]]*(len(cur_ids)+1)
                new_global_id_mapper[trk_id] = new_ids
                new_global_id_mapper[trk_id] += [new_ids[-1]]*n # some extra length to prevent bug
                

            # Write Result

            conflict = 0
            counter2 = collections.defaultdict(int)

            for frame_id,trk_id,x,y,w,h,score,_,_,_ in sct:
                
                if len(new_global_id_mapper[trk_id]) == 0:
                    continue
                
                if counter2[trk_id]>=len(new_global_id_mapper[trk_id]):
                    global_id = new_global_id_mapper[trk_id][-1]+1
                else:
                    global_id = new_global_id_mapper[trk_id][counter2[trk_id]]+1

                if frame_id != cur_frame:
                    global_ids = set()
                    cur_frame = frame_id
                if global_id in global_ids:
                    conflict+=1
                else:
                    global_ids.add(global_id)
                
                results.append(
                        f"{frame_id},{global_id},{x},{y},{w},{h},{score},-1,-1,-1\n"
                    )
                
                counter2[trk_id] += 1
            
            logger.info('conflict:{}'.format(conflict))
            
            with open(new_sct, 'w') as f:
                f.writelines(results)