import os

tracklet_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/UCMCTrack/trk_results'
emb_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/fast-reid/aic_results'
vid_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24/test'
det_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/yolov8/results'
anchor_bbox_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/clustering/results/bbox_results'


class AIC_PATH():

    def __init__(self):
        self.tracklet_path = tracklet_path
        self.emb_path = emb_path
        self.vid_path = vid_path
        self.det_path = det_path
        
    def __getitem__(self, index):
        try:
            scene, cam = index
            if isinstance(scene, int):
                scene = 'scene_%03d' % int(scene)
            if isinstance(cam, int):
                cam = 'camera_%04d' % int(cam)
            return {
                'tracklet': os.path.join(self.tracklet_path, scene, cam, 'trk_results.txt'),
                'embedding': os.path.join(self.emb_path, scene, cam+'.npy'),
                'detection': os.path.join(self.det_path, scene, cam+'.txt'),
                'video': os.path.join(self.vid_path, scene, cam, 'video.mp4'),
                'anchor_bbox': os.path.join(anchor_bbox_path, scene, cam+'.txt')
            }
        except:
            scene = index
            if isinstance(scene, int):
                scene = 'scene_%03d' % int(scene)
            return {
                'tracklet': os.path.join(self.tracklet_path, scene),
                'embedding': os.path.join(self.emb_path, scene),
                'detection': os.path.join(self.det_path, scene),
                'video': os.path.join(self.vid_path, scene),
                'anchor_bbox': os.path.join(anchor_bbox_path, scene)
            }
        
        
        