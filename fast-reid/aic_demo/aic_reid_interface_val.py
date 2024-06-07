'''
    Adapted from BOT-SORT implementation

'''


import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import sys
import os
# from torch.backends import cudnn

sys.path.append('..')

from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch


from loguru import logger
from tqdm import tqdm
# cudnn.benchmark = True


FRAME_SAMPLE_RATE = 200

def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False

    cfg.freeze()

    return cfg


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data
    return features


def preprocess(image, input_size):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[1], input_size[0], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size) * 114
    img = np.array(image)
    r = min(input_size[1] / img.shape[0], input_size[0] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    return padded_img, r


# class FastReIDInterface:
#     def __init__(self, config_file, weights_path, device, batch_size=8):
#         super(FastReIDInterface, self).__init__()
#         if device != 'cpu':
#             self.device = 'cuda'
#         else:
#             self.device = 'cpu'

#         self.batch_size = batch_size

#         self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])

#         self.model = build_model(self.cfg)
#         self.model.eval()

#         Checkpointer(self.model).load(weights_path)

#         if self.device != 'cpu':
#             self.model = self.model.eval().to(device='cuda').half()
#         else:
#             self.model = self.model.eval()

#         self.pH, self.pW = self.cfg.INPUT.SIZE_TEST

#     def inference(self, image, detections):

#         if detections is None or np.size(detections) == 0:
#             return []

#         H, W, _ = np.shape(image)

#         batch_patches = []
#         patches = []
#         for d in range(np.size(detections, 0)):
#             tlbr = detections[d, :4].astype(np.int_)
#             tlbr[0] = max(0, tlbr[0])
#             tlbr[1] = max(0, tlbr[1])
#             tlbr[2] = min(W - 1, tlbr[2])
#             tlbr[3] = min(H - 1, tlbr[3])
#             patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]

#             # the model expects RGB inputs
#             patch = patch[:, :, ::-1]

#             # Apply pre-processing to image.
#             patch = cv2.resize(patch, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_LINEAR)
#             # patch, scale = preprocess(patch, self.cfg.INPUT.SIZE_TEST[::-1])

#             # plt.figure()
#             # plt.imshow(patch)
#             # plt.show()

#             # Make shape with a new batch dimension which is adapted for network input
#             patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
#             patch = patch.to(device=self.device).half()

#             patches.append(patch)

#             if (d + 1) % self.batch_size == 0:
#                 patches = torch.stack(patches, dim=0)
#                 batch_patches.append(patches)
#                 patches = []

#         if len(patches):
#             patches = torch.stack(patches, dim=0)
#             batch_patches.append(patches)

#         features = np.zeros((0, 2048))
#         # features = np.zeros((0, 768))

#         for patches in batch_patches:

#             # Run model
#             patches_ = torch.clone(patches)
#             pred = self.model(patches)
#             pred[torch.isinf(pred)] = 1.0

#             feat = postprocess(pred)

#             nans = np.isnan(np.sum(feat, axis=1))
#             if np.isnan(feat).any():
#                 for n in range(np.size(nans)):
#                     if nans[n]:
#                         # patch_np = patches[n, ...].squeeze().transpose(1, 2, 0).cpu().numpy()
#                         patch_np = patches_[n, ...]
#                         patch_np_ = torch.unsqueeze(patch_np, 0)
#                         pred_ = self.model(patch_np_)

#                         patch_np = torch.squeeze(patch_np).cpu()
#                         patch_np = torch.permute(patch_np, (1, 2, 0)).int()
#                         patch_np = patch_np.numpy()

#                         plt.figure()
#                         plt.imshow(patch_np)
#                         plt.show()

#             features = np.vstack((features, feat))

#         return features


class aic_reid_interface():
    
    # overriding
    def __init__(self, config_file, weights_path, device, 
                 detection_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/yolov8/results', 
                 output_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/fast-reid/aic_results', batch_size=16) -> None:
        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])
        self.model = build_model(self.cfg)
        print(self.model.eval())
        
        self.batch_size = batch_size
        self.detection_path = detection_path
        self.output_path = output_path
        self.device = device
        
        Checkpointer(self.model).load(weights_path)
        
        if device != 'cuda':
            logger.warning('Running on CPU, please ensure that you do not have a CUDA device')
            
        self.model.to(device).half()
        
        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST
        
        
    
        
    @torch.no_grad()
    def __call__(self, scene, cam, verbose):
        
        if isinstance(scene, int):
            scene = 'scene_%03d' % scene
            
        if isinstance(cam, int):
            cam = 'camera_%04d' % cam
            
        # out_file = open(os.path.join(self.output_path, scene, cam + '.txt'), 'w')
        result = []
        logger.info('Running inference on scene %s, camera %s' % (scene, cam))
        
        detection_path = os.path.join(self.detection_path, scene, cam)
        output_dir = os.path.join(self.output_path, scene)
        
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cap = cv2.VideoCapture('/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24/val/%s/%s/video.mp4' % (scene, cam))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        

        feature_size = 2048 # TODO: figure this out
        
        detection_file = open(detection_path + '.txt', 'r')
        
        lines = list(detection_file.readlines())
        total=len(lines)
        logger.info('Total number of detections: %d' % total)
        
        curr_frame = 0
        patches = torch.zeros((self.batch_size, 3, self.pH, self.pW), dtype=torch.half).cuda()
        if verbose: 
            pbar = tqdm(enumerate(lines), desc=f'reid {scene}_{cam}', total = total)
        else:
            pbar = enumerate(lines)
        
        
        for i, line in pbar:
            line_spilt = line.split(' ')
            frame = int(line_spilt[0])
            xywh = [float(i) for i in line_spilt[1:5]]
            conf = float(line_spilt[-1])
            
            
            if frame != curr_frame:
                if curr_frame + 1 == frame:
                    curr_frame += 1
                    ret, image = cap.read()
                    if not ret:
                        logger.warning(f'failed to read frame {frame}, max frame is {n_frames}')
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame-1)
                    curr_frame = frame
                    ret, image = cap.read()
                    if not ret:
                        logger.warning(f'failed to read frame {frame}, max frame is {n_frames}')
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # print(f'frame: {frame}, curr_frame = {cap.get(cv2.CAP_PROP_POS_FRAMES)}')
            assert frame == cap.get(cv2.CAP_PROP_POS_FRAMES) == curr_frame
            
            x, y, w, h = xywh
            # check xywh for borders
            # note that xy resembles the center of the box
            
            # x = max(x, 0)
            # y = max(y, 0)
            # w = min(x+w, W-1) - x
            # h = min(y+h, H-1) - y
            
            x1 = max(x - w * 0.5, 0)
            y1 = max(y - h * 0.5, 0)
            x2 = min(x + w * 0.5, W-1)
            y2 = min(y + h * 0.5, H-1)
            
            # img = image[int(y):int(y+h), int(x):int(x+w), :]
            img = image[int(y1):int(y2), int(x1):int(x2), :]
            
            patch = cv2.resize(img, (self.pW, self.pH), interpolation=cv2.INTER_LINEAR)
            
            
            patch_id = i % self.batch_size
            if patch_id == 0 and i != 0:
                # with torch.no_grad():
                    # patches.to(self.device)
                features = self.model(patches)
                    
                result.append(postprocess(features))
                
                
                
                
            patches[patch_id, ...] = torch.tensor(patch.transpose(2, 0, 1))
            
            if i == total-1:
                # last patch
                
                patches.to(self.device)
                
                features = self.model(patches)
                features.cpu()
                
                if patch_id != self.batch_size - 1:
                    result.append(postprocess(features[:patch_id+1]))
                else:
                    result.append(postprocess(features))
        
        result_npy = torch.vstack(result).cpu().numpy()
        
        
        logger.warning('Done inference on scene %s, camera %s' % (scene, cam))
        logger.warning(f'result_shape {result_npy.shape}')
        
        out_file = self.output_path + '/' + scene + '/' + cam + '.npy'
        np.save(out_file, result_npy, allow_pickle=True)
            
        


# class DetectorReader(object):
    
#     def __init__(self, detector_path, n_frames):
#         self.detector_file = open(detector_path, 'r')
#         self.detections = {}
#         self.det_cnts = {}
        
#         self.n_frames = n_frames
        
#         self.counter = 0 # counts the number of detections overall
#         for line in self.detector_file:
#             frame = int(line.split()[0])
#             x, y, w, h = (float(_) for _ in line.split(' ')[1:5])
#             conf = float(line.split(' ')[-1])
#             if frame not in self.detections:
#                 self.detections[frame] = []
#             if frame not in self.det_cnts:
#                 self.det_cnts[frame] = 0
#             self.detections[frame].append([x, y, w, h, conf])
#             self.det_cnts[frame] += 1
#             self.counter += 1
            
#     @property
#     def mat(self):
#         i = 0
#         self.mat = np.zeros((self.counter, 5))
#         for frame in self.detections:
#             for d in enumerate(self.detections[frame]):
#                 self.mat[i, 0] = d[0]
#                 self.mat[i, 1] = d[1]
#                 self.mat[i, 2] = d[2]
#                 self.mat[i, 3] = d[3]
#                 self.mat[i, 4] = d[4]
                
#                 i += 1
        
            
#     def get_count(self, frame):
#         return self.det_cnts[frame]
    
#     def get_dets_from_frame(self, frame):
#         return self.detections[frame]
    
            
    
            
            
    