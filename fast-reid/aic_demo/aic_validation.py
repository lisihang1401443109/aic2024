from aic_reid_interface_val import aic_reid_interface
import argparse
import os
from loguru import logger

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', type=str, default='/WAVE/workarea/users/sli13/AIC23_MTPC/fast-reid/configs/aic_bagtricks.yml', help='config file path')
parser.add_argument('--scene', type=str, default='scene_061', help='scene')
# parser.add_argument('--cam', type=str, default='camera_0535', help='camera')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')


args = parser.parse_args()


reid = aic_reid_interface(
    config_file=args.config_file,
    weights_path='/WAVE/workarea/users/sli13/AIC23_MTPC/fast-reid/weights/duke_bot_S50.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    batch_size= args.batch_size
)

done_list = []
with open('/WAVE/workarea/users/sli13/AIC23_MTPC/fast-reid/aic_demo/done_list.log', 'r') as f:
    for line in f.readlines():
        done_list.append(line.strip('\n'))

for cam in sorted(list(os.listdir('/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24/val/%s' % args.scene))):
    if cam in done_list:
        logger.info(f'skipping {args.scene} {cam} because found in done list')
        continue
    logger.info(f'starting {args.scene} {cam}')
    reid(scene=args.scene, cam=cam, verbose=False)
    logger.info(f'done {args.scene} {cam}')
    with open('/WAVE/workarea/users/sli13/AIC23_MTPC/fast-reid/aic_demo/done_list.log', 'a+') as f:
        f.write('%s\n' % cam)