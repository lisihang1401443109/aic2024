import os
from utils import WHERE_IS_CAM, path_from_scenecam, path_from_scene
import json


image_id: int = 0
bbox_id: int = 0



def extract_frames(cam_id):
    # Extract frames using ffmpeg
    cam_path = path_from_scenecam(cam_id)
    video_path = os.path.join(cam_path, 'video.mp4')
    frame_path = os.path.join(cam_path, 'frames')
    os.makedirs(frame_path, exist_ok=True)
    logger.info(f'Extracting frames from {video_path} to {frame_path}')
    ffmpeg.input(video_path).filter('fps', fps='30').output(frame_path + '/%05d.jpg', start_number=0, **{'qscale:v': 2}).overwrite_output().run(quiet=True)


def prepare_coco_format(cam_id):
    logger.info(f'Preparing coco format from {cam_id}')
    '''
        aic label format:
            frame_id, track_id, x, y, w, h, 1, -1, -1, -1
        coco_format:
            {
                'images'[
                    {
                        file_name: 'frame_id.jpg',
                        id: ,
                        frame_id: frame_id,
                        video_id: video_id,
                        height: 
                    }
                ]
            }
    '''
    
    global image_id, bbox_id
    
    cam_path = path_from_scenecam(cam_id)
    frame_path = os.path.join(cam_path, 'frames')
    out_path = os.path.join(cam_path, 'coco.json')
    
    coco_format = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'person'}]
    }
    anno_path = os.path.join(cam_path, 'label.txt')
    
    logger.info(f'Preparing {out_path} from {anno_path}')
    
    if 'test' in cam_path:
        return
    
    with open(anno_path, 'r') as f:
        last_frame_id = None
        for line in f:
            frame_id, track_id, x, y, w, h, _, _, _, _ = (int(i) for i in line.strip().split(','))
            if frame_id != last_frame_id:
                coco_format['images'].append({
                    'file_name': f'{frame_id:05}.jpg',
                    'id': (image_id := image_id + 1),
                    'width': 1920, # assumed 1920*1080
                    'height': 1080 # assumed 1920*1080
                })
            last_frame_id = frame_id
            
            coco_format['annotations'].append({
                'image_id': image_id,
                'id': (bbox_id := bbox_id + 1),
                'category_id': 1, # since there's only one category (person)
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0
            })
    
    with open(out_path, 'w') as f:
        json.dump(coco_format, f)
                
                
                
def prepare_labels_txt(scene):
    folder_path =  path_from_scene(scene)
    label_file = os.path.join(folder_path, 'ground_truth.txt')
    labels : dict[str, list[str]] = {}
    print(f'Preparing {label_file}')
    with open(label_file, 'r') as f:
        ## assumed format: cam_id, track_id, frame_id, x, y, w, h, global_x, global_y
        for line in f:
            cam_id, track_id, frame_id, x, y, w, h, gx, gy = line.strip().split(' ')
            if cam_id not in labels:
                labels[cam_id] = []
            labels[cam_id].append(f'{frame_id} {track_id} {x} {y} {w} {h} {gx} {gy}')
    for cam_id in labels:
        with open(os.path.join(os.path.join(folder_path, 'camera_%04d' % int(cam_id)), f'label.txt'), 'w') as f:
            f.write('\n'.join(labels[cam_id]))
    
    
    
    

      
def main():
    for usage, scenes in WHERE_IS_CAM.items():
        for scene, cams in scenes.items():
            prepare_labels_txt(scene)
    
    
    
if __name__ == '__main__':
    main()