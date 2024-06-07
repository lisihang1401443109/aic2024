'''
current data folder format:

data
    aic24
        train
            scene_001
                camera_0001
                    video.mp4
                    calibration.txt
                    label.txt
                camera_0002
                    ......
            scene_002
                camera_0011
                    ......
            ......
        val
        ......
        
write a script to prepare the dataset for detector training
data
    aic24_for_trainint
        train
            camera_0001
                video.mp4
                label.txt
            camera_0003
                ......
        val
        ......
        
create another folder called aic24_for_training under the same directory
note that the videos are symbolic links

also instead of moving all the video, move one out of five of the videos (e.g. camera_0001, camera_0006 ......)
'''
import os
import shutil
from glob import glob

def prepare_dataset_for_training(src_folder, dest_folder, train_ratio=0.2):
    """
    Prepares the dataset for detector training by creating a new folder structure
    and moving a subset of videos.

    Args:
    - src_folder: the source folder containing the original dataset.
    - dest_folder: the destination folder where the prepared dataset will be stored.
    - train_ratio: the ratio of videos to be moved for each camera (default 0.2, i.e., one out of five).
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Walk through the source directory
    for root, dirs, files in os.walk(src_folder):
        for dir_name in dirs:
            # Check if the directory name starts with 'camera_'
            if dir_name.startswith('camera_'):
                # Get the numeric part of the camera name and check if it should be moved
                camera_num = int(dir_name.replace('camera_', ''))
                if camera_num % int(1/train_ratio) == 1:
                    # Create destination directory
                    dest_dir = os.path.join(dest_folder, dir_name)
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    # Copy the video file and label file using symbolic links
                    video_file = os.path.join(root, dir_name, 'video.mp4')
                    label_file = os.path.join(root, dir_name, 'label.txt')
                    if os.path.exists(video_file) and os.path.exists(label_file):
                        os.symlink(video_file, os.path.join(dest_dir, 'video.mp4'))
                        os.symlink(label_file, os.path.join(dest_dir, 'label.txt'))

if __name__ == '__main__':
    src_folder = '/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24'
    dest_folder = '/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24_for_training'
    prepare_dataset_for_training(src_folder, dest_folder)

