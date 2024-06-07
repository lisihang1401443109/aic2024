result_path = '/WAVE/workarea/users/sli13/AIC23_MTPC/yolov8/results'

from tqdm import tqdm

def fix_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = list(lines)
    curr_frame = 0
    for i, line in enumerate(lines):
        frame_number = int(line.split(' ')[0])
        if frame_number > curr_frame:
            curr_frame = frame_number
        else:
            lines = lines[:i]
            break
        
    # write back
    with open(file_path, 'w') as f:
        f.writelines(lines)
        
    
if __name__ == '__main__':
    import os
    for scene in tqdm(os.listdir(result_path)):
        for file in os.listdir(os.path.join(result_path, scene)):
            if file.endswith('.txt'):
                fix_file(os.path.join(result_path, scene, file))
                
