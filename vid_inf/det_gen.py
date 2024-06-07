import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

tracks_path = "/WAVE/workarea/users/sli13/AIC23_MTPC/stcra/results/reassignment"
vid_path = "/WAVE/workarea/users/sli13/AIC23_MTPC/data/aic24/test"

out_path = "/WAVE/workarea/users/sli13/AIC23_MTPC/vid_inf/outputs"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scene = args.scene

    # load the track path
    track_path = os.path.join(tracks_path, scene)
    cam_list = os.listdir(track_path)

    #load the vid path
    data_path = os.path.join(vid_path, scene)


    for cam in tqdm(cam_list, desc="generating inference video..."):
        cam = cam[:-4]
        outs = os.path.join(out_path, scene, "camera_"+cam)
        os.makedirs(outs, exist_ok=True)

        cap = cv2.VideoCapture(os.path.join(data_path, "camera_"+cam, "video.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(outs, "video.mp4"), fourcc, fps, (width, height))

        trk_path = os.path.join(track_path, cam+".txt")
        track_file = np.loadtxt(trk_path, delimiter = ',')

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            track_f = track_file[track_file[:, 0] == cap.get(cv2.CAP_PROP_POS_FRAMES)]

            for track in track_f:
                frame_id, track_id, x, y, w, h, _, _, fx, fy = track
                cv2.rectangle(frame, (int(x),int(y)), (int(x)+int(w), int(y)+int(h)), (0,255,0), 2)
                
                #cv2.putText(frame, str(int(track_id)), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

            out.write(frame)
        
        cap.release()
        out.release()
    #Put result video into out_path




