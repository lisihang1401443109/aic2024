from detector.detector import Detector, Detection
from detector.mapper import Mapper
from detector.gmc import GMCLoader

import numpy as np
class AIC_Detector(Detector):
    
    # overwriting method
    def load(self, cam_para_file, det_file, gmc_file = None):
        self.mapper = Mapper(cam_para_file)
        self.load_detfile(det_file)

        print(det_file)        
        # for frame_id, dets in self.dets.items():
        #     print(frame_id, len(dets))


        if gmc_file is not None:
            self.gmc = GMCLoader(gmc_file)
            
    def load_detfile(self, filename):
        self.dets = dict()
        # 打开文本文件filename
        with open(filename, 'r') as f:
            curr_frame = 0
            counter = 0
            # 读取文件中的每一行
            for line in f.readlines():
                # 将每一行的内容按照空格分开
                line = line.strip().split(' ')
                frame_id = int(line[0])
                if frame_id > self.seq_length:
                    self.seq_length = frame_id
                if frame_id != curr_frame:
                    curr_frame = frame_id
                    counter = 0
                else:
                    counter += 1
                # det_id = int(line[1])
                det_id = counter
                # 新建一个Detection对象
                # det = Detection(det_id)
                # det.bb_left = float(line[1])
                # det.bb_top = float(line[2])
                # det.bb_width = float(line[3])
                # det.bb_height = float(line[4])
                
                #? note that the xy in the input file is the center of the bbox
                x, y, w, h = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                conf = float(line[-1])
                left = x - w/2
                top = y - h/2
                det = Detection(det_id, left, top, w, h, conf, 0)
                

                det.y,det.R = self.mapper.mapto([det.bb_left,det.bb_top,det.bb_width,det.bb_height])
                

                # 将det添加到字典中
                if frame_id not in self.dets:
                    self.dets[frame_id] = []
                self.dets[frame_id].append(det)