import cv2
import numpy as np
import glob

frameSize = (640,512)

out = cv2.VideoWriter('/home/orcun/thermal_ext_board.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, frameSize)

for filename in sorted(glob.glob('/home/orcun/rosbag_extracted/*.jpg')):
    img = cv2.imread(filename)
    out.write(img)

out.release()
