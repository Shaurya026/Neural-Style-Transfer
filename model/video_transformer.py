from os.path import isfile, join
import cv2
import numpy as np
import glob
import os
pathIn=join(os.getcwd(),r'save images') 
pathOut = join(os.getcwd(),r'project.avi')
fps = 10.0

img_array = []
for filename in glob.glob(os.path.join(pathIn,r'*.png')):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'),fps, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()