import numpy as np
import pandas as pd
import cv2 as cv
import cv2
import os
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 3:
	print("Please enter valid params: ", "dst_folder, ", "src_folder1, ", "src_folder2...")
	sys.exit()

dst = sys.argv[1]


src_list = []

if len(sys.argv) > 2:
	for i in range (2, len(sys.argv)):
		src_list.append(sys.argv[i])



for directory in src_list:
	for count, filename in enumerate(os.listdir(directory)):
    	image = cv2.imread(directory + "/" + filename)
    
	    height, width = image.shape[:2]
	    res = cv.resize(image,(int(0.5*width), int(0.5*height)))
	    cv2.imwrite(dst_folder + "/" + filename, res)
