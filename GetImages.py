import numpy as np
import pandas as pd
import cv2 as cv
import cv2
import matplotlib.pyplot as plt
import time

import os
import shutil
import math
import datetime
import random
import sys


def processFrame(video, sec, thresh1=100, thresh2=500):
    video.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
    hasFrames,image = video.read()
    is_good = False
    if hasFrames:
        image = image[:-45, 375:]
        pixels = np.float32(image.reshape(-1, 3))
        n_colors = 1
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        
        dominant = palette[np.argmax(counts)]
        
        if (np.sum(dominant) > thresh1 and np.sum(dominant) < thresh2):
            is_good = True 
    
    return is_good, image

def saveFrame(frame, index, directory="images", suffix="f"):
    cv2.imwrite(directory + "/" + suffix + str(index) + ".png", frame)



def renameDirectory(directory ,suffix="f"):
    print(directory)
    for count, filename in enumerate(os.listdir(directory)):
        dst =suffix + str(count) + ".png"
        src =directory+ "/" + filename 
        dst =directory+ "/" + dst 

        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 



if len(sys.argv) < 3:
    print("Please enter correct params: ", "directory, ", "src_video1, ", "src_video2...")
    sys.exit()

samples = 1000


directory = sys.argv[1]


#videofile='werster_white_2_fast_2020.mp4'
# video =cv2.VideoCapture(videofile)
# videofile2='werster_white_2_cm_2020.mp4'
# video2 =cv2.VideoCapture(videofile2)
# frame_count2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
# fps2 = video2.get(cv2.CAP_PROP_FPS)
# duration2 = frame_count2/fps2



video_list = []

for i in range(2, len(sys.argv)):
    video = cv2.VideoCapture(sys.argv[i])
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = frame_count/fps
    video_list.append((video, duration))



i = 1
for video, duration in video_list:
    suffix = "video_" + str(i) + "_"
    i += 1
    for frame in random.sample(range(500, np.int(duration) - 1000), samples):
        is_good, pFrame = processFrame(video, frame)
        if is_good:
            saveFrame(pFrame, frame, directory, suffix)



renameDirectory(directory)






# suffix = "video1"
# for frame in random.sample(range(500, np.int(duration) - 1000), samples):
#     is_good, pFrame = processFrame(video, frame)
#     if is_good:
#         saveFrame(pFrame, frame, directory, suffix)
# suffix = "video2"
# for frame in random.sample(range(1000, np.int(duration2) - 1000), samples):

#     is_good, pFrame = processFrame(video2, frame)
#     if is_good:
#         saveFrame(pFrame, frame, directory, suffix)