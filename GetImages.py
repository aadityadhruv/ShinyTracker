import numpy as np
import pandas as pd
import cv2 as cv
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import time
from pytube import YouTube
import os
import shutil
import math
import datetime
import random




videofile='werster_white_2_fast_2020.mp4'
video =cv2.VideoCapture(videofile)
videofile2='werster_white_2_cm_2020.mp4'
video2 =cv2.VideoCapture(videofile2)

frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))


for frame in random.sample(range(1000, frame_count - 1000)):

    is_good, pFrame = processFrame(video, frame)
    if is_good:
        saveFrame(directory, pFrame, frame,  suffix)

for frame in random.sample(range(1000, frame_count - 1000)):

    is_good, pFrame = processFrame(video2, frame)
    if is_good:
        saveFrame(directory, pFrame, frame, suffix)


renameDirectory(directory, suffix)



def processFrame(video, frame, thresh1=100, thresh2=600):
    video.set(cv2.CAP_PROP_POS_MSEC,frame) 
    hasFrames,image = vidcap.read()
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
        
        if (np.sum(dominant) > thresh1 && np.sum(dominant) < thresh2):
            is_good = True 
    
    return is_good, image

def saveFrame(directory="images/", frame, index,  suffix="f"):
    cv2.imwrite(directory + suffix + str(index) + ".png")



def renameDirectory(directory ,suffix="f"):
    for count, filename in enumerate(os.listdir(directory)):
        dst =suffix + str(count) + ".png"
        src =directory+ filename 
        dst =directory+ dst 

        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 



