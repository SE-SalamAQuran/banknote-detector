import os
import cv2 as cv
import numpy as np
import time
from Preprocessor import DataAugmentor
from SlidingWindow import Slider

def augmentData(directory = 'Dataset/'):

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(directory, filename)
            augmentor = DataAugmentor(path)

            # cv.imwrite(os.path.join('Augmented/jittered/' , filename)  , augmentor.colorjitter(path))
            # cv.imwrite(os.path.join('Augmented/cropped/' , filename), augmentor.randomcrop(path))
            # cv.imwrite(os.path.join('Augmented/noise/' , filename), augmentor.noisy(path))
            # cv.imwrite(os.path.join('Augmented/filtered/' , filename), augmentor.filters(path))
        else:
            continue


(winW, winH) = (500, 220)

def object_detection_train(path):
    slider = Slider(path)
    #This is an object detection function using window-sliding
    for resized in slider.pyramid(path, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in slider.sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW
            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cv.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv.imshow("Window", clone)
            cv.waitKey(1)
            time.sleep(0.025)

#Bad Choice, use YOLO Object Detection instead
object_detection_train('Dataset/image.jpg')