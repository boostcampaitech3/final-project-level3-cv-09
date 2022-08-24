######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a video.
# It draws boxes, scores, and labels around the objects of interest in each
# frame of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import shutil
import glob
import os
import cv2
from cv2 import threshold
import numpy as np
import tensorflow as tf
import sys
from collections import deque

# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")
tf.compat.v1.disable_v2_behavior()
# Import utilites
from models.ViolenceDetectionAndLocalization.local_utils import label_map_util
# from local_utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# txtfile = open("hypotheses.txt", "w")
# Grab path to current working directory
CWD_PATH = "models/ViolenceDetectionAndLocalization"

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,'labelmap.pbtxt')



# Number of classes the object detector can identify
NUM_CLASSES = 400

boxfolderspattern='box*'
videofilespattern='labeledBOX*'

#----------------- Functions
def cleanfilesandfolders(boxfolderspattern,videofilespattern):
    # Get a list of all the file paths that ends with .txt from in specified directory
    fileList = glob.glob(boxfolderspattern)
    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            shutil.rmtree(filePath)
        except:
            print("Error while deleting folder : ", filePath)

        # Get a list of all the file paths that ends with .txt from in specified directory
        fileList = glob.glob(videofilespattern)
        # Iterate over the list of filepaths & remove each file.
        for filePath in fileList:
            try:
                os.remove(filePath)
            except:
                print("Error while deleting file : ", filePath)
#--------------------


# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=False)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def kinetics_violence_localization(threshold):
    off_path = 'data/npys/off.npy'
    original_image_path = 'data/images'
    image_path = 'data/blurred_images'
    video_name = os.listdir(image_path)[0]
    off_score = np.load(off_path)
    violent_scene = []

    for i in range(len(off_score)):
        if off_score[i] >= threshold:
            for j in range(16):
                violent_scene.append(16*i+j+1)

    if violent_scene:
        print("Start Localization Blur")
        for frame_num in violent_scene:
            # print("Frame %d" % frame_num)
            img_path = os.path.join(original_image_path, video_name, video_name+'-'+str(frame_num).zfill(6)+".jpg")
            save_path = os.path.join(image_path, video_name, video_name+'-'+str(frame_num).zfill(6)+".jpg")
            frame = cv2.imread(img_path)
            target_img = cv2.imread(save_path)
            if frame is not None:
                h,w,c = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_expanded = np.expand_dims(frame_rgb, axis=0)

                # Perform the actual detection by running the model with the image as input
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: frame_expanded})
                for box in boxes[0]:
                    if np.sum(box) == 0:
                        break
                    x1,y1,x2,y2 = box
                    x1 = int(x1*w)
                    y1 = int(y1*h)
                    x2 = int(x2*w)
                    y2 = int(y2*h)
                    ksize = 25
                    blur_img = target_img[y1:y2, x1:x2].copy()
                    target_img[y1:y2, x1:x2] = cv2.blur(blur_img, (ksize,ksize))
                    cv2.imwrite(save_path, target_img)
            else:
                # Clean up
                print("Stream Ended")
                # txtfile.close()
    print("Finish Localization Blur")
    cleanfilesandfolders(boxfolderspattern, videofilespattern)

# threshold = 0.8
# kinetics_violence_localization(threshold)