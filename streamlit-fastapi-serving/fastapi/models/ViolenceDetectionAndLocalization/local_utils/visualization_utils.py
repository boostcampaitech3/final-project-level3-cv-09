# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import abc
import collections
import matplotlib.pyplot as plt
import glob
import numpy as np
import argparse
import imutils
import shutil
import sys







# Set headless-friendly backend.
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf
from pathlib import Path
import math
# from regions import BoundingBox

from .core import keypoint_ops
from .core import standard_fields as fields
from . import shape_utils
from collections import deque
THELABEL="person"
_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

threshold = 0
maximax =1
label = ""
labelslist=[]

Violenceboxeslist=[]


txtfile = open("hypotheses.txt", "w")

# Grab path to current working directory
CWD_PATH = os.getcwd()


class ViolenceBox:
  def __init__(self, BoxNumber, ymin,xmin,ymax,xmax):
    self.BoxNumber = BoxNumber
    self.ymin = ymin
    self.xmin = xmin
    self.ymax = ymax
    self.xmax = xmax

    # def area(box):
    #     """Area of this instance"""
    #     area = abs(box.ymax-box.ymin)*abs(box.xmax-box.xmin)
    #     if (area != 0):
    #         return area
    #     else:
    #         return .0001  # small value to compinsate frames were width and hight of some object is 0
    #
    # def intersect(box1,box2 ):
    #     """Create new Rect from intersection of self and o. Cave: id and dco will be lost."""
    #     newbox= None
    #     newbox.xmin = max(box1.xmin, box2.xmin)
    #     box["y"] = max(self.y_, o.y_)
    #     box["width"] = max(0, min(self.x_ + self.w_, o.x_ + o.w_) - box["x"])
    #     box["height"] = max(0, min(self.y_ + self.h_, o.y_ + o.h_) - box["y"])
    #     box["id"] = "intersect"
    #     return Rect(box)
    #
    # def overlap(self, o):
    #     """Overlap of this and other Rect o"""
    #     ia = self.intersect(o).area()
    #     union = self.area() + o.area() - ia
    #     return float(ia) / union


def _get_multiplier_for_color_randomness():
  """Returns a multiplier to get semi-random colors from successive indices.

  This function computes a prime number, p, in the range [2, 17] that:
  - is closest to len(STANDARD_COLORS) / 10
  - does not divide len(STANDARD_COLORS)

  If no prime numbers in that range satisfy the constraints, p is returned as 1.

  Once p is established, it can be used as a multiplier to select
  non-consecutive colors from STANDARD_COLORS:
  colors = [(p * i) % len(STANDARD_COLORS) for i in range(20)]
  """
  num_colors = len(STANDARD_COLORS)
  prime_candidates = [5, 7, 11, 13, 17]

  # Remove all prime candidates that divide the number of colors.
  prime_candidates = [p for p in prime_candidates if num_colors % p]
  if not prime_candidates:
    return 1

  # Return the closest prime number to num_colors / 10.
  abs_distance = [np.abs(num_colors / 10. - p) for p in prime_candidates]
  num_candidates = len(abs_distance)
  inds = [i for _, i in sorted(zip(abs_distance, range(num_candidates)))]
  return prime_candidates[inds[0]]


def save_image_array_as_png(image, output_path):
  """Saves an image (represented as a numpy array) to PNG.

  Args:
    image: a numpy array with shape [height, width, 3].
    output_path: path to which image should be written.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  with tf.gfile.Open(output_path, 'w') as fid:
    image_pil.save(fid, 'PNG')


def encode_image_array_as_png_str(image):
  """Encodes a numpy array into a PNG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    PNG encoded image string.
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string

#This Function is called for every box found in a frame >>> for box in box_color call this function
def draw_bounding_box_on_image_array(frameno,
                                     boxnumber,
                                     image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=1,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Args:
    frameno: The frame we are currently in , this is passed from the object detection video class to parent function to this ones
    boxnumber: A count what box we are currently working on
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  myimage=image # get a copy of the passed frame
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB') #convert to rgb

  #Draw the box on image , this is called from this function which is called for every box in a frame
  draw_bounding_box_on_image(frameno,boxnumber,myimage,image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(frameno,boxnumber,myimage,image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=1,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """

  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)########################################################HERE############################################################

#Currently i'm working on this certain box that  i recived from my calling function.
# i print the coordinates then continue
#     line=str(frameno-1)+','+THELABEL+','+str(int(boxnumber-1))+','+str(int(left))+','+str(int(right))+','+str(int(top))+','+str(int(bottom))+','+"0"+"\n"
#     txtfile.write(line)

    print("Upper left corner %f" % left)
    print("Lower Right corner %f" % right)
    print("Upper Right corner %f" % top)
    print("Lower Left corner %f" % bottom)
    (left, right, top, bottom) =(int(left),int(right),int(top),int(bottom))

    # i extracted the square which the box is at and saved it as an image in a folder named Box1,box2,boxn , where n is max boxes no
    ROI = myimage[top:bottom+300, left:right+300]
    # cv2.imshow("Roi%d" % boxnumber,ROI)
    path = Path(CWD_PATH)

    boxpath="box%d" % (boxnumber)
    PATH_TO_BOXES= os.path.join(path,boxpath)
    
    if not os.path.exists(PATH_TO_BOXES): #if directory doesnt exist create it
        os.makedirs(PATH_TO_BOXES)
        
    imgloc2= os.path.join(PATH_TO_BOXES,"%d.jpg" % (frameno) ) #where to save image , image name is the frame it was snipped from
  
    global threshold  #How many frames of certain box before i do detection using kinetics model
    global maximax    #How many boxes is there
    global label      #what the result of the detetion was
    global labelslist
    global Violenceboxeslist


    if (boxnumber >= maximax):
        maximax=boxnumber    #just use the global variable to store how many boxes is there , this works because when new calls for this function are made the counter will incrase

    if(boxnumber==1):  # Now we wait for 16 frame of first box then we do classification for every video in the folders
        threshold+=1

    if(threshold != 16):
        labelslist.append("")

    if (threshold == 16):  # we got 16 frame time to pass to model

        labelslist=[]
        for count3 in range (1,maximax+1):  #range+1 cause it iterates from 1 to range-1 #All the coming code is model detection of 16 frames using kinetics dataset , resnet
            img_array = []
            searchloc= os.path.join(path,"box%d/*.jpg" % (count3) )
            
            for filename in glob.glob(searchloc):
                img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width, height)
                w = 300
                h = 300
                dim = (w, h)
                resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                img_array.append(resized)
                os.remove(filename)
            out = cv2.VideoWriter(r"labeledBOX%d.avi" % (count3), cv2.VideoWriter_fourcc(*'DIVX'), 16, dim)
            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()
            
            pathtoclss=os.path.join(path,"action_recognition_kinetics.txt")
            CLASSES = open(pathtoclss).read().strip().split("\n")
        
            SAMPLE_DURATION = 16
            SAMPLE_SIZE = 112

            frames = deque(maxlen=SAMPLE_DURATION)
            # load the human activity recognition model
            print("[INFO] loading human activity recognition model...")
            onnxmodelpath=os.path.join(path,"resnet-34_kinetics.onnx")
            net = cv2.dnn.readNet(onnxmodelpath)
            # grab a pointer to the input video stream
            print("[INFO] accessing video stream...")
            vs = cv2.VideoCapture("labeledBOX%d.avi" % count3)

            frames=[]
            while(True):
                (grabbed, frame) = vs.read()
                if not grabbed:
                    print("[INFO] no frame read from stream - exiting")
                    break;
                else:
                    frames.append(frame)

            s = len(frames)
            if (s < 16):
                labelslist.append("")
                threshold = 0
                continue

            blob = cv2.dnn.blobFromImages(frames, 1.0,(SAMPLE_SIZE, SAMPLE_SIZE),
            (114.7748, 107.7354, 99.4750),swapRB=True, crop=True)
            blob = np.transpose(blob, (1, 0, 2, 3))
            blob = np.expand_dims(blob, axis=0)
            net.setInput(blob)
            outputs = net.forward()
            label = CLASSES[np.argmax(outputs)]
            print(label) #------------>>>>
            labelslist.append(label)
            threshold=0

    #cv2.imshow("ROI%d" % boxnumber, ROI)
    cv2.imwrite(imgloc2, ROI)

  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  if thickness > 0:
    if (labelslist[boxnumber - 1] == "Violence"):

        if (len(Violenceboxeslist) >= 2):

            for i in range(0,len(Violenceboxeslist)):
                for j in range(i+1,len(Violenceboxeslist)):
                    # Centroid Calculations
                    #box1
                    cX = int((Violenceboxeslist[i].xmin + Violenceboxeslist[i].xmax) / 2.0)
                    cY = int((Violenceboxeslist[i].ymin + Violenceboxeslist[i].ymax) / 2.0)
                    #box2
                    cX1 = int((Violenceboxeslist[j].xmin + Violenceboxeslist[j].xmax) / 2.0)
                    cY1 = int((Violenceboxeslist[j].ymin + Violenceboxeslist[j].ymax) / 2.0)
                    # Caclulates Euclidean distance between centroids

                    #eDistance = math.dist([cX, cY], [cX1, cY1]) # Needs python 3.8+
                    eDistance = math.sqrt(((cX - cX1) ** 2) + ((cY - cY1) ** 2))

                    #print("Eucledean Distance is :"+str(eDistance))

                    if (eDistance < 90):  #90 is the threshold of which merging between boxes will happen
                        #print("Intersection")
                        (left, right, top, bottom) = (min(Violenceboxeslist[i].xmin ,Violenceboxeslist[j].xmin), max(Violenceboxeslist[i].xmax ,Violenceboxeslist[j].xmax),
                                                      min(Violenceboxeslist[i].ymin ,Violenceboxeslist[j].ymin), max(Violenceboxeslist[i].ymax,Violenceboxeslist[j].ymax))
        if boxnumber in Violenceboxeslist: # if box already exist then update it
            for n in range(0,len(Violenceboxeslist)):
                if Violenceboxeslist[n].boxnumber == boxnumber:
                    (Violenceboxeslist[n].ymin,Violenceboxeslist[n].xmin,Violenceboxeslist[n].ymax,Violenceboxeslist[n].xmax)=(ymin * im_height, xmin * im_width, ymax * im_height, xmax * im_width)
        else: #if not exist check if its having a new id but same box then if not append
            flag=0
            for n in range(0, len(Violenceboxeslist)):
                if abs(Violenceboxeslist[n].ymin - ymin*im_height) < 5 and abs(Violenceboxeslist[n].xmin - xmin*im_width) < 5 :
                    flag=1
            if not flag:
                    Violenceboxeslist.append(ViolenceBox(boxnumber,ymin*im_height,xmin*im_width,ymax*im_height,xmax*im_width))
            else:
                print("Box have changed its ID")

        line = str(frameno - 1) + ',' + "Violence" + ',' + str(int(boxnumber - 1)) + ',' + str(int(left)) + ',' + str(int(right)) + ',' + str(int(top)) + ',' + str(int(bottom)) + ',' + "0" + "\n"
        txtfile.write(line) #----------------> Write Violence frames here
        color = 'blue'
    # else:
    #     line = str(frameno - 1) + ',' + "person" + ',' + str(int(boxnumber - 1)) + ',' + str(int(left)) + ',' + str(
    #         int(right)) + ',' + str(int(top)) + ',' + str(int(bottom)) + ',' + "0" + "\n"
    #     txtfile.write(line)  # ----------------> Write Violence frames here   #if we need the other Boxes or lables than thsan Violence we enable this one
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                   (left, top)],
                  width=thickness,
                  fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    # print(display_str)
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,text_bottom)],fill=color)
    # if (len(labelslist) < 1):
    #     labelslist.append("")
    # print(boxnumber - 1)
    if (labelslist[boxnumber - 1] == "Violence"):
        fill = 'blue'
    else:
        fill = 'black'

    draw.text(
        (left + margin, text_bottom - text_height - margin),
        labelslist[boxnumber - 1],
        #str(boxnumber-1),   #----------------------------> changing label displayed on box
        # display_str+"<---->"+labelslist[boxnumber-1],
        fill,
        font=font)
    text_bottom -= text_height - 2 * margin

def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=(), myimage=None):
  """Draws bounding boxes on image (numpy array).

  Args:
    image: a numpy array object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  np.copyto(myimage,image)
  image_pil = Image.fromarray(image)
  draw_bounding_boxes_on_image(myimage,image_pil, boxes, color, thickness,
                               display_str_list_list)
  np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(myimage,image,
                                 boxes,
                                 color='red',
                                 thickness=1,
                                 display_str_list_list=()):
  """Draws bounding boxes on image.

  Args:
    image: a PIL.Image object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  boxes_shape = boxes.shape
  if not boxes_shape:
    return
  if len(boxes_shape) != 2 or boxes_shape[1] != 4:
    raise ValueError('Input must be of size [N, 4]')
  for i in range(boxes_shape[0]):
    display_str_list = ()
    if display_str_list_list:
      display_str_list = display_str_list_list[i]
    draw_bounding_box_on_image(0,0,myimage,image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                               boxes[i, 3], color, thickness, display_str_list)


def _resize_original_image(image, image_shape):
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_images(
      image,
      image_shape,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      align_corners=True)
  return tf.cast(tf.squeeze(image, 0), tf.uint8)


def visualize_boxes_and_labels_on_image_array(
    frameno,
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    keypoint_scores=None,
    keypoint_edges=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a uint8 numpy array of shape [N, image_height, image_width],
      can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None.
    keypoint_scores: a numpy array of shape [N, num_keypoints], can be None.
    keypoint_edges: A list of tuples with keypoint indices that specify which
      keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
      edges from keypoint 0 to 1 and from keypoint 2 to 4.
    track_ids: a numpy array of shape [N] with unique track ids. If provided,
      color-coding of boxes will be determined by these ids, and not the class
      indices.
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box or keypoint to be
      visualized.
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_boxes: whether to skip the drawing of bounding boxes.
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection
    skip_track_ids: whether to skip track id when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  box_to_keypoint_scores_map = collections.defaultdict(list)
  box_to_track_ids_map = {}
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(boxes.shape[0]):
    if max_boxes_to_draw == len(box_to_color_map):
      break
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if keypoint_scores is not None:
        box_to_keypoint_scores_map[box].extend(keypoint_scores[i])
      if track_ids is not None:
        box_to_track_ids_map[box] = track_ids[i]
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in six.viewkeys(category_index):
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
            if (display_str != "person"):
                continue
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(round(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, round(100*scores[i]))
        if not skip_track_ids and track_ids is not None:
          if not display_str:
            display_str = 'ID {}'.format(track_ids[i])
          else:
            display_str = '{}: ID {}'.format(display_str, track_ids[i])
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        elif track_ids is not None:
          prime_multipler = _get_multiplier_for_color_randomness()
          box_to_color_map[box] = STANDARD_COLORS[
              (prime_multipler * track_ids[i]) % len(STANDARD_COLORS)]
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]
  boxnumber=0
  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():############################################################################################################################33
    ymin, xmin, ymax, xmax = box
    boxnumber+=1
    print("Box number %d :: " % boxnumber)

    draw_bounding_box_on_image_array(
        frameno,
        boxnumber,
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=0 if skip_boxes else line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    

  return image
