import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

# Optional if you are using a GPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        # draw_connections(frame, person, edges, confidence_threshold)
        # draw_keypoints(frame, person, confidence_threshold)
        make_blur(frame, person, confidence_threshold)
        

def make_blur(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    # target = [7, 8]
    target = range(17)
    padding = 0.05
    # x_min, y_min = int(np.min(shaped[:,1]*(1-padding))), int(np.min(shaped[:,0]*(1-padding)))
    # x_max, y_max = int(np.max(shaped[:,1]*(1+padding))), int(np.max(shaped[:,0]*(1+padding)))
    # x_min, y_min = x,y
    # x_max, y_max = 0,0
    for idx, kp in enumerate(shaped):
        ky, kx, kp_conf = kp

        # if kp_conf > confidence_threshold:
            # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
            # cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)

            # if x_min > kx: x_min = kx
            # if x_max < kx: x_max = kx
            # if y_min > ky: y_min = ky
            # if y_max < ky: y_max = ky

            # blur_img = frame[y_min:y_max, x_min:x_max].copy()
            # blur_img = cv2.resize(blur_img, dsize=None, fx=0.15, fy=0.15, interpolation=cv2.INTER_NEAREST)
            # blur_img = cv2.resize(blur_img, dsize=(x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)

            # frame[y_min:y_max, x_min:x_max] = blur_img
        if kp_conf > 0.5:
            if idx in target:    
                offset = 20
                if offset<=ky<=y-offset and offset<=kx<=x-offset:

                    kx = int(kx)
                    ky = int(ky)

                    blur_img = frame[ky-offset:ky+offset, kx-offset:kx+offset].copy()
                    blur_img = cv2.resize(blur_img, dsize=None, fx=0.10, fy=0.10, interpolation=cv2.INTER_NEAREST)
                    blur_img = cv2.resize(blur_img, dsize=(2*offset, 2*offset), interpolation=cv2.INTER_NEAREST)

                    frame[ky-offset:ky+offset, kx-offset:kx+offset] = blur_img
                    # cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def pose_blur(target_image_path):
    print('Start Pose Blur')
    for img_path in [os.path.join(target_image_path, i) for i in os.listdir(target_image_path)]:
    
        frame = cv2.imread(img_path)
        
        # Resize image
        img = frame.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
        input_img = tf.cast(img, dtype=tf.int32)
        
        # Detection section
        results = movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
        
        # Render keypoints 
        loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
        
        cv2.imwrite(img_path, frame)
    print('Finish Pose Blur')
        

# pose_blur('/opt/ml/input/data/images/Bad.Boys.1995__#01-33-51_01-34-37_label_B2-0-0')