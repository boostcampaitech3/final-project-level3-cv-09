import numpy as np
from collections import deque
from scipy import stats
import os

import cv2
import random


def insert_image(img, gifs_rimg, count, random_point):
    r_rows = r_cols = 200
    pn, n, idx = (count % 1280) // 20, (count % 80) // 20, (count % 80) % 20
    w, h = random_point[pn][0], random_point[pn][1]

    img[h:r_cols+h, w:r_rows+w] = gifs_rimg[n][idx]

def load_gif_rimage(gifs_rimg, gif_path):
    frame_list = list()
    gif = cv2.VideoCapture(gif_path)
    ret, frame = gif.read()  # ret=True if it finds a frame else False.
    while ret:
        # something to do 'frame'
        # ...
        # 다음 frame 읽음
        frame = cv2.resize(frame, dsize=(200, 200), interpolation=cv2.INTER_AREA)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        frame_list.append(frame)
        ret, frame = gif.read()
    gifs_rimg.append(frame_list[:20])

def alternative_video(video_path, gifs_path, violent_scene):
    gifs_rimg = list()
    for gif_path in gifs_path:
        load_gif_rimage(gifs_rimg, gif_path)
    
    video_name = video_path.split("/")[-1][:-4]
    vidcap = cv2.VideoCapture(video_path)

    FPS = vidcap.get(cv2.CAP_PROP_FPS)
    WIDTH, HEIGHT = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('data/output_videos/altered_video.mp4', fourcc, FPS, (WIDTH, HEIGHT))

    # generate random gif images point list
    random_point = list()
    for _ in range(len(gifs_rimg)*16):
        w, h = random.randint(50, WIDTH-200), random.randint(50, HEIGHT-200)
        random_point.append((w, h))

    # violent_scene second to frame num
    scene_idx, gif_count = 0, 0
    for i in range(len(violent_scene)):
        for j in range(2):
            violent_scene[i][j] *= FPS

    count = 0
    success, image = vidcap.read()
    while success:
        if violent_scene[scene_idx][0] <= count and count <= violent_scene[scene_idx][1]: 
            insert_image(image, gifs_rimg, gif_count, random_point)
            gif_count += 1

        if count > violent_scene[scene_idx][1]:
            if scene_idx < len(violent_scene) - 1:
                scene_idx += 1
        # test_rimage(image, gifs_rimg, count, random_point)
        out.write(image)
        success, image = vidcap.read()

        count += 1

    output_path = "data/output_videos/"
    out.release()
    vidcap.release()
    os.system(f"ffmpeg -i {output_path}altered_video.mp4 -vcodec libx264 {output_path}altered_video_h264.mp4")
    os.system(f"ffmpeg -i {output_path}altered_video_h264.mp4 -i data/audios/*.wav -c:v copy -c:a aac -strict experimental {output_path}altered_video_final.mp4")
    print("finish! convert video to frame {name}".format(name=video_name))
    print("all convert finish!!")

def alternative(threshold):

    off_path = 'data/npys/off.npy'

    off_score = np.load(off_path)

    # 0: not violent, 1: violent
    scene = []

    for i in range(len(off_score)):
        for _ in range(16):
            if off_score[i] < threshold:
                scene.append(0)
            else:
                scene.append(1)

    scene_seconds = []

    for i in range(0, len(scene)-24, 24):
        scene_seconds.append(int(stats.mode(scene[i:24+i])[0]))

    queue = deque(scene_seconds)

    violence = queue.popleft()
    start_time = 0
    end_time = 1

    scene_snippets = []

    while queue:
        next_violence = queue.popleft()
        
        if violence != next_violence:
            scene_snippets.append((start_time, end_time - 1, violence))
            
            violence = next_violence

            start_time = end_time

        end_time += 1

    scene_snippets.append((start_time, end_time, violence))

    print(scene_snippets)

    violent_scene = []

    for scene in scene_snippets:
        if scene[2] == 1:
            violent_scene.append(list(scene[:2]))

    # Alternative image
    gifs_path = [
        "data/alter_images/mococo_0.gif",
        "data/alter_images/mococo_1.gif",
        "data/alter_images/mococo_2.gif",
        "data/alter_images/mococo_3.gif",
    ]

    alternative_video("data/output_videos/compatible_video.mp4", gifs_path, violent_scene)