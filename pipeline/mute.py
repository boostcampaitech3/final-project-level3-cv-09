import numpy as np
import streamlit as st
from collections import deque
import os

def mute(threshold):

    off_path = 'data/npys/off.npy'

    off_score = np.load(off_path)

    violent_scene = []

    for i in range(len(off_score)):
        if off_score[i] >= threshold:
            for j in range(16):
                violent_scene.append(16*i+j+1)

    violent_scene = sorted(list(set(np.array(violent_scene) // 24)))

    save_scene = []

    queue = deque(violent_scene)

    start_time = queue.popleft()
    check_time = start_time

    while queue:
        end_time = queue.popleft()

        if check_time + 1 == end_time:
            check_time = end_time
        else:
            save_scene.append((start_time, check_time))

            if queue:
                start_time = end_time
                check_time = start_time

    save_scene.append((start_time, end_time))
    print(save_scene)

    print('mute videos')

    vol_control_list = []

    for i in range(len(save_scene)):
        vol_control_list.append(f'volume=enable=\'between(t,{save_scene[i][0]},{save_scene[i][1]})\':volume=0')

    ffmpeg_cmd = ', '.join(vol_control_list)

    os.system(f'ffmpeg -hide_banner -loglevel error -y -i data/output_videos/compatible_video.mp4 -af "{ffmpeg_cmd}" data/output_videos/muted_video.mp4')

    st.video(f'data/output_videos/muted_video.mp4')
