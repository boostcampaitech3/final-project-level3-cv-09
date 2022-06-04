import numpy as np
import streamlit as st
from collections import deque
import os

def skip(threshold):

    off_path = 'data/npys/off.npy'

    off_score = np.load(off_path)

    not_violent_scene = []

    for i in range(len(off_score)):
        if off_score[i] < threshold:
            for j in range(16):
                not_violent_scene.append(16*i+j+1)

    not_violent_scene = sorted(list(set(np.array(not_violent_scene) // 24)))

    if not_violent_scene:

        save_scene = []

        queue = deque(not_violent_scene)

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

        # key_frame을 1 frame 마다
        print('changing key frame interval...')
        os.system(f'ffmpeg -hide_banner -loglevel error -y -i data/output_videos/compatible_video.mp4 -g 1 data/output_videos/key_frame.mp4')

        # 비폭력 구간 영상 만들기
        f = open('data/not_violent_videos/file_list.txt', 'w', encoding='utf-8')

        print('extracting unviolent videos...')
        for i in range(len(save_scene)):
            os.system(f'ffmpeg -hide_banner -loglevel error -y -i data/output_videos/key_frame.mp4 -ss {save_scene[i][0]} -to {save_scene[i][1]} -vcodec copy -acodec copy data/not_violent_videos/not_violent_video{i}.mp4')
            f.write(f"file 'not_violent_video{i}.mp4'\n")
        
        f.close()

        print('concatenating unviolent videos...')
        os.system('ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i data/not_violent_videos/file_list.txt -c copy data/output_videos/not_violent_video.mp4')

        st.video('data/output_videos/not_violent_video.mp4')
    else:
        st.write('No video. Please adjust the threshold.')



