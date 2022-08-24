import numpy as np
from collections import deque
from scipy import stats
import os

def mute(threshold):

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
            violent_scene.append(scene[:2])

    print('mute videos')

    vol_control_list = []

    for i in range(len(violent_scene)):
        vol_control_list.append(f'volume=enable=\'between(t,{violent_scene[i][0]},{violent_scene[i][1]})\':volume=0')

    ffmpeg_cmd = ', '.join(vol_control_list)

    os.system(f'ffmpeg -hide_banner -loglevel error -y -i data/output_videos/compatible_video.mp4 -af "{ffmpeg_cmd}" data/output_videos/muted_video.mp4')