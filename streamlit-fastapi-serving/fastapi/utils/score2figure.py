import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import deque
from scipy import stats

def make_figure_from_score(threshold_value, off_path, on_path, index_path, save_path):
    off = np.load(off_path)
    on = np.load(on_path)
    index = list(open(index_path))
    off = np.repeat(off, 16)
    on = np.repeat(on, 16)
    video = []
    for i in index:
        video.append(i.rstrip())
    data = {"video": video, "off": off, "on": on}
    df = pd.DataFrame(data)

    grouped = df.groupby("video")

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    count = 1
    for vid, score in grouped:
        save_figure(
            score, threshold_value, save_path + "/score_output_" + str(count) + ".png"
        )
        print(f'success saved : {"score_output_"+str(count)+".png"}')
        count += 1


def get_interval(off_path, threshold_value):
    off_score = np.load(off_path)
    
    # 0: not violent, 1: violent
    scene = []

    for i in range(len(off_score)):
        for _ in range(16):
            if off_score[i] < threshold_value:
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

    return violent_scene

def save_figure(data, threshold_value, save_path):

    df = data

    fig, ax = plt.subplots(1, 1, figsize=(17, 10))

    off_list = df["off"].tolist()
    on_list = df["on"].tolist()
    x = []
    time = 1
    on_score = 0
    off_score = 0
    y_on = []
    y_off = []

    for frame in range(len(df)):

        on_score += on_list[frame]
        off_score += off_list[frame]

        if frame % 24 == 23:
            x.append(time)
            time += 1

            y_on.append(on_score / 24)
            y_off.append(off_score / 24)

            on_score = 0
            off_score = 0

    save_scene = get_interval("data/npys/off.npy", threshold_value)
    ax.plot(x, y_on, label="ON (Frame level)", color='firebrick')
    ax.plot(x, y_off, label="OFF (Video level)", color='black')
    plt.xticks(fontsize=15)
    plt.yticks(np.arange(0.1, 1.1, 0.1), fontsize=15)
    plt.title('Violence Scores', fontsize=30, pad=30)
    ax.set_xlabel("time (seconds)", fontsize=25, labelpad=10)
    ax.set_ylabel("score", fontsize=25, rotation=0, labelpad=40)
    leg = ax.legend(prop={'size': 20}, bbox_to_anchor=(0.73, 1.00, 0.28, 0.2), loc="lower left", mode="expand")
    for line in leg.get_lines():
        line.set_linewidth(8.0)
    for i, (start, end) in enumerate(save_scene):
        plt.axvspan(start, end, facecolor="orange", edgecolor='gold', alpha=0.5, hatch='///')
    plt.savefig(save_path)
