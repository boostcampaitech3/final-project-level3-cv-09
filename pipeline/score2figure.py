import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def make_figure_from_score(off_path, on_path, index_path, save_path):
    off = np.load(off_path)
    on = np.load(on_path)
    index = list(open(index_path))
    off = np.repeat(off, 16)
    on = np.repeat(on, 16)
    video = []
    for i in index:
        video.append(i.rstrip())
    data = {'video':video, 'off':off, 'on':on}
    df = pd.DataFrame(data)

    grouped = df.groupby('video')

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    count=1
    for vid, score in grouped:
        save_figure(score, save_path+"/score_output_"+str(count)+".png")
        print(f'success saved : {"score_output_"+str(count)+".png"}')
        count += 1
        

def save_figure(data, save_path):

    df = data

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    off_list = df['off'].tolist()
    on_list = df['on'].tolist()
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

    ax.plot(x, y_off, label='off')
    ax.plot(x, y_on, label='on')
    ax.set_xlabel('time (seconds)')
    ax.set_ylabel('score')
    ax.legend()

    plt.savefig(save_path)
