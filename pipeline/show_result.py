import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def save_figure(result_csv_path, save_path):

    df = pd.read_csv(result_csv_path)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    x = np.arange(len(df))
    y_off = df['off']
    y_on = df['on']

    ax[0].plot(x, y_off, label='off')
    ax[0].plot(x, y_on, label='on')
    ax[0].set_xlabel('frames')
    ax[0].set_ylabel('score')
    ax[0].legend()

    off_list = df['off']
    on_list = df['on']

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

    ax[1].plot(x, y_off, label='off')
    ax[1].plot(x, y_on, label='on')
    ax[1].set_xticks(np.arange(0, x[-1], 5))
    ax[1].set_xlabel('time (seconds)')
    ax[1].set_ylabel('score')
    ax[1].legend()

    plt.savefig(save_path)
