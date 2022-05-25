import numpy as np
import pandas as pd

def make_csv_from_score(off_path, on_path, index_path, save_path):
    off = np.load(off_path)
    on = np.load(on_path)
    index = list(open(index_path))

    off = np.repeat(off, 16)
    on = np.repeat(on, 16)

    video = []
    for i in index:
        video.append(i[:-1])

    data = {'video':video, 'off':off, 'on':on}
    df = pd.DataFrame(data)

    grouped = df.groupby('video')

    for vid, score in grouped:
        score.to_csv(save_path, mode='w')
                    