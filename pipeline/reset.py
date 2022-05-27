import os
import shutil

reset_target_list = [
    # audio_features
    '/opt/ml/input/data/audio_features',
    # audios
    '/opt/ml/input/data/audios',
    # image_features
    '/opt/ml/input/data/image_features',
    # images
    '/opt/ml/input/data/images',
    # list
    '/opt/ml/input/data/list',
    # output_videos
    '/opt/ml/input/data/output_videos',
    # videos
    '/opt/ml/input/data/videos',
]

def reset_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def reset_data():
    for reset_target in reset_target_list:
        reset_dir(reset_target)

# reset_data()