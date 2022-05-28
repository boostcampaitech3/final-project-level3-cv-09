import os
import shutil

reset_target_list = [
    # audio_features
    'data/audio_features',
    # audios
    'data/audios',
    # image_features
    'data/image_features',
    # images
    'data/images',
    # list
    'data/list',
    # output_videos
    'data/output_videos',
    # videos
    'data/videos',
]

def reset_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def reset_data():
    for reset_target in reset_target_list:
        reset_dir(reset_target)

def setting_directory():
    for reset_target in reset_target_list:
        if not os.path.exists(reset_target):
            os.makedirs(reset_target)

setting_directory()