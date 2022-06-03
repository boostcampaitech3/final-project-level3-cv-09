import numpy as np
import streamlit as st
import os
import shutil

def skip(origin_image_path, threshold):

    off_path = 'data/npys/off.npy'

    off_score = np.load(off_path)

    violent_scene = []

    for i in range(len(off_score)):
        if off_score[i] >= threshold:
            for j in range(16):
                violent_scene.append(str(16*i+j+1).zfill(6))


    dest_path = 'data/not_violent_images'

    files = os.listdir(origin_image_path)
    video_name = origin_image_path.split('/')[-1]

    for file in files:
        #for frame in violent_scene:
        for i in range(len(violent_scene)):
            if violent_scene[i] in file:
                shutil.copy(os.path.join(origin_image_path, file), os.path.join(dest_path, video_name+'-'+str(i).zfill(6)+'.jpg'))

    os.system(f'ffmpeg -hide_banner -loglevel error -y -framerate 1 -f image2 -r 24 -i {dest_path}/{video_name}-%06d.jpg -vcodec libx264 -vcodec libx264 -b 800k -y data/output_videos/skipped_video.mp4')
    
    if os.listdir(dest_path):
        st.video('data/output_videos/skipped_video.mp4')
    else:
        st.write('No images. Please adjust the threshold.')