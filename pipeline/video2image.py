import os
import cv2
from utils  import createDirectory

def make_image_from_video(video_root_path, image_root_path):
    video_list = [os.path.join(video_root_path, i) for i in os.listdir(video_root_path)]
    image_list = [os.path.join(image_root_path, i) for i in os.listdir(video_root_path)]

    print('convert codec')
    for file_path in video_list:
        os.system(f"ffmpeg -hide_banner -loglevel error -y -i {file_path} -map 0 -c:v libx264 -c:a copy {os.path.join('data/output_videos', 'compatible_video.mp4')}")

    for file_path, save_path in zip(video_list, image_list):
        createDirectory(save_path[:-4])
        start = file_path.find('/', file_path.find('/') + 1)
        video_name = file_path[start+1:-4]

        os.system("ffmpeg -hide_banner -loglevel error -i "+file_path+' -vf fps=24 -s 640x360 -qscale:v 4 -b 800k '+save_path[:-4]+'/'+video_name+'-%6d.jpg'" < /dev/null")
        
        print('convert mp4 to jpg :', video_name)
