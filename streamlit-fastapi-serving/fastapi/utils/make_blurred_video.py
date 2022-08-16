import os
import time

def encoding_video(images_path, audios_path, save_video_path):
    video_name = os.listdir(images_path)[0]
    audio_name = os.listdir(audios_path)[0]
    image_target_path = os.path.join(images_path, video_name)
    audio_target_path = os.path.join(audios_path, audio_name)
    save_blurred_video_path = os.path.join(save_video_path, 'output_blurred_video.mp4')
    save_encoding_video_path = os.path.join(save_video_path, 'encoding_video.mp4')

    if len(os.listdir(images_path)) == 1 and len(os.listdir(audios_path)) == 1:
        # jpg to mp4
        os.system(f"ffmpeg -hide_banner -loglevel error -y -framerate 1 -f image2 -r 24 -i {image_target_path}/{video_name}-%06d.jpg -vcodec libx264 -b 800k -y {save_blurred_video_path}")
        
        os.system(f"ffmpeg -hide_banner -loglevel error -y -i {save_blurred_video_path} -i {audio_target_path} -c:v copy -c:a aac {save_encoding_video_path}")
    else:
        if len(os.listdir(images_path)) == 0: raise "NoneOfVideoError"
        elif len(os.listdir(audios_path)) == 0: raise "NoneOfAudioError"
        else:
            if len(os.listdir(audios_path)) == 1: raise "MultipleVideoError" 
            if len(os.listdir(images_path)) == 1: raise "MultipleAudioError" 

