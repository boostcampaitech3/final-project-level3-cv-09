import os
import cv2
from utils import createDirectory, make_feature_list
from video2image import make_image_from_video
from audio2wav import make_wav_from_audio
from score2figure import make_figure_from_score
from image_feature_extract import ImageFeatureExtractor
from audio_feature_extract import AudioFeatureExtractor
from custom_infer import ViolenceDetector

XDVioDet_path = "XDVioDet"
pytorchi3d_path = "pytorchi3d"
video_root_path = "data/videos"  # feature를 추출할 비디오들이 담긴 폴더 지정
image_root_path = "data/images"  # jpg가 저장되는 폴더 경로 지정
createDirectory(image_root_path)

output_directory = "data/audios"  # wav가 저장되는 폴더 경로
createDirectory(output_directory)


def violence_detection(threshold):
    # mp4 -> jpg
    make_image_from_video(video_root_path, image_root_path)
    # mp4 -> wav
    make_wav_from_audio(video_root_path, output_directory)

    # image feature extract object
    IFE = ImageFeatureExtractor(
        root="data/",
        mode="rgb",
        isTrain=False,
        batch_size=1,
        save_dir="data/image_features",
        load_model="models/rgb_imagenet.pt",
    )

    print("[info] image features extracting...")
    # jpg -> image feature(npy)
    IFE.extract_image_features()

    # audio feature extract object
    AFE = AudioFeatureExtractor(
        audio_path="data/audios", feature_save_path="data/audio_features"
    )

    print("[info] audio features extracting...")
    # wav -> audio feature(npy)
    AFE.extract_audio_features()

    # npy(image features) -> video.list
    image_list_path = make_feature_list(
        target_path="data/image_features",
        save_path="data/list/",
        file_name="video.list",
    )

    # npy(audio features) -> audio.list
    audio_list_path = make_feature_list(
        target_path="data/audio_features",
        save_path="data/list/",
        file_name="audio.list",
    )

    print("[info] Start Violence Detection...")

    # Violence Detection object
    Detector = ViolenceDetector(
        image_list_path=image_list_path,
        audio_list_path=audio_list_path,
        pretrained_model_path="models/wsanodet_mix2.pkl",
        save_path="data/npys",
    )
    # violence detect
    Detector.violence_detection()

    # score(npy) -> result.csv
    make_figure_from_score(
        threshold,
        off_path="data/npys/off.npy",
        on_path="data/npys/on.npy",
        index_path="data/list/output_index.list",
        save_path="data/figures",
    )

    print(f"[info] Finish Pipeline")
