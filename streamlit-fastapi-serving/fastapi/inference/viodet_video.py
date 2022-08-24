import os
import cv2
import sys
from PIL import Image

from inference.custom_inference import ViolenceDetector
sys.path.append("..")
from utils.file_handler import make_image_from_video, createDirectory, make_feature_list
from utils.audio2wav import make_wav_from_audio
from utils.score2figure import make_figure_from_score
from utils.image_feature_extract import ImageFeatureExtractor
from utils.audio_feature_extract import AudioFeatureExtractor


data_path = "data"
model_path = "models"
XDVioDet_path = model_path + "/XDVioDet"
pytorchi3d_path = model_path + "/pytorchi3d"
video_root_path = data_path + "/videos"  # feature를 추출할 비디오들이 담긴 폴더 지정
image_root_path = data_path + "/images"  # jpg가 저장되는 폴더 경로 지정
createDirectory(image_root_path)

output_directory = data_path + "/audios"  # wav가 저장되는 폴더 경로
createDirectory(output_directory)


def get_violencer():
    # image feature extract object
    IFE = ImageFeatureExtractor(
        root=data_path+"/",
        mode="rgb",
        isTrain=False,
        batch_size=1,
        save_dir=data_path+"/image_features",
        load_model=XDVioDet_path+"/ckpt/rgb_imagenet.pt",
    )

    # audio feature extract object
    AFE = AudioFeatureExtractor(
        audio_path=data_path+"/audios", feature_save_path=data_path+"/audio_features"
    )

    return IFE, AFE


def get_violence(model_IFE, model_AFE, threshold):
    threshold = 0.8 # api 따로 만들어서 처리? or 비디오랑 같이 보내줄 수 있나?
    # binary_video save 하는 부분 필요 (경로에)

    # mp4 -> jpg
    make_image_from_video(video_root_path, image_root_path)
    # mp4 -> wav
    make_wav_from_audio(video_root_path, output_directory)

    print("[info] image features extracting...")
    # jpg -> image feature(npy)
    model_IFE.extract_image_features()

    print("[info] audio features extracting...")
    # wav -> audio feature(npy)
    model_AFE.extract_audio_features()

    # npy(image features) -> video.list
    image_list_path = make_feature_list(
        target_path=data_path+"/image_features",
        save_path=data_path+"/list/",
        file_name="video.list",
    )

    # npy(audio features) -> audio.list
    audio_list_path = make_feature_list(
        target_path=data_path+"/audio_features",
        save_path=data_path+"/list/",
        file_name="audio.list",
    )

    print("[info] Start Violence Detection...")

    # Violence Detection object
    Detector = ViolenceDetector(
        image_list_path=image_list_path,
        audio_list_path=audio_list_path,
        pretrained_model_path=XDVioDet_path+"/ckpt/wsanodet_mix2.pkl",
        save_path=data_path+"/npys",
    )
    # violence detect
    Detector.violence_detection()

    # score(npy) -> result.csv
    make_figure_from_score(
        threshold,
        off_path=data_path+"/npys/off.npy",
        on_path=data_path+"/npys/on.npy",
        index_path=data_path+"/list/output_index.list",
        save_path=data_path+"/figures",
    )

    print(f"[info] Finish Pipeline")
    violence_score_image = Image.open(data_path+"/figures/score_output_1.png")
    # return image -> return npy
    return violence_score_image