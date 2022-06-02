## Overview

YOLOv5을 이용한 Custom trainning 모델입니다.  
Pre-trained Model은 **Yolov5x6**를 사용했습니다. 


## 1. Install
```
$ pip install -r requirements.txt
```

## 2. Prepare Config & Dataset
- data/hyps/hyp.p6.yaml과 같이 학습에 사용할 parameter config 작성
- data/blood.yaml에 사용할 데이터셋 경로와 클래스 정의
- train custom_dataset은 images/train, images/val하단에 사용한 jpg를 저장
- label custom_dataset은 labels/train, labels/val하단에 이미지와 동일한 이름의 txt를 저장
- 추가로 데이터셋 준비가 필요한 경우 custom_dataset/yolo_annotation_tools을 참고하여 생성

## 3. Train
- train.sh 내 인자를 조정하여 HyperParameter와 경로 등을 수정합니다.
```bash
$ ./train.sh
```

## 4. Detect
- custom train 된 가중치로 mp4영상 detect 해보기 (output = rendering된 동영상.mp4 sound(X))
- YouTube, image, webcam, Stream으로도 detect 해볼 수 있습니다.
- 아래와 같이 pt 파일과 실험해볼 동영상을 지정하여 실행할 수 있습니다.
```
python detect.py --weights ../output/yolov5/freeze10_train/weights/best.pt --source NewWorld_BestScenes.mp4
```
```
Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```