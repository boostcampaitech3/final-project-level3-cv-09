![team](https://user-images.githubusercontent.com/82289435/172589237-13de3210-f184-4a99-a23c-7dbb71a5e4ae.png)

### 손바닥도 마주쳐야 소리가 납니다.  
### 손바닥을 맞대고 싶은 우리는 팀 하이파이프입니다.

![220606_발표자료](https://user-images.githubusercontent.com/82289435/172586979-f5a45e21-058a-4464-ac7b-75957c3dd690.png)
- 프로젝트 주제
    - 아동 · 심약자를 위한 동영상 폭력 감지 및 필터링 서비스

- 프로젝트 소개
    - 동영상 내 폭력적인 장면에 대한 폭력성 스코어링
    - 폭력성 스코어가 높은 부분, 필터링 적용 
    - 필터링 적용 후, 처리된 영상 다운로드

<br>

- 서비스 개념도
    - Multi-Modal을 활용한 Violence Detection(HLNet)
    - 동영상 전 구간에 대한 Blood Detection(YOLOv5)
    - 폭력성 구간에 대한 Violence Localization(ResNet34-Kinetics)
<br>

![concept](https://user-images.githubusercontent.com/82289435/172587234-a5da4483-bc2e-49f8-a03a-a896081b557d.png)

<br>

- 프로그램 구성도
    - Web Demo : Streamlit


![structure](https://user-images.githubusercontent.com/82289435/172627065-63c85b60-adcf-488a-bd5c-0c4f92a8d998.png)

<br>

### 🎥 Result
- Violence Detection
- Korean Movie Dataset 20 videos
- PR AUC Offline : 0.8021 Online : 0.7130

![result1](https://user-images.githubusercontent.com/82289435/173094175-3fd997fd-db21-4879-84db-309d81294016.png)

![result2](https://user-images.githubusercontent.com/82289435/173094196-b5e7b1e6-f649-4aba-93bb-d9d7c920dfe6.png)

<br>

- Blood Detection

![result3](https://user-images.githubusercontent.com/82289435/173093751-d3c58dc4-7001-48a1-8118-fb9d0ee35734.png)

### ⚙ Development Environment
- GPU : Nvidia Tesla V100
- OS : Linux Ubuntu 18.04
- Runtime : Python 3.8.5

<br>

### 🎬 Get Started
- We recommanded PYTHON 3.8.5
```
conda create -n {your_env_name} python=3.8.5
```
- conda env activate
```
conda activate {your_env_name}
```
- install requirements
```
pip install -r requirements.txt
apt-get install libsndfile1
apt-get update
apt-get install ffmpeg
```

<br>

- Run Demo 
(🧨Caution) pipeline is working directory. All path setting on pipeline.
```
cd pipeline
```
```
streamlit run zzolflix.py
```
- Run Demo on Cloud Server
```
streamlit run zzolflix.py --server.port {your_server_port}
```

### Data Directory Structure

```
pipeline/data/
            |-- audio_features(wav -> npy)
            |-- audios(wav)
            |-- blurred_images(jpg)
            |-- figures(result of plot)
            |-- image_features(jpg -> npy)
            |-- images(jpg)
            |-- list(list)
            |-- not_violent_videos
            |-- npys(on.npy, off.npy)
            |-- output_videos(mp4)
            `-- videos(mp4)
```


<br/><br/>
<hr>

### 🚩 Pre-Commit Installation Guide
0. 아래 명령어를 리눅스 기본 shell에서 실행합니다.
<br/>

1. pre-commit 설치
```
$ pip install pre-commit
$ brew install pre-commit
```

2. 버전 확인
```
$ pre-commit --v
pre-commit 2.17.0
```

3. 설치 확인 및 업데이트
```
$ pre-commit autoupdate
[WARNING] The 'rev' field.......
```

4. 설치
```
$ pre-commit install
pre-commit installed at .git/hooks/pre-commit
```

5. 이후 commit부터 오토 포매팅 반영
- fix된 사항 있을 시 git status를 통해 modified 확인 후 다시 add & commit