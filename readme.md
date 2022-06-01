### Get Started
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
data/
    |-- videos(mp4)
    |-- audios(wav)
    |-- audio_features(wav -> npy)
    |-- images(jpg)
    |-- image_features(jpg -> npy)
    `-- list : *.list
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