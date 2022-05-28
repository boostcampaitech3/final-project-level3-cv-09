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