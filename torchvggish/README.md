# 1. docs/_example_download_weights.ipynb 실행해서 docs 밑의 .ckpt 파일을 .pth 형식으로 변환 ====> drive에 올렸으니 그냥 다운받자.

# 2. torchvggish/test.ipynb 를 통해 feature extract

# 3. testset.wav는 A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A__vggish.npy의 wav 파일이다.

# p.s. 위의 다운받은 .npy는 360x128 크기지만 같은 영상으로 feature를 추출하면 250x128의 크기가 나온다. + 값이 0~255로 나오는 문제가 해결 안됨.

--------------------------------------------------

**Looking for maintainers** - I no longer have the capacity to maintain this project. If you would like to take over maintenence, please get in touch. I will either forward to your fork, or add you as a maintainer for the project. Thanks.

---


# VGGish
A `torch`-compatible port of [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset)<sup>[1]</sup>, 
a feature embedding frontend for audio classification models. The weights are ported directly from the tensorflow model, so embeddings created using `torchvggish` will be identical.


## Usage

```python
import torch

model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()

# Download an example audio file
import urllib
url, filename = ("http://soundbible.com/grab.php?id=1698&type=wav", "bus_chatter.wav")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

model.forward(filename)
```

<hr>
[1]  S. Hershey et al., ‘CNN Architectures for Large-Scale Audio Classification’,\
    in International Conference on Acoustics, Speech and Signal Processing (ICASSP),2017\
    Available: https://arxiv.org/abs/1609.09430, https://ai.google/research/pubs/pub45611
    

