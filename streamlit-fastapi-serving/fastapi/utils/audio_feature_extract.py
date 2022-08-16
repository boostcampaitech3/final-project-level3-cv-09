import numpy as np
import glob

import os, sys
sys.path.append("..")
from models.torchvggish.torchvggish import vggish

class AudioFeatureExtractor():
    def __init__(self, audio_path = 'data/audios',
                       feature_save_path = 'data/audio_features'):
        self.audio_path = audio_path
        self.feature_save_path = feature_save_path

    def extract_audio_features(self):
        files = glob.glob(self.audio_path + '/*.wav')
        file_name_list = os.listdir(self.audio_path)

        # Initialise model and download weights
        model_urls = {
            'vggish': 'https://github.com/harritaylor/torchvggish/'
            'releases/download/v0.1/vggish-10086976.pth',
            'pca': 'https://github.com/harritaylor/torchvggish/'
            'releases/download/v0.1/vggish_pca_params-970ea276.pth'
            }
        embedding_model = vggish.VGGish(urls=model_urls, postprocess=False)
        embedding_model.eval()

        print("[saved list]")
        saved_list = []
        for audio, audio_name in zip(files, file_name_list):

            if os.path.exists(os.path.join(self.feature_save_path, audio_name[:-4])+".npy"):
                continue

            embeddings = embedding_model.forward(audio)
            np.save(os.path.join(self.feature_save_path, audio_name[:-4]),embeddings.detach().cpu().numpy())
            print(os.path.join(self.feature_save_path, audio_name[:-4])+".npy")
            saved_list.append(os.path.join(self.feature_save_path, audio_name[:-4])+".npy")

        for sample in saved_list:
            data = np.load(sample)
            print(f'{sample[len(self.feature_save_path)+1:]} - {data.shape}')