from torch.utils.data import DataLoader
import torch
import numpy as np
import time
import csv

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from XDVioDet.model import Model
from XDVioDet.dataset import Dataset
from XDVioDet.test import test
import XDVioDet.option as option


class ViolenceDetector():
    def __init__(self, image_list_path, audio_list_path, pretrained_model_path, save_path):
        self.image_list_path = image_list_path
        self.audio_list_path = audio_list_path
        self.pretrained_model_path = pretrained_model_path
        self.save_path = save_path

    def violence_detection(self):
        args = option.parser.parse_args()
    
        device = torch.device("cuda")

        test_loader = DataLoader(Dataset(args, test_mode=True),
                                batch_size=5, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
        model = Model(args)
        model = model.to(device)
        model_dict = model.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load(self.pretrained_model_path).items()})
        # gt = np.load(args.gt)
        st = time.time()
        pr_auc, pr_auc_online, off, on, index = test(test_loader, model, device)
        print('Time:{}'.format(time.time()-st))
        print("[info] Violence Detection Finish")
        # inference 결과 저장
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        np.save(self.save_path+'/off.npy', off)
        np.save(self.save_path+'/on.npy', on)
        
        new_index = []
        for i in index:
            for j in i:
                new_index.append(j)

        real_index = []
        for i in range(0, len(new_index), 5):
            mp4 = new_index[i][:-7] + '.mp4'
            mp4 = mp4.split('/')[-1]
            for j in range(16):
                real_index.append(mp4)
        real_index = '\n'.join(real_index)

        file = open('data/list/output_index.list', 'w')
        file.write(real_index)
        file.close()


