import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pytorchi3d.videotransforms as videotransforms
from pytorchi3d.pytorch_i3d import InceptionI3d
from pytorchi3d.charades_dataset_full import Charades as Dataset

class ImageFeatureExtractor():
    def __init__(self, root = 'data/',
                       mode = 'rgb',
                       batch_size = 1,
                       save_dir = 'data/image_features',
                       load_model = '../pytorchi3d/models/rgb_imagenet.pt'):
        self.root = root
        self.mode = mode
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.load_model = load_model

    def extract_image_features(self):
        first_transforms = transforms.Compose([videotransforms.FirstCrop(224)])
        second_transforms = transforms.Compose([videotransforms.SecondCrop(224)])
        third_transforms = transforms.Compose([videotransforms.ThirdCrop(224)])
        fourth_transforms = transforms.Compose([videotransforms.FourthCrop(224)])
        center_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

        
        first_dataset = Dataset(self.root, self.mode, first_transforms, num=-1, save_dir=self.save_dir)
        first_dataloader = torch.utils.data.DataLoader(first_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)

        second_dataset = Dataset(self.root, self.mode, second_transforms, num=-1, save_dir=self.save_dir)
        second_dataloader = torch.utils.data.DataLoader(second_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)

        third_dataset = Dataset(self.root, self.mode, third_transforms, num=-1, save_dir=self.save_dir)
        third_dataloader = torch.utils.data.DataLoader(third_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)

        fourth_dataset = Dataset(self.root, self.mode, fourth_transforms, num=-1, save_dir=self.save_dir)
        fourth_dataloader = torch.utils.data.DataLoader(fourth_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)

        center_dataset = Dataset(self.root, self.mode, center_transforms, num=-1, save_dir=self.save_dir)
        center_dataloader = torch.utils.data.DataLoader(center_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)

        # center_dataloader : center|
        # first_dataloader : top-left
        # second_dataloader : top-right
        # third_dataloader : bottom-right
        # fourth_dataloader : bottom-left

        dataloaders = {
                    '0': center_dataloader,
                    '1': first_dataloader,
                    '2': second_dataloader,
                    '4': third_dataloader,
                    '3': fourth_dataloader,
                        }

        datasets = {
                    '0': center_dataset,
                    '1': first_dataset,
                    '2': second_dataset,
                    '4': third_dataset,
                    '3': fourth_dataset,
                    }  

        i3d = InceptionI3d(400, in_channels=3)
        i3d.replace_logits(400)
        i3d.load_state_dict(torch.load(self.load_model))
        i3d.cuda()
        i3d.train(False)  # Set model to evaluate mode
                
        # Iterate over data.
        saved_list = []
        for phase in ['0', '1', '2', '3', '4']: # 병렬화 하여 속도 개선
            print(f'Data Loading - {phase}')
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels, name = data
                if not os.path.exists(f"{os.path.join(self.save_dir, name[0]+'__'+phase)}.npy"):
                    b,c,t,h,w = inputs.shape
                    features = []
                    for start in range(0, t, 16): 
                        end = start + 16 
                        ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda())
                        if ip.shape[2] != 16:
                            continue
                        features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
                    np.save(os.path.join(self.save_dir, name[0]+f"__{phase}"), np.concatenate(features, axis=0).reshape(-1, 1024))
                    saved_list.append(f"{os.path.join(self.save_dir, name[0]+'__'+phase)}.npy")
                    print(f"{os.path.join(self.save_dir, name[0]+'__'+phase)}.npy")    