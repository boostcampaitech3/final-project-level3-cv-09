import os,glob,sys
import time
from options.train_options import TrainOptions
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
#from localization_losses import *
from utils import *
import pdb
from Sound_Localization_Dataset import *
from network import *
from torch.utils.data import Dataset, DataLoader, RandomSampler
from losses import *
from PIL import Image
import matplotlib.pyplot as plt



epoch_logger = Logger('sound_localization_train.log',['epoch', 'loss'])

opt = TrainOptions().parse()

def overlay(img, heatmap, cmap = 'jet', alpha=0.5):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if isinstance(heatmap, np.ndarray):
        colorize = plt.get_cmap(cmap)
        #Normalize
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)
        heatmap = colorize(heatmap, bytes = True)
        heatmap = Image.fromarray(heatmap[:,:,:3],mode='RGB')
    # Resize the heatmap to cover whole img
    heatmap = heatmap.resize((img.size[0], img.size[1]), resample = Image.BILINEAR)
    # Display final overlayed output
    result = Image.blend(img, heatmap, alpha)
    return result

def attention_visualization(datum_val,att_map_val,vis_folder,raw_folder):
    for i in range(len(att_map_val)):
        #Get Video Name
        sample = datum_val[i]
        video_path = sample.replace('\n','')
        words = [word.replace('\n','') for word in video_path.split('/')]
        video_name = words[-1][:-4]
        # Read the image frame of the video
        all_frames = glob.glob(video_path+'/*.jpg')
        all_frames = sorted(all_frames)
        image_path = str(all_frames[0])
        # Resize it to the network input size
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((320, 320))
        # Get the predicted attention map and reshape
        att_map_t = att_map_val[i]
        att_map = att_map_t.squeeze().detach().cpu().numpy()
        att_map = np.reshape(att_map,(20,20))
        # Overlay it onto frame
        result = overlay(image_resized, att_map)
        vis_name = video_name + '.png'
        result.save(vis_folder+'/'+vis_name)
        # Save the attention map as .npy file for accuracy calculation
        raw_val_name = video_name + '.npy'
        np.save(raw_folder+'/'+raw_val_name, att_map)


def create_optimizer(net, opt):
        if opt.optimizer == 'sgd':
                return torch.optim.SGD(net.parameters(), lr = opt.lr_rate, momentum=opt.beta1, weight_decay=opt.weight_decay)
        elif opt.optimizer == 'adam':
                return torch.optim.Adam(net.parameters(), lr = opt.lr_rate, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)


def decrease_learning_rate(optimizer, decay_factor=0.1):
        for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_factor

# /opt/ml/pd/data/videos/Cold.Eyes_0-01-29_0-01-33_label_B_P.mp4
def evaluate(model, dataset_val, opt):
        # CREATE FOLDER FOR EACH EPOCH
        if not os.path.exists(os.path.join('.', 'vis_folder')):
            os.makedirs(os.path.join('.', 'vis_folder'))
        if not os.path.exists(os.path.join('.', 'raw_folder')):    
           os.makedirs(os.path.join('.', 'raw_folder'))
        raw_folder = os.path.join('.', 'raw_folder')
        vis_folder = os.path.join('.', 'vis_folder')
        val_losses = []
        val_unsup_losses = []
        val_sup_losses = []
        with torch.no_grad():
                for i, (frame_t_val, pos_audio_val, neg_audio_val, worker_gt_val, weights_t_val, datum_val) in enumerate(dataset_val):
                    frame_t_val = frame_t_val.to(opt.device)
                    pos_audio_val = pos_audio_val.to(opt.device)
                    neg_audio_val = neg_audio_val.to(opt.device)
                    worker_gt_val = worker_gt_val.to(opt.device)
                    weights_t_val = weights_t_val.to(opt.device)
                    
                    # Feed inputs into the model
                    z_val, pos_audio_embedding_val, neg_audio_embedding_val, att_map_val =  model.forward(frame_t_val, pos_audio_val, neg_audio_val)
                    
                    att_map_val = torch.squeeze(att_map_val)
                    
                    # Save Attentions and overlay them on the frames
                    attention_visualization(datum_val,att_map_val,vis_folder,raw_folder)
        # return 0
        return att_map_val

opt.device = torch.device("cuda") 


opt.mode = 'test'
dataset_test = Sound_Localization_Dataset(opt.val_dataset_file, 'test', opt.annotation_path)
dataloader_test = DataLoader(dataset_test, batch_size = 32, shuffle = False, num_workers= opt.nThreads)#, sampler = RandomSampler(dataset_test,replacement=True,num_samples=100))
dataset_size_val = len(dataloader_test)
print('#validation audios = %d' % dataset_size_val)

model = AVModel()
#model = torch.nn.DataParallel(model, device_ids=[4,5,6,7]) ###This line was commented!
model.load_state_dict(torch.load('checkpoints/sound_localization_latest.pth'))
model.to(opt.device)

# Set up optimizer
optimizer = create_optimizer(model, opt)

unsupervised_loss_criteria = UnsupervisedLoss() # margin = 1.0
supervised_loss_criteria = SupervisedLoss()

model.eval()
val_err = evaluate(model, dataloader_test, opt)