from torch.utils.data import DataLoader
import torch
import numpy as np
from model import Model
from dataset import Dataset
from test_index import test
import option
import time
import csv


if __name__ == '__main__':
    args = option.parser.parse_args()
    device = torch.device("cuda")

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=5, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    model = Model(args)
    model = model.to(device)
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('ckpt/wsanodet_mix2.pkl').items()})
    gt = np.load(args.gt)
    st = time.time()
    pr_auc, pr_auc_online, off, on, index = test(test_loader, model, device, gt)
    print('Time:{}'.format(time.time()-st))
    print('offline pr_auc:{0:.4}; online pr_auc:{1:.4}\n'.format(pr_auc, pr_auc_online))
    # inference 결과 저장
    np.save('off.npy', off)
    np.save('on.npy', on)
    
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

    file = open('output_index.list', 'w')
    file.write(real_index)
    file.close()


