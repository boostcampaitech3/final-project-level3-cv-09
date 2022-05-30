from sklearn.metrics import auc, precision_recall_curve
import numpy as np
import torch

def test(dataloader, model, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(device)
        pred2 = torch.zeros(0).to(device)
        indexes = []
        for i, input in enumerate(dataloader):
            input, index = input
            input = input.to(device)
            logits, logits2 = model(inputs=input, seq_len=None)
            logits = torch.squeeze(logits)
            sig = torch.sigmoid(logits)
            sig = torch.mean(sig, 0)
            pred = torch.cat((pred, sig))

            for j in range(len(sig)):       # 예측 값의 길이 만큼 리스트에 파일명 추가
                indexes.append(index)
            '''
            online detection
            '''
            logits2 = torch.squeeze(logits2)
            sig2 = torch.sigmoid(logits2)
            sig2 = torch.mean(sig2, 0)

            sig2 = torch.unsqueeze(sig2, 1) ##for audio
            pred2 = torch.cat((pred2, sig2))

        pred = list(pred.cpu().detach().numpy())
        pred2 = list(pred2.cpu().detach().numpy())


        # precision, recall, th = precision_recall_curve(list(gt), np.repeat(pred, 16))
        # pr_auc = auc(recall, precision)
        # precision, recall, th = precision_recall_curve(list(gt), np.repeat(pred2, 16))
        # pr_auc2 = auc(recall, precision)
        # return pr_auc, pr_auc2, pred, pred2, indexes
        return 0.5, 0.5, pred, pred2, indexes



