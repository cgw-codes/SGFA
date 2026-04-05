import os
import random
import time

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, precision_score, recall_score
from torch.utils.data import TensorDataset
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.deterministic = True
    torch.backends.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    dgl.random.seed(seed)
    dgl.seed(seed)
    torch.use_deterministic_algorithms(True)


def dataset(HSI, LiDAR, label):
    label_index = torch.where(label != 0)[-1]
    unlabel_index = torch.where(label == 0)[-1]
    label = label.long()
    label_HSI = HSI[label_index]
    label_LiDAR = LiDAR[label_index]
    labeled_label = label[label_index]
    # if uncertain_index.shape[0] == 0:
    unlabel_HSI = HSI[unlabel_index]
    unlabel_LiDAR = LiDAR[unlabel_index]

    len = label_index.shape[0]

    rand_list = [i for i in range(unlabel_HSI.shape[0])]  # 用于随机的列表
    rand_idx = random.sample(rand_list, np.ceil(len).astype('int32'))
    unlabel_HSI = unlabel_HSI[rand_idx]
    unlabel_LiDAR = unlabel_LiDAR[rand_idx]

    dataset = TensorDataset(label_HSI, label_LiDAR, labeled_label, unlabel_HSI, unlabel_LiDAR)
    return dataset

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        alpha: 控制正负样本的平衡
        gamma: 调整难易样本的关注度，gamma 越大，对难分类样本的关注度越高
        reduction: 损失计算方式，可选 'mean' 或 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, targets):
        # 计算交叉熵损失（不做 reduction）
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        # 计算 p_t，即正确类别的预测概率
        pt = torch.exp(-ce_loss)
        # 计算 Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def select(bin, result, train_label):
    device = bin.device
    unlabel_index = torch.where(train_label == 0)[-1]
    class_num = train_label.max().item()

    # 对无标签数据分数排序，取出最接近决策边界的点
    bin = torch.abs(bin - 0.3)
    _, bin_index = torch.topk(bin, k=300, largest=False)
    # 获取选择出的点的伪标签
    class_index = unlabel_index[bin_index]
    m = nn.Softmax(dim=1)
    result = m(result)
    sample_selected = result[class_index]
    [confidence, label] = torch.max(sample_selected, dim=1)
    confidence = confidence.view(-1)
    label = label.view(-1)
    confidence_index = torch.tensor([]).to(device)

    th = torch.mean(confidence)
    print('D_q中的均值为：{}'.format(th))
    confidence_th = torch.where(confidence > th)[-1]
    sample_selected_number = torch.zeros(class_num)
    for i in range(class_num):
        sample_selected_number[i] = torch.where(label[confidence_th] == i)[-1].shape[0]
    class_th_para_stand = sample_selected_number / sample_selected_number.mean()
    class_th_para_stand = class_th_para_stand.to(device)
    
    # Initialize counter for selected samples
    total_selected = 0
    
    for i in range(class_num):
        th_class = th * class_th_para_stand[i]
        if th_class > 1:
            th_class = 0.95
        confidence_index_temp = torch.where((label == i) & (confidence > th_class))[-1]
        confidence_index = torch.cat([confidence_index, confidence_index_temp])
        total_selected += len(confidence_index_temp)
        print(f'Class {i+1} selected samples: {len(confidence_index_temp)}')

    # 取出对应的伪标签以及所选取样本的索引
    confidence_index = confidence_index.long()
    select_index = class_index[confidence_index]
    label = label[confidence_index] + 1.0
    train_label[select_index] = label.long()

    print(f'Total selected samples: {total_selected}')
    return train_label

# 将全部数据送入网络
def test_all(net, test_data):
    device = next(net.parameters()).device
    result = torch.tensor([]).to(device)
    data = torch.tensor([]).to(device)
    label = torch.tensor([]).to(device)
    with torch.no_grad():
        for i, (HSI, LiDAR, label_temp) in enumerate(test_data):
            # result_temp, data_temp, _ = net(HSI, LiDAR, 'label', 'test')
            result_temp, data_temp = net(HSI, LiDAR)
            result = torch.cat([result, result_temp], dim=0)
            data = torch.cat([data, data_temp], dim=0)
            label = torch.cat([label, label_temp], dim=0)
    return result, data, label


# 计算指标
def analy(label, result):
    label = label.cpu().detach().numpy()
    result = result.cpu().detach().numpy()
    result = result + 1
    # 计算混淆矩阵
    matrix = confusion_matrix(label, result)
    # 计算总体精度
    oa = accuracy_score(label, result)
    # 计算每类精度
    each_aa = recall_score(label, result, average=None, zero_division=0.0)
    each_aa = each_aa * 100
    each_aa = [round(i, 2) for i in each_aa]
    # 计算平均精度
    aa = recall_score(label, result, average='macro', zero_division=0.0)
    # 计算kappa系数
    kappa = cohen_kappa_score(label, result)
    return round(oa * 100, 2), each_aa, round(aa * 100, 2), round(kappa * 100, 2)
