# import dgl
import argparse
import os

from datetime import datetime
import torch.nn.functional as F
import dgl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix
from thop import profile, clever_format
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


from NetWork import MMamba, FeatureAwareSuperPixelPropagation
import dataset
import utils
from utils import set_seed, analy, test_all

# device_ids = [0, 1]
print('\n')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

parse = argparse.ArgumentParser()
parse.add_argument('--dataset', default='Muufl', choices=['Houston', 'Trento', 'houston2018', 'Muufl'],
                   help='select dataset')
parse.add_argument('--seed', default=42, help='random seed')
parse.add_argument('--train_num', default=0.003, help='the number of train set')
parse.add_argument('--batchsize', default=64, help='batch-size')
parse.add_argument('--test_batchsize', default=5000, help='test-batch-size')
parse.add_argument('--patchsize', default=11, help='the size of patch')
parse.add_argument('--lr', default=0.0005, help='learning rate')
parse.add_argument('--channels', default=12, help='the first layer channel numbers')

parse.add_argument('--train_epoch', default=300, help='epochs')
parse.add_argument('--select_epoch', default=100, help='when select')
parse.add_argument('--circle_epoch', default=50, help='the iteration')

args = parse.parse_args()
times = 1
# ###################### 超参预设 ######################

module_para = './module/{}/train_num_{} best_acc.pkl'.format(args.dataset, args.train_num)
if not os.path.exists('./module/{}/'.format(args.dataset)):
    os.makedirs('./module/{}/'.format(args.dataset))

# ###################### 加载数据集 ######################
samples_type = ['ratio', 'same_num'][0]  # 训练集按照 0-按比例取训练集 1-按每类个数取训练集
# 选择数据集
datasets = args.dataset


# 加载数据
[data_HSI, data_LiDAR, gt, class_count, dataset_name] = dataset.get_dataset(datasets)
th = 1 / class_count
# 源域和目标域数据信息
height, width, bands = data_HSI.shape

#给LiDAR降一个维度
data_LiDAR = data_LiDAR[:, :, 0]
# 数据标准化
[data_HSI, data_LiDAR] = dataset.data_standard(data_HSI, data_LiDAR)


# 打印每类样本个数
print('#####源域样本个数#####')
dataset.print_data(gt, class_count)



# ###################### 参数初始化 ######################

def train(TestPatch_HSI, TestPatch_LiDAR, label_index, unlabel_index, train_label, test_label):
    set_seed(args.seed)
    best_acc = 0

    # 构建数据集
    test_data = TensorDataset(TestPatch_HSI[unlabel_index], TestPatch_LiDAR[unlabel_index], test_label[unlabel_index])
    test_data = DataLoader(test_data, batch_size=args.test_batchsize, shuffle=False)
    all_data = TensorDataset(TestPatch_HSI, TestPatch_LiDAR, test_label)
    all_data = DataLoader(all_data, batch_size=args.test_batchsize, shuffle=False, drop_last=False)

    # 构建SAGE网络
    net_mamba = MMamba(FM=args.channels, NC=bands, NCLidar=1, Classes=class_count, patchsize=args.patchsize, drop_path=0.1, depth=2, token = 12, length = args.patchsize * args.patchsize).cuda(device)

    # 初始化超像素标签扩散器
    superpixel_propagator = FeatureAwareSuperPixelPropagation(
        n_groups=50,
        n_subgroups=5,
        sigma=0.8,
        kl_weight=0.2,
        dist_weight=0.3,
        pred_weight=0.5
    )
    

    # ###################### 训练 ######################
    optimizer = torch.optim.Adam(net_mamba.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    loss_fun = nn.CrossEntropyLoss()
    focal_loss_fun = utils.FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')

    # Focal Loss 权重因子，可根据实验进行调整
    weight_focal = 1

    weight_proto = 0.1

    # 添加重建损失函数
    recon_loss_fn = nn.MSELoss()
    # 重建损失权重
    weight_recon = 0.001
    
    for epoch in range(args.train_epoch + 1):

        net_mamba.train()
        acc_num = 0
        loss_class, loss_focus, loss_proto, loss_recon, loss = [], [], [], [], []

        # 每隔一定 epoch 重新采样训练数据
        if epoch % 50 == 0 or (epoch - args.select_epoch) % args.circle_epoch == 1:
            train_data = utils.dataset(TestPatch_HSI, TestPatch_LiDAR, train_label)
            train_data = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)

        # ======== 主训练循环 ========
        for i, (HSI_lbl, LiDAR_lbl, label_lbl, HSI_ulb, LiDAR_ulb) in enumerate(train_data):
            target = (label_lbl - 1).to(device).long()

            # 前向传播（启用中间特征与重建）
            result_lbl, data_lbl, hsi_feat_mid, lidar_feat_mid, hsi_recon_feat, lidar_recon_feat = \
                net_mamba(HSI_lbl, LiDAR_lbl, target, need_reconstruction=True)

            # 分类损失
            loss_cls = loss_fun(result_lbl, target)
            loss_class.append(loss_cls.item())

            # 计算分类准确率
            result = torch.argmax(result_lbl, dim=1) + 1
            acc_num += (label_lbl == result).sum().item()

            # 原型约束损失
            proto_centers = net_mamba.prototype_memory.prototypes
            centers_batch = proto_centers[target]
            loss_proto_batch = F.mse_loss(data_lbl, centers_batch)
            loss_proto.append(loss_proto_batch.item())

            # Focal Loss
            loss_focal = focal_loss_fun(result_lbl, target)
            loss_focus.append(loss_focal.item())

            # ======== 特征重建 / 对齐损失 ========
            # 让 decoder 重建的特征与中间层特征尽量接近
            # 使用余弦相似度约束（更稳）
            recon_loss_hsi = 1 - F.cosine_similarity(hsi_recon_feat, hsi_feat_mid, dim=-1).mean()
            recon_loss_lidar = 1 - F.cosine_similarity(lidar_recon_feat, lidar_feat_mid, dim=-1).mean()
            loss_recon_batch = recon_loss_hsi + recon_loss_lidar
            loss_recon.append(loss_recon_batch.item())

            # ======== 总损失加权 ========
            # 推荐比例（经验稳定）
            # weight_focal ≈ 0.5，weight_proto ≈ 0.3，weight_recon ≈ 0.005
            loss_temp = (
                loss_cls
                + weight_focal * loss_focal
                + weight_proto * loss_proto_batch
                + weight_recon * loss_recon_batch
            )
            loss.append(loss_temp.item())

            # ======== 反向传播与优化 ========
            optimizer.zero_grad()
            loss_temp.backward()
            torch.nn.utils.clip_grad_norm_(net_mamba.parameters(), max_norm=5.0)  # 梯度裁剪防止震荡
            optimizer.step()

        acc_train = acc_num / label_index.shape[0]

        # 输出训练结果，添加重建损失信息
        out = PrettyTable()
        print('epoch:{:0>3d}'.format(epoch))
        out.add_column("loss", ['value'])
        out.add_column('acc_train', ['{:.4f}'.format(acc_train)])
        out.add_column('train loss', ['{:.4f}'.format(np.mean(loss) if len(loss) > 0 else 0)])
        out.add_column('class loss', ['{:.4f}'.format(np.mean(loss_class)) if len(loss_class) > 0 else 0])
        out.add_column('focus loss', ['{:.4f}'.format(np.mean(loss_focus)) if len(loss_focus) > 0 else 0])
        out.add_column('recon loss', ['{:.4f}'.format(np.mean(loss_recon)) if len(loss_recon) > 0 else 0])
        out.add_column('proto loss', ['{:.4f}'.format(np.mean(loss_proto)) if len(loss_proto) > 0 else 0])

        print(out)

        # 利用验证集保存最优网络结果
        if epoch % 1 == 0:
            # a = time.time()
            net_mamba.eval()
            result, _, test_label_v1 = test_all(net_mamba, test_data)
            loss_val = loss_fun(result, test_label_v1.long() - 1)
            print('| val loss:%.4f' % loss_val)
            result = torch.argmax(result, dim=1)
            accuracy, each_aa, aa, kappa = utils.analy(test_label_v1, result)
            print('| test accuracy: %.4f' % accuracy, '| aa:', aa)
            if best_acc < accuracy:
                best_acc = accuracy
                # best_oa = aa
                torch.save(net_mamba.state_dict(), module_para)
            print('best_acc:{}'.format(best_acc))
            # oa, aa, kappa, each_aa = test(net_mamba, TestPatch_HSI, TestPatch_LiDAR, test_label)
            # print('测试时间为{}s'.format(time.time() - a))

        # 在伪标签选择部分添加对比学习
        if epoch != args.train_epoch and (
                epoch - args.select_epoch) % args.circle_epoch == 0 and epoch >= args.select_epoch:
        
            net_mamba.load_state_dict(torch.load(module_para))
            net_mamba.eval()
            result, data, test_label = test_all(net_mamba, all_data)
            print(data.shape)
            
            # 使用超像素方法计算置信度
            print("使用超像素方法计算置信度...")
            
            # 计算未标记样本的置信度
            bin_unlabel = superpixel_propagator.calculate_confidence(
                data_HSI, data_LiDAR, unlabel_index, result, data
            )
            
            # 打印置信度统计信息，帮助调试
            print(f"置信度统计: 最小值={bin_unlabel.min().item():.4f}, 最大值={bin_unlabel.max().item():.4f}, 平均值={bin_unlabel.mean().item():.4f}")
            print(bin_unlabel.shape, result.shape, train_label.shape)
            
            train_label = utils.select(bin_unlabel, result, train_label)
            
            
            label_index = torch.where(train_label != 0)[-1]
            unlabel_index = torch.where(train_label == 0)[-1]

    net_mamba.load_state_dict(torch.load(module_para))
    net_mamba.eval()
    torch.save(net_mamba.state_dict(),
               './module/{}/train_num_{}_{}_{}'.format(args.dataset, args.train_num, best_acc,
                                                       formatted_datetime))
    result, data, test_true = test_all(net_mamba, test_data)
    result = torch.argmax(result, dim=1)
    oa, each_aa, aa, kappa = analy(test_true, result)
    print('OA: %.2f' % oa, 'AA: %.2f' % aa, 'kappa: %.2f' % kappa)
    print(each_aa)

    test(net_mamba, TestPatch_HSI, TestPatch_LiDAR, test_label)


def test(net_mamba, TestPatch_HSI, TestPatch_LiDAR, test_label):
    net_mamba.load_state_dict(torch.load(module_para))
    net_mamba.eval()
    torch.cuda.synchronize()

    test_data = TensorDataset(TestPatch_HSI, TestPatch_LiDAR, test_label)
    test_data = DataLoader(test_data, batch_size=args.test_batchsize, shuffle=False, drop_last=False)
    result, data, test_label = test_all(net_mamba, test_data)
    result = torch.argmax(result, dim=1)
    oa, each_aa, aa, kappa = analy(test_label, result)
    print('OA: %.4f' % oa, 'AA: %.4f' % aa, 'kappa: %.4f' % kappa)
    print(each_aa)

    Testlabel = np.zeros([height, width])
    Testlabel = np.reshape(Testlabel, [height * width])
    Testlabel[test_label_index] = result.cpu().detach().numpy() + 1
    Testlabel = np.reshape(Testlabel, [height, width])
    torch.save(net_mamba, './colorMap/' + args.dataset + '_best_model_' + '{:.4f}'.format(oa) + '.pt')
    sio.savemat('./result/{}_{}.mat'.format(args.dataset, oa), {'data': Testlabel})


current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
set_seed(args.seed)
# 对源域样本进行划分，得到训练、测试、验证集
[train_label, test_label, unlabel] = dataset.data_partition(class_count, gt, args.train_num)
TestPatch_HSI, TestPatch_LiDAR = dataset.gen_cnn_data(data_HSI, data_LiDAR, args.patchsize, train_label, unlabel,
                                                      device)
# 获得新的索引
train_label = np.reshape(train_label, (height * width))
label_index = np.where(train_label != 0)[-1]
label = train_label[label_index]
unlabel = np.reshape(unlabel, (height * width))
unlabel_index = np.where(unlabel != 0)[-1]
unlabel = unlabel[unlabel_index]
test_label = np.reshape(test_label, (height * width))
test_label_index = np.where(test_label != 0)[-1]
test_label = test_label[test_label_index]

all_index = np.concatenate([label_index, unlabel_index], axis=0)
len1 = label_index.shape[0]
index_index = np.argsort(all_index)

all_index = np.sort(all_index)
label_index = np.where(index_index < len1)[-1]
unlabel_index = np.where(index_index >= len1)[-1]
train_label = np.zeros(all_index.shape[0]).astype(np.int64)
train_label[label_index] = label

# 送进gpu
TestPatch_HSI = TestPatch_HSI[index_index].to(device)
TestPatch_LiDAR = TestPatch_LiDAR[index_index].to(device)
label_index = torch.from_numpy(label_index).to(device)
unlabel_index = torch.from_numpy(unlabel_index).to(device)
train_label = torch.from_numpy(train_label).long().to(device)
test_label = torch.from_numpy(test_label).long().to(device)

# 训练集样本个数
for i in range(1, train_label.max().item() + 1):
    print('第{}类训练样本的个数为{}：'.format(i, torch.where(train_label == i)[0].shape[0]))
# 进行训练
train(TestPatch_HSI, TestPatch_LiDAR, label_index, unlabel_index, train_label, test_label)
