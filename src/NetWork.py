

import numpy as np
import time
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# 添加超像素标签扩散相关的导入
from skimage.segmentation import slic, mark_boundaries
from scipy.spatial.distance import cdist
import numpy as np
import torch
import matplotlib.pyplot as plt





import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from scipy.sparse import csr_matrix, diags, eye
import torch

class FeatureAwareSuperPixelPropagation:
    def __init__(
        self,
        n_groups=50,        # 第一次：未标记样本分组数
        n_subgroups=5,      # 第二次：组内 SLIC 数
        sigma=0.8,
        kl_weight=0.3,
        dist_weight=0.3,
        pred_weight=0.4
    ):
        self.n_groups = n_groups
        self.n_subgroups = n_subgroups
        self.sigma = sigma
        self.kl_weight = kl_weight
        self.dist_weight = dist_weight
        self.pred_weight = pred_weight

    # ================= KL =================
    @staticmethod
    def kl_divergence(p, q):
        eps = 1e-10
        p = p + eps
        q = q + eps
        p /= p.sum()
        q /= q.sum()
        return np.sum(p * np.log(p / q))

    # ================= 第二次 SLIC（组内） =================
    def group_inner_confidence(self, feats, logits):
        """
        feats : [M, D]
        logits: [M, C]
        """
        M = feats.shape[0]
        conf = np.zeros(M)

        if M <= self.n_subgroups:
            labels = np.zeros(M, dtype=int)
        else:
            labels = KMeans(
                n_clusters=self.n_subgroups,
                n_init=5,
                random_state=0
            ).fit_predict(feats)

        for g in np.unique(labels):
            idx = np.where(labels == g)[0]
            sub_feat = feats[idx]
            sub_log = logits[idx]

            # 结构一致性
            variance = np.mean(np.var(sub_feat, axis=0))
            max_min = np.linalg.norm(sub_feat.max(0) - sub_feat.min(0))
            global_hist = np.histogram(sub_feat.reshape(-1), bins=20, density=True)[0]

            for i, pi in enumerate(idx):
                hist = np.histogram(sub_feat[i], bins=20, density=True)[0]
                kl = self.kl_divergence(hist, global_hist)
                kl_conf = np.exp(-kl / self.sigma)

                dist_conf = np.exp(-variance / self.sigma) * np.exp(-max_min / (2 * self.sigma))

                prob = torch.softmax(
                    torch.from_numpy(sub_log[i]), dim=0
                ).numpy()
                pred_conf = prob.max()

                conf[pi] = (
                    self.kl_weight * kl_conf +
                    self.dist_weight * dist_conf +
                    self.pred_weight * pred_conf
                )

        conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
        return conf

    # ================= 组间传播 =================
    @staticmethod
    def group_propagation(group_feat, group_conf):
        sim = cosine_similarity(group_feat)
        sim /= sim.sum(axis=1, keepdims=True) + 1e-8
        out = sim @ group_conf
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)
        return out

    # ================= 主接口 =================
    def calculate_confidence(
        self,
        data_HSI,       # 保留接口（不使用）
        data_LiDAR,     # 保留接口（不使用）
        unlabel_index,
        features,       # test_all 返回的 data
        logits          # test_all 返回的 result
    ):
        """
        features: Tensor [N, D]
        logits  : Tensor [N, C]
        """

        # -------- Tensor → numpy --------
        feats = features.detach().cpu().numpy()
        logs = logits.detach().cpu().numpy()
        unlabel_idx = unlabel_index.detach().cpu().numpy()

        feats = feats[unlabel_idx]
        logs = logs[unlabel_idx]

        N = feats.shape[0]
        final_conf = np.zeros(N)

        # ========== 第一次 SLIC（未标记样本级） ==========
        if N <= self.n_groups:
            group_labels = np.zeros(N, dtype=int)
        else:
            group_labels = KMeans(
                n_clusters=self.n_groups,
                n_init=10,
                random_state=0
            ).fit_predict(feats)

        group_feat_repr = []
        group_conf_repr = []

        # ========== 第二次 SLIC（组内） ==========
        for g in np.unique(group_labels):
            idx = np.where(group_labels == g)[0]
            sub_feat = feats[idx]
            sub_log = logs[idx]

            inner_conf = self.group_inner_confidence(sub_feat, sub_log)
            final_conf[idx] = inner_conf

            group_feat_repr.append(sub_feat.mean(axis=0))
            group_conf_repr.append(inner_conf.mean())

        group_feat_repr = np.vstack(group_feat_repr)
        group_conf_repr = np.array(group_conf_repr)

        # ========== 组间置信度扩散 ==========
        propagated = self.group_propagation(group_feat_repr, group_conf_repr)

        for i, g in enumerate(np.unique(group_labels)):
            idx = np.where(group_labels == g)[0]
            final_conf[idx] = 0.5 * final_conf[idx] + 0.5 * propagated[i]

        final_conf = (final_conf - final_conf.min()) / (final_conf.max() - final_conf.min() + 1e-8)

        return torch.from_numpy(final_conf).float().to(features.device)



    



from einops import rearrange
from timm.models.layers import DropPath
import torch.nn as nn
import torch



from functools import partial
import torch.backends.cudnn as cudnn
import numpy as np
from mamba import SSM
cudnn.deterministic = True
cudnn.benchmark = False




# --------------------------------------------------------
# DropPath 实现
# --------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# --------------------------------------------------------
# HetConv Layer
# --------------------------------------------------------
class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, p=64, g=64):
        super().__init__()
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=3, groups=g, padding=1)
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=p)

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)


# --------------------------------------------------------
# Mamba Block
# --------------------------------------------------------
class block_1D(nn.Module):
    def __init__(self, hidden_dim=0, drop_path=0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate=0, d_state=16, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SSM(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        residual = x
        x = self.ln_1(x)
        out_forward = self.self_attention(x)
        out_backward = torch.flip(x, dims=[1])
        out_backward = self.self_attention(out_backward)
        out_backward = torch.flip(out_backward, dims=[1])
        return residual + self.drop_path(out_forward + out_backward)


# --------------------------------------------------------
# 改进版融合模块（注意力融合）
# --------------------------------------------------------
class AdaptiveFusionModule(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(seq_len))
        self.encoder_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.channel_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        w = torch.softmax(torch.sigmoid(self.weights), dim=0)
        x1_new = torch.sum(x1 * w.view(1, -1, 1), dim=1, keepdim=True)
        x2_new = torch.sum(x2 * w.view(1, -1, 1), dim=1, keepdim=True)

        x_fused = x1_new + x2_new
        ca = self.channel_attention(x_fused)
        x_fused = self.encoder_norm(x_fused * ca)
        return x_fused


# --------------------------------------------------------
# Mamba Encoder
# --------------------------------------------------------
class MultiMambaEncoder(nn.Module):
    def __init__(self, hidden_dim, drop_path, depth, seq_len):
        super().__init__()
        self.blocks = nn.ModuleList([
            block_1D(hidden_dim=hidden_dim, drop_path=drop_path)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.encoder_norm(x)


# --------------------------------------------------------
# 动态平均池化
# --------------------------------------------------------
class DynamicAvgPool1d(nn.Module):
    def __init__(self, token, input_length=121):
        super().__init__()
        self.token = token
        self.input_length = input_length
        kernel_size = self.input_length // self.token
        stride = self.input_length // self.token
        if self.input_length % self.token != 0:
            kernel_size = self.input_length - (self.token - 1) * stride
        self.pooling = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        return self.pooling(x)


# --------------------------------------------------------
# Prototype Loss 模块（类别中心约束）
# --------------------------------------------------------
class PrototypeMemory(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.register_buffer("prototypes", torch.zeros(num_classes, feat_dim))
        self.momentum = 0.9

    def forward(self, features, labels):
        with torch.no_grad():
            for c in labels.unique():
                mask = (labels == c)
                if mask.sum() == 0:
                    continue
                cls_feat = features[mask].mean(0)
                self.prototypes[c] = self.momentum * self.prototypes[c] + (1 - self.momentum) * cls_feat
        return self.prototypes


# --------------------------------------------------------
# 改进版 MMamba 主模型
# --------------------------------------------------------
class FeatureAlignDecoder(nn.Module):
    """
    对齐特征空间的重建器
    输入：融合后的特征 (B, FM*4)
    输出：重建到中间层特征空间 (B, FM*4)
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, x):
        return self.decoder(x)


# ---------- 改进后的 MMamba ----------
class MMamba(nn.Module):
    def __init__(self, FM, NC, NCLidar, Classes, patchsize,
                 drop_path=0.0, depth=2, token=12, length=121):
        super().__init__()
        self.FM = FM
        self.token = token
        self.input_length = length
        self.patchsize = patchsize

        # ---- HSI 分支 ----
        self.hsi1 = nn.Sequential(
            nn.Conv3d(1, 18, (9, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(18),
            nn.ReLU()
        )
        self.hsi2 = nn.Sequential(
            HetConv(18 * (NC - 8), FM * 8, p=1, g=(FM * 4) // 8 if (18 * (NC - 8)) % 8 == 0 else 2),
            nn.BatchNorm2d(FM * 8),
            nn.ReLU()
        )
        self.hsi3 = nn.Sequential(
            nn.Conv2d(FM * 8, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.GELU()
        )

        # ---- LiDAR 分支 ----
        self.lidarConv = nn.Sequential(
            nn.Conv2d(NCLidar, FM * 8, 3, 1, 1),
            nn.BatchNorm2d(FM * 8),
            nn.GELU()
        )
        self.lidarConv2 = nn.Sequential(
            nn.Conv2d(FM * 8, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.GELU()
        )

        # ---- 序列建模 + 融合 ----
        self.pooling = DynamicAvgPool1d(token=self.token, input_length=self.input_length)
        self.position_embeddings1 = nn.Parameter(torch.randn(1, self.token, FM * 4))
        self.position_embeddings2 = nn.Parameter(torch.randn(1, self.token, FM * 4))
        self.dropout = nn.Dropout(0.1)
        self.fusion = AdaptiveFusionModule(self.token, FM * 4)
        self.mamba = MultiMambaEncoder(hidden_dim=FM * 4, drop_path=drop_path, depth=depth, seq_len=self.token)

        # ---- 分类头 ----
        self.out = nn.Linear(FM * 4, Classes)

        # ---- 特征重建模块（替代原像素重建）----
        self.hsi_decoder = FeatureAlignDecoder(FM * 4)
        self.lidar_decoder = FeatureAlignDecoder(FM * 4)

        # ---- Prototype Memory ----
        self.prototype_memory = PrototypeMemory(num_classes=Classes, feat_dim=FM * 4)

    def forward(self, x1, x2, labels=None, need_reconstruction=False):
        # HSI branch
        x1 = x1.reshape(x1.shape[0], -1, self.patchsize, self.patchsize).unsqueeze(1)
        x1 = self.hsi1(x1).reshape(x1.shape[0], -1, self.patchsize, self.patchsize)
        x1 = self.hsi2(x1)
        x1_mid = self.hsi3(x1)     # <- 中间特征
        x1_seq = self.pooling(x1_mid.flatten(2)).transpose(-1, -2)

        # LiDAR branch
        x2 = x2.reshape(x2.shape[0], -1, self.patchsize, self.patchsize)
        x2 = self.lidarConv(x2)
        x2_mid = self.lidarConv2(x2)  # <- 中间特征
        x2_seq = self.pooling(x2_mid.reshape(x2.shape[0], -1, self.patchsize ** 2)).transpose(-1, -2)

        # 融合 + Mamba
        x1_seq = self.dropout(x1_seq + self.position_embeddings1)
        x2_seq = self.dropout(x2_seq + self.position_embeddings2)
        x = self.fusion(x1_seq, x2_seq)
        x = self.mamba(x)

        # 全局特征
        features = x.mean(dim=1)
        logits = self.out(features)

        # Prototype Memory 更新
        if labels is not None:
            self.prototype_memory(features.detach(), labels)

        if need_reconstruction:
            # 特征重建
            hsi_recon_feat = self.hsi_decoder(features)
            lidar_recon_feat = self.lidar_decoder(features)
            return logits, features, x1_mid.mean(dim=(2,3)), x2_mid.mean(dim=(2,3)), hsi_recon_feat, lidar_recon_feat

        return logits, features





