import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import torchvision.transforms as transforms
import timm
import shutil
from einops import rearrange
from models.bra_legacy import BiLevelRoutingAttention


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                       num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                       kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                       topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, mlp_ratio=4, mlp_dwconv=False,
                       side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim,  kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                        qk_scale=qk_scale, kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio,
                                        kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode=kv_downsample_mode,
                                        topk=topk, param_attention=param_attention, param_routing=param_routing,
                                        diff_routing=diff_routing, soft_routing=soft_routing, side_dwconv=side_dwconv,
                                        auto_pad=auto_pad)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'), # compatiability
                                      nn.Conv2d(dim, dim, 1), # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim), # pseudo attention
                                      nn.Conv2d(dim, dim, 1), # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                     )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio*dim)),
                                 DWConv(int(mlp_ratio*dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio*dim), dim)
                                )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm
            

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x))) # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x))) # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x))) # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x))) # (N, H, W, C)
        else: # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x))) # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x))) # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x))) # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x))) # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x

class ResidualGuidedBiFormer(nn.Module):
    def __init__(self, dim, num_heads, n_win=7, topk=8):
        super().__init__()

        self.norm = nn.BatchNorm2d(dim)

        self.biformer = Block(
            dim=dim,
            num_heads=num_heads,
            n_win=n_win,
            topk=topk
        )

        # residual scaling (關鍵)
        self.gamma = nn.Parameter(torch.tensor(0.01))

        # fluorescence-guided gating
        self.gate_conv = nn.Conv2d(1, dim, kernel_size=1)

    def forward(self, x, fluor_heatmap):
        """
        x: [B, C, H, W]
        fluor_heatmap: [B, 1, H, W]
        """

        # ---- fluorescence-guided residual gating ----
        gate = torch.sigmoid(self.gate_conv(fluor_heatmap))
        x_guided = x + gate * x   # 不放大，只重加權

        # ---- BiFormer refinement ----
        out = self.biformer(self.norm(x_guided))

        # ---- residual refinement ----
        return x + self.gamma * out

# =========================
# PVTv2-B0 + Guided Residual BiFormer
# =========================
class PVTv2_GuidedBiFormer(nn.Module):
    def __init__(self, backbone_name='pvt_v2_b0', num_classes=3):
        super().__init__()

        # ---- backbone (unchanged) ----
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            in_chans=1
        )

        dims = self.backbone.feature_info.channels()
        # dims = [64, 128, 320, 512] for PVTv2-B0

        # ---- Stage 3 guided BiFormer (ONLY place we modify) ----
        self.guided_biformer_s3 = ResidualGuidedBiFormer(
            dim=dims[2],
            num_heads=5,
            n_win=7,
            topk=8
        )

        # ---- head ----
        self.norm = nn.BatchNorm2d(dims[-1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1] * 4, num_classes)

    # -------------------------
    # fluorescence heatmap
    # -------------------------
    def make_fluor_heatmap(self, g, y, bl):
        """
        g, y, bl: [B, 1, H, W]
        return:  [B, 1, H, W]
        """
        return torch.mean(torch.stack([g, y, bl], dim=1), dim=1)

    # -------------------------
    # forward single branch
    # -------------------------
    def forward_one(self, x, fluor_heatmap):
        feats = self.backbone(x)

        # Stage 3 feature
        f3 = feats[2]
        fluor_resize = F.interpolate(
            fluor_heatmap,
            size=f3.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        # Guided residual BiFormer (Stage 3 only)
        f3 = self.guided_biformer_s3(f3, fluor_resize)

        # Stage 4 (unchanged)
        f4 = feats[3]

        x = self.norm(f4)
        x = self.pool(x).flatten(1)
        return x

    # -------------------------
    # forward
    # -------------------------
    def forward(self, b, g, y, bl):
        """
        b  : bright-field image  [B,1,H,W]
        g  : green fluorescence  [B,1,H,W]
        y  : yellow fluorescence [B,1,H,W]
        bl : blue fluorescence   [B,1,H,W]
        """

        fluor = self.make_fluor_heatmap(g, y, bl)

        fb  = self.forward_one(b,  fluor)
        fg  = self.forward_one(g,  fluor)
        fy  = self.forward_one(y,  fluor)
        fbl = self.forward_one(bl, fluor)

        feat = torch.cat([fb, fg, fy, fbl], dim=1)
        return self.head(feat)