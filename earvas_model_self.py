# earvas_model.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
# from freRA_augment_in_train import FreRA 
from freRA_mm_cross import FreRA_MM_Cross

def l2_normalize(x, eps=1e-6):
    return x / (x.norm(dim=1, keepdim=True) + eps)

class EarVAS(nn.Module):
    def __init__(self, num_classes=9, use_freRA=False ):
        super().__init__()

        self.use_freRA = use_freRA
        if self.use_freRA:
            # 仅对imu做frera：
            # self.freRA = FreRA(F=51)  # F 根据 IMU 长度设定，T=100 → F=51
            # 同时对音频和imu做 frera：
            self.freRA = FreRA_MM_Cross(imu_F=51, mel_F=65)  # mel_F= mel.shape[2] // 2 + 1 = n_fft // 2 + 1 = 128 // 2 + 1 = 65


        # self.audio_model = models.efficientnet_b0(pretrained=True)
        self.audio_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # 修改第一层接受 2 通道
        self.audio_model.features[0][0] = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # 移除原分类头
        self.audio_model.classifier = nn.Identity()
        self.audio_proj = nn.Linear(1280, 256)

        self.imu_branch = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.gate_layer = nn.Sequential( # gating 模块 
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.classifier1 = nn.Sequential(nn.Linear(256, 128),nn.ReLU(),nn.Linear(128, num_classes))
        self.classifier2 = nn.Sequential(nn.Linear(256 * 2, 128),nn.ReLU(),nn.Linear(128, num_classes)) 
        self.classifier3 = nn.Sequential(nn.Linear(256 * 3, 128),nn.ReLU(),nn.Linear(128, num_classes)) 
        

    def forward(self, audio, imu, return_alpha=False):
        audio_feat = self.audio_model(audio)
        audio_feat = self.audio_proj(audio_feat)

        if self.use_freRA and self.training: 
            # 仅imu做 frera：
            # imu = self.freRA(imu)  # → [B, 6, 100] 
            # 同时对音频和imu做 frera：
            imu, audio, s_audio, s_imu = self.freRA(imu, audio)  # 增强 audio + imu
        imu_feat = self.imu_branch(imu)
        
        # ✅ 对特征做 L2 归一化，避免模长主导 alpha
        audio_feat = l2_normalize(audio_feat)
        imu_feat = l2_normalize(imu_feat)

        feat = torch.cat([audio_feat, imu_feat], dim=1) # [B, 512]

        # ✅ 方法一：直接拼接
        # out = self.classifier2(feat) # 两个模态的信息原样拼接

        # ✅ 方法二：gating 融合 
        alpha = self.gate_layer(feat)                    # [B, 1]
        fused = alpha * audio_feat + (1 - alpha) * imu_feat    # [B, 256]  
        # 每个都试一下
        # out = self.classifier1(fused)  # 考虑作为 baseline、但一旦 alpha 学不好，没法补救
        # out = self.classifier2(torch.cat([fused, imu_feat], dim=1))  #强调弱模态，鼓励 IMU 被用； 将融合向量 fused 和原始 imu_feat 拼接后送入分类器：final_input = [ fused || imu_feat ] → [B, 256 + 256]
        # out = self.classifier2(torch.cat([fused, audio_feat], dim=1)) # 
        out = self.classifier3(torch.cat([fused, audio_feat, imu_feat], dim=1))  # [B, 256 + 256 + 256]

        # 方法三考虑使用动态的 gate
        if return_alpha:
            return out, alpha  # ← 这是 tuple
        else:
            return out         # ← 正确是 Tensor
