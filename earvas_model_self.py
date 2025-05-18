# earvas_model.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from freRA_augment_in_train import FreRA 

class EarVAS(nn.Module):
    def __init__(self, num_classes=9, use_freRA=False ):
        super().__init__()

        self.use_freRA = use_freRA
        if self.use_freRA:
            self.freRA = FreRA(F=51)  # F 根据 IMU 长度设定，T=100 → F=51

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

        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, audio, imu):
        audio_feat = self.audio_model(audio)
        audio_feat = self.audio_proj(audio_feat)

        if self.use_freRA and self.training: 
            imu = self.freRA(imu)  # → [B, 6, 100] 
        imu_feat = self.imu_branch(imu)
        
        feat = torch.cat([audio_feat, imu_feat], dim=1)
        out = self.classifier(feat)
        return out
