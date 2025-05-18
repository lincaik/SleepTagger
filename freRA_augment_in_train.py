# freRA_augment_in_train.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FreRA(nn.Module):
    def __init__(self, F: int = 51):
        """
        - F: 频率维度，应与 IMU 的 rFFT 输出维度一致（T//2 + 1）
        """
        super().__init__()
        self.F = F
        self.s = nn.Parameter(torch.randn(F))  # 可学习频率掩码

    def forward(self, imu):  
        """
        imu: [B, C, T]，如 [batch, 6, 100]
        return: 增强后的 imu，形状仍为 [B, C, T]
        """
        B, C, T = imu.shape
        F = T // 2 + 1

        assert F == self.F, f"[FreRA] 输入时间帧 T={T} 导致 F={F} ≠ 训练设定 F={self.F}"

        # 频域变换
        imu_fft = torch.fft.rfft(imu, dim=2)  # [B, C, F]

        # # 【T 上长度不一定一致】
        # # 插值 self.s → 目标 F
        # if self.s.shape[0] != F:
        #     s_score = torch.sigmoid(F.interpolate(
        #         self.s.view(1, 1, -1), size=F, mode='linear', align_corners=True
        #     ).view(-1))
        # else:
        #     s_score = torch.sigmoid(self.s)
        # s_score = s_score.view(1, 1, -1)

        # 【长度必须一致】
        # 可学习频率重要性评分
        s_score = torch.sigmoid(self.s)           # [F]
        s_score = s_score.view(1, 1, F)           # [1, 1, F] for broadcasting

        # 重要频率保留；不重要的注入噪声扰动
        w_crit = (s_score > 0.5).float()  # 保留掩码 [1, 1, F]
        w_dist = torch.abs(s_score - 0.5) * 2 # 扰动强度 [0, 1]

        noise = torch.randn_like(imu_fft) * w_dist # 高斯噪声
        imu_fft_aug = imu_fft * w_crit + noise * (1 - w_crit)

        # 逆变换回时域
        imu_aug = torch.fft.irfft(imu_fft_aug, n=T, dim=2)

        return imu_aug  # [B, C, T]
