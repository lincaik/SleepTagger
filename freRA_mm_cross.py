import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEncoder(nn.Module):
    """一个轻量级编码器：用于从音频或IMU中提取模态特征"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)  # shape: [B, output_dim]


def gumbel_softmax_sample(logits, temperature=0.1):
    noise = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
    y = logits + gumbel_noise
    return torch.sigmoid(y / temperature)  # Approx. binary with gradient


class FreRA_MM_Cross(nn.Module):
    def __init__(self, imu_F=51, mel_F=128, feature_dim=64):
        super().__init__()
        self.imu_F = imu_F
        self.mel_F = mel_F

        self.imu_encoder = SimpleEncoder(input_dim=6, hidden_dim=32, output_dim=feature_dim)
        self.audio_encoder = SimpleEncoder(input_dim=2, hidden_dim=32, output_dim=feature_dim)

        self.imu_to_mask = nn.Linear(feature_dim, imu_F * 6)  # 每个 IMU 通道一个掩码
        self.audio_to_mask = nn.Linear(feature_dim, mel_F * 2)  # 每个 audio 通道一个掩码

    def forward(self, imu, mel):
        """
        imu:  [B, 6, 100]
        mel:  [B, 2, 512, 128]  → 按频率轴 rFFT
        return: imu_aug, mel_aug, s_audio, s_imu
 
        imu shape: torch.Size([128, 6, 100])
        mel shape: torch.Size([128, 2, 512, 128])
        """
        B, C_imu, T = imu.shape 
        _, C_audio, _, F_audio = mel.shape 
        mel = mel.permute(0, 1, 3, 2)  # [B, 2, 128, 512]

        # Step 1: encode features
        feat_imu = self.imu_encoder(imu)  # [B, d]
        feat_audio = self.audio_encoder(mel.mean(dim=-1))  # mean over time → [B, d]

        # Step 2: predict masks (logits)
        s_audio_logits = self.audio_to_mask(feat_audio).view(B, C_audio, self.mel_F)  # [B, 2, F]
        s_imu_logits = self.imu_to_mask(feat_imu).view(B, C_imu, self.imu_F)          # [B, 6, F]

        # Step 3: Sample retain masks using Gumbel-Softmax
        s_audio_mask = gumbel_softmax_sample(s_audio_logits).unsqueeze(-1)  # [B, 2, F, 1]
        s_imu_mask = gumbel_softmax_sample(s_imu_logits)  # [B, 6, F]

        # Step 4: IMU: rFFT + per-channel mask from audio
        imu_fft = torch.fft.rfft(imu, dim=2)  # [B, 6, F]
        noise = torch.randn_like(imu_fft)
        imu_fft_aug = imu_fft * s_imu_mask + (1 - s_imu_mask) * noise
        imu_aug = torch.fft.irfft(imu_fft_aug, n=T, dim=2)

        # Step 5: Mel: rFFT + per-channel mask from IMU
        mel_fft = torch.fft.rfft(mel, dim=2)  # [B, 2, F, T]
        noise_mel = torch.randn_like(mel_fft)
        mel_fft_aug = mel_fft * s_audio_mask + (1 - s_audio_mask) * noise_mel
        mel_aug = torch.fft.irfft(mel_fft_aug, n=mel.shape[2], dim=2)
        mel_aug = mel_aug.permute(0, 1, 3, 2)  # → 回到 [B, 2, 512, 128]

        # Sigmoid for logging/visualization only
        s_audio = torch.sigmoid(s_audio_logits)
        s_imu = torch.sigmoid(s_imu_logits)

        return imu_aug, mel_aug, s_audio, s_imu
