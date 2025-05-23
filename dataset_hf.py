# 从 HuggingFace 数据集中实时提取 Mel + IMU
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset
# from freRA_augment import FreRA # 预处理的增强模块

class EarVASDatasetFromHF(Dataset):
    def __init__(self, hf_dataset, n_mels=128, target_time_frames=512, imu_frames=100, apply_freRA=False):
        self.dataset = hf_dataset
        self.n_mels = n_mels
        self.target_time_frames = target_time_frames
        self.imu_frames = imu_frames
        self.counter = 0 # 只是为了debug,给异常样本计数
        self.apply_freRA = apply_freRA

        # 初始化 FreRA 增强模块（只在训练集激活）
        if self.apply_freRA:
            self.freRA = FreRA().to("cuda" if torch.cuda.is_available() else "cpu") # self.freRA = FreRA()

    def extract_mel(self, y, path=None, sr=16000): # 🎧 audio，梅尔频谱提取与标准化，含异常检测和填充/截断逻辑
        if not isinstance(y, np.ndarray): # 存在异常样本：单声道
            self.counter += 1
            # print(f"[{self.counter}] ⚠️ [extract_mel] 非法输入类型: {type(y)} | 文件路径: {path}") # 有75个audio，出现在home/v-wangzeyu/skywang/DreamCatcher_cropped/data/test/breathe
            return np.zeros((self.n_mels, self.target_time_frames), dtype=np.float32)  # 返回默认特征 

        if len(y) < 400:  # 存在异常样本：原始音频波形长度太短，连最小的 Mel 频谱都无法计算（librosa 直接报错）
        # 处理过短音频（<400采样点时生成全零频谱）
            # print(f"[{self.counter}] ⚠️ [extract_mel] 太短了: len(y) = {len(y)} | 文件路径: {path}")
            return np.zeros((self.n_mels, self.target_time_frames), dtype=np.float32) # 全 0 的 Mel 频谱张量，shape 是 (128, 512)（频率 × 时间帧）

        # 将双通道音频转换为梅尔频谱图
        # 将波形转为 Mel 频谱（功率谱）。常用参数：
        #     n_fft=400: 对应 25ms 窗口
        #     hop_length=160: 对应 10ms 步长
        #     n_mels: 频率维度数
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, n_mels=self.n_mels, window='hann') # 音频太短，librosa 根本无法计算 Mel 频谱，会直接报错或返回空值。比如说音频甚至小于400个点，ffts 计算都无法进行。
        mel_db = librosa.power_to_db(mel, ref=np.max) # 将功率谱转为对数尺度（分贝），便于模型学习。

        # Pad / truncate to fixed length (T = 512)
        # 对时间维度进行统一处理：
        #     太短的补零
        #     太长的截断为 512 帧
        # 最后返回 shape 为 (128, 512) 的 Mel 特征图。
        T = mel_db.shape[1]
        if T < self.target_time_frames:
            mel_db = np.pad(mel_db, ((0, 0), (0, self.target_time_frames - T)), mode='constant')
        else:
            mel_db = mel_db[:, :self.target_time_frames]
        return mel_db  # shape: (128, 512)

    def __getitem__(self, idx):
        sample = self.dataset[idx] # 从 HuggingFace 数据集中取出第 idx 个样本，是一个字典
        audio_array = sample["audio"]["array"]  # shape: (2, N)
        imu_array = sample["imu"]["array"]  # (7, 98)
        label = sample["label"]
        path = sample["path"]  # 路径

        # 🎧 audio
        mel0 = self.extract_mel(audio_array[0], path)  # 👈 传入是第一个通道
        mel1 = self.extract_mel(audio_array[1], path)  # 👈 传入是第二个通道
        mel_tensor = torch.tensor(np.stack([mel0, mel1]), dtype=torch.float32)  # (2, 128, 512)
        mel_tensor = mel_tensor.permute(0, 2, 1) # 重排维度为 (2, 512, 128)，方便模型输入，匹配 EarVAS 的通道 × 时间 × 频率 的设计。

        # 📈 imu
        imu_tensor = torch.tensor(imu_array[0:6], dtype=torch.float32)  #debug：imu_array[1:7] # 形状是 (6, 98)
        if imu_tensor.shape[1] < self.imu_frames: # 统一时间帧长度到 imu_frames，短了补零，长了截断。
            imu_tensor = torch.nn.functional.pad(imu_tensor, (0, self.imu_frames - imu_tensor.shape[1]))
        else:
            imu_tensor = imu_tensor[:, :self.imu_frames]

        if self.apply_freRA:
            imu_tensor = imu_tensor.unsqueeze(0).to(self.freRA.device)
            imu_tensor = self.freRA(imu_tensor).squeeze(0).detach().cpu()

        # 🎯 label
        label_tensor = torch.tensor(label)

        return mel_tensor, imu_tensor, label_tensor

    def __len__(self):
        return len(self.dataset)
