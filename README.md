project/
<!-- 
├── preprocess_one_sample.py         # 💾预处理脚本：单样本存pt文件
├── preprocess_save.py               # 💾 预处理脚本： 一次性提取特征 mel / imu / label 并保存 .pt 文件
	├── dataset_hf.py                # 包含 EarVASDatasetFromHF 
    -->
├── train.py                         # 主训练脚本（含 main() 和评估函数）
	├── dataset_lazy_split.py        # 🧊 懒加载 Dataset，从分块 .pt 文件中读取样本
├── earvas_model_self.py             # 🧩 EarVAS 模型定义（音频+IMU 双分支 + MLP） 
		├── freRA_augment_in_train.py   # 🔁 FreRA 频域增强模块（用于 IMU）训练阶段
	├── dataset_hf.py                  # 📦 HuggingFace Dataset 适配类（EarVASDatasetFromHF）
		├── freRA_augment.py               # 🔁 FreRA 频域增强模块（用于 IMU）预处理阶段