project/  
<!-- 运行耗时太长，目前最好直接加载 HF 数据集  
├── preprocess_one_sample.py         # 💾 预处理脚本：单样本存pt文件  
├── preprocess_save.py               # 💾 预处理脚本： 一次性提取特征 mel / imu / label 并保存 .pt 文件  
　　　　├── dataset_hf.py                # 包含 EarVASDatasetFromHF   
-->  
├── train.py                         # 主训练脚本（含 main() 和评估函数）  
<!--
　　　　├── dataset_lazy_split.py        # 🧊 懒加载 Dataset，从多样本的 .pt 文件中读取样本，与preprocess_save.py配合使用  
　　　　├── dataset_cached.py             # 🧊 加载单样本的pt文件，配合preprocess_one_sample.py 使用【有bug】
-->
　　　　├── dataset_hf.py                  # 📦 HuggingFace Dataset 适配类（EarVASDatasetFromHF）  
　　　　　　　　├── freRA_augment.py               # 🔁 FreRA 频域增强模块（用于 IMU）预处理阶段  
　　　　├── earvas_model_self.py             # 🧩 复刻的EarVAS模型（音频+IMU 双分支 + MLP）   
　　　　　　　　├── freRA_augment_in_train.py  # 🔁 FreRA 频域增强模块（用于 IMU）训练阶段
　　　　　　　　├── freRA_mm_cross.py          # FreRA 频域增强模块（用于 IMU 和 audio, 逐个通道处理）训练阶段

用于debug：
monitor.py
