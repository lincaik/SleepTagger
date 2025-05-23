# train.py
from datasets import load_dataset
from earvas_model_self import EarVAS                 # 复刻的模型，有差异
# from EarVAS_models import EarVAS                 # 来自开源项目 
from dataset_hf import EarVASDatasetFromHF      # 自己写的，处理数据源
# from dataset_lazy_split import CachedEarVASDataset
# from dataset_cached import CachedEarVASDataset

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import multiprocessing
from monitor import monitor_system
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import numpy as np
from fvcore.nn import FlopCountAnalysis, parameter_count
import pandas as pd
import os
from sklearn.preprocessing import label_binarize
from fvcore.nn import FlopCountAnalysis, parameter_count
import datetime 
import shutil 
import time
import torch.multiprocessing as mp 

import random

# ========= 配置区域 =========
use_freRA_in_train = True 
batch_size = 128  
num_epochs = 15
return_alpha = True # 是否返回 alpha 分数

# Early Stopping 配置
early_stop_patience = 5           # 忍耐轮数
warmup_epochs = 10                # 前几轮不判断早停

# ========= 初始化，基本不用改动 =========
# Early Stopping 相关
val_loss_history = []            # 存放最近 val loss
max_history = early_stop_patience
no_improve_epochs = 0            # 未改善计数器

# 与训练无关
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M") # 获取当前时间作为子文件夹，例如 '2025-05-09_0320'
save_dir = os.path.join('model', 'earvas_self', timestamp) # 创建带时间戳的保存目录

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", None)
if not PROJECT_ROOT:
    PROJECT_ROOT = os.path.abspath(".")
#=========

def set_seed(seed=42):
    random.seed(seed)                          # Python 原生随机数
    np.random.seed(seed)                       # NumPy 随机数
    torch.manual_seed(seed)                    # PyTorch CPU 随机数
    torch.cuda.manual_seed(seed)               # GPU 随机数（单GPU足够）
    torch.backends.cudnn.deterministic = True  # 使用确定性算法
    torch.backends.cudnn.benchmark = False     # 关闭自动优化算法（提高可重复性）

def main(): 
    set_seed(42)  # 固定随机种子，提高可复现性
    mp.set_start_method("spawn", force=True) # 设置多进程启动方法
    num_workers = min(16, os.cpu_count() // 2)   # 8 ??? 改成min(config.num_workers, os.cpu_count() // 2)
    print("num_workers:", num_workers) 
 
# 加载数据集
    print("\n==================== 加载 HF 数据集 ====================")
    dataset = load_dataset("THU-PI-Sensing/DreamCatcher", "sleep_event_classification", trust_remote_code=True)
    label_names = dataset["train"].features["label"].names
    # train_dataset = EarVASDatasetFromHF(dataset["train"]) 
    # val_dataset = EarVASDatasetFromHF(dataset["validation"])
    # test_dataset = EarVASDatasetFromHF(dataset["test"])
    train_dataset = EarVASDatasetFromHF(dataset["train"], apply_freRA=False)
    val_dataset   = EarVASDatasetFromHF(dataset["validation"], apply_freRA=False)
    test_dataset  = EarVASDatasetFromHF(dataset["test"], apply_freRA=False)
 
    
    # print("\n==================== 读取缓存数据集 ====================")
    # preprocesssed_data_dir = "preprocessed_DC/1" # 10000样本一个pt
    # preprocesssed_data_dir = "preprocessed_DreamCatcher" # 单条样本一个pt

    # 提前挪到本地 SSD，减少 NFS 读取时间【好像效果不大】
    # if "TMPDIR" in os.environ: # 检查是否在 GLab 环境（是否有本地 SSD 临时目录）
    #     tmp_data_root = os.environ["TMPDIR"]
    #     tmp_data_dir = os.path.join(PROJECT_ROOT, tmp_data_root, "preprocessed_DC")

    #     if not os.path.exists(tmp_data_dir):
    #         print(f"📦 复制预处理数据到本地 SSD: {tmp_data_dir}（这可能需要几分钟）")
    #         t0 = time.time()
    #         shutil.copytree("preprocessed_DC", tmp_data_dir)
    #         print(f"✅ 完成复制，用时 {time.time() - t0:.1f} 秒")
    #     else:
    #         print(f"📂 本地 SSD 已存在预处理数据: {tmp_data_dir}")
    # else:
    #     tmp_data_dir = "preprocessed_DC"
    #     print("⚠️ 未检测到 TMPDIR，仍然使用 NFS 目录") 
    # 使用本地路径构造 Dataset  ✅ 针对lazy_split
    # preprocesssed_data_dir = os.path.join(PROJECT_ROOT, tmp_data_dir, "1")  # 注意 原始目录结构是 preprocessed_DC/1

    # 针对dataset_cached
    # train_dataset = CachedEarVASDataset(preprocesssed_data_dir, "train") # 使用缓存数据集
    # val_dataset = CachedEarVASDataset(preprocesssed_data_dir, "validation")
    # test_dataset = CachedEarVASDataset(preprocesssed_data_dir, "test")
    # ✅ 针对lazy_split
    # train_dataset = CachedEarVASDataset(preprocesssed_data_dir, "train", apply_freRA=True) # 使用缓存数据集
    # val_dataset = CachedEarVASDataset(preprocesssed_data_dir, "validation", apply_freRA=False)
    # test_dataset = CachedEarVASDataset(preprocesssed_data_dir, "test", apply_freRA=False)


# 构建 PyTorch Dataset 和 DataLoader
    print("\n==================== 构建PyTorch Dataset 和 DataLoader ====================")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,     # 使用 CPU 核心数的一半
        pin_memory=True
    ) 
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, # 64,   # 你可以设置等于训练 batch_size
        num_workers=num_workers,
        pin_memory=True
    )

#     # 初始化模型：测试最简单的模型
#         # model = EarVAS(num_classes=9)
#         # # 测试 forward
#         # for audio, imu, label in train_loader:
#         #     print("🎧 audio shape:", audio.shape)  # (4, 2, 512, 128)
#         #     print("📈 imu shape:", imu.shape)      # (4, 6, 100)
#         #     print("🎯 label:", label.shape)        # (4, )
            
#         #     output = model(audio, imu)
#         #     print("🧠 model output:", output.shape)  # (4, 9)
#         #     break

# 初始化
    print("\n==================== 初始化设备和模型 ====================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EarVAS(num_classes=9, use_freRA=use_freRA_in_train).to(device)     # ✅ 使用自制 EarVAS 模型
    # 返回的是一个 PyTorch 模型对象，继承自 torch.nn.Module, 构造了一个神经网络结构图
    # 将模型移动到指定设备
    print("✅ 当前设备：", device)
    print("✅ 模型设备：", next(model.parameters()).device)

    # ✅ 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ✅ 评估函数
    @torch.no_grad()
    def evaluate_model(model, val_loader, device):
        print("\n==================== 进入评估函数 evaluate_model ====================")
        model.eval()
        model.to(device)  # 确保模型也在 GPU
        print("evaluate_model 模型当前所在设备：", next(model.parameters()).device)
        all_preds, all_labels = [], []

        total_loss = 0.0
        for step, (audio, imu, labels) in enumerate(val_loader):
            # if step >= 30: # 👈 只跑前 3 个 batch, 以便debug
            #     break   
            audio, imu, labels = audio.to(device), imu.to(device), labels.to(device)

            # # ✅ 只在第一个 batch 分析模态范数差异
            # if step == 0:
            #     try:
            #         # Audio 特征
            #         audio_feat = model.audio_model(audio)
            #         audio_feat = model.audio_proj(audio_feat)

            #         # IMU 特征
            #         if hasattr(model, "freRA") and model.use_freRA:
            #             imu = model.freRA(imu)
            #         imu_feat = model.imu_branch(imu)

            #         audio_norm = audio_feat.norm(dim=1)
            #         imu_norm = imu_feat.norm(dim=1)

            #         print(f"\n🎧 audio_feat mean norm: {audio_norm.mean().item():.4f}")
            #         print(f"📈 imu_feat mean norm:   {imu_norm.mean().item():.4f}")
            #         print(f"📏 平均差值:             {(audio_norm.mean() - imu_norm.mean()).item():.4f}")
            #         print(f"📏 平均绝对差（样本级）: {(audio_norm - imu_norm).abs().mean().item():.4f}")
            #     except Exception as e:
            #         print("⚠️ 模态范数分析失败：", e)

            # 正常推理
            outputs, alpha = model(audio, imu, return_alpha=return_alpha) # outputs = model(audio, imu)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # 为了早停而计算loss
            loss = criterion(outputs, labels)  
            total_loss += loss.item()

            # 每个 epoch 只打印一次 alpha 分布（第一个 batch）
            if step == 0:
                print(f"📊 alpha: mean={alpha.mean():.4f}, min={alpha.min():.4f}, max={alpha.max():.4f}")
        print("evaluate_model 内循环完了")

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        mcc = matthews_corrcoef(all_labels, all_preds)

        num_classes = dataset["train"].features["label"].num_classes
        labels = list(range(num_classes))
        cm = confusion_matrix(all_labels, all_preds, labels=labels)

        avg_val_loss = total_loss / len(val_loader)   # 为了早停 新增
        # 🎯 Macro-AUC
        try:
            y_true_bin = label_binarize(all_labels, classes=list(range(num_classes)))
            y_pred_bin = label_binarize(all_preds, classes=list(range(num_classes)))
            auc = roc_auc_score(y_true_bin, y_pred_bin, average="macro", multi_class="ovr")
        except:
            auc = -1  # fail-safe

        return {
            "accuracy": acc,
            "f1": f1,
            "confusion_matrix": cm,
            "mcc": mcc,
            "auc": auc,
            "val_loss": avg_val_loss               # 👈 新增
        }
 

# 如果想只用test，注释掉这部分-----
    # 找最佳模型
    best_f1 = 0.0
    best_acc = 0.0
    best_val_loss = float("inf") 
    
    dirs = os.path.join(PROJECT_ROOT, save_dir)
    os.makedirs(dirs, exist_ok=True)

    # ✅ 开始训练循环
    print("\n==================== 进入train ====================")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", mininterval=20, maxinterval=60)

        for step, (audio, imu, labels) in enumerate(loop):
            # if step >= 300:  # 👈 只跑前 3 个 batch, 以便debug
            #     break

            t0 = time.time() 
            #  数据转移到 GPU
            audio = audio.to(device)     # [B, 2, 512, 128]
            imu = imu.to(device)         # [B, 6, 100]
            labels = labels.to(device)   # [B]
            t1 = time.time()

            optimizer.zero_grad()

            # ✅ 前向传播
            outputs = model(audio, imu)  # [B, 9]
            loss = criterion(outputs, labels)
            t2 = time.time()

            # ✅ 反向传播 + 更新
            loss.backward()
            optimizer.step()
            t3 = time.time()

            total_loss += loss.item()

            #  打印详细耗时 Epoch 1/10:   2%|▏ | 42/2313 [09:31<7:47:56, 12.36s/batch, loss=0.8083, load=0.00s, fwd=0.01s, bwd=0.01s]  问题不大
            if step % 10 == 0:
                loop.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "load": f"{t1 - t0:.2f}s",
                    "fwd": f"{t2 - t1:.2f}s",
                    "bwd": f"{t3 - t2:.2f}s"
                }) 
            # loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] ✅ Average Loss: {avg_loss:.4f}")

        # ✅ 验证集评估
        results = evaluate_model(model, val_loader, device)
        acc, f1, cm = results["accuracy"], results["f1"], results["confusion_matrix"]
        df_cm = pd.DataFrame(cm, index=label_names, columns=label_names)
            
        print(f"[Epoch {epoch+1}] 🎯 Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print(f"[Epoch {epoch+1}] 📊 Confusion Matrix:\n{cm}")

        val_loss = results.get("val_loss", 0)   # 为了早停 初始化
        
        # ✅ 保存最佳模型
        if f1 > best_f1: # 以f1为准
            best_f1 = f1
        # if acc > best_acc: # 以acc为准
            # best_acc = acc
            no_improve_epochs = 0 
            save_path = os.path.join(PROJECT_ROOT, save_dir, f"epoch{epoch+1}_f1_{f1:.4f}_acc_{acc:.4f}.pt")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'f1': f1,
                'accuracy': acc
            }, save_path)
            print(f"💾 Saved best model at epoch {epoch+1} with F1 = {f1:.4f}, accuracy = {acc:.4f}")
        else:
            # 本次模型效果下降
            no_improve_epochs += 1
            print(f"[Epoch {epoch+1}] ❌ F1无提升（+{f1 - best_f1:.4f}），忍耐计数: {no_improve_epochs}/{early_stop_patience}")

            # Early stopping 判断（跳过 warmup）
            if epoch + 1 <= warmup_epochs: 
                print(f"[Epoch {epoch+1}] ⏳ Warmup中，不判断早停")
                continue

            # ➕ 记录 val_loss 趋势
            val_loss_history.append(val_loss)
            if len(val_loss_history) > max_history:
                val_loss_history.pop(0) 
            
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss # 以整个训练过程的最小 loss 为最佳。
            # 🎯 Early stop 条件：F1 连续不提升 + val_loss 上升
            if no_improve_epochs >= early_stop_patience and all(x >= best_val_loss for x in val_loss_history):
                print("⏹️ Early stopping: F1 无改善，且 val_loss 持续未下降")
                break
# ---
    print("\n==================== 进入test ====================")
    test_loader = DataLoader(test_dataset, batch_size=batch_size)  # 32

    def safe_load_model(model, path, device="cpu"):
        try: # 尝试直接加载为 state_dict
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✅ 直接加载模型权重成功：{path}")
        except RuntimeError as e:
            print("⚠️ 尝试直接加载失败，可能是 checkpoint 包含额外信息，是字典，切换到 checkpoint['model_state'] 方式")
            try:
                checkpoint = torch.load(path, map_location=device)
                model.load_state_dict(checkpoint["model_state"])
                print(f"✅ 从 checkpoint['model_state'] 加载成功：{path}")
            except Exception as e2:
                print(f"❌ 模型加载失败：{e2}")
                raise RuntimeError("模型文件格式不兼容，请确认是否保存了正确的 state_dict 或 checkpoint")
        return model

    # ✅ 加载最优模型
    # model_path = "epoch7_f1_0.4924_acc_0.7277.pt"
    # save_path = os.path.join(PROJECT_ROOT, 'model/earvas_self/2025-05-09_0522', model_path)
    print(f"\n==================== 加载最优模型 echo总次数:{num_epochs} 模型名称:{save_path}====================")
    # print("🔥 PyTorch 编译时用的 CUDA 版本", torch.version.cuda) # PyTorch 编译时用的 CUDA 版本
    # print("🔥 是否检测到 GPU:", torch.cuda.is_available())
    # print("🔥 当前可用 GPU 数量:", torch.cuda.device_count())
    # print("🔥 当前使用设备:", device)

    model = safe_load_model(model, save_path, device)
    model.to(device)

    # ✅ 添加 FLOPs & 参数量统计（建议在 test 前只计算一次）
    dummy_audio = torch.randn(1, 2, 512, 128).to(device)
    dummy_imu = torch.randn(1, 6, 100).to(device)

    flops = FlopCountAnalysis(model, (dummy_audio, dummy_imu))
    params = parameter_count(model)[""]

    print(f"🧮 FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    print(f"🧮 Params: {params / 1e6:.2f} M")

    # ✅ 测试评估
    print("\n==================== 测试的评估 ing ====================") 
    results = evaluate_model(model, test_loader, device)

    df_cm = pd.DataFrame(results['confusion_matrix'], index=label_names, columns=label_names)

    print("\n==================== 🧪 Final Evaluation ====================")
    print(f"{save_path}\n")
    print(f"🎯 Accuracy:   {results['accuracy']:.4f}")
    print(f"🎯 Macro-AUC:  {results['auc']:.4f}")
    print(f"🎯 Macro-F1:   {results['f1']:.4f}")
    print(f"🎯 MCC:        {results['mcc']:.4f}")
    print(f"🧮 FLOPs:      {flops.total() / 1e9:.2f} GFLOPs")
    print(f"🧮 Params:     {params / 1e6:.2f} M")
    print(f"📊 Test Confusion Matrix:\n{df_cm}")

    # 保存 confusion matrix
    confusion_csv_path = os.path.join(PROJECT_ROOT, "confusion_matrix.csv")
    df_cm.to_csv(confusion_csv_path, index=True)
    # 保存评估指标
    metrics_path = os.path.join(PROJECT_ROOT, "metrics.txt")
    with open(metrics_path, "a") as f:
        f.write(f"{save_path}\n")
        f.write(f"Accuracy:   {results['accuracy']:.4f}\n")
        f.write(f"Macro-AUC:  {results['auc']:.4f}\n")
        f.write(f"Macro-F1:   {results['f1']:.4f}\n")
        f.write(f"MCC:        {results['mcc']:.4f}\n")
        f.write(f"FLOPs:      {flops.total() / 1e9:.2f} GFLOPs\n")
        f.write(f"Params:     {params / 1e6:.2f} M\n")
        f.write(f"Confusion Matrix:\n{df_cm.to_string()}\n")

if __name__ == "__main__":  
    print("\n==================== 开始后台监控 ====================") 
    MONITOR_FILE = os.path.join(PROJECT_ROOT, f"monitor_log/monitor_log_{timestamp}.txt")
    monitor_process = multiprocessing.Process(target=monitor_system, args=(60, MONITOR_FILE )) # 每 60 秒记录一次状态
    monitor_process.start()

    try: 
        main()
    except Exception as e:
        print("❌ 报错啦：", e)
        raise
    finally:  # 无论是否报错，都终止子进程
        #  停止系统监控
        monitor_process.terminate()
        monitor_process.join()
        print("\n==================== 脚本结束 ====================")