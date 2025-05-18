# monitor.py
import psutil
import GPUtil
import time
from datetime import datetime

def monitor_system(interval=60, log_file="monitor_log.txt"):
    with open(log_file, "a") as f:
        while True:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n===== Monitor at {now} =====\n")

            # GPU 状态
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    f.write(f"[GPU {i}] Util: {gpu.load*100:.1f}% | Mem: {gpu.memoryUsed:.1f}/{gpu.memoryTotal:.1f} MB\n")
            except Exception as e:
                f.write(f"GPU info unavailable: {e}\n")

            # CPU 状态
            cpu_percent = psutil.cpu_percent(interval=1)
            f.write(f"[CPU] Usage: {cpu_percent:.1f}%\n")

            # 内存状态
            mem = psutil.virtual_memory()
            f.write(f"[RAM] Used: {mem.used/1e9:.2f} GB / {mem.total/1e9:.2f} GB | {mem.percent}%\n")

            f.flush()
            time.sleep(interval - 1)  # 已经 sleep 过1秒 for cpu_percent
