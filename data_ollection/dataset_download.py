'''
从 huggingface 的官方镜像站下载数据
'''
import os
import time

# 🚨 必须放在最前面：强行指向国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 强行增加底层 requests 的超时容忍度 (单位秒)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60" 

from huggingface_hub import snapshot_download

repo_id = "iantc104/gv_sim_insert_peg_3arms"
local_dir = "src/av-aloha/data_collection_scripts/data/sim_insert_peg/3arms"

print(f"🚀 开始定向拉取 {repo_id} 中的 videos/ 目录...")

max_retries = 100 # 给它 100 次重试机会

for attempt in range(max_retries):
    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns="videos/*",
            local_dir=local_dir,
            resume_download=True, # 核心魔法：断网后自动从上一次断开的地方继续
            max_workers=2         # 🚨 极度关键：把 8 降到 2，防止并发过高导致带宽塞车超时
        )
        print(f"\n🎉 恭喜！视频数据已全部完整下载至: {downloaded_path}")
        break  # 下载成功，跳出循环

    except Exception as e:
        print(f"\n⚠️ 第 {attempt + 1} 次下载中断。底层原因: {e}")
        print("💡 别慌，正在触发断点续传机制，5秒后自动重新连接...")
        time.sleep(5)