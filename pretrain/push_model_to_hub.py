import os
from huggingface_hub import HfApi, login

# 1. 绝对不要使用 hf-mirror 镜像！确保删除镜像环境变量
if "HF_ENDPOINT" in os.environ:
    del os.environ["HF_ENDPOINT"]

# 2. 挂上刚才测试成功的本地代理
# (如果你刚才测试时改了端口，请务必在这里也改成你刚才测试成功的端口！)
os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

MY_HF_TOKEN = os.environ.get("HF_TOKEN")
if not MY_HF_TOKEN:
    raise RuntimeError("请先设置 Hugging Face token: export HF_TOKEN=你的token，不要把 token 写进代码。")

def push_model_folder_to_hf(local_dir, repo_id, commit_message="Upload model"):
    # 强制在代码层面登录
    login(token=MY_HF_TOKEN)
    
    # 明确绑定 Token 初始化 API
    api = HfApi(token=MY_HF_TOKEN)
    
    print(f"📦 准备将本地文件夹: {local_dir} 上传至 Hugging Face Hub...")
    print(f"🎯 目标仓库: https://huggingface.co/{repo_id}")
    
    # 1. 创建仓库 (如果已存在则跳过)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)                                                                                                                                       
    
    # 2. 上传文件夹
    print("🚀 正在拼命上传中，这可能需要一点时间，请耐心等待...")
    api.upload_folder(
        folder_path=local_dir, # 本地的文件夹路径
        repo_id=repo_id,       # 目标仓库 ID
        repo_type="model",
        path_in_repo="pretrained_model",  # 在仓库中新建的子文件夹
        commit_message=commit_message,
    )
    
    print("✅ 模型上传成功！")

if __name__ == "__main__":
    # 你的本地模型文件夹路径，末尾一定要有pretrained_model
    LOCAL_MODEL_DIR = "outputs/pretrain/train/2026-05-18/21-39-48_SewNeedle-3Arms-v0_pre_zed_wrist_act/checkpoints/032000_loss=0.0729_sr=0.0_ar=-113.35/pretrained_model"
    
    # 你的目标仓库
    HF_REPO_ID = "Dc-dc/pre_sim_sew_needle_3arms_zed_wrist_act"
    
    push_model_folder_to_hf(LOCAL_MODEL_DIR, HF_REPO_ID)
