from huggingface_hub import HfApi

def push_model_folder_to_hf(local_dir, repo_id, commit_message="Upload model"):
    api = HfApi()
    
    print(f"📦 准备将本地文件夹: {local_dir} 上传至 Hugging Face Hub...")
    print(f"🎯 目标仓库: https://huggingface.co/{repo_id}")
    
    # 1. 确保仓库存在，如果不存在则自动创建 (repo_type="model" 表示这是模型仓库)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    
    # 2. 上传整个文件夹
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        # ignore_patterns=["*.log", "eval_videos/"] # 如果有不想上传的临时文件可以忽略
    )
    
    print("✅ 模型上传成功！")

if __name__ == "__main__":
    # 替换为你实际的本地模型文件夹路径
    LOCAL_MODEL_DIR = "outputs/pretrain/train/2026-05-13/23-21-10_SewNeedle-3Arms-v0_pre_zed_wrist_diffusion/checkpoints/174000_loss=0.0018_sr=70.0_ar=545.44/pretrained_model"
    
    # 替换为你的 Hugging Face 用户名和你想取的仓库名
    HF_REPO_ID = "Dc-dc/pre_zed_wrist_diffusion" # 例如: "iantc104/sew-needle-3arms-diffusion"
    
    push_model_folder_to_hf(LOCAL_MODEL_DIR, HF_REPO_ID)