import torch
import logging
import numpy as np
import imageio
import json
import shutil
from pathlib import Path
from contextlib import nullcontext
import gymnasium as gym
import yaml
from pathlib import Path
from tqdm import tqdm
# ==========================================
# 🌟 [新增] 自定义 Top-K 快照管理器(包含视频同步清理)
# ==========================================
class TopKCheckpointManager:
    """
    核心逻辑：
    1. 维护一个大小为 max_keep 的列表，按 loss 从小到大排序。
    2. 永远保留最新的 checkpoint（防止训练中断后无法续训最近的进度）。
    3. 自动扫描并删除既不在 top_k 列表，也不是 latest 的多余权重文件夹。
    """
    def __init__(self, out_dir: str, max_keep: int = 5, records_resume: bool = True):
        self.out_dir = Path(out_dir) if out_dir else Path("outputs")
        self.checkpoints_dir = self.out_dir / "checkpoints"  # 模型快照存放的总目录 
        self.eval_dir = self.out_dir / "eval"  # 评估视频存放的总目录
        self.max_keep = max_keep
        self.top_k = [] # 数据结构: [{"step": int, "loss": float, "path": Path}]
        self.latest_path = None
        self.records_file = self.checkpoints_dir / "top_k_records.json"
        self.records_resume = records_resume
        # 支持断点续训：每次实例化时从本地读取记录，保证跨 step 调用时不丢失历史信息
        if self.records_file.exists() and self.records_resume:
            try:
                with open(self.records_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.latest_path = Path(data.get("latest")) if data.get("latest") else None
                    self.top_k = [
                        {"step": item["step"], "loss": item["loss"], "path": Path(item["path"])} 
                        for item in data.get("top_k", [])
                    ]
            except Exception as e:
                logging.warning(f"⚠️ 无法读取 Top-K 记录，将重新开始统计: {e}")

    def update(self, step: int, loss: float, ckpt_path: Path):
        self.latest_path = ckpt_path
        # 防重入：如果当前路径已经在记录里了，先剔除旧的
        self.top_k = [item for item in self.top_k if item["path"].name != ckpt_path.name]
        # 将新的 checkpoint 加入 top-k 候选并排序
        self.top_k.append({"step": step, "loss": loss, "path": ckpt_path})
        self.top_k.sort(key=lambda x: x["loss"]) # loss 越小越好，排在前面
        
        # 如果超出了保留数量，把表现最差的剔除（仅仅是从内存列表中剔除）
        if len(self.top_k) > self.max_keep:
            self.top_k.pop(-1) 
            logging.info(f"🛡️ 候选列表完成 ({len(self.top_k)}/{self.max_keep})，去除loss最大的模型快照。")
        else:
            logging.info(f"🛡️ 候选列表还在收集中 ({len(self.top_k)}/{self.max_keep})，暂不执行硬盘清理。")
        if self.checkpoints_dir.exists():
            # 提取出合法的【文件夹名称】集合
            valid_names = {item["path"].name for item in self.top_k}
            if self.latest_path:
                valid_names.add(self.latest_path.name) # 最新模型必须保存

            for d in self.checkpoints_dir.iterdir():
                if d.is_dir() and d.name.split('_')[0].isdigit():
                    if d.name not in valid_names:
                        shutil.rmtree(d, ignore_errors=True)
                        logging.info(f"🗑️ 已清理未进入 Top-{self.max_keep} 的模型快照: {d.name}")
            
        
        # 2. 同步清理物理硬盘上的无用评估视频文件夹
        if self.eval_dir.exists():
            # 基于 valid_names 生成合法的视频文件夹名称
            valid_video_folder_names = {f"videos_{name}" for name in valid_names}
            
            for v_dir in self.eval_dir.iterdir():
                # 只清理以 "videos_" 开头，且后缀是数字（或带 loss 的数字）的文件夹
                if v_dir.is_dir() and v_dir.name.startswith("videos_"):
                    # 提取 videos_ 后面的部分判断是不是我们的目标文件夹
                    suffix = v_dir.name.replace("videos_", "")
                    if suffix.split('_')[0].isdigit():
                        if v_dir.name not in valid_video_folder_names:
                            shutil.rmtree(v_dir, ignore_errors=True)
                            logging.info(f"🗑️ 已同步清理失效模型的评估视频: {v_dir.name}")
                            
        self._save_records()

    def _save_records(self):
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        with open(self.records_file, "w", encoding="utf-8") as f:
            json.dump({
                "latest": str(self.latest_path) if self.latest_path else None,
                "top_k": [{"step": i["step"], "loss": i["loss"], "path": str(i["path"])} for i in self.top_k]
            }, f, indent=4, ensure_ascii=False)

def custom_eval_policy(env, policy, cfg_eval, videos_dir, device):
    """
    完全自主实现的评估代码。没有任何黑盒。
    接收标准 Gym 环境，处理图像归一化，跑策略推理，保存视频。
    """
    policy.eval() # 必须开启评估模式
    successes = []
    rewards = []

    # 用来动态记录实际保存的视频路径
    saved_video_paths = []
    
    videos_dir = Path(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # 从配置中动态读取参数，提供后备默认值防崩溃
    n_episodes = getattr(cfg_eval, "n_episodes", 10)
    max_rendered = getattr(cfg_eval, "max_episodes_rendered", 4)
    fps = getattr(cfg_eval, "fps", 25)
    max_steps = getattr(cfg_eval, "max_steps", 300)
    raw_camera = getattr(cfg_eval, "render_camera", 'overhead_cam')
    
    render_camera = raw_camera
    # 动作执行循环
    for ep in tqdm(range(n_episodes), leave=False):
        obs, _ = env.reset()
        done = False
        frames = []
        ep_reward = 0
        # LeRobot/DPPO 的 Policy 内置了 action chunking 队列
        policy.reset() # 清空模型的动作缓冲历史

        for step in range(max_steps):
            # 1. 如果还在需要渲染的额度内，才调用渲染 (提升非渲染 episode 的评估速度)
            if ep < max_rendered:
                frames.append(env.unwrapped.render(render_camera)) # gym创建需要加上 .unwrapped
                # frames.append(env.render(render_camera))  #直接创建环境不需要

            # 2. 手动处理格式，将 Numpy 字典转为 Tensor 字典送入模型
            batch = {}
            for k, v in obs.items():
                v_safe = v.copy() # 强制在内存中拷贝一份连续的数据防报错
                if "images" in k:
                    # 图片：[C, H, W] -> [1, C, H, W] -> 归一化
                    tensor_v = torch.from_numpy(v_safe).float().unsqueeze(0).to(device) / 255.0
                else:
                    # 状态：[21] -> [1, 21]
                    tensor_v = torch.from_numpy(v_safe).float().unsqueeze(0).to(device)
                batch[k] = tensor_v

            # 3. 推理获取动作
            with torch.no_grad():
                # 使用lerobot自带的推理函数
                action = policy.select_action(batch) # 这里每次取出一个动作，推理依旧一次生成8个动作，只是一个个往外取
            
            # 4. 把模型输出的 Tensor 动作转回 Numpy
            action_np = action.squeeze(0).cpu().numpy()

            # 5. 与环境交互
            obs, reward, terminated, truncated, info = env.step(action_np)
            ep_reward += float(reward)

            done = terminated or truncated

            if done:
                break
        
        # 记录指标
        successes.append(info.get("is_success", False))
        rewards.append(ep_reward)

        # 6. 根据配置的帧率和最大渲染数量保存视频
        if ep < max_rendered and len(frames) > 0:
            status = "Success" if successes[ep] else "Fail"
            # 格式：012500_reward=150.5_cam_overhead_ep_0.mp4
            video_name = f"{render_camera[0]}_ep_{ep}_reward={ep_reward:.1f}_{status}.mp4"
            video_path = videos_dir / video_name
            imageio.mimsave(str(video_path), frames, fps=fps)
            logging.info(f"🎥 保存视频: {video_path.name}")
            
            saved_video_paths.append(str(video_path))

    policy.train() # 恢复训练模式
    
    # 计算实际应该返回的视频列表长度
    actual_rendered = min(max_rendered, n_episodes)
    return {
        "aggregated": {
            "success_rate": float(np.mean(successes)),
            "average_reward": float(np.mean(rewards))
        },
        "video_paths": saved_video_paths
    }


def evaluate_and_checkpoint_if_needed(
    step, policy, optimizer, lr_scheduler, logger, cfg, device, out_dir, eval_env=None, train_loss=None, manager=None
):
    """
    主评估与保存入口
    """
    _num_digits = max(6, len(str(cfg.training.offline_steps))) 
    step_identifier = f"{step:0{_num_digits}d}" 

    if train_loss is not None:
        folder_identifier = f"{step_identifier}_loss={train_loss:.4f}"
    else:
        folder_identifier = step_identifier
    
    # 1. 评估逻辑 (优先读取 cfg.eval.eval_freq)
    eval_freq = getattr(cfg.eval, "eval_freq", 0)
    is_last_step = (step == cfg.training.offline_steps - 1)
    
    if (eval_freq > 0 and step > 0 and step % eval_freq == 0) or is_last_step:
        logging.info(f"📊 开始自主评估流程, 当前 Step: {step}")
        if eval_env is not None:
            video_dir = Path(out_dir) / "eval" / f"videos_{folder_identifier}"
            
            with torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext():
                # 传入完整的 cfg.eval 节点
                eval_info = custom_eval_policy(
                    env=eval_env,
                    policy=policy,
                    cfg_eval=cfg.eval, 
                    videos_dir=video_dir,
                    device=device
                )
            
            sr = eval_info["aggregated"]["success_rate"]
            ar = eval_info["aggregated"]["average_reward"]
            logging.info(f"✅ 评估完毕! 成功率: {sr*100:.1f}%, 平均奖励: {ar:.2f}")

            if getattr(cfg, "wandb", {}).get("enable", False) and len(eval_info["video_paths"]) > 0:
                logger.log_video(eval_info["video_paths"][0], step, mode="eval") # 只上传第一个视频到 wandb

    save_freq = getattr(cfg.training, "save_freq", 10000)
    if getattr(cfg.training, "save_checkpoint", False) and (step > 0 and step % save_freq == 0 or is_last_step):
        logging.info(f"💾 保存模型快照... Step: {step}")
        logger.save_checkpoint(step, policy, optimizer, lr_scheduler, identifier=folder_identifier)

        # ==========================================
        # 触发 Top-K 筛选与清理
        # ==========================================
        if train_loss is not None:
            
            ckpt_path = Path(out_dir) / "checkpoints" / folder_identifier
            if ckpt_path.exists():
                manager.update(step, train_loss, ckpt_path)
        else:
            logging.warning("⚠️ 警告: 未传入 train_loss，跳过 Top-K 模型清理逻辑，将保留所有权重。")

# =========================================================================
# 🌟 独立评估测试入口，推荐使用lerobot保存的快照格式，只需要给路径，环境配置会自动对齐
# =========================================================================
if __name__ == "__main__":
    import os
    import sys
    import torch
    from types import SimpleNamespace
    from contextlib import nullcontext
    from lerobot.common.policies.factory import make_policy
    from lerobot.common.utils.utils import get_safe_torch_device
    
    # 确保能够导入你的环境
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from env.sim_envs import SewNeedleEnv

    # ==========================================
    # 🎯 核心配置区：在这里自由修改你的评估参数！
    # ==========================================
    eval_cfg = SimpleNamespace(
        # 📂 模型路径设置 (直接指向 0000600_loss=0.1540 文件夹即可，代码会自动寻找内部结构)
        ckpt_path="outputs/finetune/train/2026-04-24/17-19-22_SewNeedle-2Arms-v0_ft_static_diffusion/checkpoints/000003_sr=0.30_reward=721.72",
        # ⚙️ 评估参数设置
        seed=100,
        n_episodes=100,             # 评估多少个任务                 
        max_episodes_rendered=10,  # 保存多少个视频 
        fps=25,                   # 视频帧率，和环境控制频率对齐
        max_steps=300,            # 每个任务的最大步数
        
        # 📷 相机设置
        render_camera=['overhead_cam']         # 保存video的相机视角              
    )
    USE_AMP = True  
    # ==========================================

    def main():
        ckpt_path = eval_cfg.ckpt_path
        from lerobot.common.utils.utils import set_global_seed
        set_global_seed(eval_cfg.seed)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"❌ 找不到权重路径: {ckpt_path}\n请检查路径是否正确。")
        # ==========================================
        # 🌟 1.自动探测 LeRobot 的 pretrained_model 子文件夹
        # ==========================================
        hf_model_dir = os.path.join(ckpt_path, "pretrained_model")
        if os.path.exists(hf_model_dir):
            print(f"🔍 检测到 LeRobot 标准快照结构，将自动读取子目录: pretrained_model")
            load_dir = hf_model_dir
        else:
            load_dir = ckpt_path # 兼容其他保存格式

        # 准备设备 (传入 "cuda" 防止新版本报错)
        device = get_safe_torch_device("cuda")
        print(f"🚀 初始化评估程序... 使用设备: {device}")

        
        # ==========================================
        # 🌟 2.实例化 Policy 并加载权重
        # ==========================================
        print(f"💾 正在从目录重建网络并加载权重: {load_dir}")
        try:
            #  直接使用 DiffusionPolicy 官方类，绕开所有版本不兼容的 API
            from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
            
            # 像加载常规 Hugging Face 模型一样，直接读取文件夹
            policy = DiffusionPolicy.from_pretrained(load_dir)
            
            # 手动推入 GPU
            policy.to(device)
        except Exception as e:
            raise RuntimeError(f"❌ 权重加载失败！详细报错: {e}")
        
        # ==========================================
        # 🌟 3.读取快照中的配置，使环境和训练时的对齐
        # ==========================================
        all_obs_keys = policy.config.input_shapes.keys()
        ref_cams = [k.replace("observation.images.", "") for k in all_obs_keys if "observation.images." in k]
        if not ref_cams:
            raise ValueError(f"❌ 严重冲突：模型中未找到相机相关参数。请检查模型输入是否正确。")
        
        obs_cameras = list(dict.fromkeys(ref_cams + eval_cfg.render_camera))
        # 动态读取环境元数据 (直接从 load_dir 读取)
        
        config_yaml_path = Path(load_dir) / "config.yaml"

        if config_yaml_path.exists():
            with open(config_yaml_path, "r") as f:
                full_cfg = yaml.safe_load(f)
                
                # 安全地从 YAML 的字典树中提取 env.name 和 env.task
                env_cfg = full_cfg.get("env", {})
                env_name = env_cfg.get("name", getattr(env_cfg, "name", "sim_envs"))
                env_task = env_cfg.get("task", getattr(env_cfg, "task", "SewNeedle-3Arms-v0"))
                logging.info(f"📦 成功从预训练文件夹读取完整环境配置: {env_name}/{env_task}")
        else:
            # 极限防呆后备
            env_name = getattr(eval_cfg, "name", "sim_envs")
            env_task = getattr(eval_cfg, "task", "SewNeedle-3Arms-v0")
            logging.warning(f"⚠️ 未找到 config.yaml，使用本地设定的后备环境: {env_name}/{env_task}")

        # 拼接 Gym ID
        env_id = f"{env_name}/{env_task}"
        logging.info(f"正在通过 Gym 注册表构建环境: {env_id}")
        # 使用 gym.make 创建环境，并通过 kwargs 强行覆盖你需要的相机
        eval_env = gym.make(
            id=env_id, 
            cameras=obs_cameras  # 👈 这里的传参会直接覆盖 __init__.py 里的默认套餐！
        )
        logging.info(f"✅ 环境加载成功！最终挂载的相机: {obs_cameras}")

        # ==========================================
        # 🌟 4.设置视频输出目录 (默认在 000000 文件夹同级新建 eval_videos 文件夹)
        # ==========================================
        videos_dir = os.path.join(ckpt_path, "eval_videos")
        print(f"🎬 开始测试! 录像将保存在: {videos_dir}")

        # ==========================================
        # 🌟 5.调用评估函数
        # ==========================================
        with torch.autocast(device_type=device.type) if USE_AMP else nullcontext():
            eval_info = custom_eval_policy(
                env=eval_env,
                policy=policy,
                cfg_eval=eval_cfg,     
                videos_dir=videos_dir,
                device=device
            )

        # ==========================================
        # 🌟 6.整理指标，重命名文件夹归档视频
        # ==========================================
        import shutil
        sr = eval_info["aggregated"]["success_rate"]
        ar = eval_info["aggregated"]["average_reward"]
        
        # 生成直观的文件夹名称 (例如: eval_seed100_sr85.0_ar150.5)
        new_folder_name = f"eval_seed={eval_cfg.seed}_ep={eval_cfg.n_episodes}_sr={sr*100:.1f}_ar={ar:.2f}"
        new_videos_dir = os.path.join(videos_dir, new_folder_name)
        os.makedirs(new_videos_dir, exist_ok=True)
        
        # 将刚刚生成的视频全部移动到新文件夹中
        moved_count = 0
        for video_path in eval_info["video_paths"]:
            if os.path.exists(video_path):
                file_name = os.path.basename(video_path)
                shutil.move(video_path, os.path.join(new_videos_dir, file_name))
                moved_count += 1
                

        # ==========================================
        # 🌟 7.打印最终结果
        # ==========================================
        print("\n" + "="*50)
        print(f"🎉 独立评估完成！")
        print(f"🏆 成功率 (Success Rate): {sr*100:.1f}%")
        print(f"💰 平均奖励 (Average Reward): {ar:.2f}")
        print("="*50)

    # 启动
    main()