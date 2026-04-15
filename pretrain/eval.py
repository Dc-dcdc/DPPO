import torch
import logging
import numpy as np
import imageio
from pathlib import Path
from contextlib import nullcontext
import gymnasium as gym
import yaml
from pathlib import Path
def custom_eval_policy(env, policy, cfg_eval, videos_dir, device):
    """
    完全自主实现的评估代码。没有任何黑盒。
    接收标准 Gym 环境，处理图像归一化，跑策略推理，保存视频。
    """
    policy.eval() # 必须开启评估模式
    successes = []
    rewards = []

    videos_dir = Path(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # 从配置中动态读取参数，提供后备默认值防崩溃
    n_episodes = getattr(cfg_eval, "n_episodes", 10)
    max_rendered = getattr(cfg_eval, "max_episodes_rendered", 4)
    fps = getattr(cfg_eval, "fps", 25)
    max_steps = getattr(cfg_eval, "max_steps", 300)
    render_camera = getattr(cfg_eval, "render_camera", 'overhead_cam')
    # 动作执行循环
    for ep in range(n_episodes):
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
                action = policy.select_action(batch)
            
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
            video_path = videos_dir / f"eval_{render_camera[0]}_{ep}.mp4"
            imageio.mimsave(str(video_path), frames, fps=fps)
            logging.info(f"🎥 保存视频: {video_path}")

    policy.train() # 恢复训练模式
    
    # 计算实际应该返回的视频列表长度
    actual_rendered = min(max_rendered, n_episodes)
    return {
        "aggregated": {
            "success_rate": float(np.mean(successes)),
            "average_reward": float(np.mean(rewards))
        },
        "video_paths": [str(videos_dir / f"eval_{render_camera[0]}_{i}.mp4") for i in range(actual_rendered)]
    }


def evaluate_and_checkpoint_if_needed(
    step, policy, optimizer, lr_scheduler, logger, cfg, device, out_dir, eval_env=None
):
    """
    主评估与保存入口
    """
    _num_digits = max(6, len(str(cfg.training.offline_steps))) 
    step_identifier = f"{step:0{_num_digits}d}" 

    # 1. 评估逻辑 (优先读取 cfg.eval.eval_freq)
    eval_freq = getattr(cfg.eval, "eval_freq", 0)
    
    if eval_freq > 0 and step > 0 and step % eval_freq == 0:
        logging.info(f"📊 开始自主评估流程, 当前 Step: {step}")
        if eval_env is not None:
            video_dir = Path(out_dir) / "eval" / f"videos_step_{step_identifier}"
            
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
                logger.log_video(eval_info["video_paths"][0], step, mode="eval") 

    # 2. 保存检查点逻辑
    is_last_step = (step == cfg.training.offline_steps - 1)
    save_freq = getattr(cfg.training, "save_freq", 10000)
    if getattr(cfg.training, "save_checkpoint", False) and (step > 0 and step % save_freq == 0 or is_last_step):
        logging.info(f"💾 保存模型快照... Step: {step}")
        logger.save_checkpoint(step, policy, optimizer, lr_scheduler, identifier=step_identifier)


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
        # 📂 模型路径设置 (直接指向 000000 这样的数字文件夹即可，代码会自动寻找内部结构)
        ckpt_path="outputs/pretrain/train/2026-04-15/12-42-37_sim_envs_diffusion_pretrain_zed_diffusion_2026-04-15_12-42-37/checkpoints/040000",
        name = 'sim_envs', #会自动加载，可以不改
        task = 'SewNeedle-3Arms-v0', #会自动加载，可以不改
        # ⚙️ 评估参数设置
        n_episodes=4,             # 评估多少个任务                 
        max_episodes_rendered=4,  # 保存多少个视频 
        fps=25,                   # 视频帧率，和环境控制频率对齐
        max_steps=400,            # 每个任务的最大步数
        
        # 📷 相机设置
        render_camera=['overhead_cam']         # 保存video的相机视角              
    )
    USE_AMP = True  
    # ==========================================

    def main():
        ckpt_path = eval_cfg.ckpt_path
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
        # 🌟 6.打印最终结果
        # ==========================================
        sr = eval_info["aggregated"]["success_rate"]
        ar = eval_info["aggregated"]["average_reward"]
        print("\n" + "="*50)
        print(f"🎉 独立评估完成！")
        print(f"🏆 成功率 (Success Rate): {sr*100:.1f}%")
        print(f"💰 平均奖励 (Average Reward): {ar:.2f}")
        print("="*50)

    # 启动
    main()