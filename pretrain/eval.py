import torch
import logging
import numpy as np
import imageio
from pathlib import Path
from contextlib import nullcontext

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
                frames.append(env.render())

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
            video_path = videos_dir / f"eval_ep_{ep}.mp4"
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
        "video_paths": [str(videos_dir / f"eval_ep_{i}.mp4") for i in range(actual_rendered)]
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