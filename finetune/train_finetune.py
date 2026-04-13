import os
import sys
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
from contextlib import nullcontext

# 确保能够导入你的环境
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.sim_envs import SewNeedleEnv
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.utils.utils import get_safe_torch_device

# 导入我们上一轮写的 Critic
from critic import ImageCritic

# ==========================================
# 📦 1. 定义经验回放池 (Rollout Buffer)
# ==========================================
class RolloutBuffer:
    def __init__(self):
        self.obs = []          # 观测画面和状态
        self.actions = []      # 实际执行的动作
        self.log_probs = []    # 动作的对数概率 (RL核心)
        self.rewards = []      # 环境给的真实奖励
        self.values = []       # Critic 预测的价值
        self.dones = []        # 游戏是否结束

    def clear(self):
        self.__init__()

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

# ==========================================
# 🏃 2. 数据收集主循环 (Rollout Logic)
# ==========================================
def collect_rollouts(env, actor_policy, critic_net, buffer, rollout_steps, device):
    """
    让模型在环境里跑指定的步数，并将经验存入 buffer
    """
    print(f"\n🏃 开始收集 RL 数据 (Rollout)... 目标步数: {rollout_steps}")
    actor_policy.eval()
    critic_net.eval()
    buffer.clear()

    obs, _ = env.reset()
    actor_policy.reset() # 清空 Diffusion 的动作缓冲历史

    step_count = 0
    ep_reward = 0
    ep_lengths = []
    ep_rewards = []

    while step_count < rollout_steps:
        # 1. 格式化观测数据
        batch = {}
        for k, v in obs.items():
            v_safe = v.copy()
            if "images" in k:
                tensor_v = torch.from_numpy(v_safe).float().unsqueeze(0).to(device) / 255.0
            else:
                tensor_v = torch.from_numpy(v_safe).float().unsqueeze(0).to(device)
            batch[k] = tensor_v

        with torch.no_grad():
            # 2. Critic 评估当前局势的价值 V(s)
            value = critic_net(batch)

            # 3. Actor (Diffusion) 预测动作
            # ⚠️ 注意: 这里暂时使用 Dummy 的对数概率，后续我们会换成 DPPO 的真实计算公式
            action_tensor = actor_policy.select_action(batch)
            dummy_log_prob = torch.tensor([-0.1], device=device) 

        # 处理 Action Chunking，只取当前步的动作
        action_np = action_tensor[0].cpu().numpy()
        if len(action_np.shape) > 1:
            action_np = action_np[0]

        # 4. 与环境交互
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        # 5. 存入 Buffer
        buffer.add(
            obs=obs, 
            action=action_np,
            log_prob=dummy_log_prob.cpu().numpy()[0],
            reward=reward,
            value=value.cpu().numpy()[0],
            done=done
        )

        ep_reward += float(reward)
        step_count += 1
        obs = next_obs

        # 6. 如果游戏结束，记录统计数据并重置
        if done:
            ep_lengths.append(info.get('step', 300))
            ep_rewards.append(ep_reward)
            obs, _ = env.reset()
            actor_policy.reset()
            ep_reward = 0

    print(f"✅ 收集完成! 本次收集了 {step_count} 步数据.")
    if len(ep_rewards) > 0:
        print(f"📊 期间完成了 {len(ep_rewards)} 个 Episode, 平均得分为: {np.mean(ep_rewards):.2f}")


# ==========================================
# 🌟 3. 主函数 (通过 Hydra 注入配置)
# ==========================================
@hydra.main(version_base="1.2", config_name="ft_zed_diffusion", config_path="../configs/finetune")
def main(cfg: DictConfig):
    device = get_safe_torch_device("cuda")
    print(f"🚀 初始化 DPPO 微调程序... 使用设备: {device}")

    # 1. 动态读取相机配置并初始化环境
    ref_cams = list(cfg.eval.reference_cameras)
    ren_cams = list(cfg.eval.render_camera)
    all_cams = list(dict.fromkeys(ref_cams + ren_cams))
    env = SewNeedleEnv(cameras=all_cams)

    # 2. 读取权重路径并加载预训练 Actor
    ckpt_path = cfg.training.pretrained_ckpt_path
    hf_model_dir = os.path.join(ckpt_path, "pretrained_model")
    load_dir = hf_model_dir if os.path.exists(hf_model_dir) else ckpt_path
    
    print(f"🧠 正在加载预训练的 Diffusion Actor: {load_dir}")
    actor = DiffusionPolicy.from_pretrained(load_dir).to(device)

    # 3. 初始化 Critic
    print("⚖️ 正在初始化全新的 Image Critic...")
    critic = ImageCritic(camera_names=ref_cams).to(device)

    # 4. 实例化 Buffer
    rollout_buffer = RolloutBuffer()

    # 5. 执行一次数据收集测试
    collect_rollouts(
        env=env, 
        actor_policy=actor, 
        critic_net=critic, 
        buffer=rollout_buffer, 
        rollout_steps=cfg.training.rollout_steps, # 从 yaml 动态读取步数
        device=device
    )

    print("\n🎉 第一步测试成功！Buffer 中已经充满了可以用来更新网络的 RL 数据！")

if __name__ == "__main__":
    main()