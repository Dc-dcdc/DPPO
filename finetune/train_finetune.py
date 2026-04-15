import os
import sys
import math
import torch
import numpy as np
import logging
import einops
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import hydra
import yaml
# 路径处理
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from env.sim_envs import SewNeedleEnv
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.utils.utils import get_safe_torch_device
from finetune.critic import ImageCritic

def train_dppo_finetune(cfg: DictConfig, out_dir: str):
    device = get_safe_torch_device("cuda")
    logging.info(f"🚀 启动 DPPO 强化学习微调... 设备: {device}")

    # ==========================================
    # 1. 配置对齐与初始化 (环境、Actor、Critic)
    # ==========================================
    ckpt_path = cfg.training.pretrained_ckpt_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ 找不到权重路径: {ckpt_path}\n请检查路径是否正确。")

    # 🌟 核心修改：自动探测 LeRobot 的 pretrained_model 子文件夹
    hf_model_dir = os.path.join(ckpt_path, "pretrained_model")
    if os.path.exists(hf_model_dir):
        print(f"🔍 检测到 LeRobot 标准快照结构，将自动读取子目录: pretrained_model")
        load_dir = hf_model_dir
    else:
        load_dir = ckpt_path # 兼容其他保存格式

    # 准备设备 (传入 "cuda" 防止新版本报错)
    device = get_safe_torch_device("cuda")
    print(f"🚀 初始化评估程序... 使用设备: {device}")

    # 实例化 Policy 并加载权重
    print(f"💾 正在从目录重建网络并加载权重: {load_dir}")
    try:
        # ==========================================
        # 🌟 直接使用 DiffusionPolicy 官方类，绕开所有版本不兼容的 API
        # ==========================================
        # 像加载常规 Hugging Face 模型一样，直接读取文件夹
        actor = DiffusionPolicy.from_pretrained(load_dir)
        
        # 手动推入 GPU
        actor.to(device)
    except Exception as e:
        raise RuntimeError(f"❌ 权重加载失败！详细报错: {e}")
    
    # 提取相机配置并初始化环境与 Critic
    all_obs_keys = actor.config.input_shapes.keys()
    # 读取预训练配置文件中policy.input_shapes中的相机配置，用于初始化环境，保证相机名称顺序一致
    ref_cams = [k.replace("observation.images.", "") for k in all_obs_keys if "observation.images." in k]
    action_shape = actor.config.output_shapes.get("action", [horizon_steps, 14]) # 默认 14 维
    action_dim = action_shape[-1]  # 取最后一位作为动作维度
    if not ref_cams:
        raise ValueError(f"❌ 严重冲突：模型中未找到相机相关参数。请检查模型快照重是否正确。")
    
    env = SewNeedleEnv(cameras=list(dict.fromkeys(ref_cams + list(cfg.env.render_camera))))
    critic = ImageCritic(camera_names=ref_cams).to(device)

    # ==========================================
    # 2. 初始化优化器与超参数 (对应原版 init)
    # ==========================================
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=cfg.training.actor_lr)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=cfg.training.critic_lr)

    # ==========================================
    # 🌟 主循环：DPPO 强化学习全流程
    # ==========================================
    for itr in range(cfg.training.n_train_itr):
        logging.info(f"\n========== 第 {itr+1}/{cfg.training.n_train_itr} 轮迭代 ==========")
        
        # 容器初始化
        obs_list, chains_list, rewards_list, dones_list = [], [], [], []
        
        obs, _ = env.reset()
        actor.reset()
        actor.eval()
        critic.eval()
        
        # ------------------------------------------
        # 步骤 A：Rollout 数据收集
        # ------------------------------------------
        logging.info("🏃 正在与环境交互收集数据 (Rollout)...")
        for step in range(cfg.training.rollout_steps):
            # 将 Numpy 字典转为 Tensor 字典送入模型
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
            
            with torch.no_grad():
                # ⚠️ 核心接口预留：未来的 Actor 必须返回 Diffusion 的降噪链 (chains)
                # 对应原代码 samples = self.model(cond=cond, return_chain=True)
                # 目前暂用标准 select_action 占位，链条用 dummy 代替
                action_tensor = actor.select_action(batch)
                chains_trajs = torch.zeros(  # 扩散模型降噪链轨迹缓存
                    (
                        cfg.n_envs, # 环境数
                        cfg.ft_denoising_steps + 1, # 降噪步数，第一步存的纯随机噪声
                        cfg.horizon_steps, # 预测未来步数
                        cfg.action_dim, # 动作维度
                    ), device=device)
            action_np = action_tensor[0].cpu().numpy()
            if len(action_np.shape) > 1: action_np = action_np[0]
            
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            # 存储数据
            obs_list.append(batch)
            chains_list.append(chains_trajs)
            rewards_list.append(reward)
            dones_list.append(done)

            obs = next_obs
            if done:
                obs, _ = env.reset()
                actor.reset()
                
        # ------------------------------------------
        # 步骤 B：Critic 价值评估与旧概率计算
        # ------------------------------------------
        logging.info("🧠 计算状态价值 (Value) 与旧动作概率 (Old Logprobs)...")
        values_list = []
        old_logprobs_list = []
        
        with torch.no_grad():
            for i in range(cfg.training.rollout_steps):
                b_obs = obs_list[i]
                b_chain = chains_list[i]
                
                # 1. 算出 V(s)
                val = critic(b_obs).item()
                values_list.append(val)
                
                # 2. 算出旧策略概率 log π_old(a|s)
                # ⚠️ 核心接口预留：未来的 Actor 必须具备计算 logprob 的能力
                dummy_logprob = torch.zeros(1, device=device) 
                old_logprobs_list.append(dummy_logprob)

        # ------------------------------------------
        # 步骤 C：GAE (广义优势估计) 逆向时间旅行
        # ------------------------------------------
        logging.info("⚖️ 计算 GAE 优势函数...")
        advantages = np.zeros(cfg.training.rollout_steps)
        last_gae_lam = 0
        
        # 提前准备最后一步的 Next Value
        with torch.no_grad():
            last_batch = {}
            for k, v in obs.items():
                if "images" in k: last_batch[k] = torch.from_numpy(v).float().unsqueeze(0).to(device) / 255.0
                else: last_batch[k] = torch.from_numpy(v).float().unsqueeze(0).to(device)
            last_val = critic(last_batch).item()

        for t in reversed(range(cfg.training.rollout_steps)):
            next_val = last_val if t == cfg.training.rollout_steps - 1 else values_list[t + 1]
            nonterminal = 1.0 - float(dones_list[t])
            
            # TD 误差: δ = r + γ * V(s') * (1-done) - V(s)
            delta = rewards_list[t] + cfg.training.gamma * next_val * nonterminal - values_list[t]
            
            # 优势值: A_t = δ_t + γ * λ * (1-done) * A_{t+1}
            advantages[t] = last_gae_lam = delta + cfg.training.gamma * cfg.training.gae_lambda * nonterminal * last_gae_lam
            
        returns = advantages + np.array(values_list)

        # 转换为 Tensor 准备训练
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
        val_t = torch.tensor(values_list, dtype=torch.float32, device=device)

        # ------------------------------------------
        # 步骤 D：PPO 多轮小批量更新 (Update Epochs)
        # ------------------------------------------
        logging.info(f"🔄 开始 PPO 网络更新 (Epochs: {cfg.training.update_epochs})...")
        actor.train()
        critic.train()
        
        for epoch in range(cfg.training.update_epochs):
            # 打乱索引
            indices = torch.randperm(rollout_steps, device=device)
            
            for start_idx in range(0, rollout_steps, cfg.training.batch_size):
                end_idx = start_idx + cfg.training.batch_size
                batch_inds = indices[start_idx:end_idx]
                
                # ... [由于 LeRobot 的限制，这里需要手动拼接字典 batch] ...
                # 现实中会将 obs_list 堆叠成一个大的 Tensor，这里使用伪代码展示核心逻辑
                
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                
                # 1. Critic Loss (均方误差)
                # new_values = critic(batch_obs)
                # v_loss = F.mse_loss(new_values, ret_t[batch_inds])
                v_loss = torch.tensor(0.0, requires_grad=True, device=device) # 占位
                
                # 2. Actor Loss (PPO Clip)
                # new_logprobs = actor.get_logprobs(...)
                # ratio = torch.exp(new_logprobs - old_logprobs[batch_inds])
                # surr1 = ratio * adv_t[batch_inds]
                # surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_t[batch_inds]
                # pg_loss = -torch.min(surr1, surr2).mean()
                pg_loss = torch.tensor(0.0, requires_grad=True, device=device) # 占位
                
                # 3. 反向传播
                loss = pg_loss + 0.5 * v_loss # 可以加上 entropy_loss 和 bc_loss
                loss.backward()
                
                # 梯度裁剪防爆炸
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                
                actor_optimizer.step()
                critic_optimizer.step()
                
        logging.info(f"✅ 第 {itr+1} 轮更新完成！平均奖励: {np.mean(rewards_list):.4f}")

@hydra.main(version_base="1.2", config_name="ft_default", config_path="../configs/finetune")
def train_cli(cfg: DictConfig):
    out_dir = hydra.core.hydra_config.HydraConfig.get().run.dir
    train_dppo_finetune(cfg, out_dir)

if __name__ == "__main__":
    # 命令行参数注入
    default_args = [
        # "env=sim_sew_needle_3arms",
        "policy=ft_zed_diffusion",
        "training.pretrained_ckpt_path=outputs/pretrain/train/2026-04-13/11-21-21_guided_vision_diffusion_default/checkpoints/320000",
        "training.rollout_steps=500", 
        "training.batch_size=32",     
        "wandb.enable=false",
    ]
    
    for arg in default_args:
        arg_key = arg.split("=")[0]
        if not any(arg_key in sys_arg for sys_arg in sys.argv):
            sys.argv.append(arg)

    train_cli()