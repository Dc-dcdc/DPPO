import os
# 没有显示器时使用，比如在服务器上
# os.environ["MUJOCO_GL"] = "egl"
# os.environ["EGL_DEVICE_ID"] = "0"
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
import gymnasium as gym
from lerobot.common.utils.utils import init_logging, set_global_seed
from pprint import pformat
from lerobot.common.logger import Logger
from tqdm import tqdm
# 路径处理
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from env.sim_envs import SewNeedleEnv
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.utils.utils import get_safe_torch_device 
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from finetune.critic import ImageCritic


import torch
from types import MethodType

@torch.no_grad()
def forward_dppo(self, cond: dict, return_chain=True):
    """
    专为 LeRobot DiffusionPolicy 定制的 DPPO 推理函数。
    拦截并保存扩散去噪链，同时完美兼容 LeRobot 的多帧观测缓存机制 (queues)。
    """
    self.eval()
    
    # ==========================================
    # 1. 观测输入归一化与多帧堆叠 (完美复刻 select_action 逻辑)
    # ==========================================
    batch = self.normalize_inputs(cond.copy())
    # 按照 LeRobot 源码将多个相机的图像堆叠在一个张量里
    if len(self.expected_image_keys) > 0:
        batch = dict(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        
    # 重点：把当前帧塞进历史队列 (Queue) 中
    self._queues = populate_queues(self._queues, batch)
    
    # 从队列中提取包含了 n_obs_steps 帧的历史数据
    stacked_batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
    
    # 获取动态的 batch_size (即并行的 n_envs 个数) 和设备
    batch_size = next(iter(stacked_batch.values())).shape[0]
    device = get_device_from_parameters(self)
    dtype = get_dtype_from_parameters(self)
    
    # ==========================================
    # 2. 提取视觉和状态的全局条件 (Global Conditioning)
    # ==========================================
    # 🌟 修正点：直接调用 DiffusionModel 内置的完美处理函数
    global_cond = self.diffusion._prepare_global_conditioning(stacked_batch)
    
    # ==========================================
    # 3. 初始化纯噪声 [Batch, Horizon, Action_Dim]
    # ==========================================
    action_dim = self.config.output_shapes["action"][0]
    trajectory = torch.randn(
        size=(batch_size, self.config.horizon, action_dim),
        dtype=dtype,
        device=device
    )
    
    # ==========================================
    # 4. 配置 Scheduler 步数并计算截断起点
    # ==========================================
    num_inference_steps = self.diffusion.num_inference_steps
    self.diffusion.noise_scheduler.set_timesteps(num_inference_steps)
    timesteps = self.diffusion.noise_scheduler.timesteps
    
    ft_denoising_steps = getattr(self.config, "ft_denoising_steps", 10)
    record_start_idx = len(timesteps) - ft_denoising_steps
    if record_start_idx < 0:
        record_start_idx = 0 
        
    chains = []
    
    # ==========================================
    # 5. 手动展开去噪循环 (Denoising Loop)
    # ==========================================
    for i, t in enumerate(timesteps):
        # 🌟 保存当前的带噪状态 x_t
        if return_chain and i >= record_start_idx:
            chains.append(trajectory.clone())
            
        # 🌟 修正点：调用底层的 UNet 进行预测，注意时间步的数据类型对齐
        timestep_tensor = torch.full(trajectory.shape[:1], t, dtype=torch.long, device=device)
        model_output = self.diffusion.unet(
            trajectory,
            timestep_tensor,
            global_cond=global_cond
        )
        
        # Scheduler 步进：求出更干净的 x_{t-1}
        trajectory = self.diffusion.noise_scheduler.step(
            model_output, t, trajectory
        ).prev_sample

    # 追加最后完全干净的状态 x_0 (Final Action)
    if return_chain and len(chains) < ft_denoising_steps + 1:
        chains.append(trajectory.clone())
        
    # 堆叠维度：[Batch, ft_denoising_steps + 1, Horizon, Action_Dim]
    chains_tensor = torch.stack(chains, dim=1) if return_chain else None
    
    # ==========================================
    # 6. 反归一化并输出完整的 Horizon
    # ==========================================
    out_dict = self.unnormalize_outputs({"action": trajectory})
    final_actions = out_dict["action"]
    
    return {
        "actions": final_actions,       # 完整的预测动作序列 [Batch, Horizon, Action_Dim]
        "chains": chains_tensor         # 去噪链，+1是因为保存了纯噪声  [Batch, ft_denoising_steps + 1, Horizon, Action_Dim]
    }


# ==========================================
# 🌟  定义 PPO 概率计算函数
# ==========================================
def get_logprobs(self, cond: dict, x_t: torch.Tensor, x_t_1: torch.Tensor, timesteps: torch.Tensor):
    """
    计算扩散模型从 x_t 转移到 x_{t-1} 的对数概率 (Log-Likelihood)。
    基于 DDIM (预测 Epsilon) 的数学展开。
    """
    # 1. 提取条件特征 (复用 LeRobot 底层逻辑)
    batch = self.normalize_inputs(cond.copy())
    if len(self.expected_image_keys) > 0:
        batch = dict(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
    global_cond = self.diffusion._prepare_global_conditioning(batch)

    # 2. UNet 预测噪声 (Epsilon)
    noise_pred = self.diffusion.unet(x_t, timesteps, global_cond=global_cond)

    # 3. 计算 DDIM 确定的均值 (mu)
    # 因为 Diffusers 的 step 接口不支持批量 timestep 张量运算，
    # 我们通过遍历 Batch 维度来确保安全的梯度传播和时间步映射。
    mu = torch.empty_like(x_t)
    for i in range(x_t.shape[0]):
        t_val = timesteps[i].item()
        step_out = self.diffusion.noise_scheduler.step(
            model_output=noise_pred[i:i+1],
            timestep=t_val,
            sample=x_t[i:i+1]
        )
        mu[i] = step_out.prev_sample[0]

    # 4. 强制注入 RL 探索方差 (因为 DDIM 默认方差为0)
    std = getattr(self.config, "min_sampling_denoising_std", 0.05)
    var = std ** 2

    # 5. 高斯分布的对数概率公式: -0.5 * ((x - mu)/std)^2 - log(std) - 0.5 * log(2*pi)
    log_prob = -0.5 * ((x_t_1 - mu) ** 2) / var - math.log(std) - 0.5 * math.log(2 * math.pi)
    
    # 将概率在动作序列维度求和 (累乘转化为对数累加)
    log_prob = log_prob.flatten(start_dim=1).sum(dim=-1)

    # 6. 计算高斯分布的香农熵 (用于 PPO 探索正则化)
    action_dim_total = x_t.shape[1] * x_t.shape[2]
    entropy = action_dim_total * (0.5 * math.log(2 * math.pi * math.e * var))

    return log_prob, entropy

def train_dppo_finetune(cfg: DictConfig, out_dir: str | None = None, job_name: str | None = None):
    """
    DPPO 第二阶段：在预训练参数上采用PPO算法进行微调 
    """
    # ==========================================
    # 1. 基础配置与日志初始化
    # ==========================================
    init_logging()
    logging.info("🚀 启动 DPPO 微调程序...")
    logging.info(f"配置参数:\n{pformat(OmegaConf.to_container(cfg))}")

    # 初始化日志记录器与全局随机种子
    logger = Logger(cfg, out_dir, wandb_job_name=job_name)
    set_global_seed(cfg.seed)
    
    # 获取设备 
    device = get_safe_torch_device(cfg.device, log=True)
    logging.info(f"💻 运行设备已绑定: {device}")

    # ==========================================
    # 2. 权重路径检测与 Actor 网络加载
    # ==========================================
    ckpt_path = cfg.training.pretrained_ckpt_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ 找不到权重路径: {ckpt_path}\n请检查路径是否正确。")

    # 自动探测 LeRobot 的 pretrained_model 子文件夹
    hf_model_dir = os.path.join(ckpt_path, "pretrained_model")
    if os.path.exists(hf_model_dir):
        print(f"🔍 检测到 LeRobot 标准快照结构，将自动读取子目录: pretrained_model")
        load_dir = hf_model_dir
    else:
        load_dir = ckpt_path # 兼容其他保存格式
    logging.info(f"💾 正在从目录重建网络并加载权重: {load_dir}")
    try:
        # 直接使用 DiffusionPolicy 官方类加载权重
        actor = DiffusionPolicy.from_pretrained(load_dir)
        actor.to(device)  # 手动推入 GPU

        # 动态挂载 DPPO 专用的前向与概率计算函数
        actor.forward_dppo = MethodType(forward_dppo, actor) # 将forward_dppo绑定到实例中
        actor.get_logprobs = MethodType(get_logprobs, actor) # 将get_logprobs绑定到实例中
        logging.info("✅ 成功加载 Actor (DiffusionPolicy) 并挂载 DPPO 专用接口！")
    except Exception as e:
        logging.error(f"❌ 权重加载失败！详细报错: {e}")
        raise RuntimeError(f"❌ 权重加载失败！详细报错: {e}")
    
    # ==========================================
    # 3. 读取预训练配置文件中输入输出配置，保证与环境对齐
    # ==========================================
    ref_cams = [k.replace("observation.images.", "") for k in actor.config.input_shapes.keys() if "observation.images." in k]
    horizon_steps = getattr(actor.config, "horizon", None)
    action_dim = actor.config.output_shapes.get("action", [None])[-1]
    state_dim = actor.config.input_shapes.get("observation.state", [None])[0]
    # 优化 1：更严谨的校验，允许纯视觉策略 (state_dim=None)
    if not ref_cams or horizon_steps is None or action_dim is None:
        raise ValueError(f"❌ 严重冲突：模型快照中缺少关键参数 (ref_cams={ref_cams}, horizon={horizon_steps}, action_dim={action_dim})。")
        
    if state_dim is None:
        logging.warning("⚠️ 模型配置中未检测到 observation.state，如果这是纯视觉策略，请忽略此警告。")

    # 动态读取环境配置
    config_yaml_path = Path(load_dir) / "config.yaml"
    if config_yaml_path.exists():
        with open(config_yaml_path, "r") as f:
            full_cfg = yaml.safe_load(f)
            # 从 YAML 字典中提取 env_name 和 env_task
            env_cfg = full_cfg.get("env", {})
            env_name = env_cfg.get("name")
            env_task = env_cfg.get("task")
            
            if not env_name or not env_task:
                raise ValueError("❌ config.yaml 中缺少环境的 name 或 task 字段！")
    else:
        raise ValueError(f"❌ 严重错误: 在 {load_dir} 中未找到 config.yaml，无法确定微调环境配置！")
    
    env_id = f"{env_name}/{env_task}" 
    
    logging.info(f"🔄 准备通过 Gym 注册表构建环境: {env_id}")
    # ==========================================
    # 4. 初始化环境与 Critic
    # ==========================================

    # ---------------------------------------------------------
    # 4.1 提取并清洗相机参数 (防止单一字符串被误拆为字母列表)
    # ---------------------------------------------------------
    n_envs = getattr(cfg.env, "n_envs", 1)
    render_cams = getattr(cfg.env, "render_camera", [])
    
    # 安全处理：如果是纯字符串 "top"，转换为 ["top"]；如果是列表则保持；None 则设为空列表
    if render_cams is None:
        render_cams = []
    elif isinstance(render_cams, str):
        render_cams = [render_cams]
    else:
        render_cams = list(render_cams)
        
    # 合并训练视角与渲染视角，并利用字典去重 (保留原始顺序)
    obs_cameras = list(dict.fromkeys(ref_cams + render_cams))
    logging.info(f"📷 最终绑定的环境相机视角: {obs_cameras}")

    # ---------------------------------------------------------
    # 4.2 启动 Gym 物理环境
    # ---------------------------------------------------------
    if n_envs > 1:
        # 使用 AsyncVectorEnv 自动拉起多个进程 
        # 🌟 优化：使用 lambda 延迟初始化，保证每个进程拿到的都是绝对独立的环境实例
        env = gym.vector.AsyncVectorEnv(
            [lambda: gym.make(id=env_id, cameras=obs_cameras) for _ in range(n_envs)],
            shared_memory=True,  # 👈 开启官方内置的共享内存优化，防止传图像时卡顿
            context="spawn"      # 👈 强制安全启动子进程，防止 OpenGL/CUDA 崩溃
        )
        logging.info(f"✅ 成功启动 {n_envs} 个并行多进程环境 (AsyncVectorEnv) ...")
    else:
        env = gym.make(id=env_id, cameras=obs_cameras)
        logging.info("✅ 成功启动单环境模式...")

    # ---------------------------------------------------------
    # 4.3 初始化 Critic 网络
    # ---------------------------------------------------------
    critic = ImageCritic(
        camera_names=ref_cams, # 注意：网络输入只需要 ref_cams，不需要 render_cams
        state_dim=state_dim    # 👈 传入前面动态提取的真实状态维度
    ).to(device)
    
    logging.info("✅ 成功初始化 Critic 网络！")

    # ==========================================
    # 5. 初始化优化器与超参数
    # ==========================================
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=cfg.training.actor_lr)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=cfg.training.critic_lr)
    logging.info(f"⚙️ 扩散模型调度器类型 (Scheduler Type): {actor.config.noise_scheduler_type}")
    logging.info(f"⚙️ 预测目标类型 (Prediction Type): {actor.config.prediction_type}")
    # 从配置中提取 RL 收集参数 (提供后备默认值)
    n_steps = getattr(cfg.training, "rollout_steps", 256)   # 每次更新前收集的步数
    act_steps = getattr(cfg.policy, "n_action_steps", 8) 
    denoising_steps = getattr(cfg.policy, "ft_denoising_steps", 10)

    # 在进入训练循环前，仅全局重置一次环境，保证后续 MDP (马尔可夫决策过程) 的连续性
    prev_obs, _ = env.reset()
    # 记录当前每个环境正在跑的回合的累计分数
    running_ep_rewards = np.zeros(n_envs, dtype=np.float32)
    # ==========================================
    # 🌟 主循环：DPPO 强化学习全流程
    # ==========================================
    for itr in range(cfg.training.n_train_itr):
        logging.info(f"\n========== 第 {itr+1}/{cfg.training.n_train_itr} 轮迭代 ==========")
        
        # ==========================================
        # 1. 初始化 DPPO Rollout 缓冲区 (Buffers)
        # ==========================================
        # 预分配轨迹内存，k对应相机/状态,v对应数据形状
        obs_trajs = {
            k: np.zeros((n_steps, n_envs, *v.shape), dtype=np.float32)
            for k, v in prev_obs.items()
        }
        
        # 核心：保存去噪链 (Chains) 用于计算 Logprob
        chains_trajs = np.zeros(
            # (步数，环境数，去噪步数，预测步数，动作维度)
            (n_steps, n_envs, denoising_steps + 1, horizon_steps, action_dim),
            dtype=np.float32
        )
        
        reward_trajs = np.zeros((n_steps, n_envs), dtype=np.float32)
        terminated_trajs = np.zeros((n_steps, n_envs), dtype=np.float32)
        completed_ep_rewards = []                                # 存放所有【已经跑完】的回合的总分
        completed_ep_successes = []                              # 存放成功标志 (假设环境 info 里有 success 信息)
        # ==========================================
        # 2. 开始收集环境交互数据 (Rollout Loop)
        # ==========================================
        logging.info(f"🏃 开始进入数据收集循环 (共 {n_steps} 步)...")
        for step in range(n_steps):
            if step > 0 and step % 10 == 0:
                logging.info(f"  > 已收集 {step}/{n_steps} 步...")

            # 1. 格式化当前观测：Numpy -> PyTorch Tensor
            batch_obs = {}
            for k, v in prev_obs.items():
                tensor_v = torch.from_numpy(v.copy()).float().to(device)
                # 兼容单环境:n_envs=1,数据形状=[C, H, W]，不符合输入，需要增加维度，多环境形状为[n_envs, C, H, W]
                if n_envs == 1 and tensor_v.dim() == len(v.shape):
                    # [C, H, W] -> [1, C, H, W]   [state] -> [1, state]
                    tensor_v = tensor_v.unsqueeze(0)
                
                # 图像归一化处理
                if "images" in k:
                    tensor_v = tensor_v / 255.0
                    
                batch_obs[k] = tensor_v

            # 2. 网络前向传播获取动作与去噪链
            with torch.no_grad():
                # ⚠️ 注意：此处需确保 actor 有返回 chains 的接口
                # 标准的 actor.select_action 不返回 chains，DPPO 通常需要调用底层的 forward 或特定生成函数
                # 此处仿照参考代码：返回 deterministic=False 的采样以及 return_chain=True
                samples = actor.forward_dppo(cond=batch_obs, return_chain=True) 
                
                output_venv = samples["actions"].cpu().numpy()  # shape: [n_envs, horizon, action_dim]
                chains_venv = samples["chains"].cpu().numpy()   # shape: [n_envs, denoising_steps+1, horizon, action_dim]

            # 截取实际执行的动作长度 (Action Chunking)
            action_venv = output_venv[:, :act_steps]

            # ==========================================
            # 3. 手动展开动作序列块 (Chunking Loop)
            # 网络一次预测了 8 步，我们必须让物理环境分 8 次真实执行
            # ==========================================
            chunk_reward = np.zeros(n_envs, dtype=np.float32)
            # 🌟 修复 1：区分 "任何结束" 和 "真实终止"
            any_done_accum = np.zeros(n_envs, dtype=bool)   # 用于停止累加当前块的奖励
            true_term_accum = np.zeros(n_envs, dtype=bool)  # 用于告诉 GAE 抹除未来价值 (V=0)

            for step_i in range(act_steps):
                # 1. 提取当前这一小步的动作，形状变为: [n_envs, action_dim]
                curr_action = action_venv[:, step_i, :]
                
                # 单环境降维处理: 从 [1, 14] 降为 [14] 给原生 Gym 识别
                action_to_step = curr_action[0] if n_envs == 1 else curr_action

                # 2. 与物理引擎交互 (执行 1 步)
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = env.step(action_to_step)

                # 单环境标量兼容处理
                if n_envs == 1:
                    reward_venv = np.array([reward_venv])
                    terminated_venv = np.array([terminated_venv])
                    truncated_venv = np.array([truncated_venv])
                
                # 3. 极其重要的细节：计算掩码 (Mask)
                # 如果某个并行环境在第 3 步就已经 done 了，Gym 会在底层自动复活它。
                # 为了防止网络误把后续第 4-8 步的动作算进新任务的奖励里，我们需要掩蔽已结束的环境。
                active_mask = ~any_done_accum
                chunk_reward += reward_venv * active_mask
                
                # 只有在 active 的状态下发生 terminated，才算是真实的死亡/通关
                true_term_accum = true_term_accum | (terminated_venv & active_mask)
                
                # 4. 单环境的特殊处理：如果跑完了，必须手动重置并强行跳出当前 Chunk
                if n_envs == 1 and any_done_accum[0]:
                    obs_venv, _ = env.reset()
                    break

            # 4. 顺手统计回合总奖励 (用于日志打印)
            running_ep_rewards += chunk_reward
            for env_idx in range(n_envs):
                if any_done_accum[env_idx]:
                    completed_ep_rewards.append(running_ep_rewards[env_idx])
                    running_ep_rewards[env_idx] = 0.0
            
            # 5. 写入轨迹 Buffer，将这 8 步累积的完整奖励和结束标志，交给外部的 PPO 缓冲区
            for k in obs_trajs:
                # 写入执行动作前的观测
                obs_trajs[k][step] = prev_obs[k] if n_envs > 1 else np.expand_dims(prev_obs[k], 0)
                
            chains_trajs[step] = chains_venv            # [n_steps, n_envs, denoising_steps + 1, horizon_steps, action_dim]
            reward_trajs[step] = chunk_reward            # [n_steps, n_envs]
            terminated_trajs[step] = true_term_accum    # [n_steps, n_envs]

            # 5. 状态流转
            prev_obs = obs_venv

        logging.info("✅ 数据收集 (Rollout) 完成，准备进入 PPO 网络更新阶段！")
                
        # =========================================================
        # 5. 批量计算价值 (Values) 与 GAE 优势函数
        # =========================================================
        logging.info("🧠 计算状态价值 (Values) 与优势函数 (GAE)...")
        
        # 1. 将收集到的字典张量展平，形状由 [n_steps, n_envs, ...] 变为 [n_steps*n_envs, ...] ，相当于是n_steps*n_envs个样本
        # 为防止显存爆掉，先不加 .to(device)，让它留在 CPU 内存中！
        obs_k_cpu = {
            k: einops.rearrange(torch.from_numpy(v), "s e ... -> (s e) ...")
            for k, v in obs_trajs.items()
        }
        
        with torch.no_grad():
            # 🌟 修复 2：分批次 (Mini-batch) 计算 Critic 价值，防止显存爆炸
            total_samples = n_steps * n_envs
            val_batch_size = getattr(cfg.training, "batch_size", 32) * 2  # 评估不算梯度，batch 可以开大点
            values_flat = np.zeros(total_samples, dtype=np.float32)
            
            for i in range(0, total_samples, val_batch_size):
                end_i = min(i + val_batch_size, total_samples)
                
                # 临时切下一小块推入 GPU，并顺手除以 255.0
                obs_chunk = {}
                for k, v in obs_k_cpu.items():
                    tensor_v = v[i:end_i].float().to(device)
                    if "images" in k:
                        tensor_v = tensor_v / 255.0
                        # 🌟 新增：如果是图像，且是 5D 张量 [Batch, Obs_Steps, C, H, W]
                        # 我们只取最新的一帧 (索引 -1) 喂给 Critic
                        if tensor_v.dim() == 5:
                            tensor_v = tensor_v[:, -1]
                    else:
                        # 🌟 新增修复：剥离 State 的历史帧: [B, T, StateDim] -> [B, StateDim]
                        if tensor_v.dim() == 3:
                            tensor_v = tensor_v[:, -1]
                    obs_chunk[k] = tensor_v
                
                # 计算这批数据的 Value 并塞回 CPU 数组
                values_flat[i:end_i] = critic(obs_chunk).cpu().numpy().flatten()
                
            values_trajs = values_flat.reshape(n_steps, n_envs)
            
            # 计算最后一步的 Next Value (Bootstrap)
            last_obs_ts = {}
            for k, v in prev_obs.items():
                tensor_v = torch.from_numpy(v).float().to(device)
                if n_envs == 1 and tensor_v.dim() == len(v.shape):
                    tensor_v = tensor_v.unsqueeze(0)
                if "images" in k: 
                    tensor_v = tensor_v / 255.0
                    # 🌟 新增：处理最后一步的帧缓存，取最新的一帧
                    if tensor_v.dim() == 5:
                        tensor_v = tensor_v[:, -1]
                last_obs_ts[k] = tensor_v
                
            next_values_last = critic(last_obs_ts).cpu().numpy().flatten()

        # 2. 计算 GAE (逆向时间推导)
        advantages_trajs = np.zeros_like(reward_trajs)
        last_gae_lam = 0
        gamma = getattr(cfg.training, "gamma", 0.99)
        gae_lambda = getattr(cfg.training, "gae_lambda", 0.95)

        for t in reversed(range(n_steps)):
            # 获取下一步的 观测评估价值
            next_val = next_values_last if t == n_steps - 1 else values_trajs[t + 1]
            # 判断游戏是否结束，# 如果nonterminal为0，表示死亡或通关，那么未来价值都是0
            nonterminal = 1.0 - terminated_trajs[t]

            # 单步TD误差 = 第t步的奖励*缩放系数 + 折旧因子*下一步观测的价值 - 当前步观测的价值
            # TD 误差: δ = r + γ * V(s') * (1-done) - V(s)
            delta = reward_trajs[t] + gamma * next_val * nonterminal - values_trajs[t]

            # 优势值计算：t-1步的优势值 = TD误差 + 双重衰减系数*t-1步的优势值   一直迭代，一轮(n_steps步)就能算出所有步的优势值 
            # 优势值: A_t = δ_t + γ * λ * (1-done) * A_{t+1}
            advantages_trajs[t] = last_gae_lam = delta + gamma * gae_lambda * nonterminal * last_gae_lam
        
        # Advantage = Return - Value,       优势 = 实际总回报 - 预期总回报
        # aritic网络输出的是当前画面对应的未来总回报  Return = Advantage + Value，
        returns_trajs = advantages_trajs + values_trajs # 用于训练critic网络

        # =========================================================
        # 6. DPPO 多轮小批量更新 (Update Epochs)
        # =========================================================
        logging.info(f"🔄 开始 PPO 网络更新 (Epochs: {getattr(cfg.training, 'update_epochs', 4)})...")
        actor.train()
        critic.train()

        # 1. 准备训练用的展平张量
        returns_k = torch.tensor(returns_trajs, device=device).float().reshape(-1)
        advantages_k = torch.tensor(advantages_trajs, device=device).float().reshape(-1)
        
        # 优势函数归一化 (极大地提升 PPO 训练稳定性)
        advantages_k = (advantages_k - advantages_k.mean()) / (advantages_k.std() + 1e-8)
        
        # 将 Chains 展平为 [(步数*环境数), 去噪步数, 预测视野, 动作维度]
        chains_k = einops.rearrange(
            torch.tensor(chains_trajs, device=device).float(),
            "s e t h d -> (s e) t h d"
        )

        total_steps = n_steps * n_envs * denoising_steps      # 包含去噪的步数 例如：15000=300*5*10

        # 获取与去噪步对应的真实 TimeSteps
        actor.diffusion.noise_scheduler.set_timesteps(actor.diffusion.num_inference_steps)
        all_timesteps = actor.diffusion.noise_scheduler.timesteps
        record_start_idx = max(0, len(all_timesteps) - denoising_steps)
        recorded_timesteps = all_timesteps[record_start_idx:].to(device)

        # =========================================================
        # 7. 预计算旧策略的对数概率 (Old Logprobs)
        # =========================================================
        # 🌟 修复 3：预计算旧概率时，同样按需从 CPU 搬运数据
        logging.info("🧠 正在预计算旧策略概率基准...")
        old_logprobs_k = torch.zeros(total_steps, device=device)
        with torch.no_grad():
            eval_batch_size = getattr(cfg.training, "batch_size", 32) * 2
            for i in range(0, total_steps, eval_batch_size):
                inds = torch.arange(i, min(i + eval_batch_size, total_steps), device=device)
                b_inds, d_inds = torch.unravel_index(inds, (n_steps * n_envs, denoising_steps))
                
                # 从 CPU 取出这一小批的画面放入 GPU
                obs_eval = {}
                for k, v in obs_k_cpu.items():
                    tensor_v = v[b_inds.cpu()].float().to(device)
                    if "images" in k:
                        tensor_v = tensor_v / 255.0

                    obs_eval[k] = tensor_v

                logprobs, _ = actor.get_logprobs(
                    cond=obs_eval, 
                    x_t=chains_k[b_inds, d_inds], 
                    x_t_1=chains_k[b_inds, d_inds + 1], 
                    timesteps=recorded_timesteps[d_inds]
                )
                old_logprobs_k[inds] = logprobs

        # 3. 开始 Epoch 循环
        batch_size = getattr(cfg.training, "batch_size", 50)
        update_epochs = getattr(cfg.training, "update_epochs", 10)
        clip_ratio = getattr(cfg.training, "clip_ratio", 0.25)
        entropy_coef = getattr(cfg.training, "entropy_coef", 1e-4)

        # 2. 开始 Epoch 循环，PPO的数据集可以训练多轮，数据利用率高
        for epoch in tqdm(range(update_epochs), desc=f"⏳ PPO 更新中 (Iter {itr+1})", leave=False):
            # 打乱所有数据点 (不仅打乱时间步，还打乱去噪步)
            indices = torch.randperm(total_steps, device=device)
            num_batch = max(1, total_steps // batch_size) #分批次训练
            
            for batch_idx in range(num_batch):
                start = batch_idx * batch_size
                end = start + batch_size
                inds_b = indices[start:end] #从打乱的总数据中提取一个batch的索引
                
                # 将inds_b索引值对应到哪批样本batch_inds_b 的 哪个去噪步denoising_inds_b
                batch_inds_b, denoising_inds_b = torch.unravel_index(
                    inds_b, 
                    (n_steps * n_envs, denoising_steps) #n_steps * n_envs行，denoising_steps列
                )
                
                # 切片提取 Mini-batch
                # 🌟 修复：创建两个字典，实现数据分流
                obs_b = {}           # 给 Actor 用的（原汁原味，带历史帧）
                obs_b_critic = {}    # 给 Critic 用的（单帧，降维后）
                
                for k, v in obs_k_cpu.items():
                    tensor_v = v[batch_inds_b.cpu()].float().to(device)
                    if "images" in k:
                        tensor_v = tensor_v / 255.0
                    
                    # 1. 完整数据给 Actor
                    obs_b[k] = tensor_v
                    
                    # 2. 剥离历史帧后给 Critic
                    tensor_c = tensor_v
                    if "images" in k and tensor_c.dim() == 5:
                        tensor_c = tensor_c[:, -1]
                    elif "images" not in k and tensor_c.dim() == 3:
                        tensor_c = tensor_c[:, -1]
                    
                    obs_b_critic[k] = tensor_c
                
                # obs_b = {k: v[batch_inds_b] for k, v in obs_k.items()}        # 取出对应样本的观测，也就是对应环境和步数的观测
                chains_prev_b = chains_k[batch_inds_b, denoising_inds_b]      # 对应样本的当前去噪步 动作
                chains_next_b = chains_k[batch_inds_b, denoising_inds_b + 1]  # 对应样本的下一个去噪步 动作
                returns_b = returns_k[batch_inds_b]                           # 对应样本的总回报
                advantages_b = advantages_k[batch_inds_b]                     # 对应样本的优势值
                timesteps_b = recorded_timesteps[denoising_inds_b]
                
                old_logprobs_b = old_logprobs_k[inds_b]
                

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                
                # ----------------------------------------------------
                # 🌟 网络 Loss 计算区 (真正的 PPO 数学魔法)
                # ----------------------------------------------------
                # 1. 计算当前网络对动作的新概率预测
                new_logprobs_b, entropy_b = actor.get_logprobs(
                    cond=obs_b, 
                    x_t=chains_prev_b, 
                    x_t_1=chains_next_b, 
                    timesteps=timesteps_b
                )
                
                # 2. PPO 概率比 (Ratio): Ratio = exp(new_logprob - old_logprob)
                ratio = torch.exp(new_logprobs_b - old_logprobs_b)
                
                # 3. PPO 截断代理损失 (Clipped Surrogate Objective)
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_b
                pg_loss = -torch.min(surr1, surr2).mean()
                
                # 4. Critic MSE 损失
                values_pred = critic(obs_b_critic).squeeze(-1)
                v_loss = torch.nn.functional.mse_loss(values_pred, returns_b)
                
                # 5. 总 Loss 汇总
                loss = pg_loss + 0.5 * v_loss - entropy_coef * entropy_b
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                
                actor_optimizer.step()
                critic_optimizer.step()
                
        # ------------------------------------------
        # 步骤 E：打印评估指标 (替代原论文代码)
        # ------------------------------------------
        if len(completed_ep_rewards) > 0:
            avg_ep_return = np.mean(completed_ep_rewards)
            max_ep_return = np.max(completed_ep_rewards)
            logging.info(f"✅ 第 {itr+1} 轮完成！")
            logging.info(f"   🏃 本轮共完成回合数: {len(completed_ep_rewards)}")
            logging.info(f"   💰 平均回合总奖励 (Return): {avg_ep_return:.2f}")
            logging.info(f"   🏆 最高回合奖励: {max_ep_return:.2f}")
            
            # 如果配置了 WandB
            # logger.log_dict({"train/avg_return": avg_ep_return}, step=itr)
        else:
            logging.info(f"⚠️ 第 {itr+1} 轮结束，但没有环境跑完一个完整回合 (考虑增加 rollout_steps)")

@hydra.main(version_base="1.2", config_name="ft_default", config_path="../configs/finetune")
def train_cli(cfg: DictConfig):
    train_dppo_finetune(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,  # 获取当前训练运行的输出目录，用于保存训练输出的数据
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name, # 获取当前训练运行的作业名称，用于wandb
    )
if __name__ == "__main__":
    # 命令行参数注入
    default_args = [
        # "env=sim_sew_needle_3arms",
        "policy=ft_zed_diffusion",
        "training.pretrained_ckpt_path=outputs/pretrain/train/2026-04-15/22-11-55_sim_envs_diffusion_pretrain_zed_diffusion_2026-04-15_22-11-55/checkpoints/0670000",
        "training.rollout_steps=50", 
        "training.batch_size=5",     
        "wandb.enable=false",
    ]
    
    for arg in default_args:
        arg_key = arg.split("=")[0]
        if not any(arg_key in sys_arg for sys_arg in sys.argv):
            sys.argv.append(arg)

    train_cli()