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
from collections import deque
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
from pretrain.eval import custom_eval_policy, TopKCheckpointManager
from contextlib import nullcontext

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

        # 注入探索噪声 (除了最后一步)
        std = getattr(self.config, "min_sampling_denoising_std", 0.05)
        if i < len(timesteps) - 1:
            trajectory = trajectory + torch.randn_like(trajectory) * std
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
    基于 DDIM (预测 Epsilon) 的数学展开，使用DDPM需要重新变化实现公式。
    """
    # 1. 提取条件特征 (复用 LeRobot 底层逻辑)，包括视觉和状态
    batch = self.normalize_inputs(cond.copy())
    # 堆叠后形状完美契合 LeRobot 底层要求: [Batch, Time, Num_Cams, C, H, W]
    if len(self.expected_image_keys) > 0:
        batch = dict(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
    global_cond = self.diffusion._prepare_global_conditioning(batch)

    # 2. 预测x_t中的噪声 (Epsilon)
    noise_pred = self.diffusion.unet(x_t, timesteps, global_cond=global_cond)

    # 3. 计算 DDIM 确定的均值 (mu)

    alphas_cumprod = self.diffusion.noise_scheduler.alphas_cumprod.to(x_t.device)

    # 3.1： 提取当前步的 alpha (注意对齐形状以支持广播)
    alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1)

    # 3.2： 动态计算 DDIM 的真实“上一步”时间
    # 根据配置的 训练总步数 和 实际推理步数 算出跳跃步长
    scheduler = self.diffusion.noise_scheduler
    step_ratio = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_timesteps = timesteps - step_ratio

    # 3.3： 提取上一步的 alpha
    # 注意细节：当 prev_timesteps < 0 时（也就是最后一步），意味着要抵达完全无噪的 x_0
    # 在数学上，x_0 的 alpha_cumprod 应该绝对等于 1.0
    alpha_prod_t_prev = torch.where(
        prev_timesteps >= 0,
        alphas_cumprod[torch.clamp(prev_timesteps, min=0)],
        torch.tensor(1.0, device=x_t.device, dtype=x_t.dtype)
    ).view(-1, 1, 1)

    # 4. DDIM 核心推导公式：
    # 步骤 A：预测出纯净的 x_0 (Pred Original Sample)
    pred_original_sample = (x_t - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
    
    # 步骤 B：用 x_0 和 epsilon 重新组合出 DDIM 路径上的上一帧均值 (mu)
    mu = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + torch.sqrt(1 - alpha_prod_t_prev) * noise_pred

    # 5. 强制注入 RL 探索方差 (因为 DDIM 默认方差为0)
    std = getattr(self.config, "min_sampling_denoising_std", 0.05)
    var = std ** 2

    # 6. 高斯分布的对数概率公式: -0.5 * ((x - mu)/std)^2 - log(std) - 0.5 * log(2*pi)
    log_prob = -0.5 * ((x_t_1 - mu) ** 2) / var - math.log(std) - 0.5 * math.log(2 * math.pi)
    
    # 将概率在动作序列维度求和 (累乘转化为对数累加)
    log_prob = log_prob.flatten(start_dim=1).sum(dim=-1)

    return log_prob

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

        # 单独注入模型微调需要的超参数
        actor.config.min_sampling_denoising_std = getattr(
            cfg.training, 
            "min_sampling_denoising_std", 
            0.05
        )
        logging.info("✅ 成功将微调配置 (YAML) 注入到 Actor 内部 config 中！")

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

    # ---------------------------------------------------------
    # 4.4 初始化独立的评估环境与 Top-K 快照管理器
    # ---------------------------------------------------------
    logging.info("🎬 正在初始化独立的评估环境 (Eval Env)...")
    # 评估环境不需要向量化并发，只需一个单例环境即可
    eval_env = gym.make(id=env_id, cameras=obs_cameras)
    
    # 初始化 Top-K 快照管理器 (比如最多保留表现最好的 3 个模型)
    max_checkpoints = getattr(cfg.eval, "max_checkpoints", 3)
    records_resume = getattr(cfg.eval, "records_resume", True)
    ckpt_manager = TopKCheckpointManager(out_dir=out_dir, max_keep=max_checkpoints, records_resume=records_resume)

    # ==========================================
    # 5. 初始化优化器与超参数
    # ==========================================
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=cfg.training.actor_lr)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=cfg.training.critic_lr)
    logging.info(f"⚙️ 扩散模型调度器类型 (Scheduler Type): {actor.config.noise_scheduler_type}")
    logging.info(f"⚙️ 预测目标类型 (Prediction Type): {actor.config.prediction_type}")
    # 从配置中提取 RL 收集参数 (提供后备默认值)
    n_steps = getattr(cfg.training, "rollout_steps", 300)   # 每次更新前收集的步数
    act_steps = getattr(cfg.policy, "n_action_steps", 8) 
    denoising_steps = getattr(cfg.policy, "ft_denoising_steps", 10)
    critic_warmup_iters = getattr(cfg.training, "n_critic_warmup_itr", 2) # critic网络先默认热身 2 轮
    ema_alpha = getattr(cfg.training, "reward_ema_alpha", 0.05) # 推荐 0.05 或 0.01
    target_kl = getattr(cfg.training, "target_kl", 0.02)
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
        # 保留最近 n_obs_steps 帧历史观测
        n_obs_steps = getattr(actor.config, "n_obs_steps", 2)
        raw_obs_queue = {k: deque(maxlen=n_obs_steps) for k in prev_obs.keys()}
        obs_trajs = None

        # 核心：保存去噪链 (Chains) 用于计算 Logprob
        chains_trajs = np.zeros(
            # (步数，环境数，去噪步数，预测步数，动作维度)
            (n_steps, n_envs, denoising_steps + 1, horizon_steps, action_dim),
            dtype=np.float32
        )
        
        reward_trajs = np.zeros((n_steps, n_envs), dtype=np.float32)
        terminated_trajs = np.zeros((n_steps, n_envs), dtype=np.float32)
        completed_ep_rewards = []                                # 存放所有【已经跑完】的回合的总分
        # ==========================================
        # 2. 开始收集环境交互数据 (Rollout Loop)
        # ==========================================
        actor.reset()
        logging.info(f"🏃 开始进入数据收集循环 (共 {n_steps} 步)...")
        for step in tqdm(range(n_steps), leave=False):

            # 将当前物理环境的画面压入历史队列
            for k, v in prev_obs.items():
                if len(raw_obs_queue[k]) == 0:
                    for _ in range(n_obs_steps):  # 第一步时，把画面复制填满队列
                        raw_obs_queue[k].append(v)
                else:
                    raw_obs_queue[k].append(v)

            # 打包出带有时间维度 T 的状态 [n_envs, n_obs_steps, ...]
            stacked_raw_obs = {}
            for k in prev_obs.keys():
                stacked_v = np.stack(list(raw_obs_queue[k]), axis=0 if n_envs == 1 else 1)
                # 兼容单环境的 Batch 维度：确保最终形状是 [1, T, C, H, W]
                if n_envs == 1:
                    stacked_v = np.expand_dims(stacked_v, axis=0)
                stacked_raw_obs[k] = stacked_v
            
            # 使用包含了完整 T 维度的 stacked_raw_obs 初始化 obs_trajs
            if obs_trajs is None:
                obs_trajs = {
                    k: np.zeros((n_steps, *v.shape), dtype=np.float32)
                    for k, v in stacked_raw_obs.items()
                }

            # 1. 格式化当前单帧观测给 Actor 去推断 
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
            any_done_accum = np.zeros(n_envs, dtype=bool)   # 用于停止累加当前块的奖励
            true_term_accum = np.zeros(n_envs, dtype=bool)  # 用于告诉 GAE 抹除未来价值 (V=0)

            # 🌟 用于存放每个环境真正的 "原地待命" 动作
            # 形状需要和动作维度一致
            if n_envs > 1:
                safe_actions = np.zeros((n_envs, action_venv.shape[-1]), dtype=np.float32)

            for step_i in range(act_steps):
                curr_action = action_venv[:, step_i, :].copy() # 注意加 .copy()，防止修改原张量
                
                # 多环境动作冻结
                if n_envs > 1:
                    for env_idx in range(n_envs):
                        # 如果某个环境已经死亡并重置，我们不能给它喂后续的垃圾动作。
                        if any_done_accum[env_idx]:
                            # 策略：一直给它发送死亡前最后一刻的安全动作，让机器人在重置原点尽量保持静止
                            curr_action[env_idx] = safe_actions[env_idx]
                
                action_to_step = curr_action[0] if n_envs == 1 else curr_action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = env.step(action_to_step)

                if n_envs == 1:
                    reward_venv = np.array([reward_venv])
                    terminated_venv = np.array([terminated_venv])
                    truncated_venv = np.array([truncated_venv])
                
                active_mask = ~any_done_accum
                chunk_reward += reward_venv * active_mask
                
                # 只在环境第一次真正终止时标记
                just_done = (terminated_venv | truncated_venv) & active_mask
                true_term_accum = true_term_accum | (terminated_venv & active_mask)
                
                # 在环境重置的瞬间，提取它真实的初始位姿作为安全动作！ (仅针对多进程环境)
                if n_envs > 1:
                    for env_idx in range(n_envs):
                        if just_done[env_idx]:
                            # 将重置后状态对应的真实关节角度作为安全动作
                            safe_actions[env_idx] = obs_venv["observation.state"][env_idx][:action_venv.shape[-1]]

                any_done_accum = any_done_accum | terminated_venv | truncated_venv   
                
                # 完成任务或者超时后，重新初始化环境，继续收集数据
                if n_envs == 1 and any_done_accum[0]:
                    obs_venv, _ = env.reset()
                    break

            prev_obs = obs_venv
            
            # 4. 顺手统计回合总奖励 (用于日志打印)
            running_ep_rewards += chunk_reward
            for env_idx in range(n_envs):
                if any_done_accum[env_idx]: # 判断当前环境是否结束
                    completed_ep_rewards.append(running_ep_rewards[env_idx]) # 记录回合总奖励
                    running_ep_rewards[env_idx] = 0.0 # 重置回合总奖励，以便下一回合计算
            
            # 5. 写入轨迹 Buffer，将这 8 步累积的完整奖励和结束标志，交给外部的 PPO 缓冲区
            for k in obs_trajs:
                # 写入刚刚打包好的完整历史帧
                obs_trajs[k][step] = stacked_raw_obs[k]
                
            chains_trajs[step] = chains_venv             # [n_steps, n_envs, denoising_steps + 1, horizon_steps, action_dim]
            reward_trajs[step] = chunk_reward            # [n_steps, n_envs]
            terminated_trajs[step] = true_term_accum     # [n_steps, n_envs]

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
            # 分批次 (Mini-batch) 计算 Critic 价值，防止显存爆炸
            total_samples = n_steps * n_envs
            val_batch_size = getattr(cfg.training, "batch_size", 32) * 2  # 评估不算梯度，batch 可以开大点
            values_flat = np.zeros(total_samples, dtype=np.float32)
            # 每次取 val_batch_size 个样本
            for i in range(0, total_samples, val_batch_size):
                end_i = min(i + val_batch_size, total_samples)
                
                # 临时切下一小块推入 GPU，并顺手除以 255.0
                obs_chunk = {}
                for k, v in obs_k_cpu.items():
                    tensor_v = v[i:end_i].float().to(device)
                    if "images" in k:
                        tensor_v = tensor_v / 255.0
                        # 如果是图像，且是 5D 张量 [Batch, Obs_Steps, C, H, W]
                        # 我们只取最新的一帧观测 (索引 -1) 喂给 Critic
                        if tensor_v.dim() == 5:
                            tensor_v = tensor_v[:, -1]
                    else:
                        # 🌟 新增修复：剥离 State 的历史帧: [B, T, StateDim] -> [B, StateDim]
                        if tensor_v.dim() == 3:
                            tensor_v = tensor_v[:, -1]
                    obs_chunk[k] = tensor_v
                
                # 在送入 Critic 前，使用 Actor 的统计信息进行状态归一化
                obs_chunk_norm = actor.normalize_inputs(obs_chunk.copy())
                values_flat[i:end_i] = critic(obs_chunk_norm).cpu().numpy().flatten()
                
            values_trajs = values_flat.reshape(n_steps, n_envs)
            
            # 计算最后一步的 Next Value (Bootstrap)
            last_obs_ts = {}
            for k, v in prev_obs.items():
                tensor_v = torch.from_numpy(v.copy()).float().to(device)
                if n_envs == 1 and tensor_v.dim() == len(v.shape):
                    tensor_v = tensor_v.unsqueeze(0)
                if "images" in k: 
                    tensor_v = tensor_v / 255.0
                    # 处理最后一步的帧缓存，取最新的一帧
                    if tensor_v.dim() == 5:
                        tensor_v = tensor_v[:, -1]
                else:
                    # 剥离 State 的历史帧: [B, T, StateDim] -> [B, StateDim]
                    if tensor_v.dim() == 3:
                        tensor_v = tensor_v[:, -1]
                last_obs_ts[k] = tensor_v
                
            last_obs_ts_norm = actor.normalize_inputs(last_obs_ts.copy())
            next_values_last = critic(last_obs_ts_norm).cpu().numpy().flatten()


        # 使用 EMA 动态缩放全局 Reward
        batch_reward_std = reward_trajs.std()
        if batch_reward_std > 1e-8:
            if itr == 0:
                # 🚀 冷启动修复：第一轮直接使用真实的批次标准差，瞬间对齐量级！
                running_reward_std = batch_reward_std
            else:
                # 动态更新全局标准差 (95% 的历史记忆 + 5% 的新知识)
                running_reward_std = (1 - ema_alpha) * running_reward_std + ema_alpha * batch_reward_std
            
        # 使用平滑后的全局标尺进行缩放，确保 Critic 的目标是平稳的
        reward_trajs = reward_trajs / running_reward_std

        # 2. 计算 GAE (逆向时间推导)
        advantages_trajs = np.zeros_like(reward_trajs)
        last_gae_lam = 0
        gamma = getattr(cfg.training, "gamma", 0.99)
        gae_lambda = getattr(cfg.training, "gae_lambda", 0.95)

        for t in reversed(range(n_steps)):
            # 获取下一步的 观测评估价值
            next_val = next_values_last if t == n_steps - 1 else values_trajs[t + 1]
            # 判断游戏是否结束，如果nonterminal为0，表示死亡或通关，那么未来价值都是0
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
        # 🌟 优化 1：统一提取 PPO 核心超参数，确保日志与实际运行绝对一致
        batch_size = getattr(cfg.training, "batch_size", 32)
        update_epochs = getattr(cfg.training, "update_epochs", 10)
        clip_ratio = getattr(cfg.training, "clip_ratio", 0.25)

        logging.info(f"🔄 开始 PPO 网络更新 (Epochs: {getattr(cfg.training, 'update_epochs', 4)})...")
        actor.train()
        critic.train()

        # 1. 准备训练用的展平张量
        returns_k = torch.from_numpy(returns_trajs).float().to(device).reshape(-1)
        advantages_k = torch.from_numpy(advantages_trajs).float().to(device).reshape(-1)
        
        # 提取旧的价值预估张量，用于 价值截断Value Clipping，让critic网络更新更稳定
        values_k = torch.from_numpy(values_trajs).float().to(device).reshape(-1)

        # 优势函数归一化 (防止太小更新太慢，极大地提升 PPO 训练稳定性)
        advantages_k = (advantages_k - advantages_k.mean()) / (advantages_k.std() + 1e-8)
        
        # 将 Chains 展平为 [(步数*环境数), 去噪步数, 预测视野, 动作维度]
        chains_k = einops.rearrange(
            torch.from_numpy(chains_trajs).float().to(device),
            "s e t h d -> (s e) t h d"
        )
        # total_steps 表示包含去噪步数的总状态转移次数 (例如：300 * 5 * 10)
        total_steps = n_steps * n_envs * denoising_steps  

        # 获取与去噪步对应的真实 TimeSteps
        actor.diffusion.noise_scheduler.set_timesteps(actor.diffusion.num_inference_steps)
        all_timesteps = actor.diffusion.noise_scheduler.timesteps
        record_start_idx = max(0, len(all_timesteps) - denoising_steps) # 只保留最后 denoising_steps 步
        recorded_timesteps = all_timesteps[record_start_idx:].to(device)

        # =========================================================
        # 7. 预计算旧策略的对数概率 (Old Logprobs)
        # =========================================================
        # 预计算旧概率时，同样按需从 CPU 搬运数据
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

                logprobs = actor.get_logprobs(
                    cond=obs_eval, 
                    x_t=chains_k[b_inds, d_inds], 
                    x_t_1=chains_k[b_inds, d_inds + 1], 
                    timesteps=recorded_timesteps[d_inds]
                )
                old_logprobs_k[inds] = logprobs

        # 2. 开始 Epoch 循环，PPO的数据集可以训练多轮，数据利用率高
        early_stop = False
        for epoch in tqdm(range(update_epochs), desc=f"⏳ PPO 更新中 (Iter {itr+1})", leave=False):
            # 每一轮 Epoch 开始前检查，如果已熔断，彻底跳出 Epoch 循环，开启新的一轮迭代
            if early_stop:
                break
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
                # 创建两个字典，实现数据分流
                obs_b = {}           # 给 Actor （带历史帧）
                obs_b_critic = {}    # 给 Critic（单帧，降维后）
                
                for k, v in obs_k_cpu.items():
                    tensor_v = v[batch_inds_b.cpu()].float().to(device)
                    if "images" in k:
                        tensor_v = tensor_v / 255.0
                    
                    # 1. 完整数据给 Actor
                    obs_b[k] = tensor_v
                    
                    # 取最新一帧给 Critic (完美兼容 Image 和 State)
                    if tensor_v.dim() in [3, 5]: # [B, T, D] 或 [B, T, C, H, W]
                        obs_b_critic[k] = tensor_v[:, -1]
                    else:
                        obs_b_critic[k] = tensor_v
                
                # obs_b = {k: v[batch_inds_b] for k, v in obs_k.items()}        # 取出对应样本的观测，也就是对应环境和步数的观测
                chains_prev_b = chains_k[batch_inds_b, denoising_inds_b]      # 对应样本的当前去噪步 动作
                chains_next_b = chains_k[batch_inds_b, denoising_inds_b + 1]  # 对应样本的下一个去噪步 动作
                returns_b = returns_k[batch_inds_b]                           # 对应样本的总回报
                advantages_b = advantages_k[batch_inds_b]                     # 每一轮去噪步公用一个优势值
                timesteps_b = recorded_timesteps[denoising_inds_b]
                
                # 取出对应样本的旧价值预估
                old_values_b = values_k[batch_inds_b]
                # 取出旧策略的对数概率
                old_logprobs_b = old_logprobs_k[inds_b]
                

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                
                # ----------------------------------------------------
                # 🌟 网络 Loss 计算区 (真正的 PPO 数学魔法)
                # ----------------------------------------------------
                # 1. 计算当前网络对动作的新概率预测
                new_logprobs_b = actor.get_logprobs(
                    cond=obs_b, 
                    x_t=chains_prev_b, 
                    x_t_1=chains_next_b, 
                    timesteps=timesteps_b
                )
                
                # 2. PPO 概率比截断，Ratio = exp(new_logprob - old_logprob)
                log_ratio = new_logprobs_b - old_logprobs_b
                log_ratio = torch.clamp(log_ratio, min=-20.0, max=5.0)
                ratio = torch.exp(log_ratio)
                
                # 3. PPO 截断代理损失 (Clipped Surrogate Objective)
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_b
                pg_loss = -torch.min(surr1, surr2).mean()
                
                # 4. Critic MSE 损失，统一归一化，彻底消灭量纲方差灾难
                obs_b_critic_norm = actor.normalize_inputs(obs_b_critic.copy())
                values_pred = critic(obs_b_critic_norm).squeeze(-1)
                
                # 限制当前价值预测值相对于旧价值的变动幅度，防止价值更新步子太大、训练震荡。
                values_pred_clipped = old_values_b + torch.clamp(
                    values_pred - old_values_b, -clip_ratio, clip_ratio
                )
                # 用smooth_l1_loss代替mse_loss计算两个 Loss，对异常值不敏感，训练更稳
                v_loss_unclipped = torch.nn.functional.smooth_l1_loss(values_pred, returns_b, reduction="none")
                v_loss_clipped = torch.nn.functional.smooth_l1_loss(values_pred_clipped, returns_b, reduction="none")
                
                # 取最大值，意味着只要截断前或截断后的误差有一个变大了，我们就用大的那个惩罚它
                v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                # 5. 总 Loss 汇总
                loss = pg_loss + 0.5 * v_loss   


                with torch.no_grad():
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                # 早期熔断 (Early Stopping)
                if approx_kl > 1.5 * target_kl:
                    logging.warning(f"⚠️ 策略偏离过大 (KL: {approx_kl:.4f} > {1.5 * target_kl})，触发早期熔断！")
                    early_stop = True # 用于跳出当前epoch
                    break # 跳出当前batch


                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                
                # 只有当迭代轮次超过热身期，才允许更新 Actor 的预训练权重
                if itr >= critic_warmup_iters:
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
        
        # ==========================================
        # 步骤 F：定期策略评估、录像与模型快照保存
        # ==========================================
        # 默认每 5 轮评估一次，最后一步强制评估
        eval_freq = getattr(cfg.eval, "eval_freq", 5) 
        is_last_step = (itr + 1) == cfg.training.n_train_itr
        
        if eval_freq > 0 and ((itr + 1) % eval_freq == 0 or is_last_step):
            logging.info(f"\n🎬 开始第 {itr+1} 轮的策略评估与录像...")
            
            # 1. 设定本轮视频的保存路径
            tmp_videos_dir = Path(out_dir) / "eval" / f"videos_{itr+1:06d}"
            
            # 2. 调用 eval.py 中的评估函数 (内部已自动处理 actor.eval() 和 actor.train() 切换)
            # 使用 getattr 安全获取 cfg.eval，如果 yaml 里没配就传一个空字典
            eval_cfg_node = getattr(cfg, "eval", OmegaConf.create())
            
            with torch.autocast(device_type=device.type) if getattr(cfg, "use_amp", False) else nullcontext():
                eval_info = custom_eval_policy(
                    env=eval_env,
                    policy=actor,
                    cfg_eval=eval_cfg_node,   
                    videos_dir=tmp_videos_dir,  # 👈 先存到临时文件夹
                    device=device
                )
            
            # 3. 提取测试成绩
            sr = eval_info["aggregated"]["success_rate"]
            ar = eval_info["aggregated"]["average_reward"]
            logging.info(f"📊 评估完成! 成功率: {sr*100:.1f}%, 平均奖励: {ar:.2f}")

            # 4. 保存模型权重快照 (LeRobot 标准格式)
            ckpt_name = f"{itr+1:06d}_sr={sr:.2f}_reward={ar:.2f}"
            save_path = Path(out_dir) / "checkpoints" / ckpt_name
            final_videos_dir = Path(out_dir) / "eval" / f"videos_{ckpt_name}"

            # 执行文件夹重命名 (把 tmp_videos_... 改成 videos_000005_sr=...)
            if tmp_videos_dir.exists() and tmp_videos_dir != final_videos_dir:
                import shutil
                # 使用 shutil.move 比 Path.rename 更安全，能兼容跨盘操作
                shutil.move(str(tmp_videos_dir), str(final_videos_dir))
                logging.info(f"🎞️ 视频文件夹已重命名为: {final_videos_dir.name}")
            
            actor.save_pretrained(save_path)
            logging.info(f"💾 模型快照已保存至: {save_path}")
            
            # 5. 交给 TopKCheckpointManager 进行同步清理
            # 💡 核心技巧：因为 manager 的逻辑是 loss 越小越保留 (从小到大排序)
            # 我们想保留 Average Reward 最高的，所以传入负的 reward (-ar) 作为假 loss
            ckpt_manager.update(step=itr+1, loss=-ar, ckpt_path=save_path)


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
        "policy=ft_static_diffusion",
        "training.pretrained_ckpt_path='outputs/pretrain/train/2026-04-23/20-30-23_SewNeedle-2Arms-v0_pre_static_wrist_diffusion/checkpoints/0126000_loss=0.0042'",
        "env.n_envs=2",
        "training.rollout_steps=200", 
        "training.batch_size=8",     
        "training.update_epochs=5",     
        "wandb.enable=false",
    ]
    
    for arg in default_args:
        arg_key = arg.split("=")[0]
        if not any(arg_key in sys_arg for sys_arg in sys.argv):
            sys.argv.append(arg)

    train_cli()