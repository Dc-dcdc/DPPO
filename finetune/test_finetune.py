import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # 开启 PyTorch CUDA 显存动态分片 + 可扩展内存分配，不再一次性预占用大块连续显存
# 没有显示器时使用，比如在服务器上
# os.environ["MUJOCO_GL"] = "egl"
# os.environ["EGL_DEVICE_ID"] = "0"
import sys
import math
import torch
import numpy as np
import logging
import einops
import tempfile
import hydra
import yaml
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from lerobot.common.utils.utils import init_logging, set_global_seed
from pprint import pformat
from lerobot.common.logger import Logger
from tqdm import tqdm
from collections import deque
from lerobot.common.policies.factory import make_policy
# 路径处理
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
import env.sim_envs
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.utils.utils import get_safe_torch_device 
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from finetune.critic import ImageCritic, SharedFeatureCritic
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
    # for i, t in enumerate(timesteps):
    #     # 保存当前的带噪状态 x_t
    #     if return_chain and i >= record_start_idx:
    #         chains.append(trajectory.clone())
            
    #     # 调用底层的 UNet 进行预测，注意时间步的数据类型对齐
    #     timestep_tensor = torch.full(trajectory.shape[:1], t, dtype=torch.long, device=device)
    #     model_output = self.diffusion.unet(
    #         trajectory,
    #         timestep_tensor,
    #         global_cond=global_cond
    #     )
        
    #     # Scheduler 步进：求出更干净的 x_{t-1}
    #     trajectory = self.diffusion.noise_scheduler.step(
    #         model_output, t, trajectory
    #     ).prev_sample

    #     # 注入探索噪声 (除了最后一步)
    #     std = getattr(self.config, "min_sampling_denoising_std", 0.05)
    #     if i < len(timesteps) - 1:
    #         trajectory = trajectory + torch.randn_like(trajectory) * std
    for i, t in enumerate(timesteps):
        # 🌟 保存当前的带噪状态 x_t
        if return_chain and i >= record_start_idx:
            chains.append(trajectory.clone())
            
        timestep_tensor = torch.full(trajectory.shape[:1], t, dtype=torch.long, device=device)
        model_output = self.diffusion.unet(
            trajectory,
            timestep_tensor,
            global_cond=global_cond
        )
        
        # 🌟 核心修复 1：抛弃黑盒 scheduler，使用与 get_logprobs 100% 对齐的 DDIM 数学公式！
        alphas_cumprod = self.diffusion.noise_scheduler.alphas_cumprod.to(device)
        alpha_prod_t = alphas_cumprod[t].view(-1, 1, 1)
        
        step_ratio = self.diffusion.noise_scheduler.config.num_train_timesteps // self.diffusion.num_inference_steps
        prev_t = t - step_ratio
        
        alpha_prod_t_prev = torch.where(
            prev_t >= 0,
            alphas_cumprod[torch.clamp(prev_t, min=0)],
            torch.tensor(1.0, device=device, dtype=trajectory.dtype)
        ).view(-1, 1, 1)
        
        # 严格计算 DDIM 均值 (mu)
        pred_original_sample = (trajectory - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        mu = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + torch.sqrt(1 - alpha_prod_t_prev) * model_output

        #  所有步骤（包括最后一步）必须强制注入探索噪声！
        std = getattr(self.config, "min_sampling_denoising_std", 0.05)
        trajectory = mu + torch.randn_like(mu) * std

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
def get_logprobs(self, cond: dict, x_t: torch.Tensor, x_t_1: torch.Tensor, timesteps: torch.Tensor, return_global_cond=False):
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
    # 这里的 global_cond 就是带着 Actor 视觉权重的特征
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
    std = getattr(self.config, "min_logprob_denoising_std", 0.05)
    var = std ** 2

    # 6. 高斯分布的对数概率公式: -0.5 * ((x - mu)/std)^2 - log(std) - 0.5 * log(2*pi)
    log_prob = -0.5 * ((x_t_1 - mu) ** 2) / var - math.log(std) - 0.5 * math.log(2 * math.pi)
    
    # ==========================================
    # 🌟 官方对齐 1：Action Chunking 真实执行步数截断
    # ==========================================
    act_steps = getattr(self.config, "n_action_steps", 8) 

    log_prob = log_prob[:, :act_steps, :]
    
    # ==========================================
    # 🌟 官方对齐 2：概率限幅 (防止极端负数导致梯度崩塌)
    # ==========================================
    log_prob = torch.clamp(log_prob, min=-5.0, max=2.0)
    
    # ==========================================
    # 🌟 官方对齐 3：在 Horizon 和 ActionDim 维度求均值 (替代 sum)
    # 官方源码：newlogprobs.mean(dim=(-1, -2)).view(-1)
    # ==========================================
    log_prob = log_prob.mean(dim=(-1, -2))

    # 按需返回特征，用于评价网络的resnet视觉底座
    if return_global_cond:
        return log_prob, global_cond
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
        logging.info(f"🔍 检测到 LeRobot 标准快照结构，将自动读取子目录: pretrained_model")
        load_dir = hf_model_dir
    else:
        load_dir = ckpt_path # 兼容其他保存格式
    logging.info(f"💾 正在从目录重建网络并加载权重: {load_dir}")

    hydra_cfg = None
    try:
        from pathlib import Path
        from lerobot.common.utils.utils import init_hydra_config

        # 1. 寻找 config.yaml 路径
        config_yaml_path = Path(load_dir) / "config.yaml"
        if not config_yaml_path.exists():
            config_yaml_path = Path(load_dir).parent / "config.yaml"
            
        if not config_yaml_path.exists():
            raise FileNotFoundError(f"找不到 config.yaml，无法初始化 hydra_cfg！")

        # 2. 根据 yaml 文件初始化 hydra_cfg 对象
        hydra_cfg = init_hydra_config(str(config_yaml_path))

        # 3. 🌟 核心：直接使用 make_policy，让框架接管底层张量与 EMA 加载
        actor = make_policy(
            hydra_cfg=hydra_cfg, 
            pretrained_policy_name_or_path=str(load_dir)
        )
        
        logging.info("✅ 成功使用 make_policy 加载策略！底层 Normalizer 与平滑权重已自动生效。")
        
        # 单独注入模型微调需要的超参数
        actor.config.min_sampling_denoising_std = getattr(
            cfg.training, 
            "min_sampling_denoising_std", 
            0.05
        )
        actor.config.min_logprob_denoising_std = getattr(
            cfg.training, 
            "min_logprob_denoising_std", 
            0.05
        )
        logging.info("✅ 成功将微调配置 (YAML) 注入到 Actor 内部 config 中！")

        # 动态挂载 DPPO 专用的前向与概率计算函数
        actor.forward_dppo = MethodType(forward_dppo, actor) # 将forward_dppo绑定到实例中
        actor.get_logprobs = MethodType(get_logprobs, actor) # 将get_logprobs绑定到实例中
        logging.info("✅ 成功加载 Actor (DiffusionPolicy) 并挂载 DPPO 专用接口！")

    except Exception as e:
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
    render_cams = getattr(cfg.eval, "render_camera", [])
    
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
    # 4.3 动态推断特征维度并初始化 Shared Critic
    # ---------------------------------------------------------
    # 利用预热的空输入，找 Actor 问一下它输出的 global_cond 是多少维
    with torch.no_grad():
        dummy_batch = {k: torch.zeros((1, 2, *v), device=device) for k, v in actor.config.input_shapes.items()}
        if len(actor.expected_image_keys) > 0:
            dummy_batch["observation.images"] = torch.stack([dummy_batch[k] for k in actor.expected_image_keys], dim=-4)
        dummy_cond = actor.diffusion._prepare_global_conditioning(dummy_batch)
        global_cond_dim = dummy_cond.shape[-1]
        
    logging.info(f"🧠 动态探测到 Actor 视觉底座输出特征维度为: {global_cond_dim}")

    actor.to(device)  # 在所有 Gym 子进程全部 Spawn 安全启动后再 手动推入 GPU
    critic = SharedFeatureCritic(global_cond_dim=global_cond_dim).to(device)
    logging.info("✅ 成功初始化 Shared Critic (完美视觉底座共享模式)！")

    # ---------------------------------------------------------
    # 4.4 初始化独立的评估环境与 Top-K 快照管理器
    # ---------------------------------------------------------
    logging.info("🎬 正在初始化独立的评估环境 (Eval Env)...")
    # 评估环境不需要向量化并发，只需一个单例环境即可
    eval_env = gym.make(id=env_id, cameras=obs_cameras)
    
    # 初始化 Top-K 快照管理器 (比如最多保留表现最好的 3 个模型)
    max_checkpoints = getattr(cfg.eval, "max_checkpoints", 5)
    records_resume = getattr(cfg.eval, "records_resume", True)
    checkpoint_metric = getattr(cfg.eval, "checkpoint_metric", "loss")
    manager = TopKCheckpointManager(out_dir=out_dir, 
                                    max_keep=max_checkpoints, 
                                    records_resume=records_resume, 
                                    metric=checkpoint_metric)
    # ==========================================
    # 5. 初始化优化器与超参数
    # ==========================================
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=cfg.training.actor_lr, weight_decay=getattr(cfg.training, "weight_decay", 1e-6))
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

    # Gym 嵌套字典展平工具
    def flatten_lerobot_obs(obs_dict):
        """将环境吐出的嵌套字典翻译成 LeRobot 认识的扁平结构
            {
            "observation.images.zed": [96, 96, 3],
            "observation.images.wrist": [96, 96, 3],
            "observation.state": [0.1, 0.2, 0.5, ...]
            }
        """
        flat_obs = {}
        # 拆解图像字典
        if "pixels" in obs_dict:
            for cam_name, img_array in obs_dict["pixels"].items():
                flat_obs[f"observation.images.{cam_name}"] = img_array
        # 换名本体状态
        if "agent_pos" in obs_dict:
            flat_obs["observation.state"] = obs_dict["agent_pos"]
        # 保留环境可能吐出的其他一维信息（兜底）
        for k, v in obs_dict.items():
            if k not in ["pixels", "agent_pos"]:
                flat_obs[k] = v
        return flat_obs
    
    # 在进入训练循环前，仅全局重置一次环境，保证后续 MDP (马尔可夫决策过程) 的连续性
    prev_obs, _ = env.reset()
    # 将初始画面展平
    prev_obs = flatten_lerobot_obs(prev_obs)
    # 记录当前每个环境正在跑的回合的累计分数
    running_ep_rewards = np.zeros(n_envs, dtype=np.float32)
    # ==========================================
    # 🌟 主循环：DPPO 强化学习全流程
    # ==========================================
    n_obs_steps = getattr(actor.config, "n_obs_steps", 2)
    use_disk_cache = getattr(cfg.training, "use_disk_cache", False)
    for itr in range(cfg.training.n_train_itr):
        logging.info(f"\n========== 第 {itr+1}/{cfg.training.n_train_itr} 轮迭代 ==========")
        
        # ==========================================
        # 1. 初始化 DPPO Rollout 缓冲区 (Buffers)
        # ==========================================
        # 保留最近 n_obs_steps 帧历史观测，deque会自动移除最早的观测
        """
        {"observation.images.top": deque([], maxlen=2),
        "observation.images.wrist": deque([], maxlen=2),
        "observation.state": deque([], maxlen=2)}
        """
        raw_obs_queue = {k: deque(maxlen=n_obs_steps) for k in prev_obs.keys()}
        obs_trajs = None

        if use_disk_cache:
            # 使用硬盘缓存
            temp_buffer_dir = tempfile.TemporaryDirectory()
            buffer_path = temp_buffer_dir.name
            logging.info(f"💾 [缓存模式] 已开启硬盘缓存 (Memmap)，临时目录: {buffer_path}")
            
            chains_trajs = np.memmap(
                os.path.join(buffer_path, 'chains_trajs.npy'), 
                dtype=np.float32, 
                mode='w+',
                shape=(n_steps, n_envs, denoising_steps + 1, horizon_steps, action_dim)
            )         # (步数，环境数，去噪步数，预测步数，动作维度)
        else:
            temp_buffer_dir = None
            logging.info("💾 [缓存模式] 使用纯内存 (RAM) 收集数据，追求极致速度...")
            # (步数，环境数，去噪步数，预测步数，动作维度)
            chains_trajs = np.zeros(
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
                    """
                    # 里面装着两张一模一样的 zed 相机图片矩阵 (H, W, C)
                    "observation.images.zed": deque([
                        array([[[...]]]), # 第 1 帧 (复制的)
                        array([[[...]]])  # 第 2 帧 (当前的)
                    ], maxlen=2),
                    """
                    for _ in range(n_obs_steps):  # 第一步时，复制初始画面填满队列
                        raw_obs_queue[k].append(v)
                else:
                    raw_obs_queue[k].append(v) # 将当前画面压入队列，并且移除最早的一帧

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
                obs_trajs = {}
                for k, v in stacked_raw_obs.items():
                    if use_disk_cache:
                        safe_k = k.replace(".", "_")
                        obs_trajs[k] = np.memmap(
                            os.path.join(buffer_path, f'obs_{safe_k}.npy'),
                            dtype=v.dtype,
                            mode='w+',
                            shape=(n_steps, *v.shape)
                        )
                    else:
                        obs_trajs[k] = np.zeros((n_steps, *v.shape), dtype=v.dtype)

            # 1. 格式化当前单帧观测给 Actor 去推断 ，forward_dppo中会补成2帧进行推理
            batch_obs = {}
            for k, v in prev_obs.items():
                # 只提取模型 config 中真正需要的输入特征，防止 environment_state 引发报错
                if k not in actor.config.input_shapes:
                    continue
                safe_v = np.ascontiguousarray(v)
                tensor_v = torch.from_numpy(safe_v).float().to(device)

                # 兼容单环境
                if n_envs == 1 and tensor_v.dim() == len(v.shape):
                    # [H, W, C] -> [1, H, W, C]  [state] -> [1, state]
                    tensor_v = tensor_v.unsqueeze(0)
                
                # 图像归一化与通道换位 (HWC -> CHW)
                if "images" in k:
                    #  [Batch, H, W, C] -> [Batch, C, H, W]
                    tensor_v = tensor_v.permute(0, 3, 1, 2) / 255.0
                    
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
                # 环境交互后，需要将 obs 展平成 LeRobot 认识的扁平结构
                obs_venv = flatten_lerobot_obs(obs_venv)
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

        logging.info("✅ 数据收集 (Rollout) 完成，准备进入 PPO 网络更新阶段！")
                
        # =========================================================
        # 5. 批量计算价值 (Values) 与 GAE 优势函数
        # =========================================================
        logging.info("🧠 计算状态价值 (Values) 与优势函数 (GAE)...")
        
        # 利用 reshape 虚拟展平 (不管底层是 RAM 还是 Disk，均不触发全量内存加载)
        # 将 Chains 展平为 [(步数*环境数), 去噪步数, 预测视野, 动作维度]
        chains_flat = chains_trajs.reshape(n_steps * n_envs, denoising_steps + 1, horizon_steps, action_dim)
        
        obs_flat = {}
        for k, v in obs_trajs.items():
            obs_flat[k] = v.reshape(n_steps * n_envs, *v.shape[2:])
        
        with torch.no_grad():
            # 分批次 (Mini-batch) 计算 Critic 价值，防止显存爆炸
            total_samples = n_steps * n_envs
            val_batch_size = getattr(cfg.training, "batch_size", 32) * 2  # 评估不算梯度，batch 可以开大点
            values_flat = np.zeros(total_samples, dtype=np.float32)
            # 每次取 val_batch_size 个样本
            for i in range(0, total_samples, val_batch_size):
                end_i = min(i + val_batch_size, total_samples)
                
                # 临时切下一小块推入 GPU
                obs_chunk = {}
                for k, v in obs_flat.items():
                    tensor_v = torch.from_numpy(v[i:end_i]).float().to(device)
                    if "images" in k:
                        # [B, T, H, W, C] -> [B, T, C, H, W]
                        tensor_v = tensor_v.permute(0, 1, 4, 2, 3) / 255.0
                    obs_chunk[k] = tensor_v
                
                # 🌟 调用 Actor 的特征提取底座
                obs_chunk_norm = actor.normalize_inputs(obs_chunk)
                if len(actor.expected_image_keys) > 0:
                    obs_chunk_norm["observation.images"] = torch.stack([obs_chunk_norm[k] for k in actor.expected_image_keys], dim=-4)
                
                # 获取融合特征并给 Critic 估值
                global_cond = actor.diffusion._prepare_global_conditioning(obs_chunk_norm)
                values_flat[i:end_i] = critic(global_cond.detach()).cpu().numpy().flatten()
                
            values_trajs = values_flat.reshape(n_steps, n_envs)
            
            # 计算最后一步的 Next Value (Bootstrap)
            # 1. 补齐时间维度：将最新的单帧 prev_obs 塞入历史队列
            # 因为 Actor 的视觉底座需要 n_obs_steps 层历史帧才能前向传播
            for k, v in prev_obs.items():
                raw_obs_queue[k].append(v)
                
            last_obs_ts = {}
            for k, v in prev_obs.items():
                # 过滤不需要的键值
                if k not in actor.config.input_shapes:
                    continue
                    
                # 从队列中提取包含了完整历史帧 (Time) 的数据
                stacked_v = np.stack(list(raw_obs_queue[k]), axis=0 if n_envs == 1 else 1)
                tensor_v = torch.from_numpy(stacked_v).float().to(device)
                
                # 兼容单环境: [Time, ...] -> [1, Time, ...]
                if n_envs == 1 and tensor_v.dim() == len(stacked_v.shape):
                    tensor_v = tensor_v.unsqueeze(0)
                    
                if "images" in k: 
                    # 此时已经是 5D 张量: [Batch, Time, H, W, C] -> [Batch, Time, C, H, W]
                    tensor_v = tensor_v.permute(0, 1, 4, 2, 3) / 255.0
                    
                last_obs_ts[k] = tensor_v
                
            # 2. 调用 Actor 逻辑进行统计学归一化
            last_obs_ts_norm = actor.normalize_inputs(last_obs_ts.copy())
            
            # 3. 按 LeRobot 的规范，将字典里的所有相机图像按 Num_Cams 维度堆叠
            if len(actor.expected_image_keys) > 0:
                last_obs_ts_norm["observation.images"] = torch.stack(
                    [last_obs_ts_norm[k] for k in actor.expected_image_keys], dim=-4
                )
                
            # 4. 调用共享底座提取终极特征 (global_cond)
            global_cond_last = actor.diffusion._prepare_global_conditioning(last_obs_ts_norm)
            
            # 5. 让盲人军师（Critic）对最后一步打分！
            next_values_last = critic(global_cond_last.detach()).cpu().numpy().flatten()


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

        # 🌟 核心修复：强制冻结视觉底座的 BatchNorm 层，防止小 Batch 引起的统计量震荡导致的虚假 KL 爆炸
        def freeze_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
                # 可选：如果完全不想更新底座特征，连梯度也关掉
                # m.weight.requires_grad = False
                # m.bias.requires_grad = False

        # 应用到 Actor 的视觉底座和 Critic 网络
        actor.apply(freeze_bn)
        critic.apply(freeze_bn)

        # 1. 准备训练用的展平张量
        returns_k = torch.from_numpy(returns_trajs).float().to(device).reshape(-1)
        advantages_k = torch.from_numpy(advantages_trajs).float().to(device).reshape(-1)
        
        # 提取旧的价值预估张量，用于 价值截断Value Clipping，让critic网络更新更稳定
        values_k = torch.from_numpy(values_trajs).float().to(device).reshape(-1)

        # 优势函数归一化 (防止太小更新太慢，极大地提升 PPO 训练稳定性)
        advantages_k = (advantages_k - advantages_k.mean()) / (advantages_k.std() + 1e-8)

        # 强行剃掉前 5% 和后 5% 的极端数据，防止网络被带偏
        adv_lower = torch.quantile(advantages_k, 0.05)
        adv_upper = torch.quantile(advantages_k, 0.95)
        advantages_k = torch.clamp(advantages_k, min=adv_lower, max=adv_upper)

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
                # ===================================================
                # 🌟 懒加载兼容修改 1：将 GPU 索引转为 CPU Numpy 索引，才能切片 memmap
                # ===================================================
                b_inds_np = b_inds.cpu().numpy()
                d_inds_np = d_inds.cpu().numpy()

                # 从硬盘/内存按需拉取画面到 GPU
                obs_eval = {}
                for k, v in obs_flat.items():
                    tensor_v = torch.from_numpy(v[b_inds_np]).float().to(device)
                    if "images" in k:
                        # [Batch, Time, H, W, C] -> [Batch, Time, C, H, W]
                        tensor_v = tensor_v.permute(0, 1, 4, 2, 3)/ 255.0
                    obs_eval[k] = tensor_v
                chains_prev_eval = torch.from_numpy(chains_flat[b_inds_np, d_inds_np]).float().to(device)
                chains_next_eval = torch.from_numpy(chains_flat[b_inds_np, d_inds_np + 1]).float().to(device)
                logprobs = actor.get_logprobs(
                    cond=obs_eval, 
                    x_t=chains_prev_eval, 
                    x_t_1=chains_next_eval, 
                    timesteps=recorded_timesteps[d_inds],
                    return_global_cond=False
                )
                old_logprobs_k[inds] = logprobs

        running_v_loss = []
        running_pg_loss = []

        # 2. 开始 Epoch 循环，PPO的数据集可以训练多轮，数据利用率高
        early_stop = False
        for epoch in tqdm(range(update_epochs), desc=f"⏳ PPO 更新中 (Iter {itr+1})", leave=False):
            # 每一轮 Epoch 开始前检查，如果已熔断，彻底跳出 Epoch 循环，开启新的一轮迭代
            if early_stop:
                break
            # 打乱所有数据点 (不仅打乱时间步，还打乱去噪步)
            indices = torch.randperm(total_steps, device=device)
            num_batch = max(1, total_steps // batch_size) #分批次训练
            
            for batch_idx in tqdm(range(num_batch), desc=f"   📦 Batch 更新", leave=False):
                start = batch_idx * batch_size
                end = start + batch_size
                inds_b = indices[start:end] #从打乱的总数据中提取一个batch的索引
                
                # 将inds_b索引值对应到哪批样本batch_inds_b 的 哪个去噪步denoising_inds_b
                batch_inds_b, denoising_inds_b = torch.unravel_index(
                    inds_b, 
                    (n_steps * n_envs, denoising_steps) #n_steps * n_envs行，denoising_steps列
                )
                
                # ===================================================
                # 将 GPU 上的批量索引转为 CPU Numpy 格式
                # ===================================================
                b_inds_np = batch_inds_b.cpu().numpy()
                d_inds_np = denoising_inds_b.cpu().numpy()
                
                # 切片提取 Mini-batch
                obs_b = {}           # 给 Actor （带历史帧）
                
                for k, v in obs_flat.items():
                    tensor_v = torch.from_numpy(v[b_inds_np]).float().to(device)
                    if "images" in k:
                        # [Batch, Time, H, W, C] -> [Batch, Time, C, H, W]
                        tensor_v = tensor_v.permute(0, 1, 4, 2, 3)/ 255.0
                    
                    # 完整数据给 Actor
                    obs_b[k] = tensor_v
                


                # ===================================================
                # 从硬盘 (或系统内存) 按需拉取这一小批动作切片到 GPU
                # ===================================================
                chains_prev_b = torch.from_numpy(chains_flat[b_inds_np, d_inds_np]).float().to(device)      
                chains_next_b = torch.from_numpy(chains_flat[b_inds_np, d_inds_np + 1]).float().to(device)
                returns_b = returns_k[batch_inds_b]                           # 对应样本的总回报
                advantages_b = advantages_k[batch_inds_b]                     # 每一轮去噪步公用一个优势值
                timesteps_b = recorded_timesteps[denoising_inds_b]
                
                # ==========================================
                # 去噪步骤的优势衰减折扣
                # 越接近输出端 (denoising_inds_b 越大)，折扣越接近 1.0
                # ==========================================
                gamma_denoising = getattr(cfg.training, "gamma_denoising", 0.99)
                # 公式完全复刻官方: gamma ** (ft_denoising_steps - i - 1)
                discount = gamma_denoising ** (denoising_steps - denoising_inds_b - 1)
                advantages_b = advantages_b * discount

                # 取出对应样本的旧价值预估
                old_values_b = values_k[batch_inds_b]
                # 取出旧策略的对数概率
                old_logprobs_b = old_logprobs_k[inds_b]
                
                #  使用 set_to_none=True 节省大量显存
                actor_optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)
                
                # ----------------------------------------------------
                # 🌟 网络 Loss 计算区 (共享计算底座)
                # ----------------------------------------------------
                # 1. 计算当前网络对动作的新概率预测，同时返回全局特征
                new_logprobs_b, global_cond_b = actor.get_logprobs(
                    cond=obs_b, 
                    x_t=chains_prev_b, 
                    x_t_1=chains_next_b, 
                    timesteps=timesteps_b,
                    return_global_cond=True  
                )
                
                # 2. PPO 概率比截断，Ratio = exp(new_logprob - old_logprob)
                log_ratio = new_logprobs_b - old_logprobs_b
                log_ratio = torch.clamp(log_ratio, min=-20.0, max=5.0)
                ratio = torch.exp(log_ratio)
                
                # 3. PPO 截断代理损失，把控单次更新范围
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_b
                pg_loss = -torch.min(surr1, surr2).mean()
                
                # 4. Critic 的打分损失 (🌟 军师直接看 Actor 传来的带有梯度的特征图！)
                values_pred = critic(global_cond_b).squeeze(-1)
                
                # 限制当前价值预测值相对于旧价值的变动幅度，防止价值更新步子太大、训练震荡。
                values_pred_clipped = old_values_b + torch.clamp(
                    values_pred - old_values_b, -clip_ratio, clip_ratio
                )
                # 用smooth_l1_loss代替mse_loss计算两个 Loss，对异常值不敏感，训练更稳
                v_loss_unclipped = torch.nn.functional.smooth_l1_loss(values_pred, returns_b, reduction="none")
                v_loss_clipped = torch.nn.functional.smooth_l1_loss(values_pred_clipped, returns_b, reduction="none")
                
                # 取最大值，意味着只要截断前或截断后的误差有一个变大了，我们就用大的那个惩罚它
                v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                # 5. 总 Loss 汇总, 反向传播时，pg_loss流向策略网络，v_loss流向价值网络
                loss = pg_loss + 0.5 * v_loss   

                running_v_loss.append(v_loss.item())
                running_pg_loss.append(pg_loss.item())

                with torch.no_grad():
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                # 早期熔断 (Early Stopping)，把控整体策略偏移量
                if approx_kl > target_kl:
                    logging.warning(f"⚠️ 策略偏离过大 (KL: {approx_kl:.4f} > {target_kl})，触发早期熔断！")
                    early_stop = True # 用于跳出当前epoch

                    # 🌟 修复 2：极其重要！必须手动销毁所有带梯度的局部变量，释放几 GB 的计算图！
                    try:
                        del loss, pg_loss, v_loss, new_logprobs_b, log_ratio, ratio, surr1, surr2, values_pred, v_loss_unclipped, v_loss_clipped
                    except Exception:
                        pass
                    
                    # 清空优化器里的残余状态，并强制清空显存缓存
                    actor_optimizer.zero_grad(set_to_none=True)
                    critic_optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                    break # 跳出当前batch


                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)

                # critic网络需要一直更新
                critic_optimizer.step()
                # 只有当迭代轮次超过热身期，才允许更新 Actor 的预训练权重
                if itr >= critic_warmup_iters:
                    actor_optimizer.step()
                    # 🌟 核心修复 3：同步更新模型的 EMA 平滑权重！这是评估成绩的关键！
                    if hasattr(actor, "update"):
                        with torch.no_grad():
                            actor.update()
                
        # =========================================================
        # 🌟 修复 4：PPO 阶段结束，清扫全部巨大的训练经验张量，让显存归还给评估和下一次 Rollout
        # =========================================================
        try:
            del chains_flat, returns_k, advantages_k, values_k, old_logprobs_k, obs_flat
            # 如果开启了硬盘缓存，必须显式调用 cleanup() 删除物理文件
            if use_disk_cache and temp_buffer_dir is not None:
                temp_buffer_dir.cleanup()
        except Exception:
            pass
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        # ------------------------------------------
        # 步骤 E：打印评估指标 (替代原论文代码)
        # ------------------------------------------
        if len(completed_ep_rewards) > 0:
            avg_ep_return = np.mean(completed_ep_rewards)
            max_ep_return = np.max(completed_ep_rewards)
            avg_v_loss = np.mean(running_v_loss) if running_v_loss else 0.0
            avg_pg_loss = np.mean(running_pg_loss) if running_pg_loss else 0.0
            logging.info(f"✅ 第 {itr+1} 轮完成！")
            logging.info(f"   🏃 本轮共完成回合数: {len(completed_ep_rewards)}")
            logging.info(f"   💰 平均回合总奖励 (Return): {avg_ep_return:.2f}")
            logging.info(f"   🏆 最高回合奖励: {max_ep_return:.2f}")
            # 🌟 打印平均 Loss
            logging.info(f"   📉 Critic (Value) Loss: {avg_v_loss:.4f}")
            logging.info(f"   📉 Actor (Policy) Loss: {avg_pg_loss:.4f}")
            
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
        
        if ((itr + 1) > critic_warmup_iters) and ((itr + 1) % eval_freq == 0 or is_last_step):
            logging.info(f"\n🎬 开始第 {itr+1} 轮的策略评估与录像...")
            
            # 1. 设定本轮视频的保存路径
            tmp_videos_dir = Path(out_dir) / "eval" / f"videos_{itr+1:06d}"
            
            # 2. 调用 eval.py 中的评估函数 (内部已自动处理 actor.eval() 和 actor.train() 切换)
            # 使用 getattr 安全获取 cfg.eval，如果 yaml 里没配就传一个空字典
            eval_cfg_node = getattr(cfg, "eval", OmegaConf.create())
            
            with torch.no_grad():
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
            ckpt_name = f"{itr+1:06d}_sr={sr:.2f}_reward={ar:.2f}_Ploss={avg_pg_loss:.4f}_Vloss={avg_v_loss:.4f}"
            ckpt_path = Path(out_dir) / "checkpoints" / ckpt_name
            save_path = Path(out_dir) / "checkpoints" / ckpt_name / "pretrained_model"
            final_videos_dir = Path(ckpt_path) / "eval" / f"eval_videos"

            # 执行文件夹重命名 (把 tmp_videos_... 改成 videos_000005_sr=...)
            if tmp_videos_dir.exists() and tmp_videos_dir != final_videos_dir:
                import shutil
                # 使用 shutil.move 比 Path.rename 更安全，能兼容跨盘操作
                shutil.move(str(tmp_videos_dir), str(final_videos_dir))
                logging.info(f"🎞️ 视频文件夹已重命名为: {final_videos_dir.name}")
            
            actor.save_pretrained(save_path)

            # 融合预训练底层网络配置与当前微调环境，生成完美的 config.yaml
            save_path.mkdir(parents=True, exist_ok=True) # 确保目录已创建
            config_out_path = save_path / "config.yaml"
            
            if hydra_cfg is not None:
                # 1. 以包含完美网络结构的预训练配置为基础字典
                final_config_dict = OmegaConf.to_container(hydra_cfg, resolve=True)
                
                # 2. 将当前微调使用的真实环境、评估参数以及 policy 微调字段覆盖进去
                current_ft_dict = OmegaConf.to_container(cfg, resolve=True)
                final_config_dict["env"] = current_ft_dict.get("env", final_config_dict.get("env"))
                final_config_dict["eval"] = current_ft_dict.get("eval", final_config_dict.get("eval"))
                
                # 3. 融合 policy 节点（保留预训练结构的同时，加入微调可能新增的参数）
                if "policy" in final_config_dict and "policy" in current_ft_dict:
                    final_config_dict["policy"].update(current_ft_dict["policy"])
            else:
                # 容错降级方案
                final_config_dict = OmegaConf.to_container(cfg, resolve=True)

            # 4. 正式写出文件
            with open(config_out_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    final_config_dict, 
                    f, 
                    allow_unicode=True,
                    sort_keys=False # 保持原始 yaml 顺序，方便人类阅读
                )
            
            logging.info(f"💾 模型快照及完美的混合 config.yaml 已完整保存至: {save_path}")

            
            # 5. 交给 TopKCheckpointManager 进行同步清理
            manager.update(step=itr+1, loss=avg_pg_loss,  ckpt_path=ckpt_path, reward=ar)
        elif (itr + 1) <= critic_warmup_iters:
            logging.info(f"Critic 正在预热阶段 ({itr+1}/{critic_warmup_iters})，本轮暂不评估 Actor 模型")
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
        "policy=ft_zed_wrist_diffusion",
        "training.pretrained_ckpt_path='outputs/1.hugging_model/pre_sim_sew_needle_3arms_zed_wrist_diffusion'",
        "env.n_envs=10",
        "training.rollout_steps=200", 
        "training.batch_size=8",     
        "training.update_epochs=2",     
        "wandb.enable=false",
    ]
    
    for arg in default_args:
        arg_key = arg.split("=")[0]
        if not any(arg_key in sys_arg for sys_arg in sys.argv):
            sys.argv.append(arg)

    train_cli()