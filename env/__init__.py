from gymnasium.envs.registration import register

# ==========================================
# 🌟 环境配置字典
# ==========================================
ENVS = {
    # ==========================================
    #  穿针任务 (3臂)
    # ==========================================
    "sim_envs/SewNeedle-3Arms-v0": {   
        "entry_point": "env.sim_envs:SewNeedleEnv",  # 指向你的文件路径和类名
        "num_arms": 3,
        "episode_length": 300,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
    },
    # 穿针任务 (2臂)
    "sim_envs/SewNeedle-2Arms-v0": {
        "entry_point": "env.sim_envs:SewNeedleEnv",
        "num_arms": 2,
        "episode_length": 300,
        "cameras": ["overhead_cam", "worms_eye_cam", "wrist_cam_left", "wrist_cam_right"],
        "observation_height": 480,
        "observation_width": 640,
    },
    
    # 💡 如果你有其他的任务（比如插孔），可以继续在这里添加：
    # "sim_envs/InsertPeg-3Arms-v0": {
    #     "entry_point": "env.sim_envs:InsertPegEnv",
    #     "num_arms": 3,
    #     "cameras": ["zed_cam_left", "zed_cam_right", "overhead_cam"],
    #     "observation_height": 480,
    #     "observation_width": 640,
    # },
}

# ==========================================
# 🚀 批量注册环境
# ==========================================
for env_id, env_kwargs in ENVS.items():
    register(
        id=env_id,
        entry_point=env_kwargs["entry_point"],
        # nondeterministic=True 告诉 Gym 这个环境的渲染/物理可能不是绝对确定的，防止 check_env 测试报错
        nondeterministic=True, 
        # kwargs 里的参数会直接传递给你 SewNeedleEnv 类的 __init__ 方法
        kwargs={
            "num_arms": env_kwargs["num_arms"],
            "cameras": env_kwargs["cameras"],
            "episode_length": env_kwargs["episode_length"],
            "observation_height": env_kwargs["observation_height"],
            "observation_width": env_kwargs["observation_width"],
        }
    )