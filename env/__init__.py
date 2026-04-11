from gymnasium.envs.registration import register

# 注册你的穿针环境
register(
    id='SewNeedleEnv',                      # 这个名字必须和你 yaml 配置文件里写的一模一样
    entry_point='env.sim_envs:SewNeedleEnv', # 指向你刚才写的类
    max_episode_steps=300,
)