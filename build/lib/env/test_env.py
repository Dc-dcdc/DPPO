import gymnasium as gym
import sim_envs  # 这行很重要，它会触发 __init__.py 里的 register
import numpy as np

def main():
    print("环境注册成功，准备加载...")
    # 注意这里传参，把你的 xml 绝对或相对路径传进去
    env = sim_envs.make_sim_env('sim_sew_needle')
    
    obs, info = env.reset()
    print("✅ Reset 成功!")
    print(f"👀 观察到左相机图像形状: {obs['observation.images.zed_cam_left'].shape}")
    print(f"🤖 观察到机械臂状态维度: {obs['observation.state'].shape}")

    print("🚀 开始随机动作测试...")
    for i in range(50):
        # 随机生成一个 21 维的动作
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)
        
        if terminated or truncated:
            print(f"回合在第 {i} 步结束。")
            break

    print("✅ 环境 Step 测试通过！")
    env.close()

if __name__ == "__main__":
    main()

   