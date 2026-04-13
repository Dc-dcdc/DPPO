import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import mjcf
import mujoco.viewer


from env.constants import (
    XML_DIR,
    SIM_DT, SIM_PHYSICS_DT, SIM_PHYSICS_ENV_STEP_RATIO,
    LEFT_ARM_POSE, RIGHT_ARM_POSE, MIDDLE_ARM_POSE,
    LEFT_JOINT_NAMES, RIGHT_JOINT_NAMES, MIDDLE_JOINT_NAMES,
    LEFT_ACTUATOR_NAMES, RIGHT_ACTUATOR_NAMES, MIDDLE_ACTUATOR_NAMES,
    LEFT_EEF_SITE, RIGHT_EEF_SITE, MIDDLE_EEF_SITE, MIDDLE_BASE_LINK,
    LEFT_GRIPPER_JOINT_NAMES, RIGHT_GRIPPER_JOINT_NAMES
)

CAMERAS = ['zed_cam_left', 'zed_cam_right', 'wrist_cam_left', 'wrist_cam_right', 'overhead_cam', 'worms_eye_cam']

class GuidedVisionEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, xml_path, cameras=CAMERAS): 
        super().__init__()
        # self.num_envs = 1
        # ==========================================
        # 🌟 1. 加载物理模型
        # ==========================================
        self._mjcf_root = mjcf.from_path(xml_path)  
        self._mjcf_root.option.timestep = SIM_PHYSICS_DT  
        self._physics = mjcf.Physics.from_mjcf_model(self._mjcf_root) 
        self.cameras = cameras # 使用的摄像头列表
        # ==========================================
        # 🌟 2. 构建观察空间 (Observation Space)
        # ==========================================
        obs_spaces = {}
        # 动态遍历并注册所有的相机图像空间
        for cam_name in self.cameras:
            # 这里的键名严格对齐 LeRobot 的规范：observation.images.xxxx
            obs_spaces[f'observation.images.{cam_name}'] = spaces.Box(
                low=0, high=255, shape=(3, 480, 640), dtype=np.uint8
            )
            
        # 注册 21维本体状态
        obs_spaces['observation.state'] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        ) 
        
        self.observation_space = spaces.Dict(obs_spaces)
        
        # ==========================================
        # 🌟 3. 定义动作空间 (Action Space): 21维关节目标角度
        # ==========================================
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )   
        
        # ==========================================
        # 🌟 4. 寻址与绑定 MJCF 节点，和底层的 MuJoCo XML 物理模型之间建立连接
        # ==========================================
        self._middle_base_link = self._mjcf_root.find('body', MIDDLE_BASE_LINK)
        self._middle_base_link_init_pos = self._middle_base_link.pos.copy()
        # 绑定关节，用于读取关节角度
        self._left_joints = [self._mjcf_root.find('joint', name) for name in LEFT_JOINT_NAMES]
        self._right_joints = [self._mjcf_root.find('joint', name) for name in RIGHT_JOINT_NAMES]
        self._middle_joints = [self._mjcf_root.find('joint', name) for name in MIDDLE_JOINT_NAMES]
        # 绑定动作，用于发送控制指令
        self._left_actuators = [self._mjcf_root.find('actuator', name) for name in LEFT_ACTUATOR_NAMES]
        self._right_actuators = [self._mjcf_root.find('actuator', name) for name in RIGHT_ACTUATOR_NAMES]
        self._middle_actuators = [self._mjcf_root.find('actuator', name) for name in MIDDLE_ACTUATOR_NAMES]
        # 绑定夹爪关节进行单独控制
        self._left_gripper_joints = [self._mjcf_root.find('joint', name) for name in LEFT_GRIPPER_JOINT_NAMES]
        self._right_gripper_joints = [self._mjcf_root.find('joint', name) for name in RIGHT_GRIPPER_JOINT_NAMES]

        # ==========================================
        # 🌟 5. 夹爪归一化函数
        # ==========================================
        # 读取真实物理引擎中夹爪电机的控制限位
        self.left_gripper_range = self._physics.bind(self._left_actuators[-1]).ctrlrange 
        self.right_gripper_range = self._physics.bind(self._right_actuators[-1]).ctrlrange
        # 归一化：(x - min) / (max - min)
        self.left_gripper_norm_fn = lambda x: (x - self.left_gripper_range[0]) / (self.left_gripper_range[1] - self.left_gripper_range[0])
        self.right_gripper_norm_fn = lambda x: (x - self.right_gripper_range[0]) / (self.right_gripper_range[1] - self.right_gripper_range[0])
        # 反归一化:将模型输出映射到实际控制量 y = x * (max - min) + min
        self.left_gripper_unnorm_fn = lambda x: x * (self.left_gripper_range[1] - self.left_gripper_range[0]) + self.left_gripper_range[0]
        self.right_gripper_unnorm_fn = lambda x: x * (self.right_gripper_range[1] - self.right_gripper_range[0]) + self.right_gripper_range[0]

        self._viewer = None 

    def get_obs(self) -> dict:
        """获取并格式化模型输入所需的字典"""
        # ==========================================
        # 🌟 1. 提取本体感知状态 (Proprioceptive State)
        # ==========================================
        left_qpos = self._physics.bind(self._left_joints).qpos.copy() 
        left_qpos[6] = self.left_gripper_norm_fn(left_qpos[6]) # 夹爪数据归一化
        
        right_qpos = self._physics.bind(self._right_joints).qpos.copy()
        right_qpos[6] = self.right_gripper_norm_fn(right_qpos[6])
        
        middle_qpos = self._physics.bind(self._middle_joints).qpos.copy()
        
        state_21d = np.concatenate([left_qpos, right_qpos, middle_qpos]).astype(np.float32)

        # ==========================================
        # 🌟 2. 渲染相机图像并转换通道为 (C, H, W)
        # ==========================================
        # 2. 准备返回的字典
        obs_dict = {'observation.state': state_21d}
        for cam_name in self.cameras:
            # 注意：这里的 camera_id 必须和你的 XML 文件里 <camera name="..."> 的名字完全一致！
            try:
                img = self._physics.render(height=480, width=640, camera_id=cam_name)
                img = np.transpose(img, (2, 0, 1))# 转换通道 (H, W, C) -> (C, H, W)
                # 存入字典，键名严格对齐
                obs_dict[f'observation.images.{cam_name}'] = img
            except Exception as e:
                raise ValueError(f"❌ 渲染相机 '{cam_name}' 失败！请检查 XML 文件中是否有这个名字的相机。报错详情: {e}")

        return obs_dict

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)
        self._physics.reset()
        
        # 恢复默认位姿
        self._physics.bind(self._left_joints).qpos = LEFT_ARM_POSE 
        self._physics.bind(self._left_gripper_joints).qpos = self.left_gripper_unnorm_fn(1) # 夹爪张开到最大
        self._physics.bind(self._right_joints).qpos = RIGHT_ARM_POSE
        self._physics.bind(self._right_gripper_joints).qpos = self.right_gripper_unnorm_fn(1)
        self._physics.bind(self._middle_joints).qpos = MIDDLE_ARM_POSE
        # 初始化控制器
        self._physics.bind(self._left_actuators).ctrl = LEFT_ARM_POSE 
        self._physics.bind(self._left_actuators[6]).ctrl = self.left_gripper_unnorm_fn(1) 
        self._physics.bind(self._right_actuators).ctrl = RIGHT_ARM_POSE
        self._physics.bind(self._right_actuators[6]).ctrl = self.right_gripper_unnorm_fn(1)
        self._physics.bind(self._middle_actuators).ctrl = MIDDLE_ARM_POSE
        # 强制物理引擎进行一次正向运动学计算
        self._physics.forward() 
        # 读取当前环境的观测
        observation = self.get_obs()
        info = {"message": "Environment reset successfully."}
        return observation, info   

    def step(self, action: np.ndarray) -> tuple:
        """Gymnasium 标准步进函数"""
        # 1. 动作拆包
        left_joints = action[:6]
        left_gripper = np.clip(action[6], 0.0, 1.0) # 0.0 到 1.0 之间的归一化值
        right_joints = action[7:13]
        right_gripper = np.clip(action[13], 0.0, 1.0)
        middle_joints = action[14:21]

        # 2. 映射到物理引擎执行器
        self._physics.bind(self._left_actuators[:6]).ctrl = left_joints
        self._physics.bind(self._right_actuators[:6]).ctrl = right_joints
        self._physics.bind(self._middle_actuators).ctrl = middle_joints
        self._physics.bind(self._left_actuators[6]).ctrl = self.left_gripper_unnorm_fn(left_gripper)
        self._physics.bind(self._right_actuators[6]).ctrl = self.right_gripper_unnorm_fn(right_gripper)

        # 3. 步进物理引擎
        for _ in range(SIM_PHYSICS_ENV_STEP_RATIO): self._physics.step()
        # self._physics.step(nstep=SIM_PHYSICS_ENV_STEP_RATIO)
        
        # 4. 获取观察与奖励
        observation = self.get_obs()
        reward = self.get_reward() if hasattr(self, 'get_reward') else 0.0
        
        # 5. 判断终止条件
        max_rwd = getattr(self, 'max_reward', -1)
        terminated = (reward == max_rwd) # 达到最大奖励则任务成功结束
        truncated = False
        
        info = {"is_success": terminated, "reward": reward}

        return observation, reward, terminated, truncated, info

    def render(self,render_camera):
        """
        Gymnasium 标准渲染接口，供 LeRobot 等高级框架录制视频时调用。
        必须返回 (H, W, C) 维度的 numpy 图像矩阵。
        """
        # 选择一个最好的相机视角用来生成测试录像（比如用左目相机）
        # 这里默认使用 self.cameras 列表里的第一个相机
        # render_cam = self.cameras[0] if len(self.cameras) > 0 else 'zed_cam_left'
        render_cam = render_camera[0] if len(render_camera) > 0 else 'overhead_cam'
        
        try:
            # MuJoCo 的 render 默认输出的就是标准的 (H, W, C) rgb_array
            img = self._physics.render(height=480, width=640, camera_id=render_cam)
            return img
        except Exception as e:
            # 防止万一没找到相机导致崩溃
            print(f"⚠️ 渲染视频帧失败: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
    def render_viewer(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr, self._physics.data.ptr,
                show_left_ui=True, show_right_ui=True,
            )
        self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()


class SewNeedleEnv(GuidedVisionEnv):
    """
    缝合针穿引任务专用环境
    """
    def __init__(self,cameras):
        xml_path = os.path.join(XML_DIR, 'task_sew_needle.xml')
        super().__init__(xml_path, cameras=cameras)

        self.max_reward = 5
        self._needle_joint = self._mjcf_root.find('joint', 'needle_joint')
        self._wall_joint = self._mjcf_root.find('joint', 'wall_joint')
        self._threaded_needle = False

    def reset(self, seed=None, options=None) -> tuple:     
        super().reset(seed=seed)
        # 随机化针的位置
        x_range = [0.15, 0.2]
        y_range = [-.025, 0.1]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        needle_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        needle_quat = np.array([1, 0, 0, 0])

        # 随机化墙(洞)的位置
        x_range = [-0.025, 0.025]
        y_range = [-.025, 0.1]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        wall_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        wall_quat = np.array([1, 0, 0, 0]) 

        self._physics.bind(self._needle_joint).qpos = np.concatenate([needle_position, needle_quat])
        self._physics.bind(self._wall_joint).qpos = np.concatenate([wall_position, wall_quat])

        self._physics.forward()
        self._threaded_needle = False

        observation = self.get_obs()
        info = {"message": "SewNeedle env reset."}
        return observation, info
    
    def get_reward(self):
        touch_left_gripper = False
        touch_right_gripper = False
        needle_touch_table = False
        needle_touch_wall = False
        needle_touch_pin = False

        # 碰撞检测
        contact_pairs = []
        for i_contact in range(self._physics.data.ncon):
            id_geom_1 = self._physics.data.contact[i_contact].geom1
            id_geom_2 = self._physics.data.contact[i_contact].geom2
            geom1 = self._physics.model.id2name(id_geom_1, 'geom')
            geom2 = self._physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1 == "needle" and geom2.startswith("right"):
                touch_right_gripper = True    # 右手碰到了针
            if geom1 == "needle" and geom2.startswith("left"):
                touch_left_gripper = True     # 左手碰到了针
            if geom1 == "table" and geom2 == "needle":
                needle_touch_table = True     # 针接触到桌面
            if geom1 == "needle" and geom2.startswith("wall-"):
                needle_touch_wall = True      # 针接触到墙
            if geom1 == "pin-needle" and geom2 == "pin-wall":
                self._threaded_needle = True  # 针穿过墙
            if geom1 == "needle" and geom2 == "pin-wall":
                needle_touch_pin = True       # 针接触到墙

        # 塑造奖励 (Reward Shaping)
        reward = 0
        if touch_right_gripper:                                 # 右臂碰到了针
            reward = 1
        if touch_right_gripper and (not needle_touch_table):    # 右臂成功抓起了针
            reward = 2
        if needle_touch_wall and (not needle_touch_table):      # 针接触到墙壁，且未掉落
            reward = 3
        if self._threaded_needle:                               # 穿针成功
            reward = 4
        if touch_left_gripper and (not touch_right_gripper) and (not needle_touch_table) and (not needle_touch_pin) and self._threaded_needle: # 完美穿针并成功换手交接
            reward = 5
            
        return reward

# ==========================================
# 本地测试代码 (确保环境逻辑完美运行)
# ==========================================
if __name__ == '__main__':
    print("🚀 初始化 SewNeedle 环境...")
    env = SewNeedleEnv(cameras=['zed_cam_left', 'zed_cam_right'])
    obs, info = env.reset()
    
    print("✅ 环境初始化成功！")
    print(f"📷 左目相机输出维度: {obs['observation.images.zed_cam_left'].shape} (应为 3, 480, 640)")
    print(f"🤖 关节状态输出维度: {obs['observation.state'].shape} (应为 21,)")
    
    # 获取初始夹爪状态，确保它们是归一化的 (0到1之间)
    left_gripper_state = obs['observation.state'][6]
    right_gripper_state = obs['observation.state'][13]
    
    print("\n▶️ 开始执行 100 步随机游走测试...")
    for i in range(100):
        step_start = time.time()

        # 生成合法的随机动作并执行
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)
        
        # 启动可视化窗口
        env.render_viewer()

        if terminated or truncated:
            print(f"🎉 任务在第 {i} 步完成！获得了最大奖励 {reward}")
            obs, info = env.reset()

        time_until_next_step = SIM_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step))

    print("✅ 步进测试通过！环境可以交付 PPO 和 LeRobot 进行训练了。")
    env.close()