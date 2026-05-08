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

    metadata = {"render_modes": ["rgb_array"], "render_fps": 1/SIM_DT}

    def __init__(self, 
            xml_path: str,
            num_arms: int = 3,
            episode_length: int = 300,
            cameras: list[str] = CAMERAS,
            observation_height: int = 480,
            observation_width: int = 640,
        ):
        super().__init__()
        assert num_arms in [2, 3], f"Invalid number of arms: {num_arms}"
        assert all([camera in CAMERAS for camera in cameras]), f"Invalid camera names: {cameras}"
        # self.num_envs = 1
        # ==========================================
        # 🌟 1. 加载物理模型
        # ==========================================
        self.cameras = cameras # 使用的摄像头列表
        self.num_arms = num_arms
        self._mjcf_root = mjcf.from_path(xml_path)
        self._physics = mjcf.Physics.from_mjcf_model(self._mjcf_root)  
        self.observation_height = observation_height
        self.observation_width = observation_width   
        self._mjcf_root.option.timestep = SIM_PHYSICS_DT  
        
        self.episode_length = episode_length
        self._middle_base_link = self._mjcf_root.find('body', MIDDLE_BASE_LINK)
        self._middle_base_link_init_pos = self._middle_base_link.pos.copy()

        if self.num_arms == 2:
            self.hide_middle_arm() # HACK, 隐藏中央机械臂
            self.num_joints = 14
        elif self.num_arms == 3:
            self.num_joints = 21
        # ==========================================
        # 🌟 2. 构建观察空间 (Observation Space)
        # ==========================================
        """
        {
            "observation.images.cam_1": Box(...),
            "observation.images.cam_2": Box(...),
            "observation.state": Box(...)
        }
        """
        obs_spaces = {}
        # 动态遍历并注册所有的相机图像空间
        for cam_name in self.cameras:
            # 这里的键名严格对齐 LeRobot 的规范：observation.images.xxxx
            obs_spaces[f'observation.images.{cam_name}'] = spaces.Box(
                low=0, high=255, shape=(3, self.observation_height, self.observation_width), dtype=np.uint8
            )
            
        # 注册 21维本体状态
        obs_spaces['observation.state'] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_joints,), dtype=np.float64
        ) 
        
        self.observation_space = spaces.Dict(obs_spaces)
        
        # ==========================================
        # 🌟 3. 定义动作空间 (Action Space): 21维关节目标角度
        # ==========================================
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_joints,), dtype=np.float64
        )   
        
        # ==========================================
        # 🌟 4. 寻址与绑定 MJCF 节点，和底层的 MuJoCo XML 物理模型之间建立连接
        # ==========================================

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
        
        if self.num_arms == 2:
            agent_pos = np.concatenate([left_qpos, right_qpos]).astype(np.float64)
        elif self.num_arms == 3:
            agent_pos = np.concatenate([left_qpos, right_qpos, middle_qpos]).astype(np.float64) 
        # state_21d = np.concatenate([left_qpos, right_qpos, middle_qpos]).astype(np.float32)

        # ==========================================
        # 🌟 2. 渲染相机图像并转换通道为 (C, H, W)
        # ==========================================
        # 2. 准备返回的字典
        obs_dict = {'observation.state': agent_pos}
        for cam_name in self.cameras:
            # 注意：这里的 camera_id 必须和你的 XML 文件里 <camera name="..."> 的名字完全一致！
            try:
                img = self._physics.render(height=self.observation_height, width=self.observation_width, camera_id=cam_name)
                img = np.transpose(img, (2, 0, 1))# 转换通道 (H, W, C) -> (C, H, W)
                # 存入字典，键名严格对齐
                obs_dict[f'observation.images.{cam_name}'] = img
            except Exception as e:
                raise ValueError(f"❌ 渲染相机 '{cam_name}' 失败！请检查 XML 文件中是否有这个名字的相机。报错详情: {e}")

        return obs_dict

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)
        self._physics.reset()
        # 🌟 新增：重置回合内部的步数计数器
        self._current_step = 0
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
        self.terminated = False
        # 读取当前环境的观测
        observation = self.get_obs()
        info = {"message": "Environment reset successfully."}
        return observation, info   

    def step(self, action: np.ndarray) -> tuple:
        """Gymnasium 标准步进函数"""
        # 1. 引擎防爆护盾 (拦截 NaN 和无穷大)
        if np.isnan(action).any() or np.isinf(action).any():
            print("⚠️ 警告：检测到非法动作 (NaN/Inf)，已启动安全降级为全 0 动作！")
            action = np.zeros_like(action)
        
        # 2. 动作拆包
        left_joints = action[:6]
        # left_gripper = action[6]
        left_gripper = np.clip(action[6], 0.0, 1.0) # 0.0 到 1.0 之间的归一化值
        right_joints = action[7:13]
        # right_gripper = action[13]
        right_gripper = np.clip(action[13], 0.0, 1.0)
        if self.num_arms == 3:
            middle_joints = action[14:21]
            self._physics.bind(self._middle_actuators).ctrl = middle_joints

        # 3. 映射到物理引擎执行器
        self._physics.bind(self._left_actuators[:6]).ctrl = left_joints
        self._physics.bind(self._right_actuators[:6]).ctrl = right_joints

        self._physics.bind(self._left_actuators[6]).ctrl = self.left_gripper_unnorm_fn(left_gripper)
        self._physics.bind(self._right_actuators[6]).ctrl = self.right_gripper_unnorm_fn(right_gripper)

        # 4. 步进物理引擎
        for _ in range(SIM_PHYSICS_ENV_STEP_RATIO): self._physics.step()
        self._current_step += 1   # 步数追踪

        # 5. 获取观察与奖励
        observation = self.get_obs()
        reward = self.get_reward() if hasattr(self, 'get_reward') else 0.0
        
        # 6. 判断终止条件
        # max_rwd = getattr(self, 'max_reward', -1)
        # terminated = bool(reward >= max_rwd - 1e-4) # 达到最大奖励则任务成功结束
        truncated = bool(self._current_step >= self.episode_length) # 超出最大步数
        
        info = {"is_success": self.terminated, "reward": reward, "step": self._current_step}

        return observation, float(reward),  self.terminated, truncated, info

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
            img = self._physics.render(height=self.observation_height, width=self.observation_width, camera_id=render_cam)
            return img
        except Exception as e:
            # 防止万一没找到相机导致崩溃
            print(f"⚠️ 渲染视频帧失败: {e}")
            return np.zeros((self.observation_height, self.observation_width, 3), dtype=np.uint8)
        
    def render_viewer(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr, self._physics.data.ptr,
                show_left_ui=True, show_right_ui=True,
            )
        self._viewer.sync()

    # 将中间臂隐藏（移出视角外）
    def hide_middle_arm(self):
        self._physics.bind(self._middle_base_link).pos = np.array([0, -2.4, -0.4]) # HACK


    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()


class SewNeedleEnv(GuidedVisionEnv):
    """
    缝合针穿引任务专用环境
    """
    def __init__(self,**kwargs):
        xml_path = os.path.join(XML_DIR, 'task_sew_needle.xml')
        super().__init__(xml_path, **kwargs)

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
        self.needle_reached_exit = False  # 针头穿墙标志位
        self.left_has_grasped = False     # 左臂成功接针标志位
        self.needle_completely_through = False # 针完全过孔标志位
        self.needle_start_through = False # 针开始过孔标志位        
        observation = self.get_obs()
        info = {"message": "SewNeedle env reset."}
        return observation, info
    
    def get_reward(self):

        touch_left_gripper = False
        touch_right_gripper = False
        # 存放当前帧里所有碰到针的物体名字
        objects_touching_needle = set()

        # 遍历底层物理引擎计算出的所有有效接触点 (Contacts)
        for i_contact in range(self._physics.data.ncon):
            # 提取发生碰撞的两个物体的底层 ID
            id_geom_1 = self._physics.data.contact[i_contact].geom1
            id_geom_2 = self._physics.data.contact[i_contact].geom2
            
            # 将底层 ID 翻译成你在 XML 环境文件中定义的名字 (如 "peg", "right_finger_1")
            geom1 = self._physics.model.id2name(id_geom_1, 'geom')
            geom2 = self._physics.model.id2name(id_geom_2, 'geom')
            
            # 只有两个物体都有名字时，才加入判定列表(双向记录)
            if geom1 and geom2:
                # 2. 只要有一方是针，就把另一方的名字扔进篮子
                if geom1 == "needle":
                    objects_touching_needle.add(geom2)
                elif geom2 == "needle":
                    objects_touching_needle.add(geom1)

        # 检查左夹爪是否成功捏住：左半边手指碰到了，且右半边手指也碰到了left_left中包含主模型right_right_finger和g0、g1、g2
        touched_left_left = any(obj.startswith("left_left") for obj in objects_touching_needle)
        touched_left_right = any(obj.startswith("left_right") for obj in objects_touching_needle)
        if touched_left_left and touched_left_right:
            touch_left_gripper = True
            
        # 检查右夹爪是否成功捏住
        touched_right_left = any(obj.startswith("right_left") for obj in objects_touching_needle)
        touched_right_right = any(obj.startswith("right_right") for obj in objects_touching_needle)
        if touched_right_left and touched_right_right:
            touch_right_gripper = True

        # ==========================================
        # 提取相关坐标
        # ==========================================
        needle_head = self._physics.named.data.geom_xpos['needle_head']                #针头
        needle_tail = self._physics.named.data.geom_xpos['needle_tail']                #针尾
        needle_left_pos = self._physics.named.data.geom_xpos['needle_mark_1_4']        #左臂抓取标记点 1/4处
        needle_right_pos = self._physics.named.data.geom_xpos['needle_mark_3_4']       #右臂抓取标记点 3/4处
        
        left_left_finger = self._physics.named.data.geom_xpos['left_left_g2']
        left_right_finger = self._physics.named.data.geom_xpos['left_right_g2']        
        right_left_finger = self._physics.named.data.geom_xpos['right_left_g2']        #右臂左指尖
        right_right_finger = self._physics.named.data.geom_xpos['right_right_g2']      #右臂右指尖

        hole_entrance = self._physics.named.data.geom_xpos['hole_entrance']            #出洞口
        hole_exit = self._physics.named.data.geom_xpos['hole_exit']                    #出洞口

        # ==========================================
        # 计算左右臂和针距离奖励
        # ==========================================
        # 计算右臂夹爪的中心点 (Pinch Center)
        left_gripper_center = (left_left_finger + left_right_finger) / 2.0
        right_gripper_center = (right_left_finger + right_right_finger) / 2.0

        # 计算右臂夹爪中心到针标记点的距离
        dist_left_to_mark = np.linalg.norm(left_gripper_center - needle_left_pos)
        dist_right_to_mark = np.linalg.norm(right_gripper_center - needle_right_pos)
        
        # ==========================================
        # 计算针穿过墙洞的奖励
        # ==========================================
        # 计算针头到入口的距离
        dist_head_to_entrance = np.linalg.norm(needle_head - hole_entrance)

        # 计算针头到“洞口出口(hole_exit)”的距离
        dist_head_to_exit = np.linalg.norm(needle_head - hole_exit)
        
        # 计算针尾到出口的距离
        dist_tail_to_exit = np.linalg.norm(needle_tail - hole_exit)

        # ==========================================
        # 阶段流转奖励逻辑
        # ==========================================
        # 每走一步扣 0.5 分，降低步数
        reward = -1

        # 触发 1：针头到达入口 (进洞),给个很小的值防止在外面就判断进去了 ，加上3d距离防止绕墙
        if needle_head[0] - hole_entrance[0] < 0.001 and dist_head_to_entrance < 0.01:
            if not self.needle_start_through: # 确保只触发一次
                self.needle_start_through = True
                reward += 25.0  # 突破瞬间的巨额奖励

        # 触发 2：针头到达出口 (露头)
        if needle_head[0] < hole_exit[0] and dist_head_to_exit < 0.02:
            if not self.needle_reached_exit:
                self.needle_reached_exit = True
                reward += 50.0  # 突破瞬间的巨额奖励

        # 触发 3：左手成功接应
        if self.needle_reached_exit and touch_left_gripper:
            if not self.left_has_grasped:
                self.left_has_grasped = True
                reward += 75.0  # 突破瞬间的巨额奖励
        
        # 触发 4：针尾完全拔出
        # # 穿孔方向为沿着 X 轴向负方向移动，针尾小于出口 且 距离小于设定值（防止从墙外面绕过）
        if needle_tail[0] < hole_exit[0] and dist_tail_to_exit < 0.02:
            if not self.needle_completely_through:
                self.needle_completely_through = True
                reward += 100.0  # 突破瞬间的巨额奖励
        
        if not self.left_has_grasped:
            # --- 前半场：右臂主导 ---

            if not touch_right_gripper:  # 右臂未抓针
                # 引导右臂接近 针3/4 处的奖励
                reward += 2.0 * np.exp(-15.0 * dist_right_to_mark)
            else: 
                reward += 0.25 # 保持抓取的微弱奖励
                if not self.needle_start_through:
                    # 没到达洞口，引导针头到达入口 （最高5分）
                    reward += 5 * np.exp(-15.0 * dist_head_to_entrance)
                elif not self.needle_reached_exit:
                    # 到达洞口，引导针头到达出口
                    reward += 5 * np.exp(-20.0 * dist_head_to_exit)
                else:
                    # 到达出口，引导左臂接近针标记
                    reward += 3 * np.exp(-15.0 * dist_left_to_mark)

        else: 
            # --- 后半场：左臂主导 ---
            reward += 0.25 
            # 惩罚右手，逼迫其松开并让开空间
            if touch_right_gripper:
                reward -= 0.5 

            if not touch_left_gripper:
                # 左手只是碰了一下但没抓稳，或者中途脱手了
                # 掉落惩罚，但因为有之前的一次性奖励撑腰，它依然敢于尝试交接
                # reward -= 0.5
                # 重新提供一个引导左臂去抓针的低保底引力，逼迫它捏紧双指
                reward += 3.0 * np.exp(-15.0 * dist_left_to_mark)
                
            else:
                reward += 0.25 # 保持抓紧的微弱奖励
                # 3. 终极判定：针尾是否完全越过出口

                if not self.needle_completely_through:
                    # 左手往外拔出
                    # 连续奖励：引导左臂继续往外拉，直到针尾到达出口
                    reward += 5.0 * np.exp(-15.0 * dist_tail_to_exit)
                else:
                    # print("✅ 针尾完全越过出口！")
                    # 针已完全拔出，执行抬举动作验证
                    wall_center_x = (hole_entrance[0] + hole_exit[0]) / 2.0
                    target_z_target = hole_exit[2] + 0.12 # 最低要求抬起 12 厘米，总高度17cm
                    
                    # 2. 计算针的中心点
                    needle_center = (needle_head + needle_tail) / 2.0
                    
                    # 3. 独立计算各轴误差 (解耦 Z 轴)
                    # Z轴误差：巧妙使用 max()，只要当前高度超过 target_z_min，误差就归零！举多高都不会受罚。
                    z_error = max(0.0, target_z_target - needle_center[2])
                    x_error = abs(needle_center[0] - wall_center_x)
                    y_error = abs(needle_center[1] - hole_exit[1])
                    
                    # 将解耦后的误差重新合成为复合距离，专供连续奖励计算使用
                    composite_error_dist = np.sqrt(x_error**2 + y_error**2 + z_error**2)
                    
                    # 4. 独立且宽容的成功判定条件
                    # 只要高度超过下限 (z_error == 0)，且 XY 平面没有偏离墙体中心太远 (比如 2 厘米内容差)
                    is_above_wall = (z_error == 0.0) and (x_error < 0.01) #and (y_error < 0.02)
                    
                    if not is_above_wall:
                        # 连续奖励：此时距离变量变成了 composite_error_dist
                        reward += 10.0 * np.exp(-15.0 * composite_error_dist)
                    else:
                        # 阶段 6：终极胜利！针被成功拔出、抬高，并稳稳处于墙的正上方
                        reward += 500.0 
                        self.terminated = True

        return float(reward)


class SlotInsertionEnv(GuidedVisionEnv):
    def __init__(self, **kwargs):
        xml = os.path.join(XML_DIR, 'task_slot_insertion.xml')
        super().__init__(xml, **kwargs)

        self.max_reward = 4

        self._slot_joint = self._mjcf_root.find('joint', 'slot_joint')
        self._stick_joint = self._mjcf_root.find('joint', 'stick_joint')

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed, options=options)

        # reset physics
        x_range = [-0.05, 0.05]
        y_range = [0.1, 0.15]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        slot_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        slot_quat = np.array([1, 0, 0, 0])


        peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        peg_quat = np.array([1, 0, 0, 0])

        x_range = [-0.08, 0.08]
        y_range = [-0.1, 0.0]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        stick_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        stick_quat = np.array([1, 0, 0, 0]) 

        self._physics.bind(self._slot_joint).qpos = np.concatenate([slot_position, slot_quat])
        self._physics.bind(self._stick_joint).qpos = np.concatenate([stick_position, stick_quat])

        self._physics.forward()

        observation = self.get_obs()
        info = {"is_success": False}

        return observation, info
    

    def get_reward(self):

        touch_left_gripper = False
        touch_right_gripper = False
        stick_touch_table = False
        stick_touch_slot = False
        pins_touch = False

        # return whether peg touches the pin
        contact_pairs = []
        for i_contact in range(self._physics.data.ncon):
            id_geom_1 = self._physics.data.contact[i_contact].geom1
            id_geom_2 = self._physics.data.contact[i_contact].geom2
            geom1 = self._physics.model.id2name(id_geom_1, 'geom')
            geom2 = self._physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1 == "stick" and geom2.startswith("right"):
                touch_right_gripper = True
            
            if geom1 == "stick" and geom2.startswith("left"):
                touch_left_gripper = True

            if geom1 == "table" and geom2 == "stick":
                stick_touch_table = True

            if geom1 == "stick" and geom2.startswith("slot-"):
                stick_touch_slot = True

            if geom1 == "pin-stick" and geom2 == "pin-slot":
                pins_touch = True

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not stick_touch_table): # grasp stick
            reward = 2
        if stick_touch_slot and (not stick_touch_table): # peg and socket touching
            reward = 3
        if pins_touch: # successful insertion
            reward = 4
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
    while True:
        step_start = time.time()

        # 生成合法的随机动作并执行
        # random_action = env.action_space.sample()
        # obs, reward, terminated, truncated, info = env.step(random_action)
        
        # 启动可视化窗口
        env.render_viewer()

        # if terminated or truncated:
        #     print(f"🎉 任务在第 {i} 步完成！获得了最大奖励 {reward}")
        #     obs, info = env.reset()

        time_until_next_step = SIM_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step))

    print("✅ 步进测试通过！环境可以交付 PPO 和 LeRobot 进行训练了。")
    env.close()