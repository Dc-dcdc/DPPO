"""
原论文使用的环境
用于测试成功率
和原环境的评估条件保持一致
"""

import time
import numpy as np
import mujoco.viewer
from dm_control import mjcf
import gymnasium as gym
from gymnasium import spaces
from env.constants import (
    XML_DIR,
    SIM_DT, SIM_PHYSICS_DT, SIM_PHYSICS_ENV_STEP_RATIO,
    LEFT_ARM_POSE, RIGHT_ARM_POSE, MIDDLE_ARM_POSE,
    LEFT_JOINT_NAMES, RIGHT_JOINT_NAMES, MIDDLE_JOINT_NAMES,
    LEFT_ACTUATOR_NAMES, RIGHT_ACTUATOR_NAMES, MIDDLE_ACTUATOR_NAMES,
    LEFT_EEF_SITE, RIGHT_EEF_SITE, MIDDLE_EEF_SITE, MIDDLE_BASE_LINK,
    LEFT_GRIPPER_JOINT_NAMES, RIGHT_GRIPPER_JOINT_NAMES
)
import os


CAMERAS = [
    "zed_cam_left",
    "zed_cam_right",
    "wrist_cam_left",
    "wrist_cam_right",
    "overhead_cam",
    "worms_eye_cam",
]
RENDER_CAMERA = "overhead_cam"


def make_sim_env(task_name, **kwargs):
    # 这个文件只作为临时评估入口使用，避免牵动正式 Gym 注册环境。
    if 'sim_sew_needle' in task_name:
        return SewNeedleEnv(**kwargs)
    raise NotImplementedError(
        "env/task/env.py 是本地临时评估版，目前只适配 sim_sew_needle。"
    )

class GuidedVisionEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 1/SIM_DT}

    def __init__(self, 
            xml: str,
            num_arms: int = 3,
            episode_length: int = 300,
            cameras: list[str] = CAMERAS,
            observation_height: int = 480,
            observation_width: int = 640,
        ):
        super().__init__()

        assert num_arms in [2, 3], f"Invalid number of arms: {num_arms}"
        assert all([camera in CAMERAS for camera in cameras]), f"Invalid camera names: {cameras}"
        self.cameras = cameras
        self.num_arms = num_arms
        self.xml = xml
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.episode_length = episode_length
        self._current_step = 0
        self.terminated = False
        self.is_success = False

        self._mjcf_root = mjcf.from_path(self.xml)  
        self._mjcf_root.option.timestep = SIM_PHYSICS_DT  
        
        self._physics = mjcf.Physics.from_mjcf_model(self._mjcf_root)

        self._middle_base_link = self._mjcf_root.find('body', MIDDLE_BASE_LINK)
        self._middle_base_link_init_pos = self._middle_base_link.pos.copy()
        if self.num_arms == 2:
            self.hide_middle_arm() # HACK
            self.num_joints = 14
        elif self.num_arms == 3:
            self.num_joints = 21



        """
        {
        "pixels": {
            "cam_1": Box(...),
            "cam_2": Box(...)
        },
        "agent_pos": Box(...)
        }
        """
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        camera : spaces.Box(
                            low=0,
                            high=255,
                            shape=(self.observation_height, self.observation_width, 3),
                            dtype=np.uint8,
                        ) 
                        for camera in self.cameras
                    }
                ),
                "agent_pos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_joints,),
                    dtype=np.float64,
                ),
            }
        )
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_joints,), dtype=np.float32) 
        # 在 Python 代码和底层的 MuJoCo XML 物理模型之间建立连接
        # 绑定关节，用于读取关节角度
        self._left_joints = [self._mjcf_root.find('joint', name) for name in LEFT_JOINT_NAMES]
        self._right_joints = [self._mjcf_root.find('joint', name) for name in RIGHT_JOINT_NAMES]
        self._middle_joints = [self._mjcf_root.find('joint', name) for name in MIDDLE_JOINT_NAMES]
        # 绑定动作，用于发送控制指令
        self._left_actuators = [self._mjcf_root.find('actuator', name) for name in LEFT_ACTUATOR_NAMES]
        self._right_actuators = [self._mjcf_root.find('actuator', name) for name in RIGHT_ACTUATOR_NAMES]
        self._middle_actuators = [self._mjcf_root.find('actuator', name) for name in MIDDLE_ACTUATOR_NAMES]
        # 绑定末端执行器，用于读取末端执行器位姿
        self._left_eef_site = self._mjcf_root.find('site', LEFT_EEF_SITE)
        self._right_eef_site = self._mjcf_root.find('site', RIGHT_EEF_SITE)
        self._middle_eef_site = self._mjcf_root.find('site', MIDDLE_EEF_SITE)
        # 绑定夹爪，用于读取夹爪角度
        self._left_gripper_joints = [self._mjcf_root.find('joint', name) for name in LEFT_GRIPPER_JOINT_NAMES]
        self._right_gripper_joints = [self._mjcf_root.find('joint', name) for name in RIGHT_GRIPPER_JOINT_NAMES]
        # self._left_fk_fn = create_fk_fn(self._physics, self._left_joints[:6], self._left_eef_site)
        # self._right_fk_fn = create_fk_fn(self._physics, self._right_joints[:6], self._right_eef_site)
        # self._middle_fk_fn = create_fk_fn(self._physics, self._middle_joints[:7], self._middle_eef_site)

        # set up controllers
        # self._left_controller = GradIK(
        #     physics=self._physics,
        #     joints = self._left_joints[:6],
        #     actuators=self._left_actuators[:6],
        #     eef_site=self._left_eef_site,
        #     step_size=0.0001, 
        #     min_cost_delta=1.0e-12, 
        #     max_iterations=50, 
        #     position_weight=500.0,
        #     rotation_weight=100.0,
        #     joint_center_weight=np.array([10.0, 10.0, 1.0, 50.0, 1.0, 1.0]),
        #     joint_displacement_weight=np.array(6*[50.0]),
        #     position_threshold=0.001,
        #     rotation_threshold=0.001,
        #     max_pos_diff=0.1,
        #     max_rot_diff=0.3,
        #     joint_p = 0.9,
        # )
        # self._right_controller = GradIK(
        #     physics=self._physics,
        #     joints=self._right_joints[:6],
        #     actuators=self._right_actuators[:6],
        #     eef_site=self._right_eef_site,
        #     step_size=0.0001, 
        #     min_cost_delta=1.0e-12, 
        #     max_iterations=50, 
        #     position_weight=500.0,
        #     rotation_weight=100.0,
        #     joint_center_weight=np.array([10.0, 10.0, 1.0, 50.0, 1.0, 1.0]),
        #     joint_displacement_weight=np.array(6*[50.0]),
        #     position_threshold=0.001,
        #     rotation_threshold=0.001,
        #     max_pos_diff=0.1,
        #     max_rot_diff=0.3,
        #     joint_p = 0.9,
        # )
        # self._middle_controller = DiffIK(
        #     physics=self._physics,
        #     joints=self._middle_joints,
        #     actuators=self._middle_actuators,
        #     eef_site=self._middle_eef_site,
        #     k_pos=0.9,
        #     k_ori=0.9,
        #     damping=1.0e-4,
        #     k_null=np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0]),
        #     q0=np.array(MIDDLE_ARM_POSE),
        #     max_angvel=3.14,
        #     integration_dt=SIM_DT,
        #     iterations=10,
        # )

        self.left_gripper_range = self._physics.bind(self._left_actuators[-1]).ctrlrange
        self.right_gripper_range = self._physics.bind(self._right_actuators[-1]).ctrlrange
        self.left_gripper_norm_fn = lambda x: (x - self.left_gripper_range[0]) / (self.left_gripper_range[1] - self.left_gripper_range[0])
        self.right_gripper_norm_fn = lambda x: (x - self.right_gripper_range[0]) / (self.right_gripper_range[1] - self.right_gripper_range[0])
        self.left_gripper_unnorm_fn = lambda x: x * (self.left_gripper_range[1] - self.left_gripper_range[0]) + self.left_gripper_range[0]
        self.right_gripper_unnorm_fn = lambda x: x * (self.right_gripper_range[1] - self.right_gripper_range[0]) + self.right_gripper_range[0]

        self.max_reward = 0

        # for GUI
        self._viewer = None 

    def get_obs(self) -> np.ndarray:
        left_qpos = self._physics.bind(self._left_joints).qpos.copy()
        left_qpos[6] = self.left_gripper_norm_fn(left_qpos[6])
        right_qpos = self._physics.bind(self._right_joints).qpos.copy()
        right_qpos[6] = self.right_gripper_norm_fn(right_qpos[6])
        middle_qpos = self._physics.bind(self._middle_joints).qpos.copy()

        if self.num_arms == 2:
            agent_pos = np.concatenate([left_qpos, right_qpos])
        elif self.num_arms == 3:
            agent_pos = np.concatenate([left_qpos, right_qpos, middle_qpos])            

        return {
            'pixels': {
                camera: self._physics.render(
                    height=self.observation_height, 
                    width=self.observation_width, 
                    camera_id=camera
                )
                for camera in self.cameras
            },
            'agent_pos': agent_pos,
        }
    
    def get_reward(self):
        return 0
    
    def _resolve_render_camera(self, render_camera=None):
        if render_camera is None:
            return RENDER_CAMERA
        if isinstance(render_camera, str):
            return render_camera
        if len(render_camera) > 0:
            return render_camera[0]
        return RENDER_CAMERA

    def render(self, render_camera=None):
        render_cam = self._resolve_render_camera(render_camera)
        return self._physics.render(
                    height=self.observation_height,
                    width=self.observation_width,
                    camera_id=render_cam
                )

    
    def step(self, action: np.ndarray) -> tuple:
        if np.isnan(action).any() or np.isinf(action).any():
            print("警告：检测到非法动作 (NaN/Inf)，已替换为全 0 动作。")
            action = np.zeros_like(action)

        left_joints = action[:6]
        left_gripper = np.clip(action[6], 0.0, 1.0) # val from 0 to 1   (1 is open, 0 is closed)
        right_joints = action[7:13]
        right_gripper = np.clip(action[13], 0.0, 1.0) # val from 0 to 1         (1 is open, 0 is closed)
        self._physics.bind(self._left_actuators[:6]).ctrl = left_joints
        self._physics.bind(self._right_actuators[:6]).ctrl = right_joints
        self._physics.bind(self._left_actuators[6]).ctrl = self.left_gripper_unnorm_fn(left_gripper)
        self._physics.bind(self._right_actuators[6]).ctrl = self.right_gripper_unnorm_fn(right_gripper)

        if self.num_arms == 3:
            middle_joints = action[14:21]
            self._physics.bind(self._middle_actuators).ctrl = middle_joints

        # step physics
        self._physics.step(nstep=SIM_PHYSICS_ENV_STEP_RATIO)
        self._current_step += 1
        
        observation = self.get_obs()
        reward = self.get_reward()
        truncated = bool(self._current_step >= self.episode_length)
        info = {
            "is_success": bool(getattr(self, "is_success", False)),
            "reward": float(reward),
            "step": self._current_step,
        }

        return observation, float(reward), self.terminated, truncated, info

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed, options=options)

        # reset physics
        self._physics.reset()
        self._current_step = 0
        self.terminated = False
        self.is_success = False
        self._physics.bind(self._left_joints).qpos = LEFT_ARM_POSE
        self._physics.bind(self._left_gripper_joints).qpos = self.left_gripper_unnorm_fn(1)
        self._physics.bind(self._right_joints).qpos = RIGHT_ARM_POSE
        self._physics.bind(self._right_gripper_joints).qpos = self.right_gripper_unnorm_fn(1)
        self._physics.bind(self._middle_joints).qpos = MIDDLE_ARM_POSE
        self._physics.bind(self._left_actuators).ctrl = LEFT_ARM_POSE
        self._physics.bind(self._left_actuators[6]).ctrl = self.left_gripper_unnorm_fn(1)
        self._physics.bind(self._right_actuators).ctrl = RIGHT_ARM_POSE
        self._physics.bind(self._right_actuators[6]).ctrl = self.right_gripper_unnorm_fn(1)
        self._physics.bind(self._middle_actuators).ctrl = MIDDLE_ARM_POSE

        self._physics.forward()

        observation = self.get_obs()
        info = {"is_success": False}

        return observation, info   

    def set_qpos(self, qpos: np.ndarray):
        self._physics.data.qpos[:] = qpos
        self._physics.forward()

    def step_action(self, action: np.ndarray) -> tuple:
        if np.isnan(action).any() or np.isinf(action).any():
            print("警告：检测到非法动作 (NaN/Inf)，已替换为全 0 动作。")
            action = np.zeros_like(action)

        left_joints = action[:6]
        left_gripper = np.clip(action[6], 0.0, 1.0) # val from 0 to 1   (1 is open, 0 is closed)
        right_joints = action[7:13]
        right_gripper = np.clip(action[13], 0.0, 1.0) # val from 0 to 1         (1 is open, 0 is closed)
        self._physics.bind(self._left_actuators[:6]).ctrl = left_joints
        self._physics.bind(self._right_actuators[:6]).ctrl = right_joints
        self._physics.bind(self._left_actuators[6]).ctrl = self.left_gripper_unnorm_fn(left_gripper)
        self._physics.bind(self._right_actuators[6]).ctrl = self.right_gripper_unnorm_fn(right_gripper)

        if self.num_arms == 3:
            middle_joints = action[14:21]
            self._physics.bind(self._middle_actuators).ctrl = middle_joints

        self._physics.step(nstep=SIM_PHYSICS_ENV_STEP_RATIO)



    def render_viewer(self) -> np.ndarray:
        
        # render viewer
        if self._viewer is None:
            self.create_viewer()

        # render
        self._viewer.sync()

    def create_viewer(self, key_callback=None) -> None:
        if self._viewer is None:
            # launch viewer
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
                show_left_ui=True,
                show_right_ui=True,
                key_callback=key_callback
            )

    # 将中间臂隐藏（移出视角外）
    def hide_middle_arm(self):
        self._physics.bind(self._middle_base_link).pos = np.array([0, -2.4, -0.4]) # HACK
    
    def show_middle_arm(self):
        self._physics.bind(self._middle_base_link).pos = self._middle_base_link_init_pos

    def close(self) -> None:
        """
        Closes the viewer if it's open.
        """
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


        



class SewNeedleEnv(GuidedVisionEnv):
    def __init__(self, **kwargs):
        xml = os.path.join(XML_DIR, 'task_sew_needle.xml')
        super().__init__(xml, **kwargs)

        self.max_reward = 5

        self._needle_joint = self._mjcf_root.find('joint', 'needle_joint')
        self._wall_joint = self._mjcf_root.find('joint', 'wall_joint')
        self._threaded_needle = False
        self.is_success = False

    def _needle_reached_exit(self):
        """用本地 XML 的几何点近似原始 pin-needle 穿孔判定。"""
        needle_head = self._physics.named.data.geom_xpos['needle_head']
        hole_exit = self._physics.named.data.geom_xpos['hole_exit']
        return (
            needle_head[0] < hole_exit[0]
            and np.linalg.norm(needle_head - hole_exit) < 0.02
        )

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed, options=options)
        rng = self.np_random

        # reset physics
        x_range = [0.15, 0.2]
        y_range = [-.025,0.1]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        needle_position = rng.uniform(ranges[:, 0], ranges[:, 1])
        needle_quat = np.array([1, 0, 0, 0])

        x_range = [-0.025, 0.025]
        y_range = [-.025,0.1]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        wall_position = rng.uniform(ranges[:, 0], ranges[:, 1])
        wall_quat = np.array([1, 0, 0, 0]) 

        self._physics.bind(self._needle_joint).qpos = np.concatenate([needle_position, needle_quat])
        self._physics.bind(self._wall_joint).qpos = np.concatenate([wall_position, wall_quat])

        self._physics.forward()

        self._threaded_needle = False
        self.is_success = False
        self.terminated = False

        observation = self.get_obs()
        info = {"is_success": False}

        return observation, info
    

    def get_reward(self):

        touch_left_gripper = False
        touch_right_gripper = False
        needle_touch_table = False
        needle_touch_wall = False
        pins_touch = False
        needle_touch_pin = False

        needle_geoms = {
            "needle",
            "needle_head",
            "needle_tail",
            "needle_mid",
            "needle_mark_1_4",
            "needle_mark_3_4",
        }
        threaded_by_contact = False

        contact_pairs = []
        for i_contact in range(self._physics.data.ncon):
            id_geom_1 = self._physics.data.contact[i_contact].geom1
            id_geom_2 = self._physics.data.contact[i_contact].geom2
            geom1 = self._physics.model.id2name(id_geom_1, 'geom')
            geom2 = self._physics.model.id2name(id_geom_2, 'geom')
            if geom1 and geom2:
                contact_pairs.append((geom1, geom2))
                contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1 == "needle" and geom2.startswith("right"):
                touch_right_gripper = True

            if geom1 == "needle" and geom2.startswith("left"):
                touch_left_gripper = True

            if geom1 == "table" and geom2 == "needle":
                needle_touch_table = True

            if geom1 == "needle" and geom2.startswith("wall-"):
                needle_touch_wall = True

            # 兼容旧 XML：如果存在 pin-needle，仍按原始接触判定穿孔。
            if geom1 == "pin-needle" and geom2 == "pin-wall":
                threaded_by_contact = True
                pins_touch = True

            if geom1 in needle_geoms and geom2 == "pin-wall":
                needle_touch_pin = True

        # 本地 XML 没有 pin-needle，因此用针头到达出口附近作为穿孔替代判定。
        if threaded_by_contact or self._needle_reached_exit():
            self._threaded_needle = True
            pins_touch = True

        reward = 0
        if touch_right_gripper: # touch needle
            reward = 1
        if touch_right_gripper and (not needle_touch_table): # grasp needle
            reward = 2
        if needle_touch_wall and (not needle_touch_table): # needle and wall touching
            reward = 3
        if self._threaded_needle: # needle threaded
            reward = 4
        if (
            touch_left_gripper
            and (not touch_right_gripper)
            and (not needle_touch_table)
            and (not needle_touch_pin)
            and self._threaded_needle
        ): # grasped needle on other side
            reward = 5

        # 成功标志位采用锁存逻辑：一旦达到 env.py 原始最高阶段 reward=5，本回合保持成功。
        if reward == self.max_reward:
            self.is_success = True
            self.terminated = True
        return reward
    
