#!/home/dc/miniforge3/envs/DPPO/bin/python
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import gymnasium as gym
import mujoco.viewer
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_COLLECT_DIR = ROOT_DIR / "data_collect"
ENV_DIR = ROOT_DIR / "env"

for path in (ROOT_DIR, DATA_COLLECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quest_receive import QuestReceive
from quest_send import UnityImageStreamer
from quest_control import QuestControl
from data_collect.robot_ik_solver import ArmIKState, PoseActionIKSolver
from headset_utils import HeadsetData

if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

import env as _register_guided_vision_envs  # noqa: F401
from env.constants import SIM_DT


DEFAULT_CONFIG = {
    "head_control": True,               # 是否用头显位姿控制中间臂。
    "individual_hand_anchors": True,    # 左右臂是否使用各自手柄初始位姿作为锚点。
    "lock_roll": True,                  # 是否只锁定中间臂的 roll。
    "unity_image_stream": True,         # 是否默认向 Unity 发送 MuJoCo/ZED 图像。
    "allow_partial_anchor": False,      # 是否允许只有部分设备可用时开始锚定。
    "start_on_first_packet": False,     # 是否收到第一帧有效 Quest3 数据后自动开始。
    "no_convert_to_mujoco": False,      # 是否跳过 Unity/Quest 坐标到 MuJoCo 坐标的转换。
    "viewer": True,                     # 是否打开 MuJoCo viewer。
    "camera_window": True,              # 是否打开本地 OpenCV 相机窗口。
    "hand_position_scale": 1.0,         # 手部位移映射缩放。
    "hand_max_delta": 1,                # 手部控制末端最大偏移，单位米。
    "head_position_scale": 1.0,         # 头部位移映射缩放。
    "head_max_delta": 1,                # 头部控制末端最大偏移，单位米。
}


def _str_to_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on", "是", "开", "开启"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off", "否", "关", "关闭"}:
        return False
    raise argparse.ArgumentTypeError("布尔参数请填写 true/false、1/0、yes/no、on/off、是/否")

def _draw_status(frame_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    y = 26
    for line in lines:
        cv2.putText(
            frame_bgr,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 245, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24
    return frame_bgr


def _target_summary(states: list[ArmIKState]) -> str:
    labels = {"left": "L", "right": "R", "middle": "M"}
    return " ".join(
        f"{labels[state.name]}=({state.target_pos[0]:+.3f},{state.target_pos[1]:+.3f},{state.target_pos[2]:+.3f})"
        for state in states
    )


def _joint_tracking_summary(states: list[ArmIKState]) -> str:
    labels = {"left": "L", "right": "R", "middle": "M"}
    parts = []
    for state in states:
        if state.last_q_error is None or state.last_q_current is None or state.last_q_target is None:
            continue
        idx = int(np.argmax(np.abs(state.last_q_error)))
        parts.append(
            f"{labels[state.name]}q{idx}:{state.last_q_current[idx]:+.2f}->{state.last_q_target[idx]:+.2f}"
            f" e={state.last_q_error[idx]:+.2f}"
        )
    return " ".join(parts)


def _hand_joint_summary(states: list[ArmIKState]) -> str:
    labels = {"left": "L", "right": "R"}
    parts = []
    for state in states:
        if state.name not in labels or state.last_q_current is None or state.last_q_target is None:
            continue
        for idx in (3, 4, 5):
            if idx >= state.last_q_current.shape[0]:
                continue
            parts.append(
                f"{labels[state.name]}q{idx}:{state.last_q_current[idx]:+.2f}->{state.last_q_target[idx]:+.2f}"
            )
    return " ".join(parts)


# 打印配置参数和控制说明。
def _print_header(args) -> None:
    print("\nQuest3 -> MuJoCo three-arm teleop test")
    print("-" * 78)
    print(f"UDP:           {args.host}:{args.port}")
    print(f"Env:           {args.env_id}")
    mapping = "left controller -> left arm | right controller -> right arm"
    mapping += " | head -> middle arm" if args.head_control else " | middle arm fixed"
    print(f"Pipeline:      QuestReceive -> QuestControl -> PoseActionIKSolver -> env.step")
    print(f"Mapping:       {mapping}")
    print(f"Hand anchors:  {args.individual_hand_anchors}")
    print(f"Hand scale:    {args.hand_position_scale:.3f}, max delta {args.hand_max_delta:.3f} m")
    print(f"Head control:  {args.head_control}")
    if args.head_control:
        print(f"Head scale:    {args.head_position_scale:.3f}, max delta {args.head_max_delta:.3f} m")
    print(f"Middle roll:   {args.lock_roll}")
    print(f"Camera:        {args.display_camera}")
    print(f"Unity stream:  {'on' if args.unity_image_stream else 'off'}")
    if args.unity_image_stream:
        host_text = str(args.unity_image_host).strip().lower()
        if host_text == "auto":
            target = f"broadcast:{args.unity_image_port} -> auto from Quest pose packet"
        elif host_text in {"broadcast", "255.255.255.255"}:
            target = f"broadcast:{args.unity_image_port}"
        else:
            target = f"{args.unity_image_host}:{args.unity_image_port}"
        print(f"Unity target:  {target}, {args.unity_image_hz:.1f} Hz, JPEG q={args.unity_image_jpeg_quality}")
    print(f"Coord convert: {not args.no_convert_to_mujoco}")
    print(f"Partial anchor:{args.allow_partial_anchor}")
    print("-" * 78)
    print("Controls:")
    print("  A/X or P: anchor valid Quest poses and start")
    print("  B/Y or R/Space: reset MuJoCo env and pause")
    print("  Controller trigger/grip: close that side gripper")
    print("  Q or Esc: quit")
    print("-" * 78)


def _normalize_legacy_args(args: argparse.Namespace) -> None:
    if args.position_scale is not None:
        args.hand_position_scale = args.position_scale
        args.head_position_scale = args.position_scale
    if args.max_delta is not None:
        args.hand_max_delta = args.max_delta
        args.head_max_delta = args.max_delta


def run(args: argparse.Namespace) -> None:
    _normalize_legacy_args(args)

    if args.mujoco_gl != "auto":
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    elif not args.viewer and not args.camera_window and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    _print_header(args)

    env_obj = gym.make(
        args.env_id,
        disable_env_checker=True,
        cameras=[args.display_camera],
        episode_length=args.episode_length,
        observation_height=args.render_height,
        observation_width=args.render_width,
    )
    sim_env = env_obj.unwrapped
    obs, _ = env_obj.reset()

    physics = sim_env._physics
    ik_solver = PoseActionIKSolver.from_args(sim_env, args)
    states = ik_solver.states

    quest_control = QuestControl(
        use_head_control=args.head_control,
        use_individual_hand_anchors=args.individual_hand_anchors,
    )

    command = {"anchor": False, "reset": False, "quit": False}

    def key_callback(keycode: int) -> None:
        if keycode in (ord("p"), ord("P")):
            command["anchor"] = True
        elif keycode in (ord("r"), ord("R"), 32):
            command["reset"] = True
        elif keycode in (ord("q"), ord("Q"), 256):
            command["quit"] = True

    viewer_cm = (
        mujoco.viewer.launch_passive(
            physics.model.ptr,
            physics.data.ptr,
            show_left_ui=True,
            show_right_ui=True,
            key_callback=key_callback,
        )
        if args.viewer
        else nullcontext(None)
    )

    receiver = QuestReceive(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        convert_to_mujoco=not args.no_convert_to_mujoco,
    )
    image_streamer = (
        UnityImageStreamer(
            host=args.unity_image_host,
            port=args.unity_image_port,
            send_hz=args.unity_image_hz,
            jpeg_quality=args.unity_image_jpeg_quality,
            chunk_size=args.unity_image_chunk_size,
            log_interval=args.unity_image_log_interval,
        )
        if args.unity_image_stream
        else None
    )

    if args.camera_window:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(args.window_name, args.render_width, args.render_height)

    started = False
    latest_data: HeadsetData | None = None
    latest_feedback = None
    prev_start_button = False
    prev_reset_button = False
    last_status_t = 0.0
    step_count = 0

    try:
        with viewer_cm as viewer:
            while True:
                loop_start = time.time()

                if viewer is not None and not viewer.is_running():
                    break
                if command["quit"]:
                    break

                try:
                    latest_data = receiver.receive_latest_data()
                except socket.timeout:
                    latest_data = None
                except json.JSONDecodeError as exc:
                    print(f"Invalid Quest JSON packet: {exc}")
                    latest_data = None

                if latest_data is not None:
                    if image_streamer is not None and receiver.latest_address is not None:
                        image_streamer.update_auto_host(receiver.latest_address[0])

                    start_button = quest_control.should_start(latest_data)
                    reset_button = quest_control.should_reset(latest_data)
                    if start_button and not prev_start_button:
                        command["anchor"] = True
                    if reset_button and not prev_reset_button:
                        command["reset"] = True
                    prev_start_button = start_button
                    prev_reset_button = reset_button

                if command["reset"]:
                    obs, _ = env_obj.reset()
                    ik_solver.reset()
                    quest_control.reset()
                    started = False
                    latest_feedback = None
                    step_count = 0
                    command["reset"] = False
                    print("Reset MuJoCo env. Waiting for a new QuestControl anchor.")

                can_auto_anchor = latest_data is not None and (
                    ik_solver.can_anchor_from_data(latest_data, allow_partial=args.allow_partial_anchor)
                )
                if args.start_on_first_packet and can_auto_anchor and not started:
                    command["anchor"] = True

                if command["anchor"]:
                    if latest_data is None:
                        print("Cannot anchor yet: no Quest data.")
                    else:
                        active_count = ik_solver.activate_from_data(
                            latest_data,
                            require_all=not args.allow_partial_anchor,
                        )
                        if active_count > 0:
                            left_pose, right_pose, middle_pose = ik_solver.current_three_arm_poses()
                            quest_control.start(latest_data, middle_pose, left_pose, right_pose)
                            started = True
                    command["anchor"] = False

                if started and latest_data is not None:
                    left_pose, right_pose, middle_pose = ik_solver.current_three_arm_poses()
                    pose_action, latest_feedback = quest_control.run(latest_data, left_pose, right_pose, middle_pose)
                    action, active_count = ik_solver.pose_action_to_joint_action(pose_action, obs)
                    if active_count > 0:
                        obs, _, terminated, truncated, _ = env_obj.step(action)
                        step_count += 1

                        if terminated or truncated:
                            obs, _ = env_obj.reset()
                            ik_solver.reset()
                            quest_control.reset()
                            started = False
                            latest_feedback = None
                            step_count = 0
                            print("Episode ended. Env reset, waiting for a new QuestControl anchor.")

                if args.camera_window or image_streamer is not None:
                    frame_rgb = physics.render(
                        height=args.render_height,
                        width=args.render_width,
                        camera_id=args.display_camera,
                    )
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    if image_streamer is not None:
                        image_streamer.maybe_send_bgr(frame_bgr)

                    if args.camera_window:
                        status = "RUNNING" if started else "PAUSED"
                        active_text = " ".join(f"{state.name[0].upper()}:{'on' if state.active else 'off'}" for state in states)
                        lines = [
                            f"{status} | A/X/P anchor | B/Y/R reset | Q quit",
                            f"active: {active_text} | steps: {step_count}",
                        ]
                        if latest_feedback is not None:
                            lines.append(
                                f"sync: H={int(latest_feedback.head_out_of_sync)} L={int(latest_feedback.left_out_of_sync)} R={int(latest_feedback.right_out_of_sync)}"
                            )
                        for state in states:
                            label = state.name[0].upper()
                            lines.append(
                                f"{label} target: {state.target_pos[0]:+.3f} {state.target_pos[1]:+.3f} {state.target_pos[2]:+.3f}"
                            )
                        joint_summary = _joint_tracking_summary(states)
                        if joint_summary:
                            lines.append(f"joint: {joint_summary}")
                        hand_summary = _hand_joint_summary(states)
                        if hand_summary:
                            lines.append(f"hand: {hand_summary}")
                        cv2.imshow(args.window_name, _draw_status(frame_bgr.copy(), lines))
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break
                        if key in (ord("p"), ord("P")):
                            command["anchor"] = True
                        if key in (ord("r"), ord("R"), 32):
                            command["reset"] = True

                if viewer is not None:
                    viewer.sync()

                now = time.time()
                if now - last_status_t >= args.status_hz_interval:
                    packet_state = "data" if latest_data is not None else "no-data"
                    run_state = "running" if started else "paused"
                    joint_summary = _joint_tracking_summary(states)
                    joint_text = f" joints={joint_summary}" if joint_summary else ""
                    hand_summary = _hand_joint_summary(states)
                    hand_text = f" hand={hand_summary}" if hand_summary else ""
                    print(f"[{run_state:7s}] {packet_state:9s} {_target_summary(states)} steps={step_count}{joint_text}{hand_text}")
                    last_status_t = now

                sleep_t = SIM_DT - (time.time() - loop_start)
                if sleep_t > 0:
                    time.sleep(sleep_t)
    finally:
        receiver.close()
        if image_streamer is not None:
            image_streamer.close()
        env_obj.close()
        if args.camera_window:
            cv2.destroyAllWindows()

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 Quest3 UDP 位姿数据控制 MuJoCo 三臂仿真。")
    parser.add_argument("--host", default="0.0.0.0", help="Quest3 位姿 UDP 监听地址。")
    parser.add_argument("--port", type=int, default=5005, help="Quest3 位姿 UDP 监听端口。")
    parser.add_argument("--timeout", type=float, default=0.05, help="UDP 接收超时时间，单位秒。")
    parser.add_argument("--env-id", default="guided_vision/SewNeedle-3Arms-v0", help="Gymnasium 环境 ID。")
    parser.add_argument("--hand-position-scale", type=float, default=DEFAULT_CONFIG["hand_position_scale"], help="左右手控制机械臂末端位移的缩放系数。")
    parser.add_argument("--head-position-scale", type=float, default=DEFAULT_CONFIG["head_position_scale"], help="头显控制中间臂位移的缩放系数。")
    parser.add_argument("--hand-max-delta", type=float, default=DEFAULT_CONFIG["hand_max_delta"], help="左右手控制末端相对锚点的最大偏移，单位米。")
    parser.add_argument("--head-max-delta", type=float, default=DEFAULT_CONFIG["head_max_delta"], help="头显控制中间臂相对锚点的最大偏移，单位米。")
    parser.add_argument("--head-control", type=_str_to_bool, default=DEFAULT_CONFIG["head_control"], help="是否用头显位姿控制中间臂，填写 true/false。")
    parser.add_argument("--individual-hand-anchors", type=_str_to_bool, default=DEFAULT_CONFIG["individual_hand_anchors"], help="左右臂是否使用各自手柄初始位姿作为锚点，填写 true/false。")
    parser.add_argument("--lock-roll", type=_str_to_bool, default=DEFAULT_CONFIG["lock_roll"], help="是否只锁定中间臂的 roll，填写 true/false。")
    parser.add_argument("--position-scale", type=float, default=None, help="兼容旧参数：统一覆盖手部和头部位移缩放。")
    parser.add_argument("--max-delta", type=float, default=None, help="兼容旧参数：统一覆盖手部和头部最大偏移。")
    parser.add_argument("--episode-length", type=int, default=100000, help="交互测试时的最大 episode 长度。")
    parser.add_argument("--display-camera", default="zed_cam_left", help="本地窗口和 Unity 图像流使用的 MuJoCo 相机名称，默认左侧 ZED 相机。")
    parser.add_argument("--render-width", type=int, default=640, help="mujoco窗口渲染图像宽度。")
    parser.add_argument("--render-height", type=int, default=480, help="mujoco窗口渲染图像高度。")
    parser.add_argument("--window-name", default="Quest3 MuJoCo Teleop", help="本地 OpenCV 窗口标题。")
    parser.add_argument("--unity-image-stream", type=_str_to_bool, default=DEFAULT_CONFIG["unity_image_stream"], help="是否向 Unity 发送渲染图像，填写 true/false。")
    parser.add_argument("--unity-image-host", default="auto", help="Unity 图像接收端 IP；auto 表示自动使用 Quest 位姿包来源 IP。")
    parser.add_argument("--unity-image-port", type=int, default=5010, help="Unity 图像接收端 UDP 端口。")
    parser.add_argument("--unity-image-hz", type=float, default=25.0, help="发送给 Unity 的图像帧率。")
    parser.add_argument("--unity-image-jpeg-quality", type=int, default=55, help="发送给 Unity 的 JPEG 压缩质量，范围 1-100。")
    parser.add_argument("--unity-image-chunk-size", type=int, default=1000, help="UDP 图像分片大小，单位字节。")
    parser.add_argument("--unity-image-log-interval", type=float, default=2.0, help="Unity 图像发送状态打印间隔，0 表示不打印。")
    parser.add_argument("--workspace-low", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"), help="机械臂目标位置工作空间下限。")
    parser.add_argument("--workspace-high", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"), help="机械臂目标位置工作空间上限。")
    parser.add_argument("--status-hz-interval", type=float, default=0.5, help="终端状态打印间隔，单位秒。")
    parser.add_argument("--mujoco-gl", choices=("auto", "glfw", "egl", "osmesa"), default="auto", help="MuJoCo OpenGL 后端。")
    parser.add_argument("--allow-partial-anchor", type=_str_to_bool, default=DEFAULT_CONFIG["allow_partial_anchor"], help="是否允许部分 Quest 设备可用时就开始锚定，填写 true/false。")
    parser.add_argument("--start-on-first-packet", type=_str_to_bool, default=DEFAULT_CONFIG["start_on_first_packet"], help="是否收到第一帧有效数据后自动开始，填写 true/false。")
    parser.add_argument("--no-convert-to-mujoco", type=_str_to_bool, default=DEFAULT_CONFIG["no_convert_to_mujoco"], help="是否跳过 Unity/Quest 坐标到 MuJoCo 坐标的转换，填写 true/false。")
    parser.add_argument("--viewer", type=_str_to_bool, default=DEFAULT_CONFIG["viewer"], help="是否打开 MuJoCo viewer，填写 true/false。")
    parser.add_argument("--camera-window", type=_str_to_bool, default=DEFAULT_CONFIG["camera_window"], help="是否打开本地 OpenCV 相机窗口，填写 true/false。")
    return parser


def run_quest3_mujoco_test(**overrides) -> None:
    """Run teleoperation with file-editable parameters instead of CLI input."""
    args = build_arg_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    run(args)


if __name__ == "__main__":
    # ===== 1. Quest3 位姿通信 =====
    HOST = "0.0.0.0"       # Python 监听地址；0.0.0.0 表示监听所有网卡。
    PORT = 5005            # 需要和 Quest3PoseUdpSender.cs 的 receiverPort 一致。

    # ===== 2. 机械臂控制手感 =====
    HEAD_CONTROL = True    # 是否用头显控制中间臂。
    INDIVIDUAL_HAND_ANCHORS = True  # 左右臂是否使用各自手柄初始位姿作为锚点。
    LOCK_ROLL = True       # 是否锁定 roll，只保留 yaw/pitch。

    HAND_POSITION_SCALE = 1   # 手部位移缩放系数，调节手部控制的灵敏度。
    HAND_MAX_DELTA = 1       # 手部最大位移量
    HEAD_POSITION_SCALE = 1.0   # 头部位移缩放系数，调节头显控制的灵敏度。
    HEAD_MAX_DELTA = 1       # 头部最大位移量

    # ===== 3. 发送给 Unity/Quest 的 ZED 画面 =====
    DISPLAY_CAMERA = "zed_cam_left"
    RENDER_WIDTH = 640
    RENDER_HEIGHT = 480
    UNITY_IMAGE_STREAM = True
    UNITY_IMAGE_HOST = "auto"        # auto 表示自动使用 Quest 位姿包来源 IP。
    UNITY_IMAGE_PORT = 5010          # 需要和 ZedImageUdpReceiver.cs 的 listenPort 一致。
    UNITY_IMAGE_HZ = 25.0
    UNITY_IMAGE_JPEG_QUALITY = 55

    # ===== 4. 本地调试窗口 =====
    VIEWER = True
    CAMERA_WINDOW = True

    run_quest3_mujoco_test(
        host=HOST,
        port=PORT,
        head_control=HEAD_CONTROL,
        individual_hand_anchors=INDIVIDUAL_HAND_ANCHORS,
        lock_roll=LOCK_ROLL,
        hand_position_scale=HAND_POSITION_SCALE,
        hand_max_delta=HAND_MAX_DELTA,
        head_position_scale=HEAD_POSITION_SCALE,
        head_max_delta=HEAD_MAX_DELTA,
        display_camera=DISPLAY_CAMERA,
        render_width=RENDER_WIDTH,
        render_height=RENDER_HEIGHT,
        unity_image_stream=UNITY_IMAGE_STREAM,
        unity_image_host=UNITY_IMAGE_HOST,
        unity_image_port=UNITY_IMAGE_PORT,
        unity_image_hz=UNITY_IMAGE_HZ,
        unity_image_jpeg_quality=UNITY_IMAGE_JPEG_QUALITY,
        viewer=VIEWER,
        camera_window=CAMERA_WINDOW,
    )
