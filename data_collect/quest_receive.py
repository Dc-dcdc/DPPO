from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/dppo_numba_cache")

from headset_utils import HeadsetData, convert_left_to_right_coordinates


__all__ = ["QuestReceive"]


class QuestReceive:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5005,
        timeout: float | None = 1.0,
        buffer_size: int = 65535,
        convert_to_mujoco: bool = True,
    ):
        self.host = host
        self.port = int(port)
        self.timeout = timeout
        self.buffer_size = int(buffer_size)
        self.convert_to_mujoco = bool(convert_to_mujoco)
        self.latest_data: HeadsetData | None = None
        self.latest_address: tuple[str, int] | None = None

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(self.timeout)

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # 拿下一帧
    def receive_data(self) -> HeadsetData:
        payload, address = self.sock.recvfrom(self.buffer_size)
        raw = json.loads(payload.decode("utf-8"))
        data = self.parse_headset_data(raw)
        self.latest_data = data
        self.latest_address = address
        return data

    # 拿最新帧，丢掉旧帧
    def receive_latest_data(self) -> HeadsetData:
        data = self.receive_data()
        old_timeout = self.sock.gettimeout()
        self.sock.settimeout(0.0)
        try:
            while True:
                data = self.receive_data()
        except (BlockingIOError, socket.timeout):
            pass
        finally:
            self.sock.settimeout(old_timeout)
        return data

    def parse_headset_data(self, raw: dict[str, Any]) -> HeadsetData:
        data = HeadsetData()

        h_pos, h_quat = _get_pose(raw, "head")
        l_pos, l_quat = _get_pose(raw, "left")
        r_pos, r_quat = _get_pose(raw, "right")

        if self.convert_to_mujoco:
            # 将头部、左手柄、右手柄的位姿从左手坐标系转换为右手坐标系，以便在机器人世界中使用
            h_pos, h_quat = convert_left_to_right_coordinates(h_pos, h_quat)
            l_pos, l_quat = convert_left_to_right_coordinates(l_pos, l_quat)
            r_pos, r_quat = convert_left_to_right_coordinates(r_pos, r_quat)

        data.h_pos = h_pos
        data.h_quat = h_quat
        data.l_pos = l_pos
        data.l_quat = l_quat
        data.r_pos = r_pos
        data.r_quat = r_quat

        data.l_thumbstick_x = _get_float(raw, "left", "thumbstick_x", 0.0)
        data.l_thumbstick_y = _get_float(raw, "left", "thumbstick_y", 0.0)
        data.r_thumbstick_x = _get_float(raw, "right", "thumbstick_x", 0.0)
        data.r_thumbstick_y = _get_float(raw, "right", "thumbstick_y", 0.0)

        data.l_index_trigger = _get_float(raw, "left", "index_trigger", 0.0)
        data.r_index_trigger = _get_float(raw, "right", "index_trigger", 0.0)
        data.l_hand_trigger = _get_float(raw, "left", "hand_trigger", 0.0)
        data.r_hand_trigger = _get_float(raw, "right", "hand_trigger", 0.0)

        data.l_button_one = _get_bool(raw, "left", "button_one", False)
        data.l_button_two = _get_bool(raw, "left", "button_two", False)
        data.r_button_one = _get_bool(raw, "right", "button_one", False)
        data.r_button_two = _get_bool(raw, "right", "button_two", False)
        data.l_button_thumbstick = _get_bool(raw, "left", "thumbstick_button", False)
        data.r_button_thumbstick = _get_bool(raw, "right", "thumbstick_button", False)

        return data


def _get_group(raw: dict[str, Any], group: str) -> dict[str, Any]:
    value = raw.get(group, {})
    return value if isinstance(value, dict) else {}


def _get_pose(raw: dict[str, Any], group: str) -> tuple[np.ndarray, np.ndarray]:
    obj = _get_group(raw, group)
    pos = _as_vec(obj.get("pos"), 3, default=(0.0, 0.0, 0.0))
    quat = _normalize_quat(_as_vec(obj.get("quat"), 4, default=(0.0, 0.0, 0.0, 1.0)))
    return pos, quat


def _get_float(raw: dict[str, Any], group: str, key: str, default: float) -> float:
    obj = _get_group(raw, group)
    value = obj.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _get_bool(raw: dict[str, Any], group: str, key: str, default: bool) -> bool:
    obj = _get_group(raw, group)
    value = obj.get(key, default)
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def _as_vec(value: Any, length: int, default: tuple[float, ...]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    if isinstance(value, dict):
        if length == 3:
            value = [value.get("x", 0.0), value.get("y", 0.0), value.get("z", 0.0)]
        elif length == 4:
            value = [value.get("x", 0.0), value.get("y", 0.0), value.get("z", 0.0), value.get("w", 1.0)]
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != length:
        return np.asarray(default, dtype=np.float64)
    return arr


def _normalize_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat_xyzw)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat_xyzw / norm




# =======================================================================
# 以下是测试内容
# =======================================================================
def _fmt_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(x): .4f}" for x in vec) + "]"


def format_headset_data(data: HeadsetData) -> str:
    lines = [
        "-" * 86,
        f"head  pos={_fmt_vec(data.h_pos)} quat_xyzw={_fmt_vec(data.h_quat)}",
        f"left  pos={_fmt_vec(data.l_pos)} quat_xyzw={_fmt_vec(data.l_quat)}", 
        f"trig={data.l_index_trigger:.3f} grip={data.l_hand_trigger:.3f} stick=({data.l_thumbstick_x:.3f},{data.l_thumbstick_y:.3f}) btn1={int(data.l_button_one)} btn2={int(data.l_button_two)} stick_btn={int(data.l_button_thumbstick)}",
        f"right pos={_fmt_vec(data.r_pos)} quat_xyzw={_fmt_vec(data.r_quat)}", 
        f"trig={data.r_index_trigger:.3f} grip={data.r_hand_trigger:.3f} stick=({data.r_thumbstick_x:.3f},{data.r_thumbstick_y:.3f}) btn1={int(data.r_button_one)} btn2={int(data.r_button_two)} stick_btn={int(data.r_button_thumbstick)}",
    ]
    return "\n".join(lines)


def _str_to_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    return value in {"1", "true", "t", "yes", "y", "on", "是", "开", "开启"}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receive Meta Quest 3 tracking data over UDP.")
    parser.add_argument("--host", default="0.0.0.0", help="UDP bind host.")
    parser.add_argument("--port", type=int, default=5005, help="UDP bind port.")
    parser.add_argument("--timeout", type=float, default=1.0, help="Socket timeout in seconds.")
    parser.add_argument("--print-hz", type=float, default=20.0, help="Max terminal print rate.")
    parser.add_argument(
        "--convert-to-mujoco",
        type=_str_to_bool,
        default=True,
        help="Convert Unity/Quest coordinates to MuJoCo coordinates.",
    )
    args = parser.parse_args()

    with QuestReceive(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        convert_to_mujoco=args.convert_to_mujoco,
    ) as receiver:
        print(f"Listening for Quest3 UDP JSON on {args.host}:{args.port}")
        print("Press Ctrl+C to stop.")
        last_print = 0.0
        try:
            while True:
                try:
                    data = receiver.receive_latest_data()  # 
                except socket.timeout:
                    print(f"No packet received within {args.timeout:.1f}s...")
                    continue
                except json.JSONDecodeError as exc:
                    print(f"Invalid JSON packet: {exc}")
                    continue

                now = time.time()
                if args.print_hz <= 0 or now - last_print >= 1.0 / args.print_hz:
                    print(format_headset_data(data))
                    last_print = now
        except KeyboardInterrupt:
            print("\nStopped QuestReceive.")
