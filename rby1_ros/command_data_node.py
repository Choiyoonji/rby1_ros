#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import h5py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, UInt64, String
from rby1_ros.qos_profiles import qos_ctrl_latched, qos_tick, qos_state_latest

# Import the RBY1 State message

from rby1_interfaces.msg import EEpos, FTsensor, StateRBY1, CommandRBY1, CommandHand, Action


class RBY1DataNode(Node):
    def __init__(self):
        super().__init__("rby1_data_node_joint_both")

        # -------- Parameters --------
        self.declare_parameter("topic_state", '/control/command')

        self.topic_state: str = self.get_parameter("topic_state").get_parameter_value().string_value

        # -------- ROS I/O --------
        self.sub_record = self.create_subscription(Bool, "/record", self._on_record, qos_ctrl_latched)
        self.sub_tick   = self.create_subscription(UInt64, "/tick", self._on_tick, qos_tick)
        self.sub_path   = self.create_subscription(String, "/dataset_path", self._on_data_path, qos_ctrl_latched)
        self.sub_state  = self.create_subscription(CommandRBY1, self.topic_state, self._on_state, qos_state_latest)

        # -------- State --------
        self.recording: bool = False
        self.session_start_mono_ns: Optional[int] = None

        # latest state cache
        self.latest_state: Optional[CommandRBY1] = None
        self.latest_state_seq: int = 0          # increases on each new /rby1/state
        self._seen_seq_at_last_tick: int = 0    # seq observed at last tick
        self.last_ts: float = -1.0               # timestamp of last recorded state

        # HDF5 path & handle
        self.dataset_dir: Optional[Path] = None
        self.save_path: Optional[Path] = None

        # column buffers (in-memory until stop)
        self.buf_now_mono_ns: List[int] = []
        self.buf_tick: List[int] = []
        self.buf_state_updated: List[int] = []  # 0/1
        self.buf_state_timestamp: List[float] = []

        # vlen buffers for arrays
        self.buf_joint_positions: List[np.ndarray] = []

        self.buf_flags_is_active: List[int] = []
        self.buf_flags_left_btn: List[int] = []
        self.buf_flags_right_btn: List[int] = []
        self.buf_flags_ready: List[int] = []
        self.buf_flags_move: List[int] = []
        self.buf_flags_stop: List[int] = []
        self.buf_flags_estop: List[int] = []

        self.buf_right_gripper_pos: List[float] = []
        self.buf_left_gripper_pos: List[float] = []

        self.dataset_path = None

        self.get_logger().info(f"[init] RBY1 Data Node initialized, subscribing to '{self.topic_state}'")

    # ----------------- Callbacks -----------------
    def _on_record(self, msg: Bool):
        if msg.data and not self.recording:
            self.recording = True
        elif (not msg.data) and self.recording:
            self._stop_and_flush()

    def _on_data_path(self, msg: String):
        self.dataset_path = Path(msg.data)
        self.save_path = str(self.dataset_path / "command_state.h5")
        self.get_logger().info(f"Received dataset path: {self.dataset_path}")
        self._start_session()

    def _on_tick(self, msg: UInt64):
        if not self.recording:
            return
        
        if self.dataset_path is None:
            self.get_logger().warn("Tick received but dataset_path is None, skipping tick.")
            return

        now_mono_ns = time.monotonic_ns()

        # Snapshot the latest state (may be None initially)
        st = self.latest_state

        # Check if state updated since last tick
        # updated = 1 if (self.latest_state_seq != self._seen_seq_at_last_tick) else 0
        # self._seen_seq_at_last_tick = self.latest_state_seq
        
        timestamp = float(getattr(st, "timestamp", 0.0))

        updated = 1 if (timestamp != self.last_ts) else 0
        self.last_ts = timestamp
        
        # Append scalars
        self.buf_now_mono_ns.append(int(now_mono_ns))
        self.buf_tick.append(int(msg.data))
        self.buf_state_updated.append(int(updated))
        self.buf_state_timestamp.append(timestamp)

        # Append arrays (use empty arrays if state missing)
        def as_np(arr_like, dtype=np.float32):
            try:
                return np.asarray(arr_like, dtype=dtype)
            except Exception:
                return np.asarray([], dtype=dtype)

        if st is None:
            # push empties
            self.buf_joint_positions.append(np.asarray([], dtype=np.float32))

            self.buf_flags_is_active.append(0)
            self.buf_flags_left_btn.append(0)
            self.buf_flags_right_btn.append(0)
            self.buf_flags_ready.append(0)
            self.buf_flags_move.append(0)
            self.buf_flags_stop.append(0)
            self.buf_flags_estop.append(0)

            self.buf_right_gripper_pos.append(np.nan)
            self.buf_left_gripper_pos.append(np.nan)
            return

        # For Float32MultiArray fields, take the `.data` part
        jp = getattr(st.desired_joint_positions, "data", [])

        self.buf_joint_positions.append(as_np(jp))

        # Flags
        self.buf_flags_is_active.append(1 if getattr(st, "is_active", False) else 0)
        self.buf_flags_left_btn.append(1 if getattr(st, "left_btn", False) else 0)
        self.buf_flags_right_btn.append(1 if getattr(st, "right_btn", False) else 0)
        self.buf_flags_ready.append(1 if getattr(st, "ready", False) else 0)
        self.buf_flags_move.append(1 if getattr(st, "move", False) else 0)
        self.buf_flags_stop.append(1 if getattr(st, "stop", False) else 0)
        self.buf_flags_estop.append(1 if getattr(st, "estop", False) else 0)

        # Gripper
        self.buf_right_gripper_pos.append(float(getattr(st, "desired_right_gripper_pos", np.nan)))
        self.buf_left_gripper_pos.append(float(getattr(st, "desired_left_gripper_pos", np.nan)))

    def _on_state(self, msg: CommandRBY1):
        self.latest_state = msg
        self.latest_state_seq += 1

    # ----------------- Session handling -----------------
    def _start_session(self):
        self.session_start_mono_ns = time.monotonic_ns()
        self.recording = True
        self._seen_seq_at_last_tick = self.latest_state_seq  # snapshot
        self.get_logger().info(f"[record] START → '{self.save_path}'")

    def _stop_and_flush(self):
        self.recording = False
        try:
            self._write_h5(Path(self.save_path))
            self.get_logger().info(f"[record] STOP → wrote {len(self.buf_tick)} ticks to '{self.save_path}'")
        except Exception as e:
            self.get_logger().error(f"HDF5 write failed: {e}")
        finally:
            self._clear_buffers()
            self.dataset_dir = None
            self.save_path = None
            self.session_start_mono_ns = None

    # ----------------- HDF5 I/O -----------------
    def _write_h5(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare vlen dtype for float32 arrays
        vlen_f32 = h5py.special_dtype(vlen=np.dtype('float32'))

        with h5py.File(str(path), "w") as f:
            # Root attributes
            f.attrs["base_dir"] = str(self.dataset_dir)
            f.attrs["topic_state"] = self.topic_state
            f.attrs["session_start_mono_ns"] = int(self.session_start_mono_ns) if self.session_start_mono_ns else -1
            f.attrs["created_wall_time_ns"] = int(time.time_ns())

            g = f.create_group("samples")


            # ---- 1) 스칼라형 데이터셋 (압축 사용) ----
            def write_scalar(name: str, data, dtype):
                arr = np.asarray(data, dtype=dtype)
                g.create_dataset(
                    name,
                    data=arr,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    fletcher32=True,
                    chunks=True,
                )

            write_scalar("now_mono_ns",      self.buf_now_mono_ns,      np.int64)
            write_scalar("tick",             self.buf_tick,             np.uint64)
            write_scalar("state_updated",    self.buf_state_updated,    np.uint8)
            write_scalar("state_timestamp",  self.buf_state_timestamp,  np.float64)

            # ---- 2) 고정 길이 2D 배열로 저장 (NaN padding) ----
            def write_fixed_2d(name: str, seq: List[np.ndarray]):
                """
                seq: length N 리스트, 각 원소는 1D np.ndarray (길이가 서로 다를 수도 있음)
                => (N, max_len) float32 배열로 만들고, 모자란 부분은 NaN으로 패딩.
                """
                if not seq:
                    # 데이터가 전혀 없으면 굳이 dataset 안 만듦
                    return

                # 각 row 길이
                lengths = [len(np.asarray(a)) for a in seq]
                max_len = max(lengths)

                # (N, max_len) 배열 만들고 NaN으로 채우기
                arr = np.full((len(seq), max_len), np.nan, dtype=np.float32)
                for i, a in enumerate(seq):
                    a = np.asarray(a, dtype=np.float32).ravel()
                    if a.size == 0:
                        continue
                    L = min(a.size, max_len)
                    arr[i, :L] = a[:L]

                g.create_dataset(
                    name,
                    data=arr,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    fletcher32=True,
                    chunks=True,
                )

            # ---- joint / ee / ft / com 전부 고정 2D로 저장 ----
            write_fixed_2d("joint_positions",   self.buf_joint_positions)

            # Flags & gripper
            g.create_dataset("is_active", data=np.asarray(self.buf_flags_is_active, dtype=np.uint8), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)
            g.create_dataset("left_btn", data=np.asarray(self.buf_flags_left_btn, dtype=np.uint8), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)
            g.create_dataset("right_btn", data=np.asarray(self.buf_flags_right_btn, dtype=np.uint8), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)
            g.create_dataset("ready", data=np.asarray(self.buf_flags_ready, dtype=np.uint8), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)
            g.create_dataset("move", data=np.asarray(self.buf_flags_move, dtype=np.uint8), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)
            g.create_dataset("stop", data=np.asarray(self.buf_flags_stop, dtype=np.uint8), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)
            g.create_dataset("estop", data=np.asarray(self.buf_flags_estop, dtype=np.uint8), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)
            
            g.create_dataset("right_gripper_pos", data=np.asarray(self.buf_right_gripper_pos, dtype=np.float32), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)
            g.create_dataset("left_gripper_pos", data=np.asarray(self.buf_left_gripper_pos, dtype=np.float32), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)

    def _clear_buffers(self):
        self.buf_now_mono_ns.clear()
        self.buf_tick.clear()
        self.buf_state_updated.clear()
        self.buf_state_timestamp.clear()

        self.buf_joint_positions.clear()

        self.buf_flags_is_active.clear()
        self.buf_flags_left_btn.clear()
        self.buf_flags_right_btn.clear()
        self.buf_flags_ready.clear()
        self.buf_flags_move.clear()
        self.buf_flags_stop.clear()
        self.buf_flags_estop.clear()

        self.buf_right_gripper_pos.clear()
        self.buf_left_gripper_pos.clear()

    # ----------------- Shutdown -----------------
    def destroy_node(self):
        if self.recording:
            self.get_logger().warn("Node shutting down while recording — finalizing HDF5.")
            self._stop_and_flush()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RBY1DataNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt — exit")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()