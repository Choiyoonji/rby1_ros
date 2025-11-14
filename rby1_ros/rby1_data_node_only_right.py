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
from rby1_interfaces.msg import State as RBY1State


class RBY1DataNode(Node):
    def __init__(self):
        super().__init__("rby1_data_node")

        # -------- Parameters --------
        self.declare_parameter("topic_state", "/rby1/state")

        self.topic_state: str = self.get_parameter("topic_state").get_parameter_value().string_value

        # -------- ROS I/O --------
        self.sub_record = self.create_subscription(Bool, "/record", self._on_record, qos_ctrl_latched)
        self.sub_tick   = self.create_subscription(UInt64, "/tick", self._on_tick, qos_tick)
        self.sub_path   = self.create_subscription(String, "/dataset_path", self._on_data_path, qos_ctrl_latched)
        self.sub_state  = self.create_subscription(RBY1State, self.topic_state, self._on_state, qos_state_latest)

        # -------- State --------
        self.recording: bool = False
        self.session_start_mono_ns: Optional[int] = None

        # latest state cache
        self.latest_state: Optional[RBY1State] = None
        self.latest_state_seq: int = 0          # increases on each new /rby1/state
        self._seen_seq_at_last_tick: int = 0    # seq observed at last tick

        # HDF5 path & handle
        self.dataset_dir: Optional[Path] = None
        self.h5_path: Optional[Path] = None

        # column buffers (in-memory until stop)
        self.buf_now_mono_ns: List[int] = []
        self.buf_tick: List[int] = []
        self.buf_state_updated: List[int] = []  # 0/1
        self.buf_state_timestamp: List[float] = []

        # vlen buffers for arrays
        self.buf_joint_positions: List[np.ndarray] = []
        self.buf_joint_velocities: List[np.ndarray] = []
        self.buf_joint_currents: List[np.ndarray] = []
        self.buf_joint_torques: List[np.ndarray] = []

        self.buf_right_ee_pos: List[np.ndarray] = []      # 3
        self.buf_right_ee_quat: List[np.ndarray] = []     # 4
        self.buf_torso_ee_pos: List[np.ndarray] = []      # 3
        self.buf_torso_ee_quat: List[np.ndarray] = []     # 4

        self.buf_right_ft_force: List[np.ndarray] = []    # 3
        self.buf_right_ft_torque: List[np.ndarray] = []   # 3

        self.buf_center_of_mass: List[np.ndarray] = []    # 3

        self.buf_flags_initialized: List[int] = []
        self.buf_flags_stopped: List[int] = []
        self.buf_flags_right_follow: List[int] = []

        self.buf_right_gripper_pos: List[float] = []

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
        self.save_path = str(self.dataset_path / "rby1_state.h5")
        self.get_logger().info(f"Received dataset path: {self.dataset_path}")
        self._start_session()

    def _on_tick(self, msg: UInt64):
        if not self.recording:
            return
        
        if self.dataset_path is None:
            self.get_logger().warn("Tick received but dataset_path is None, skipping tick.")
            return

        now_mono_ns = time.monotonic_ns()
        # Check if state updated since last tick
        updated = 1 if (self.latest_state_seq != self._seen_seq_at_last_tick) else 0
        self._seen_seq_at_last_tick = self.latest_state_seq

        # Snapshot the latest state (may be None initially)
        st = self.latest_state

        # Append scalars
        self.buf_now_mono_ns.append(int(now_mono_ns))
        self.buf_tick.append(int(msg.data))
        self.buf_state_updated.append(int(updated))
        self.buf_state_timestamp.append(float(getattr(st, "timestamp", 0.0)) if st else np.nan)

        # Append arrays (use empty arrays if state missing)
        def as_np(arr_like, dtype=np.float32):
            try:
                return np.asarray(arr_like, dtype=dtype)
            except Exception:
                return np.asarray([], dtype=dtype)

        if st is None:
            # push empties
            self.buf_joint_positions.append(np.asarray([], dtype=np.float32))
            self.buf_joint_velocities.append(np.asarray([], dtype=np.float32))
            self.buf_joint_currents.append(np.asarray([], dtype=np.float32))
            self.buf_joint_torques.append(np.asarray([], dtype=np.float32))

            self.buf_right_ee_pos.append(np.asarray([], dtype=np.float32))
            self.buf_right_ee_quat.append(np.asarray([], dtype=np.float32))
            self.buf_torso_ee_pos.append(np.asarray([], dtype=np.float32))
            self.buf_torso_ee_quat.append(np.asarray([], dtype=np.float32))

            self.buf_right_ft_force.append(np.asarray([], dtype=np.float32))
            self.buf_right_ft_torque.append(np.asarray([], dtype=np.float32))

            self.buf_center_of_mass.append(np.asarray([], dtype=np.float32))

            self.buf_flags_initialized.append(0)
            self.buf_flags_stopped.append(0)
            self.buf_flags_right_follow.append(0)

            self.buf_right_gripper_pos.append(np.nan)
            return

        # For Float32MultiArray fields, take the `.data` part
        jp = getattr(st.joint_positions, "data", [])
        jv = getattr(st.joint_velocities, "data", [])
        jc = getattr(st.joint_currents, "data", [])
        jt = getattr(st.joint_torques, "data", [])

        self.buf_joint_positions.append(as_np(jp))
        self.buf_joint_velocities.append(as_np(jv))
        self.buf_joint_currents.append(as_np(jc))
        self.buf_joint_torques.append(as_np(jt))

        # EE poses
        def pos_quat(obj):
            # obj: EEpos with .position (Float32MultiArray) and .quaternion
            pos = getattr(getattr(obj, "position", None), "data", [])
            quat = getattr(getattr(obj, "quaternion", None), "data", [])
            return as_np(pos), as_np(quat)

        rp, rq = pos_quat(st.right_ee_pos)
        tp, tq = pos_quat(st.torso_ee_pos)
        self.buf_right_ee_pos.append(rp); self.buf_right_ee_quat.append(rq)
        self.buf_torso_ee_pos.append(tp); self.buf_torso_ee_quat.append(tq)

        # FT sensors
        def ft(obj):
            force = getattr(getattr(obj, "force", None), "data", [])
            torque = getattr(getattr(obj, "torque", None), "data", [])
            return as_np(force), as_np(torque)

        rff, rft = ft(st.right_ft_sensor)
        self.buf_right_ft_force.append(rff); self.buf_right_ft_torque.append(rft)

        # Center of mass
        com = getattr(st.center_of_mass, "data", [])
        self.buf_center_of_mass.append(as_np(com))

        # Flags
        self.buf_flags_initialized.append(1 if getattr(st, "is_initialized", False) else 0)
        self.buf_flags_stopped.append(1 if getattr(st, "is_stopped", False) else 0)
        self.buf_flags_right_follow.append(1 if getattr(st, "is_right_following", False) else 0)

        # Gripper
        self.buf_right_gripper_pos.append(float(getattr(st, "right_gripper_pos", np.nan)))

    def _on_state(self, msg: RBY1State):
        self.latest_state = msg
        self.latest_state_seq += 1

    # ----------------- Session handling -----------------
    def _start_session(self):
        self.h5_path = self.dataset_dir / self.h5_name
        self.session_start_mono_ns = time.monotonic_ns()
        self.recording = True
        self._seen_seq_at_last_tick = self.latest_state_seq  # snapshot
        self.get_logger().info(f"[record] START → '{self.h5_path}'")

    def _stop_and_flush(self):
        self.recording = False
        try:
            self._write_h5(self.h5_path)
            self.get_logger().info(f"[record] STOP → wrote {len(self.buf_tick)} ticks to '{self.h5_path}'")
        except Exception as e:
            self.get_logger().error(f"HDF5 write failed: {e}")
        finally:
            self._clear_buffers()
            self.dataset_dir = None
            self.h5_path = None
            self.session_start_mono_ns = None

    # ----------------- HDF5 I/O -----------------
    def _write_h5(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare vlen dtype for float32 arrays
        vlen_f32 = h5py.special_dtype(vlen=np.dtype('float32'))

        with h5py.File(str(path), "w") as f:
            # Root attributes
            f.attrs["task"] = self.task
            f.attrs["base_dir"] = str(self.base_dir)
            f.attrs["topic_state"] = self.topic_state
            f.attrs["session_start_mono_ns"] = int(self.session_start_mono_ns) if self.session_start_mono_ns else -1
            f.attrs["created_wall_time_ns"] = int(time.time_ns())

            g = f.create_group("samples")

            # Scalars
            g.create_dataset("now_mono_ns", data=np.asarray(self.buf_now_mono_ns, dtype=np.int64), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)
            g.create_dataset("tick", data=np.asarray(self.buf_tick, dtype=np.uint64), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)
            g.create_dataset("state_updated", data=np.asarray(self.buf_state_updated, dtype=np.uint8), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)
            g.create_dataset("state_timestamp", data=np.asarray(self.buf_state_timestamp, dtype=np.float64), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)

            # Helper to write vlen float32 list
            def write_vlen(name: str, seq: List[np.ndarray]):
                dset = g.create_dataset(name, (len(seq),), dtype=vlen_f32, compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)
                for i, arr in enumerate(seq):
                    dset[i] = np.asarray(arr, dtype=np.float32)

            # Arrays
            write_vlen("joint_positions", self.buf_joint_positions)
            write_vlen("joint_velocities", self.buf_joint_velocities)
            write_vlen("joint_currents", self.buf_joint_currents)
            write_vlen("joint_torques", self.buf_joint_torques)

            write_vlen("right_ee_pos", self.buf_right_ee_pos)
            write_vlen("right_ee_quat", self.buf_right_ee_quat)
            write_vlen("torso_ee_pos", self.buf_torso_ee_pos)
            write_vlen("torso_ee_quat", self.buf_torso_ee_quat)

            write_vlen("right_ft_force", self.buf_right_ft_force)
            write_vlen("right_ft_torque", self.buf_right_ft_torque)

            write_vlen("center_of_mass", self.buf_center_of_mass)

            # Flags & gripper
            g.create_dataset("is_initialized", data=np.asarray(self.buf_flags_initialized, dtype=np.uint8), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)
            g.create_dataset("is_stopped", data=np.asarray(self.buf_flags_stopped, dtype=np.uint8), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)
            g.create_dataset("is_right_following", data=np.asarray(self.buf_flags_right_follow, dtype=np.uint8), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)

            g.create_dataset("right_gripper_pos", data=np.asarray(self.buf_right_gripper_pos, dtype=np.float32), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True, chunks=True)

    def _clear_buffers(self):
        self.buf_now_mono_ns.clear()
        self.buf_tick.clear()
        self.buf_state_updated.clear()
        self.buf_state_timestamp.clear()

        self.buf_joint_positions.clear()
        self.buf_joint_velocities.clear()
        self.buf_joint_currents.clear()
        self.buf_joint_torques.clear()

        self.buf_right_ee_pos.clear()
        self.buf_right_ee_quat.clear()
        self.buf_torso_ee_pos.clear()
        self.buf_torso_ee_quat.clear()

        self.buf_right_ft_force.clear()
        self.buf_right_ft_torque.clear()

        self.buf_center_of_mass.clear()

        self.buf_flags_initialized.clear()
        self.buf_flags_stopped.clear()
        self.buf_flags_right_follow.clear()

        self.buf_right_gripper_pos.clear()

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