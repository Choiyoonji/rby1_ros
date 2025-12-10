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
from rby1_interfaces.msg import StateHand as InspireState

class HandDataNode(Node):
    def __init__(self):
        super().__init__("inspire_hand_data_node")

        # -------- Parameters --------
        self.declare_parameter("topic_state", "/hand/state")

        self.topic_state: str = self.get_parameter("topic_state").get_parameter_value().string_value

        # -------- ROS I/O --------
        self.sub_record = self.create_subscription(Bool, "/record", self._on_record, qos_ctrl_latched)
        self.sub_tick   = self.create_subscription(UInt64, "/tick", self._on_tick, qos_tick)
        self.sub_path   = self.create_subscription(String, "/dataset_path", self._on_data_path, qos_ctrl_latched)
        self.sub_state  = self.create_subscription(InspireState, self.topic_state, self._on_state, qos_state_latest)

        # -------- State --------
        self.recording: bool = False
        self.session_start_mono_ns: Optional[int] = None

        # latest state cache
        self.latest_state: Optional[InspireState] = None
        self.latest_state_seq: int = 0          # increases on each new /rby1/state
        self._seen_seq_at_last_tick: int = 0    # seq observed at last tick

        # HDF5 path & handle
        self.dataset_path: Optional[Path] = None
        self.h5_path: Optional[Path] = None

        self._init_buffers()

        self.get_logger().info(f"[init] Inspire Hand Data Node initialized, subscribing to '{self.topic_state}'")
        
    def _init_buffers(self):
        self.buf_now_mono_ns = []
        self.buf_tick = []
        self.buf_state_updated = []
        self.buf_state_timestamp = []
        
        self.buf_l_angle = []
        self.buf_l_angle_force = [] 
        self.buf_l_norm_force = []
        self.buf_l_tang_force = []
        self.buf_l_temp = []
        self.buf_l_cur = []
        
        self.buf_r_angle = []
        self.buf_r_angle_force = []
        self.buf_r_norm_force = []
        self.buf_r_tang_force = []
        self.buf_r_temp = []
        self.buf_r_cur = []
        
    # ----------------- Callbacks -----------------
    def _on_record(self, msg: Bool):
        if msg.data and not self.recording:
            if self.dataset_path:
                self._start_recording()
            else:
                self.recording = True
        elif (not msg.data) and self.recording:
            self._stop_and_flush()

    def _on_data_path(self, msg: String):
        self.dataset_path = msg.data
        self.h5_path = os.path.join(self.dataset_path, "inspire_hand_data.h5")
        if self.recording:
            self._start_recording()

    def _start_recording(self):
        self.recording = True
        self._init_buffers()
        self.get_logger().info("[Record] START")

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
            self.buf_l_angle.append(np.asarray([]))
            self.buf_l_angle_force.append(np.asarray([]))
            self.buf_l_norm_force.append(np.asarray([]))
            self.buf_l_tang_force.append(np.asarray([]))
            self.buf_l_temp.append(np.asarray([]))
            self.buf_l_cur.append(np.asarray([]))

            self.buf_r_angle.append(np.asarray([]))
            self.buf_r_angle_force.append(np.asarray([]))
            self.buf_r_norm_force.append(np.asarray([]))
            self.buf_r_tang_force.append(np.asarray([]))
            self.buf_r_temp.append(np.asarray([]))
            self.buf_r_cur.append(np.asarray([]))
            return

        self.buf_l_angle.append(as_np(st.act_angle_l))
        self.buf_l_angle_force.append(as_np(st.act_angle_force_l))
        self.buf_l_norm_force.append(as_np(st.act_norm_force_l))
        self.buf_l_tang_force.append(as_np(st.act_tang_force_l))
        self.buf_l_temp.append(as_np(st.act_temp_l))
        self.buf_l_cur.append(as_np(st.act_cur_l))

        self.buf_r_angle.append(as_np(st.act_angle_r))
        self.buf_r_angle_force.append(as_np(st.act_angle_force_r))
        self.buf_r_norm_force.append(as_np(st.act_norm_force_r))
        self.buf_r_tang_force.append(as_np(st.act_tang_force_r))
        self.buf_r_temp.append(as_np(st.act_temp_r))
        self.buf_r_cur.append(as_np(st.act_cur_r))

    def _on_state(self, msg: InspireState):
        self.latest_state = msg
        self.latest_state_seq += 1

    # ----------------- Session handling -----------------
    def _start_session(self):
        self.session_start_mono_ns = time.monotonic_ns()
        self.recording = True
        self._seen_seq_at_last_tick = self.latest_state_seq  # snapshot
        self.get_logger().info(f"[record] START → '{self.h5_path}'")
        
    def _stop_and_flush(self):
        self.recording = False
        try:
            self._write_h5(Path(self.h5_path))
            self.get_logger().info(f"[record] STOP → wrote {len(self.buf_tick)} ticks to '{self.h5_path}'")
        except Exception as e:
            self.get_logger().error(f"HDF5 write failed: {e}")
        finally:
            self._clear_buffers()
            self.dataset_path = None
            self.h5_path = None
            self.session_start_mono_ns = None

    # ----------------- HDF5 I/O -----------------
    def _write_h5(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare vlen dtype for float32 arrays
        vlen_f32 = h5py.special_dtype(vlen=np.dtype('float32'))

        with h5py.File(str(path), "w") as f:
            # Root attributes
            f.attrs["base_dir"] = str(self.dataset_path)
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
            write_vlen("left_angles", self.buf_l_angle)
            write_vlen("left_angle_forces", self.buf_l_angle_force)
            write_vlen("left_norm_forces", self.buf_l_norm_force)
            write_vlen("left_tang_forces", self.buf_l_tang_force)
            write_vlen("left_temps", self.buf_l_temp)
            write_vlen("left_currents", self.buf_l_cur)

            write_vlen("right_angles", self.buf_r_angle)
            write_vlen("right_angle_forces", self.buf_r_angle_force)
            write_vlen("right_norm_forces", self.buf_r_norm_force)
            write_vlen("right_tang_forces", self.buf_r_tang_force)
            write_vlen("right_temps", self.buf_r_temp)
            write_vlen("right_currents", self.buf_r_cur)
            
    def _clear_buffers(self):
        self.buf_now_mono_ns.clear()
        self.buf_tick.clear()
        self.buf_state_updated.clear()
        self.buf_state_timestamp.clear()

        self.buf_l_angle.clear()
        self.buf_l_angle_force.clear()
        self.buf_l_norm_force.clear()
        self.buf_l_tang_force.clear()
        self.buf_l_temp.clear()
        self.buf_l_cur.clear()

        self.buf_r_angle.clear()
        self.buf_r_angle_force.clear()
        self.buf_r_norm_force.clear()
        self.buf_r_tang_force.clear()
        self.buf_r_temp.clear()
        self.buf_r_cur.clear()

    # ----------------- Shutdown -----------------
    def destroy_node(self):
        if self.recording:
            self.get_logger().warn("Node shutting down while recording — finalizing HDF5.")
            self._stop_and_flush()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HandDataNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt — exit")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()