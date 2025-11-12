#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import signal
import subprocess
from typing import Optional, List
from pathlib import Path

import numpy as np
import h5py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, UInt64, String
from rby1_ros.qos_profiles import qos_ctrl_latched, qos_tick

from .cam_utils.shm_util import NamedSharedNDArray
from .cam_utils.digit_mp import DIGIT_MP
from .cam_utils.frame_saver_mp import FrameSaver_MP

TS_LEN = 3

class DIGITRecordNode(Node):
    def __init__(self):
        super().__init__("DIGIT_record_node")

        # ---------- Parameters ----------
        self.declare_parameter("shm_name", "digit_frame")
        self.declare_parameter("serial_number", "D20669")
        self.declare_parameter("stream", "QVGA")
        self.declare_parameter("intensity_rgb", [10, 10, 10])
        self.declare_parameter("fps", 30)
        self.declare_parameter("preprocess", True)

        self.declare_parameter("codec", "mp4v")

        # Read params
        self.camera_model = "DIGIT"
        self.shm_name = self.get_parameter("shm_name").get_parameter_value().string_value
        self.shm_name_meta = f"{self.shm_name}_meta"
        self.serial_number = self.get_parameter("serial_number").get_parameter_value().string_value
        self.stream_type = self.get_parameter("stream").get_parameter_value().string_value
        self.intensity_rgb = self.get_parameter("intensity_rgb").get_parameter_value().integer_array_value
        self.fps = int(self.get_parameter("fps").get_parameter_value().integer_value)
        self.preprocess = bool(self.get_parameter("preprocess").get_parameter_value().bool_value)

        self.width = 320 if self.stream_type == "QVGA" else 640
        self.height = 240 if self.stream_type == "QVGA" else 480

        self.codec = self.get_parameter("codec").get_parameter_value().string_value

        # ---------- ROS I/O ----------
        self.sub_record = self.create_subscription(Bool, "/record", self._on_record, qos_ctrl_latched)
        self.sub_tick = self.create_subscription(UInt64, "/tick", self._on_tick, qos_tick)
        self.sub_path = self.create_subscription(String, "/dataset_path", self._on_data_path, qos_ctrl_latched)

        # ---------- State ----------
        self.ts_shm: Optional[NamedSharedNDArray] = None
        self.ts_view: Optional[np.ndarray] = None

        self.recording = False
        self.session_start_mono_ns: Optional[int] = None
        self.prev_frame_index: Optional[int] = None
        self.dataset_path = None
        self.h5_path: Optional[str] = None

        # Buffers
        self.buf_now_mono_ns: List[int] = []
        self.buf_tick: List[int] = []
        self.buf_frame_index: List[int] = []
        self.buf_frame_index_mp4: List[int] = []
        self.buf_frame_updated: List[int] = []
        self.buf_get_timestamp_ms: List[float] = []
        self.buf_host_time_ns: List[float] = []

        self.cam = DIGIT_MP(
            shm=self.shm_name,
            stream_type=self.stream_type,
            intensity_rgb=self.intensity_rgb,
            fps=self.fps,
            serial_number=self.serial_number,
            preprocess=self.preprocess,
            verbose=True,
        )

        self.cam.start()

        time.sleep(5.0) # Wait for camera to initialize

        self.get_logger().info(
            f"[init] model='{self.camera_model}',"
            f"stream={self.stream_type} {self.width}x{self.height}@{self.fps}"
        )

    # ---------- Callbacks ----------
    def _on_record(self, msg: Bool):
        if msg.data and not self.recording:
            self.recording = True
        elif (not msg.data) and self.recording:
            self._stop_all_and_dump()

    def _on_data_path(self, msg: String):
        self.dataset_path = Path(msg.data)
        self.save_path = str(self.dataset_path / f"{self.shm_name}.mp4")
        self.h5_path = str(self.dataset_path / f"{self.shm_name_meta}.h5")
        self.get_logger().info(f"Received dataset path: {self.dataset_path}")
        self._start_all()

    def _on_tick(self, msg: UInt64):
        if not self.recording:
            return

        if self.dataset_path is None:
            self.get_logger().warn("Tick received but dataset_path is None, skipping tick.")
            return

        now_mono_ns = time.monotonic_ns()
        v = self._read_ts_vector()

        def getf(i, default=np.nan):
            try:
                x = float(v[i])
                if np.isnan(x):
                    return default
                return x
            except Exception:
                return default

        frame_number_f = getf(0, np.nan)
        frame_index = int(frame_number_f) if not np.isnan(frame_number_f) and frame_number_f >= 0 else -1
        frame_index_mp4 = self.frame_saver.saved_index()
        frame_updated = 1 if (self.prev_frame_index is None or frame_index_mp4 > self.prev_frame_index) and frame_index_mp4 >= 0 else 0

        # Append
        self.buf_now_mono_ns.append(int(now_mono_ns))
        self.buf_tick.append(int(msg.data))
        self.buf_frame_index.append(int(frame_index))
        self.buf_frame_index_mp4.append(int(frame_index_mp4))
        self.buf_frame_updated.append(int(frame_updated))

        self.buf_get_timestamp_ms.append(getf(1))
        self.buf_host_time_ns.append(getf(2))

        if frame_index >= 0:
            self.prev_frame_index = frame_index_mp4

    # ---------- Helpers ----------
    def _start_all(self):
        # Start frame saver
        self.frame_saver = FrameSaver_MP(
            shm_name=self.shm_name,
            save_path=self.save_path,
            codec=self.codec,
            fps=self.fps,
            ready_timeout=10.0,
            verbose=True,
        )
        
        # Open ts shared memory
        ts_name = self.shm_name_meta
        try:
            try:
                self.ts_shm = NamedSharedNDArray.open(ts_name)
                arr = self.ts_shm.as_array()
            except Exception:
                self.ts_shm = NamedSharedNDArray(name=ts_name, shape=(TS_LEN,), dtype=np.float64, create=False)
                arr = self.ts_shm.ndarray
            self.ts_view = arr
        except Exception as e:
            self.get_logger().error(f"Failed to open ts shm '{ts_name}': {e}")
            self.ts_view = None

        # Reset buffers
        self._reset_buffers()
        self.frame_saver.start()
        self.session_start_mono_ns = time.monotonic_ns()
        self.recording = True
        self.get_logger().info("[record] START — camera + saver launched")

    def _stop_all_and_dump(self):
        self.recording = False
        self.get_logger().info("[record] STOP — stopping camera and saver, dumping HDF5...")

        # Stop camera then saver
        self.cam.stop()
        self.frame_saver.stop()

        # Dump HDF5
        self._write_h5()

        # Cleanup
        self.ts_view = None
        try:
            if self.ts_shm is not None:
                self.ts_shm.close()
        except Exception:
            pass
        self.ts_shm = None
        self.prev_frame_index = None
        self.session_start_mono_ns = None
    
    def _write_h5(self):
        def arr_int64(lst): return np.asarray(lst, dtype=np.int64)
        def arr_uint64(lst): return np.asarray(lst, dtype=np.uint64)
        def arr_int32(lst): return np.asarray(lst, dtype=np.int32)
        def arr_uint8(lst): return np.asarray(lst, dtype=np.uint8)
        def arr_f64(lst): return np.asarray(lst, dtype=np.float64)

        os.makedirs(os.path.dirname(self.h5_path) or ".", exist_ok=True)
        with h5py.File(self.h5_path, "w") as f:
            f.attrs["camera_model"] = self.camera_model
            f.attrs["shm_name"] = self.shm_name
            f.attrs["serial_number"] = self.serial_number
            f.attrs["session_start_mono_ns"] = int(self.session_start_mono_ns) if self.session_start_mono_ns else -1
            f.attrs["created_wall_time_ns"] = int(time.time_ns())

            g = f.create_group("samples")
            comp = "gzip"; clevel = 4
            def dset(name, data):
                return g.create_dataset(name, data=data, compression=comp, compression_opts=clevel,
                                        shuffle=True, fletcher32=True, chunks=True)

            dset("now_mono_ns",          arr_int64(self.buf_now_mono_ns))
            dset("tick",                 arr_uint64(self.buf_tick))
            dset("frame_index",          arr_int32(self.buf_frame_index))
            dset("frame_index_mp4",     arr_int32(self.buf_frame_index_mp4))
            dset("frame_updated",   arr_uint8(self.buf_frame_updated))

            dset("digit_get_timestamp_ms",    arr_f64(self.buf_get_timestamp_ms))
            dset("digit_host_time_ns",        arr_f64(self.buf_host_time_ns))

        self.get_logger().info(f"[record] HDF5 written → '{self.h5_path}' ({len(self.buf_tick)} samples)")

        # Clear buffers
        self._reset_buffers()

    def _reset_buffers(self):
        self.buf_now_mono_ns.clear()
        self.buf_tick.clear()
        self.buf_frame_index.clear()
        self.buf_frame_index_mp4.clear()
        self.buf_frame_updated.clear()
        self.buf_get_timestamp_ms.clear()
        self.buf_host_time_ns.clear()

    def _read_ts_vector(self) -> np.ndarray:
        if self.ts_view is None:
            return np.full((TS_LEN,), np.nan, dtype=np.float64)
        try:
            return self.ts_view.copy()
        except Exception:
            return np.full((TS_LEN,), np.nan, dtype=np.float64)

    # ---------- Shutdown ----------
    def destroy_node(self):
        if self.recording:
            self.get_logger().warn("Shutting down while recording — stopping and saving.")
            self._stop_all_and_dump()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DIGITRecordNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt — exit")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()