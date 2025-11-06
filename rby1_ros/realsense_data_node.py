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

from cam_utils.shm_util import NamedSharedNDArray
from cam_utils.realsense_mp_full_ts import Realsense_MP
from cam_utils.frame_saver_mp import FrameSaver_MP

TS_LEN = 7

class RealsenseRecordNode(Node):
    def __init__(self):
        super().__init__("realsense_record_node")

        # ---------- Parameters ----------
        self.declare_parameter("shm_name", "camera_frame")
        self.declare_parameter("camera_model", "D405")
        self.declare_parameter("serial_number", None)

        self.declare_parameter("stream", "color")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)

        self.declare_parameter("codec", "mp4v")

        # Read params
        self.shm_name = self.get_parameter("shm_name").get_parameter_value().string_value
        self.camera_model = self.get_parameter("camera_model").get_parameter_value().string_value
        self.serial_number = self.get_parameter("serial_number").get_parameter_value().string_value

        self.stream = self.get_parameter("stream").get_parameter_value().string_value
        self.width = int(self.get_parameter("width").get_parameter_value().integer_value)
        self.height = int(self.get_parameter("height").get_parameter_value().integer_value)
        self.fps = int(self.get_parameter("fps").get_parameter_value().integer_value)

        self.codec = self.get_parameter("codec").get_parameter_value().string_value

        # ---------- ROS I/O ----------
        self.sub_record = self.create_subscription(Bool, "/record", self._on_record, 10)
        self.sub_tick = self.create_subscription(UInt64, "/tick", self._on_tick, 200)
        self.sub_path = self.create_subscription(String, "/dataset_path", self._on_data_path, 10)

        # ---------- State ----------
        self.proc_cam: Optional[subprocess.Popen] = None
        self.proc_saver: Optional[subprocess.Popen] = None
        self.ts_shm: Optional[NamedSharedNDArray] = None
        self.ts_view: Optional[np.ndarray] = None

        self.recording = False
        self.session_start_mono_ns: Optional[int] = None
        self.prev_frame_index: Optional[int] = None
        self.dataset_path = None

        # Buffers
        self.buf_now_mono_ns: List[int] = []
        self.buf_tick: List[int] = []
        self.buf_frame_index: List[int] = []
        self.buf_frame_index_mp4: List[int] = []
        self.buf_frame_updated: List[int] = []
        self.buf_get_timestamp_ms: List[float] = []
        self.buf_frame_timestamp_ms: List[float] = []
        self.buf_backend_timestamp_ms: List[float] = []
        self.buf_time_of_arrival_ms: List[float] = []
        self.buf_timestamp_domain: List[float] = []
        self.buf_host_time_ns: List[float] = []

        self.cam = Realsense_MP(
            shm_name=self.shm_name,
            device_name=self.camera_model,
            width=self.width,
            height=self.height,
            fps=self.fps,
            serial_number=self.serial_number,
        )

        self.cam.start()

        time.sleep(2.0) # Wait for camera to initialize

        self.get_logger().info(
            f"[init] shm='{self.shm_name}', model='{self.camera_model}', serial='{self.serial_number}', "
            f"stream={self.stream} {self.width}x{self.height}@{self.fps}, save='{self.save_path}', h5='{self.h5_path}'"
        )

    # ---------- Callbacks ----------
    def _on_record(self, msg: Bool):
        if msg.data and not self.recording:
            while self.dataset_dir is None:
                self.get_logger().warn("Record command received but dataset_dir is None, waiting...")
                time.sleep(0.0001)
            self._start_all()
        elif (not msg.data) and self.recording:
            self._stop_all_and_dump()

    def _on_data_path(self, msg: String):
        self.dataset_dir = Path(msg.data)
        self.save_path = str(self.dataset_dir / f"rs_{self.camera_model}_{self.serial_number}.mp4")
        self.h5_path = str(self.dataset_dir / f"rs_{self.camera_model}_{self.serial_number}.h5")
        self.get_logger().info(f"Received dataset path: {self.dataset_dir}")

    def _on_tick(self, msg: UInt64):
        if not self.recording:
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
        frame_updated = 1 if (self.prev_frame_index is None or frame_index > self.prev_frame_index) and frame_index >= 0 else 0

        # Append
        self.buf_now_mono_ns.append(int(now_mono_ns))
        self.buf_tick.append(int(msg.data))
        self.buf_frame_index.append(int(frame_index))
        self.buf_frame_index_mp4.append(int(self.frame_saver.saved_index()))
        self.buf_frame_updated.append(int(frame_updated))

        self.buf_get_timestamp_ms.append(getf(1))
        self.buf_frame_timestamp_ms.append(getf(2))
        self.buf_backend_timestamp_ms.append(getf(3))
        self.buf_time_of_arrival_ms.append(getf(4))
        self.buf_timestamp_domain.append(getf(5))
        self.buf_host_time_ns.append(getf(6))

        if frame_index >= 0:
            self.prev_frame_index = frame_index

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
        ts_name = f"{self.shm_name}_ts"
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
            f.attrs["serial_number"] = self.serial_number
            f.attrs["shm_name"] = self.shm_name
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
            dset("frame_updated",        arr_uint8(self.buf_frame_updated))

            dset("rs_get_timestamp_ms",    arr_f64(self.buf_get_timestamp_ms))
            dset("rs_frame_timestamp_ms",  arr_f64(self.buf_frame_timestamp_ms))
            dset("rs_backend_timestamp_ms",arr_f64(self.buf_backend_timestamp_ms))
            dset("rs_time_of_arrival_ms",  arr_f64(self.buf_time_of_arrival_ms))
            dset("rs_timestamp_domain",    arr_f64(self.buf_timestamp_domain))
            dset("rs_host_time_ns",        arr_f64(self.buf_host_time_ns))

        self.get_logger().info(f"[record] HDF5 written → '{self.h5_path}' ({len(self.buf_tick)} samples)")

        # Clear buffers
        self._reset_buffers()

    def _reset_buffers(self):
        self.buf_now_mono_ns.clear()
        self.buf_tick.clear()
        self.buf_frame_index.clear()
        self.buf_frame_updated.clear()
        self.buf_get_timestamp_ms.clear()
        self.buf_frame_timestamp_ms.clear()
        self.buf_backend_timestamp_ms.clear()
        self.buf_time_of_arrival_ms.clear()
        self.buf_timestamp_domain.clear()
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
    node = RealsenseRecordNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt — exit")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()