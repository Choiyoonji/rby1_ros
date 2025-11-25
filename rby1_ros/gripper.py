import rby1_sdk as rby
import numpy as np
import time
import threading


class Gripper:
    """
    Class for gripper
    """

    GRIPPER_DIRECTION = False

    def __init__(self):
        self.bus = rby.DynamixelBus(rby.upc.GripperDeviceName)
        self.bus.open_port()
        self.bus.set_baud_rate(2_000_000)
        self.bus.set_torque_constant([13, 13])
        self.min_q = np.array([np.inf, np.inf])
        self.max_q = np.array([-np.inf, -np.inf])
        self.target_q: np.typing.NDArray = None
        self.current_q: np.typing.NDArray = None
        self._running = False
        self._thread = None

    def initialize(self, verbose=False):
        print("[Gripper] Initializing gripper...")
        rv = True
        for dev_id in [0, 1]:
            if not self.bus.ping(dev_id):
                if verbose:
                    print(f"[Gripper] Dynamixel ID {dev_id} is not active")
                rv = False
            else:
                if verbose:
                    print(f"[Gripper] Dynamixel ID {dev_id} is active")
        if rv:
            print("[Gripper] Servo on gripper")
            self.bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in [0, 1]])
        return rv

    def set_operating_mode(self, mode):
        self.bus.group_sync_write_torque_enable([(dev_id, 0) for dev_id in [0, 1]])
        self.bus.group_sync_write_operating_mode([(dev_id, mode) for dev_id in [0, 1]])
        self.bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in [0, 1]])

    def homing(self):
        self.set_operating_mode(rby.DynamixelBus.CurrentControlMode)
        direction = 0
        q = np.array([0, 0], dtype=np.float64)
        prev_q = np.array([0, 0], dtype=np.float64)
        counter = 0
        while direction < 2:
            self.bus.group_sync_write_send_torque(
                [(dev_id, 5 * (1 if direction == 0 else -1)) for dev_id in [0, 1]]
            )
            rv = self.bus.group_fast_sync_read_encoder([0, 1])
            if rv is not None:
                for dev_id, enc in rv:
                    q[dev_id] = enc
            self.min_q = np.minimum(self.min_q, q)
            self.max_q = np.maximum(self.max_q, q)
            if np.array_equal(prev_q, q):
                counter += 1
            prev_q = q
            # A small value (e.g., 5) was too short and failed to detect limits properly, so a reasonably larger value was chosen.
            if counter >= 30:
                direction += 1
                counter = 0
            time.sleep(0.1)
        return True

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._running = True
            self._thread = threading.Thread(target=self.loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def loop(self):
        self.set_operating_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
        self.bus.group_sync_write_send_torque([(dev_id, 5) for dev_id in [0, 1]])
        while self._running:
            if self.target_q is not None:
                self.current_q = self.bus.group_fast_sync_read_encoder([0, 1])
                self.bus.group_sync_write_send_position(
                    [(dev_id, q) for dev_id, q in enumerate(self.target_q.tolist())]
                )
                target_torque = []
                for dev_id, enc in self.current_q:
                    if enc < self.target_q[dev_id]:
                        target_torque.append(0.3)
                    else:
                        target_torque.append(1.0)
                        
                self.bus.group_sync_write_send_torque([(dev_id, tq) for dev_id, tq in enumerate(target_torque)])
                self.bus.group_sync_write_send_position(
                    [(dev_id, q) for dev_id, q in enumerate(self.target_q.tolist())]
                )
                
            time.sleep(0.01)

    def get_target(self):
        return self.target_q
    
    def get_current(self):
        if self.current_q is None:
            return None
        current_positions = np.zeros(2)
        for dev_id, enc in self.current_q:
            current_positions[dev_id] = (enc - self.min_q[dev_id]) / (self.max_q[dev_id] - self.min_q[dev_id])
        return current_positions
    
    def get_normalized_target(self):
        return (self.target_q - self.min_q)/(self.max_q - self.min_q)

    def set_normalized_target(self, normalized_q):
        # self.target_q = normalized_q * (self.max_q - self.min_q) + self.min_q
        if not np.isfinite(self.min_q).all() or not np.isfinite(self.max_q).all():
            print("[Gripper] Cannot set target. min_q or max_q is not valid.")
            return

        if Gripper.GRIPPER_DIRECTION:
            self.target_q = normalized_q * (self.max_q - self.min_q) + self.min_q
        else:
            self.target_q = (1 - normalized_q) * (self.max_q - self.min_q) + self.min_q