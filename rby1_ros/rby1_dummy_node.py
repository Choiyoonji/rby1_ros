import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import time

# Message Imports
from std_msgs.msg import Float32MultiArray
from rby1_interfaces.msg import State, Command, EEpos, FTsensor
from rby1_ros.qos_profiles import qos_state_latest, qos_cmd, qos_ctrl_latched, qos_image_stream

# Define QoS profiles compatible with your main node
# qos_state_latest = QoSProfile(
#     reliability=ReliabilityPolicy.BEST_EFFORT,
#     history=HistoryPolicy.KEEP_LAST,
#     depth=1,
#     durability=DurabilityPolicy.VOLATILE
# )

# qos_cmd = QoSProfile(
#     reliability=ReliabilityPolicy.RELIABLE,
#     history=HistoryPolicy.KEEP_LAST,
#     depth=10,
#     durability=DurabilityPolicy.VOLATILE
# )

class DummyRBY1Node(Node):
    def __init__(self):
        super().__init__('dummy_rby1_node')
        self.get_logger().info("Dummy RBY1 Robot Node Started")

        # --- Publishers & Subscribers ---
        self.state_pub = self.create_publisher(State, '/rby1/state', qos_state_latest)
        self.cmd_sub = self.create_subscription(Command, '/control/command', self.command_callback, qos_cmd)

        # --- Internal Robot State Simulation ---
        self.hz = 30.0
        self.dt = 1.0 / self.hz
        
        # State flags
        self.is_initialized = False
        self.is_stopped = True
        
        # 22 DOF (Based on your impedance control code indices)
        # 0-8: Base/Torso?, 8-15: Right Arm, 15-22: Left Arm
        self.dof_count = 22
        
        # Current State
        self.curr_joints = np.zeros(self.dof_count)
        self.curr_ee_right_pos = np.array([0.5, -0.3, 0.5]) # Arbitrary starting pose
        self.curr_ee_right_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.curr_ee_left_pos = np.array([0.5, 0.3, 0.5])
        self.curr_ee_left_quat = np.array([0.0, 0.0, 0.0, 1.0])
        
        self.curr_grip_right = 0.0
        self.curr_grip_left = 0.0

        # Target State (What the main node commands)
        self.target_joints = self.curr_joints.copy()
        self.target_ee_right_pos = self.curr_ee_right_pos.copy()
        self.target_ee_right_quat = self.curr_ee_right_quat.copy()
        self.target_ee_left_pos = self.curr_ee_left_pos.copy()
        self.target_ee_left_quat = self.curr_ee_left_quat.copy()
        
        self.target_grip_right = 0.0
        self.target_grip_left = 0.0

        # Simulation Speed (0.1 = moves 10% of the way to target per tick)
        self.alpha = 0.1 

        # Timer
        self.timer = self.create_timer(self.dt, self.timer_callback)

    def command_callback(self, msg: Command):
        """Receive commands from main_node_command.py"""
        
        # Handle Flags
        if msg.ready:
            self.get_logger().info("CMD: Ready received. Initializing robot.")
            self.is_initialized = True
            self.is_stopped = False
            # Set a default "Ready" pose for arms (Indices 8-22)
            # Just setting some non-zero values so it looks alive
            self.target_joints[8:15] = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 0.0, 0.0]) # Right
            self.target_joints[15:22] = np.array([0.0, 0.5, 0.0, 1.5, 0.0, 0.0, 0.0])  # Left
            
        if msg.stop or msg.estop:
            self.get_logger().info("CMD: Stop received.")
            self.is_stopped = True

        if not msg.is_active or self.is_stopped:
            return

        # Handle Control Modes
        if msg.control_mode == 'joint_position':
            # msg.desired_joint_positions is Float32MultiArray
            if msg.desired_joint_positions.data:
                cmds = np.array(msg.desired_joint_positions.data)
                # Map commands based on main node logic:
                # Main node sends [Right(7) + Left(7)] = 14 values usually
                if len(cmds) == 14:
                    if msg.right_btn:
                        self.target_joints[8:15] = cmds[0:7]
                    if msg.left_btn:
                        self.target_joints[15:22] = cmds[7:14]
                elif len(cmds) == self.dof_count:
                    self.target_joints = cmds

        elif msg.control_mode == 'ee_position':
            if msg.right_btn:
                self.target_ee_right_pos = np.array(msg.desired_right_ee_pos.position.data)
                self.target_ee_right_quat = np.array(msg.desired_right_ee_pos.quaternion.data)
            if msg.left_btn:
                self.target_ee_left_pos = np.array(msg.desired_left_ee_pos.position.data)
                self.target_ee_left_quat = np.array(msg.desired_left_ee_pos.quaternion.data)

        # Handle Grippers
        self.target_grip_right = msg.desired_right_gripper_pos
        self.target_grip_left = msg.desired_left_gripper_pos

    def timer_callback(self):
        """Simulation Loop"""
        self.update_physics()
        self.publish_state()

    def update_physics(self):
        """Simple Low-Pass Filter to simulate movement inertia"""
        if self.is_stopped:
            return

        # Interpolate Joints
        self.curr_joints += (self.target_joints - self.curr_joints) * self.alpha
        
        # Interpolate EE Pos (Simple Linear)
        self.curr_ee_right_pos += (self.target_ee_right_pos - self.curr_ee_right_pos) * self.alpha
        self.curr_ee_left_pos += (self.target_ee_left_pos - self.curr_ee_left_pos) * self.alpha
        
        # Interpolate Grippers
        self.curr_grip_right += (self.target_grip_right - self.curr_grip_right) * self.alpha
        self.curr_grip_left += (self.target_grip_left - self.curr_grip_left) * self.alpha
        
        # Note: We are NOT doing Inverse Kinematics here. 
        # If the main node sends EE commands, the EE state updates, but Joints stay still.
        # If the main node sends Joint commands, Joint state updates, but EE stays still.
        # This is usually sufficient for testing the *logic* of the main node.

    def publish_state(self):
        msg = State()
        msg.timestamp = time.time()
        msg.is_initialized = self.is_initialized
        msg.is_stopped = self.is_stopped

        # Joints
        msg.joint_positions = Float32MultiArray(data=self.curr_joints.tolist())
        
        # EE Pose Right
        msg.right_ee_pos = EEpos()
        msg.right_ee_pos.position = Float32MultiArray(data=self.curr_ee_right_pos.tolist())
        msg.right_ee_pos.quaternion = Float32MultiArray(data=self.curr_ee_right_quat.tolist()) # We don't interpolate quats properly here for simplicity
        
        # EE Pose Left
        msg.left_ee_pos = EEpos()
        msg.left_ee_pos.position = Float32MultiArray(data=self.curr_ee_left_pos.tolist())
        msg.left_ee_pos.quaternion = Float32MultiArray(data=self.curr_ee_left_quat.tolist())

        # EE Pose Torso (Static for now)
        msg.torso_ee_pos = EEpos()
        msg.torso_ee_pos.position = Float32MultiArray(data=[0.0, 0.0, 1.0])
        msg.torso_ee_pos.quaternion = Float32MultiArray(data=[0.0, 0.0, 0.0, 1.0])

        # Grippers
        msg.right_gripper_pos = float(self.curr_grip_right)
        msg.left_gripper_pos = float(self.curr_grip_left)

        # FT Sensors (Zeros)
        zero_wrench = Float32MultiArray(data=[0.0, 0.0, 0.0])
        msg.right_ft_sensor = FTsensor(force=zero_wrench, torque=zero_wrench)
        msg.left_ft_sensor = FTsensor(force=zero_wrench, torque=zero_wrench)

        self.state_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DummyRBY1Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()