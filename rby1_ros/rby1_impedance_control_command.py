import os
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rby1_interfaces.msg import State, Command, EEpos, FTsensor
from rby1_ros.qos_profiles import qos_state_latest, qos_cmd

import numpy as np
from rby1_ros.utils import *
from dataclasses import dataclass, field
from typing import Union
import logging
import rby1_sdk as rby
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from rby1_ros.rby1_status import RBY1Status as RBY1State
from rby1_ros.control_status import ControlStatus as ControlState
from rby1_ros.gripper import Gripper

logging.basicConfig(level=logging.INFO)

global rby1_node, first_time

rby1_node = None
first_time = None


@dataclass(frozen=True)
class Settings:
    dt: float = 1/30.0  # 30 Hz
    initial_dt: float = 1.0  # 1 Hz
    update_dt : float = 0.01   # 100 Hz
    hand_offset: np.ndarray = np.array([0.0, 0.0, 0.0])

    no_gripper : bool = True
    no_head : bool = True

    T_hand_offset = np.identity(4)
    T_hand_offset[0:3, 3] = hand_offset

    rby1_ip = "192.168.0.83"
    # rby1_ip = "192.168.30.1"
    port = 50051
    model = "a"

    damping_ratio: float = 1.0

    mobile_linear_acceleration_gain: float = 0.15
    mobile_angular_acceleration_gain: float = 0.15
    mobile_linear_damping_gain: float = 0.3
    mobile_angular_damping_gain: float = 0.3

    right_arm_start_position = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 0.0])


class SystemContext:
    robot_model: Union[rby.Model_A, rby.Model_M] = None
    rby1_state = RBY1State()
    control_state = ControlState()



def robot_state_callback(robot_state: rby.RobotState_A):
    global rby1_node, first_time
    SystemContext.rby1_state.timestamp = robot_state.timestamp.timestamp()
    # SystemContext.rby1_state.timestamp = ((time.time() * 1000.0) - first_time)
    # print(f"CPU Timestamp: {(time.time() * 1000.0):.2f} ms")
    # print(f"Robot State Timestamp: {SystemContext.rby1_state.timestamp:.2f} ms")

    SystemContext.rby1_state.joint_positions = robot_state.position
    SystemContext.rby1_state.joint_velocities = robot_state.velocity
    SystemContext.rby1_state.joint_currents = robot_state.current
    SystemContext.rby1_state.joint_torques = robot_state.torque

    SystemContext.rby1_state.right_force_sensor = robot_state.ft_sensor_right.force
    SystemContext.rby1_state.right_torque_sensor = robot_state.ft_sensor_right.torque

    SystemContext.rby1_state.left_force_sensor = robot_state.ft_sensor_left.force
    SystemContext.rby1_state.left_torque_sensor = robot_state.ft_sensor_left.torque

    SystemContext.rby1_state.center_of_mass = robot_state.center_of_mass

    if rby1_node is not None:
        rby1_node.publish_state()


class RBY1Node(Node):
    def __init__(self):
        super().__init__("rby1_node_command")
        self.get_logger().info("RBY1 Impedance Control Node Initialized")

        self.rby1_pub = self.create_publisher(
            State,
            '/rby1/state',
            qos_state_latest
        )

        self.control_sub = self.create_subscription(
            Command,
            '/control/command',
            self.control_callback,
            qos_cmd
        )

        self.gripper = None
        self.settings = Settings()

        self.connect_rby1()

        self.stream = None
        self.torso_reset = False
        self.right_reset = False
        self.left_reset = False
        self.reset_done = False 
        
        self.link_idx = {
            "base": 0,
            "link_torso_5": 1,
            "link_right_arm_6": 2,
            "link_left_arm_6": 3,
        }

        self.control_timer = self.create_timer(self.settings.dt, self.run)

    def calc_ee_pose(self):
        self.dyn_state.set_q(SystemContext.rby1_state.joint_positions.copy())
        self.dyn_robot.compute_forward_kinematics(self.dyn_state)

        SystemContext.rby1_state.right_ee_position = self.dyn_robot.compute_transformation(
            self.dyn_state,
            self.link_idx["base"],
            self.link_idx["link_right_arm_6"]
        )

        SystemContext.rby1_state.left_ee_position = self.dyn_robot.compute_transformation(
            self.dyn_state,
            self.link_idx["base"],
            self.link_idx["link_left_arm_6"]
        )
        
        SystemContext.rby1_state.torso_ee_position = self.dyn_robot.compute_transformation(
            self.dyn_state,
            self.link_idx["base"],
            self.link_idx["link_torso_5"]
        )

    def publish_state(self):
        if not SystemContext.rby1_state.is_robot_connected:
            self.get_logger().warning("Robot not connected yet. Cannot publish state.")
            return
        # self.get_logger().warning("Publishing robot state.")
        
        msg = State()
        msg.timestamp = SystemContext.rby1_state.timestamp

        msg.joint_positions = Float32MultiArray(data=SystemContext.rby1_state.joint_positions.tolist())
        msg.joint_velocities = Float32MultiArray(data=SystemContext.rby1_state.joint_velocities.tolist())
        msg.joint_currents = Float32MultiArray(data=SystemContext.rby1_state.joint_currents.tolist())
        msg.joint_torques = Float32MultiArray(data=SystemContext.rby1_state.joint_torques.tolist())

        self.calc_ee_pose()

        right_ee_pos, right_ee_quat = se3_to_pos_quat(SystemContext.rby1_state.right_ee_position)
        left_ee_pos, left_ee_quat = se3_to_pos_quat(SystemContext.rby1_state.left_ee_position)
        torso_ee_pos, torso_ee_quat = se3_to_pos_quat(SystemContext.rby1_state.torso_ee_position)

        msg.right_ee_pos = EEpos()
        msg.right_ee_pos.position = Float32MultiArray(data=right_ee_pos.tolist())
        msg.right_ee_pos.quaternion = Float32MultiArray(data=right_ee_quat.tolist())

        msg.left_ee_pos = EEpos()
        msg.left_ee_pos.position = Float32MultiArray(data=left_ee_pos.tolist())
        msg.left_ee_pos.quaternion = Float32MultiArray(data=left_ee_quat.tolist())

        msg.torso_ee_pos = EEpos()
        msg.torso_ee_pos.position = Float32MultiArray(data=torso_ee_pos.tolist())
        msg.torso_ee_pos.quaternion = Float32MultiArray(data=torso_ee_quat.tolist())

        msg.right_ft_sensor = FTsensor()
        msg.right_ft_sensor.force = Float32MultiArray(data=SystemContext.rby1_state.right_force_sensor.tolist())
        msg.right_ft_sensor.torque = Float32MultiArray(data=SystemContext.rby1_state.right_torque_sensor.tolist())

        msg.left_ft_sensor = FTsensor()
        msg.left_ft_sensor.force = Float32MultiArray(data=SystemContext.rby1_state.left_force_sensor.tolist())
        msg.left_ft_sensor.torque = Float32MultiArray(data=SystemContext.rby1_state.left_torque_sensor.tolist())

        msg.center_of_mass = Float32MultiArray(data=SystemContext.rby1_state.center_of_mass.tolist())

        msg.is_initialized = SystemContext.rby1_state.is_initialized
        msg.is_stopped = SystemContext.rby1_state.is_stopped
        msg.is_torso_following = SystemContext.rby1_state.is_torso_following
        msg.is_right_following = SystemContext.rby1_state.is_right_following
        msg.is_left_following = SystemContext.rby1_state.is_left_following

        if self.gripper is not None:
            gripper_position = self.gripper.get_current()
            if gripper_position is not None:
                SystemContext.rby1_state.right_gripper_position = gripper_position[0]
                SystemContext.rby1_state.left_gripper_position = gripper_position[1]
                msg.right_gripper_pos = SystemContext.rby1_state.right_gripper_position
                msg.left_gripper_pos = SystemContext.rby1_state.left_gripper_position
                
        self.rby1_pub.publish(msg)

    def control_callback(self, msg: Command):
        # self.get_logger().info(f"Received control command:")
        SystemContext.control_state.is_controller_connected = True
        SystemContext.control_state.is_active = msg.is_active
        # self.get_logger().info(f"  is_controller_connected: {msg.is_active}")

        # toggle
        SystemContext.control_state.move = msg.move

        # pulse
        if msg.ready:
            self.get_logger().info("  Ready command received.")
            SystemContext.control_state.ready = True
        if msg.stop:
            self.get_logger().info("  Stop command received.")
            SystemContext.control_state.stop = True
        if msg.estop:
            self.get_logger().info("  E-Stop command received.")
            SystemContext.control_state.estop = True

        if msg.move and msg.is_active:
            SystemContext.control_state.control_mode = msg.control_mode

            SystemContext.control_state.is_button_right_pressed = msg.right_btn
            SystemContext.control_state.is_button_left_pressed = msg.left_btn

            if SystemContext.control_state.control_mode == 'joint_position':
                self.get_logger().warning("This node is not handling joint position control.")
                return
                if msg.desired_joint_positions.data is None or len(msg.desired_joint_positions.data) != 14:
                    self.get_logger().warning("Invalid desired_joint_positions data received.")
                    return
                if SystemContext.control_state.is_button_right_pressed:
                    SystemContext.control_state.desired_joint_positions[:7] = np.array(msg.desired_joint_positions.data)[:7]
                if SystemContext.control_state.is_button_left_pressed:
                    SystemContext.control_state.desired_joint_positions[7:14] = np.array(msg.desired_joint_positions.data)[7:14]
                
            elif SystemContext.control_state.control_mode == 'ee_position':
                desired_right_pos = np.array(msg.desired_right_ee_pos.position.data)
                desired_right_quat = np.array(msg.desired_right_ee_pos.quaternion.data)
                desired_left_pos = np.array(msg.desired_left_ee_pos.position.data)
                desired_left_quat = np.array(msg.desired_left_ee_pos.quaternion.data)

                if SystemContext.control_state.is_button_right_pressed:
                    SystemContext.control_state.desired_right_ee_position["position"] = desired_right_pos
                    SystemContext.control_state.desired_right_ee_position["quaternion"] = desired_right_quat
                if SystemContext.control_state.is_button_left_pressed:
                    SystemContext.control_state.desired_left_ee_position["position"] = desired_left_pos
                    SystemContext.control_state.desired_left_ee_position["quaternion"] = desired_left_quat
                    
            if self.gripper is not None:
                SystemContext.control_state.desired_right_gripper_position = msg.desired_right_gripper_pos
                SystemContext.control_state.desired_left_gripper_position = msg.desired_left_gripper_pos
                

    def set_limits(self):
        SystemContext.rby1_state.q_limits_upper = self.dyn_robot.get_limit_q_upper(self.dyn_state)[8:22]
        SystemContext.rby1_state.q_limits_lower = self.dyn_robot.get_limit_q_lower(self.dyn_state)[8:22]
        SystemContext.rby1_state.qdot_limits_upper = self.dyn_robot.get_limit_qdot_upper(self.dyn_state)[8:22] * 10
        SystemContext.rby1_state.qddot_limits_upper = self.dyn_robot.get_limit_qddot_upper(self.dyn_state)[8:22]
        
        print(len(SystemContext.rby1_state.q_limits_lower))

    def is_connected(self):
        return self.robot is not None

    def connect_rby1(self):
        global first_time
        first_time = time.time() * 1000.0
        address = f"{self.settings.rby1_ip}:{self.settings.port}"
        model = self.settings.model

        logging.info(f"Attempting to connect to RB-Y1... (Address: {address}, Model: {model})")

        self.robot = rby.create_robot(address, model)
        connected = self.robot.connect()
        if not connected:
            logging.critical("Failed to connect to RB-Y1. Exiting program.")
            exit(1)
        logging.info("Successfully connected to RB-Y1.")

        servo_pattern = "^(?!head_).*" if self.settings.no_head else ".*"
        if not self.robot.is_power_on(servo_pattern):
            logging.warning("Robot power is off. Turning it on...")
            if not self.robot.power_on(servo_pattern):
                logging.critical("Failed to power on. Exiting program.")
                exit(1)
            logging.info("Power turned on successfully.")
        else:
            logging.info("Power is already on.")

        if not self.robot.is_servo_on(servo_pattern):
            logging.warning("Servo is off. Turning it on...")
            if not self.robot.servo_on(servo_pattern):
                logging.critical("Failed to turn on the servo. Exiting program.")
                exit(1)
            logging.info("Servo turned on successfully.")
        else:
            logging.info("Servo is already on.")

        cm_state = self.robot.get_control_manager_state().state
        if cm_state in [
            rby.ControlManagerState.State.MajorFault,
            rby.ControlManagerState.State.MinorFault,
        ]:
            logging.warning(f"Control Manager is in Fault state: {cm_state.name}. Attempting reset...")
            if not self.robot.reset_fault_control_manager():
                logging.critical("Failed to reset Control Manager. Exiting program.")
                exit(1)
            logging.info("Control Manager reset successfully.")
        if not self.robot.enable_control_manager(unlimited_mode_enabled=True):
            logging.critical("Failed to enable Control Manager. Exiting program.")
            exit(1)
        logging.info("Control Manager successfully enabled. (Unlimited Mode: enabled)")

        SystemContext.robot_model = self.robot.model()
        self.robot.start_state_update(robot_state_callback, 1 / self.settings.update_dt)

        self.dyn_robot = self.robot.get_dynamics()
        self.dyn_state = self.dyn_robot.make_state(["base", "link_torso_5", "link_right_arm_6", "link_left_arm_6"],
                                     SystemContext.robot_model.robot_joint_names)
        
        self.set_limits()

        if not self.settings.no_gripper:
            for arm in ["right", "left"]:
                if not self.robot.set_tool_flange_output_voltage(arm, 12):
                    logging.error(f"Failed to supply 12V to tool flange. ({arm})")
            time.sleep(0.5)
            self.gripper = Gripper()
            if not self.gripper.initialize(verbose=True):
                logging.critical("Failed to initialize gripper. Exiting program.")
                exit(1)
            self.gripper.homing()
            self.gripper.start()
            self.gripper.set_normalized_target(np.array([1.0, 1.0]))

        SystemContext.rby1_state.is_robot_connected = True
        
    def ready(self):
        if self.robot.get_control_manager_state().control_state != rby.ControlManagerState.ControlState.Idle:
            self.robot.cancel_control()
            
        if self.robot.wait_for_control_ready(1000):
            ready_pose = np.deg2rad(
                [0.0, 40.0, -80.0, 40.0, 0.0, 0.0] +
                [0.0, -15.0, 0.0, -120.0, 0.0, 30.0, 75.0] +
                [0.0, 15.0, 0.0, -120.0, 0.0, 30.0, -75.0])
            
            cbc = (
                rby.BodyComponentBasedCommandBuilder()
                .set_torso_command(
                    rby.JointImpedanceControlCommandBuilder()
                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1))
                    .set_position(ready_pose[0:6])
                    .set_stiffness([400.] * 6)
                    .set_torque_limit([500] * 6)
                    .set_minimum_time(2)
                )
                .set_right_arm_command(
                    rby.JointImpedanceControlCommandBuilder()
                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1))
                    .set_position(ready_pose[6:13])
                    .set_stiffness([60] * 7)
                    .set_torque_limit([30] * 7)
                    .set_minimum_time(2)
                )
                .set_left_arm_command(
                    rby.JointImpedanceControlCommandBuilder()
                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1))
                    .set_position(ready_pose[13:20])
                    .set_stiffness([60] * 7)
                    .set_torque_limit([30] * 7)
                    .set_minimum_time(2)
                )
            )
            if not self.settings.no_head:
                cbc.set_head_command(
                    rby.JointPositionCommandBuilder()
                    .set_position([0.] * len(SystemContext.robot_model.head_idx))
                    .set_minimum_time(2)
                )
            self.robot.send_command(
                rby.RobotCommandBuilder().set_command(
                    rby.ComponentBasedCommandBuilder()
                        .set_body_command(
                            cbc
                    )
                )
            ).get()

        SystemContext.rby1_state.is_initialized = True
        SystemContext.rby1_state.is_stopped = False

        SystemContext.control_state.ready = False
        
        self.get_logger().warning("Robot is ready.")

    def stop(self):
        logging.info("Stopping the robot...")
        SystemContext.rby1_state.is_stopped = True
        SystemContext.control_state.stop = False
        SystemContext.rby1_state.dt = self.settings.initial_dt

    def estop(self):
        logging.critical("Emergency Stop triggered! Shutting down the robot immediately.")
        try:
            servo_pattern = "^(?!head_).*" if self.settings.no_head else ".*"
            self.robot.power_off(servo_pattern)
        finally:
            rclpy.shutdown()
            sys.exit(1)  # or sys.exit(1)

    def handle_signals(self):
        if SystemContext.control_state.ready:
            self.ready()
        elif SystemContext.control_state.stop:
            self.stop()
        elif SystemContext.control_state.estop:
            self.estop()
        else:
            return False
        return True

    def run(self):
        if not SystemContext.rby1_state.is_robot_connected:
            self.get_logger().warning("Robot not connected yet.")
            return
        
        if SystemContext.rby1_state.joint_positions.size == 0:
            self.get_logger().warning("Waiting for robot state...")
            return
        
        if self.handle_signals():
            if self.stream is not None:
                self.get_logger().info("Stream cancelled due to signal handling.")
                self.stream.cancel()
                self.stream = None

        if SystemContext.rby1_state.is_stopped:
            if self.stream is not None:
                self.get_logger().info("Stream cancelled due to signal handling.")
                self.stream.cancel()
                self.stream = None
            SystemContext.rby1_state.is_initialized = False
            return

        # if not SystemContext.control_state.is_active:
        #     return

        if self.stream is None:
            if self.robot.wait_for_control_ready(0):
                self.get_logger().info("Starting impedance control stream...")
                self.stream = self.robot.create_command_stream()

                SystemContext.rby1_state.is_right_following = False
                SystemContext.rby1_state.is_left_following = False
                SystemContext.rby1_state.is_torso_following = False
                SystemContext.rby1_state.torso_locked_pose = SystemContext.rby1_state.torso_ee_position.copy()
                SystemContext.rby1_state.right_arm_locked_pose = SystemContext.rby1_state.right_ee_position.copy()
                SystemContext.rby1_state.left_arm_locked_pose = SystemContext.rby1_state.left_ee_position.copy()

        if SystemContext.control_state.is_controller_connected:
            if SystemContext.control_state.move and SystemContext.control_state.is_active:
                SystemContext.rby1_state.is_right_following = SystemContext.control_state.is_button_right_pressed
                SystemContext.rby1_state.is_left_following = SystemContext.control_state.is_button_left_pressed
                SystemContext.rby1_state.is_torso_following = False
                if self.reset_done == False:
                    self.right_reset = True
                    self.left_reset = True
                    self.torso_reset = True
                    self.reset_done = True
            else:
                SystemContext.rby1_state.is_right_following = False
                SystemContext.rby1_state.is_left_following = False
                SystemContext.rby1_state.is_torso_following = False
                self.reset_done = False
        else:
            SystemContext.rby1_state.is_right_following = False
            SystemContext.rby1_state.is_left_following = False
            SystemContext.rby1_state.is_torso_following = False
            self.reset_done = False
            
        if self.stream:
            try:
                if self.gripper is not None:
                    gripper_target = np.array([
                        SystemContext.control_state.desired_right_gripper_position,
                        SystemContext.control_state.desired_left_gripper_position,
                    ])
                    self.gripper.set_normalized_target(gripper_target)

                if SystemContext.rby1_state.is_right_following:
                    SystemContext.control_state.desired_right_ee_T = pos_to_se3(
                        SystemContext.control_state.desired_right_ee_position["position"],
                        SystemContext.control_state.desired_right_ee_position["quaternion"]
                    )
                    right_T = SystemContext.control_state.desired_right_ee_T
                    SystemContext.rby1_state.right_arm_locked_pose = right_T.copy()
                else:
                    right_T = SystemContext.rby1_state.right_arm_locked_pose
                
                # if SystemContext.rby1_state.is_left_following:
                #     SystemContext.control_state.desired_left_ee_T = pos_to_se3(
                #         SystemContext.control_state.desired_left_ee_position["position"],
                #         SystemContext.control_state.desired_left_ee_position["quaternion"]
                #     )
                #     left_T = SystemContext.control_state.desired_left_ee_T
                #     SystemContext.rby1_state.left_arm_locked_pose = left_T.copy()
                # else:
                left_T = SystemContext.rby1_state.left_arm_locked_pose
                
                if SystemContext.rby1_state.is_torso_following:
                    SystemContext.control_state.desired_torso_ee_T = pos_to_se3(
                        SystemContext.control_state.desired_torso_ee_position["position"],
                        SystemContext.control_state.desired_torso_ee_position["quaternion"]
                    )
                    torso_T = SystemContext.control_state.desired_torso_ee_T
                    SystemContext.rby1_state.torso_locked_pose = torso_T.copy()
                else:
                    torso_T = SystemContext.rby1_state.torso_locked_pose

                
                self.get_logger().info("component control mode")
                torso_builder = (
                    rby.CartesianImpedanceControlCommandBuilder()
                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(self.settings.dt * 10))
                    .set_minimum_time(Settings.dt * 1.02)
                    .set_joint_stiffness([500.] * 6)
                    .set_joint_torque_limit([600] * 6)
                    .add_joint_limit("torso_1", -0.523598776, 1.3)
                    .add_joint_limit("torso_2", -2.617993878, -0.2)
                    .set_stop_joint_position_tracking_error(0)
                    .set_stop_orientation_tracking_error(0)
                    .set_stop_joint_position_tracking_error(0)
                    .set_joint_damping_ratio(0.7)
                    .set_reset_reference(self.torso_reset)
                )
                self.get_logger().info("Torso builder created.")
                right_builder = (
                    rby.CartesianImpedanceControlCommandBuilder()
                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(self.settings.dt * 10))
                    .set_minimum_time(Settings.dt * 1.02)
                    .set_joint_stiffness([80, 80, 80, 60, 60, 60, 60])
                    .set_joint_torque_limit([40, 40, 40, 30, 30, 30, 30])
                    .add_joint_limit("right_arm_3", -2.6, -0.5)
                    # .add_joint_limit("right_arm_5", 0.2, 1.9)
                    .set_nullspace_joint_target(np.deg2rad([0.0, -15.0, 0.0, -120.0, 0.0, 40.0, -15.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]) * 2, 0.2, 0.3)
                    .set_stop_joint_position_tracking_error(0)
                    .set_stop_orientation_tracking_error(0)
                    .set_stop_joint_position_tracking_error(0)
                    .set_joint_damping_ratio(0.85)
                    .set_reset_reference(self.right_reset)
                )
                self.get_logger().info("Right arm builder created.")
                left_builder = (
                    rby.CartesianImpedanceControlCommandBuilder()
                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(self.settings.dt * 10))
                    .set_minimum_time(Settings.dt * 1.02)
                    .set_joint_stiffness([80, 80, 80, 60, 60, 60, 60])
                    .set_joint_torque_limit([40, 40, 40, 30, 30, 30, 30])
                    .add_joint_limit("left_arm_3", -2.6, -0.5)
                    # .add_joint_limit("left_arm_5", 0.2, 1.9)
                    .set_nullspace_joint_target(np.deg2rad([0.0, 15.0, 0.0, -120.0, 0.0, 40.0, 15.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]) * 2, 0.2, 0.3)
                    .set_stop_joint_position_tracking_error(0)
                    .set_stop_orientation_tracking_error(0)
                    .set_stop_joint_position_tracking_error(0)
                    .set_joint_damping_ratio(0.85)
                    .set_reset_reference(self.left_reset)
                )
                self.get_logger().info("Component control mode")
                
                torso_builder.add_target("base", "link_torso_5", torso_T, 1, np.pi * 0.5, 10, np.pi * 20)
                right_builder.add_target("base", "link_right_arm_6", right_T @ np.linalg.inv(self.settings.T_hand_offset),
                                            2, np.pi * 2, 100, np.pi * 80)
                left_builder.add_target("base", "link_left_arm_6", left_T @ np.linalg.inv(self.settings.T_hand_offset),
                                        2, np.pi * 2, 100, np.pi * 80)
                
                print(SystemContext.rby1_state.left_arm_locked_pose)
                self.get_logger().info("Targets added to builders.")
                ctrl_builder = (
                    rby.BodyComponentBasedCommandBuilder()
                    .set_torso_command(torso_builder)
                    .set_right_arm_command(right_builder)
                    .set_left_arm_command(left_builder)
                )
                self.get_logger().info("Sending impedance control command...")
                self.stream.send_command(
                    rby.RobotCommandBuilder().set_command(
                        rby.ComponentBasedCommandBuilder()
                        .set_mobility_command(
                            rby.SE2VelocityCommandBuilder()
                            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(self.settings.dt * 100))
                            .set_velocity(SystemContext.rby1_state.mobile_linear_velocity,
                                            SystemContext.rby1_state.mobile_angular_velocity)
                            .set_minimum_time(self.settings.dt * 1.01)
                        )
                        .set_body_command(
                            ctrl_builder
                        )
                    )
                )

                # self.get_logger().info(SystemContext.control_state.is_active)
                if SystemContext.control_state.is_active:
                    SystemContext.rby1_state.dt -= self.settings.dt
                    SystemContext.rby1_state.dt = max(SystemContext.rby1_state.dt, self.settings.dt)
                else:
                    SystemContext.rby1_state.dt = self.settings.initial_dt
                # self.get_logger().info(f"Impedance control command sent. dt: {SystemContext.rby1_state.dt:.4f} sec")

            except Exception as e:
                logging.error(e)
                self.get_logger().info("Error in command stream. Cancelling stream...")
                self.stream = None
                # exit(1)
                
def main():
    global rby1_node
    rclpy.init()
    rby1_node = RBY1Node()
    
    rclpy.spin(rby1_node)
    rby1_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()