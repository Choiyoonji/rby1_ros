import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rby1_interfaces.msg import State, Command, EEpos, FTsensor

import numpy as np
from utils import *
from dataclasses import dataclass, field
from typing import Union
import logging
import rby1_sdk as rby

from rby1_ros.rby1_status import RBY1Status as RBY1State
from rby1_ros.control_status import ControlStatus as ControlState

logging.basicConfig(level=logging.INFO)

@dataclass(frozen=True)
class Settings:
    dt: float = 0.1  # 10 Hz
    update_dt : float = 0.01   # 100 Hz
    hand_offset: np.ndarray = np.array([0.0, 0.0, 0.0])

    T_hand_offset = np.identity(4)
    T_hand_offset[0:3, 3] = hand_offset

    rby1_ip = "192.168.0.83"
    port = 50051
    model = "a"

    mobile_linear_acceleration_gain: float = 0.15
    mobile_angular_acceleration_gain: float = 0.15
    mobile_linear_damping_gain: float = 0.3
    mobile_angular_damping_gain: float = 0.3

    right_arm_start_position = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 0.0])
    left_arm_start_position = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 0.0])


class SystemContext:
    robot_model: Union[rby.Model_A, rby.Model_M] = None
    rby1_state = RBY1State()
    control_state = ControlState()


def robot_state_callback(robot_state: rby.RobotState_A):
    SystemContext.rby1_state.timestamp = robot_state.timestamp.timestamp()

    SystemContext.rby1_state.joint_positions = robot_state.position
    SystemContext.rby1_state.joint_velocities = robot_state.velocity
    SystemContext.rby1_state.joint_currents = robot_state.current
    SystemContext.rby1_state.joint_torques = robot_state.torque

    SystemContext.rby1_state.right_force_sensor = robot_state.ft_sensor_right.force
    SystemContext.rby1_state.left_force_sensor = robot_state.ft_sensor_left.force

    SystemContext.rby1_state.right_torque_sensor = robot_state.ft_sensor_right.torque
    SystemContext.rby1_state.left_torque_sensor = robot_state.ft_sensor_left.torque

    SystemContext.rby1_state.center_of_mass = robot_state.center_of_mass


class RBY1Node(Node):
    def __init__(self, no_head=True, no_gripper=True):
        super().__init__("rby1_node")
        self.get_logger().info("RBY1 Impedance Control Node Initialized")

        self.rby1_pub = self.create_publisher(
            State,
            '/rby1/state',
            10
        )
        self.control_sub = self.create_subscription(
            Command,
            '/control/command',
            self.control_callback,
            10
        )

        self.no_head = no_head
        self.no_gripper = no_gripper

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
        self.publish_timer = self.create_timer(self.settings.update_dt, self.publish_state)


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
            return
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
            SystemContext.control_state.ready = True
        if msg.stop:
            SystemContext.control_state.stop = True
        if msg.estop:
            SystemContext.control_state.estop = True

        if msg.move and msg.is_active:
            SystemContext.control_state.control_mode = msg.control_mode

            SystemContext.control_state.desired_right_ee_position["position"] = np.array(msg.desired_right_ee_pos.position.data)
            SystemContext.control_state.desired_right_ee_position["quaternion"] = np.array(msg.desired_right_ee_pos.quaternion.data)
            SystemContext.control_state.desired_left_ee_position["position"] = np.array(msg.desired_left_ee_pos.position.data)
            SystemContext.control_state.desired_left_ee_position["quaternion"] = np.array(msg.desired_left_ee_pos.quaternion.data)
            SystemContext.control_state.desired_head_ee_position["position"] = np.array(msg.desired_head_ee_pos.position.data)
            SystemContext.control_state.desired_head_ee_position["quaternion"] = np.array(msg.desired_head_ee_pos.quaternion.data)

            SystemContext.control_state.desired_joint_positions = np.array(msg.desired_joint_positions.data)

    def is_connected(self):
        return self.robot is not None

    def connect_rby1(self):
        address = f"{self.settings.rby1_ip}:{self.settings.port}"
        model = self.settings.model

        logging.info(f"Attempting to connect to RB-Y1... (Address: {address}, Model: {model})")

        self.robot = rby.create_robot(address, model)
        connected = self.robot.connect()
        if not connected:
            logging.critical("Failed to connect to RB-Y1. Exiting program.")
            exit(1)
        logging.info("Successfully connected to RB-Y1.")

        servo_pattern = "^(?!head_).*" if self.no_head else ".*"
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

        SystemContext.rby1_state.is_robot_connected = True
        
    def ready(self):
        if self.robot.get_control_manager_state().control_state != rby.ControlManagerState.ControlState.Idle:
            self.robot.cancel_control()
            
        if self.robot.wait_for_control_ready(1000):
            ready_pose = np.deg2rad(
                [0.0, 45.0, -90.0, 45.0, 0.0, 0.0] +
                [0.0, -15.0, 0.0, -120.0, 0.0, 30.0, -15.0] +
                [0.0, 15.0, 0.0, -120.0, 0.0, 30.0, 15.0])
            cbc = (
                rby.ComponentBasedCommandBuilder()
                .set_body_command(
                    rby.JointImpedanceControlCommandBuilder()
                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1))
                    .set_position(ready_pose)
                    .set_stiffness([400.] * 6 + [60] * 7 + [60] * 7)
                    .set_torque_limit([500] * 6 + [30] * 7 + [30] * 7)
                    .set_minimum_time(2)
                )
            )
            if not self.no_head:
                cbc.set_head_command(
                    rby.JointPositionCommandBuilder()
                    .set_position([0.] * len(SystemContext.robot_model.head_idx))
                    .set_minimum_time(2)
                )
            self.robot.send_command(
                rby.RobotCommandBuilder().set_command(
                    cbc
                )
            ).get()

        SystemContext.rby1_state.is_initialized = True
        SystemContext.rby1_state.is_stopped = False

        SystemContext.control_state.ready = False

    def stop(self):
        logging.info("Stopping the robot...")
        SystemContext.rby1_state.is_stopped = True
        SystemContext.control_state.stop = False

    def estop(self):
        logging.critical("Emergency Stop triggered! Shutting down the robot immediately.")
        try:
            self.robot.power_off(".*")
        finally:
            rclpy.shutdown()
            os._exit(1)  # or sys.exit(1)

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
                self.stream.cancel()
                self.stream = None
                print("Stream cancelled due to signal handling.")

        if SystemContext.rby1_state.is_stopped:
            if self.stream is not None:
                self.stream.cancel()
                self.stream = None
                print("Stream cancelled due to signal handling.")
            SystemContext.rby1_state.is_initialized = False
            return

        if not SystemContext.control_state.is_active:
            return

        if self.stream is None:
            if self.robot.wait_for_control_ready(0):
                self.stream = self.robot.create_command_stream()
                self.get_logger().info("Starting impedance control stream...")

                SystemContext.rby1_state.is_right_following = False
                SystemContext.rby1_state.is_left_following = False
                SystemContext.rby1_state.right_arm_locked_pose = SystemContext.rby1_state.right_ee_position.copy()
                SystemContext.rby1_state.left_arm_locked_pose = SystemContext.rby1_state.left_ee_position.copy()
                SystemContext.rby1_state.torso_locked_pose = SystemContext.rby1_state.torso_ee_position.copy()


        if SystemContext.control_state.is_controller_connected:
            SystemContext.control_state.desired_right_ee_T = pos_to_se3(
                SystemContext.control_state.desired_right_ee_position["position"],
                SystemContext.control_state.desired_right_ee_position["quaternion"]
            )
            SystemContext.control_state.desired_left_ee_T = pos_to_se3(
                SystemContext.control_state.desired_left_ee_position["position"],
                SystemContext.control_state.desired_left_ee_position["quaternion"]
            )
            SystemContext.control_state.desired_head_ee_T = pos_to_se3(
                SystemContext.control_state.desired_head_ee_position["position"],
                SystemContext.control_state.desired_head_ee_position["quaternion"]
            )

            if SystemContext.control_state.move:
                SystemContext.rby1_state.is_right_following = True
                SystemContext.rby1_state.is_left_following = True
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
                if SystemContext.rby1_state.is_right_following:
                    right_T = SystemContext.control_state.desired_right_ee_T
                    SystemContext.rby1_state.right_arm_locked_pose = right_T.copy()
                else:
                    right_T = SystemContext.rby1_state.right_arm_locked_pose
                
                if SystemContext.rby1_state.is_left_following:
                    left_T = SystemContext.control_state.desired_left_ee_T
                    SystemContext.rby1_state.left_arm_locked_pose = left_T.copy()
                else:
                    left_T = SystemContext.rby1_state.left_arm_locked_pose
                
                if SystemContext.rby1_state.is_torso_following:
                    torso_T = SystemContext.control_state.desired_head_ee_T
                    SystemContext.rby1_state.torso_locked_pose = torso_T.copy()
                else:
                    torso_T = SystemContext.rby1_state.torso_locked_pose

                if SystemContext.control_state.control_mode == "whole_body":
                    ctrl_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(self.settings.dt * 10))
                        .set_minimum_time(self.settings.dt * 1.01)
                        .set_joint_stiffness([400.] * 6 + [60] * 7 + [60] * 7)
                        .set_joint_torque_limit([500] * 6 + [30] * 7 + [30] * 7)
                        .add_joint_limit("right_arm_3", -2.6, -0.5)
                        .add_joint_limit("right_arm_5", 0.2, 1.9)
                        .add_joint_limit("left_arm_3", -2.6, -0.5)
                        .add_joint_limit("left_arm_5", 0.2, 1.9)
                        .add_joint_limit("torso_1", -0.523598776, 1.3)
                        .add_joint_limit("torso_2", -2.617993878, -0.2)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_reset_reference(self.right_reset | self.left_reset | self.torso_reset)
                    )
                    ctrl_builder.add_target("base", "link_torso_5", torso_T, 1, np.pi * 0.5, 10, np.pi * 20)
                    ctrl_builder.add_target("base", "link_right_arm_6", right_T @ np.linalg.inv(self.settings.T_hand_offset),
                                            2, np.pi * 2, 20, np.pi * 80)
                    ctrl_builder.add_target("base", "link_left_arm_6", left_T @ np.linalg.inv(self.settings.T_hand_offset),
                                            2, np.pi * 2, 20, np.pi * 80)

                else:
                    torso_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(self.settings.dt * 100))
                        .set_minimum_time(self.settings.dt * 1.01)
                        .set_joint_stiffness([400.] * 6)
                        .set_joint_torque_limit([500] * 6)
                        .add_joint_limit("torso_1", -0.523598776, 1.3)
                        .add_joint_limit("torso_2", -2.617993878, -0.2)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(self.torso_reset)
                    )
                    right_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(self.settings.dt * 100))
                        .set_minimum_time(self.settings.dt * 1.01)
                        .set_joint_stiffness([80, 80, 80, 80, 80, 80, 40])
                        .set_joint_torque_limit([30] * 7)
                        .add_joint_limit("right_arm_3", -2.6, -0.5)
                        .add_joint_limit("right_arm_5", 0.2, 1.9)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(self.right_reset)
                    )
                    left_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(self.settings.dt * 100))
                        .set_minimum_time(self.settings.dt * 1.01)
                        .set_joint_stiffness([80, 80, 80, 80, 80, 80, 40])
                        .set_joint_torque_limit([30] * 7)
                        .add_joint_limit("left_arm_3", -2.6, -0.5)
                        .add_joint_limit("left_arm_5", 0.2, 1.9)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(self.left_reset)
                    )
                    torso_builder.add_target("base", "link_torso_5", torso_T, 1, np.pi * 0.5, 10, np.pi * 20)
                    right_builder.add_target("base", "link_right_arm_6",
                                                right_T @ np.linalg.inv(self.settings.T_hand_offset),
                                                2, np.pi * 2, 20, np.pi * 80)
                    left_builder.add_target("base", "link_left_arm_6", left_T @ np.linalg.inv(self.settings.T_hand_offset),
                                            2, np.pi * 2, 20, np.pi * 80)

                    ctrl_builder = (
                        rby.BodyComponentBasedCommandBuilder()
                        .set_torso_command(torso_builder)
                        .set_right_arm_command(right_builder)
                        .set_left_arm_command(left_builder)
                    )

                self.torso_reset = False
                self.right_reset = False
                self.left_reset = False

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

            except Exception as e:
                logging.error(e)
                self.stream = None
                # exit(1)


if __name__ == "__main__":
    rclpy.init()
    rby1_node = RBY1Node(no_head=True, no_gripper=True)
    rclpy.spin(rby1_node)
    rby1_node.destroy_node()
    rclpy.shutdown()