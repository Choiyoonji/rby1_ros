import numpy as np
from rby1_sdk.dynamics import Robot, load_robot_from_urdf
import rby1_sdk as rby


RBY1_LINK_NAMES: list[str] = ['base',
                                'wheel_l', 'wheel_r',
                                'link_torso_0', 'link_torso_1', 'link_torso_2', 'link_torso_3', 'link_torso_4', 'link_torso_5',
                                'link_head_1', 'link_left_arm_0', 'link_right_arm_0', 'link_head_2',
                                'link_left_arm_1', 'link_right_arm_1',
                                'link_left_arm_2', 'link_right_arm_2',
                                'link_left_arm_3', 'link_right_arm_3',
                                'link_left_arm_4', 'link_right_arm_4',
                                'link_left_arm_5', 'link_right_arm_5',
                                'link_left_arm_6', 'link_right_arm_6']

RBY1_JOINT_NAMES: list[str] = ['left_wheel', 'right_wheel',
                                'torso_0', 'torso_1', 'torso_2', 'torso_3', 'torso_4', 'torso_5',
                                'right_arm_0', 'right_arm_1', 'right_arm_2', 'right_arm_3',
                                'right_arm_4', 'right_arm_5', 'right_arm_6',
                                'left_arm_0', 'left_arm_1', 'left_arm_2', 'left_arm_3',
                                'left_arm_4', 'left_arm_5', 'left_arm_6',
                                'head_0', 'head_1']

class RBY1Dyn:
    def __init__(self):
        self.robot = load_robot_from_urdf("/home/choiyj/rby1-sdk/models/rby1a/urdf/model_v1.0.urdf", base_link_name="base")
        self.dyn_robot = Robot(self.robot)
        self.links: list[str] = ["base", "link_torso_5", "link_right_arm_6", "link_left_arm_6"]
        self.state = self.dyn_robot.make_state(self.links, RBY1_JOINT_NAMES)
        robot_max_q = self.dyn_robot.get_limit_q_upper(self.state)
        robot_min_q = self.dyn_robot.get_limit_q_lower(self.state)
        robot_max_qdot = self.dyn_robot.get_limit_qdot_upper(self.state)
        robot_max_qddot = self.dyn_robot.get_limit_qddot_upper(self.state)
        print("Robot max q:", robot_max_q)
        print("Robot min q:", robot_min_q)
        print("Robot max qdot:", robot_max_qdot)
        print("Robot max qddot:", robot_max_qddot)


    def get_fk(self, joint_positions):
        self.state.set_q(joint_positions)
        self.state.set_qdot(np.zeros_like(joint_positions))
        self.dyn_robot.compute_forward_kinematics(self.state)
        self.dyn_robot.compute_diff_forward_kinematics(self.state)
        fk_results = {}
        for link_idx in range(1, 4):
            link_name = self.links[link_idx]
            fk_results[link_name] = self.dyn_robot.compute_transformation(self.state, 0, link_idx)
        return fk_results
    
    def get_jacobian(self):
        J = self.dyn_robot.compute_body_jacobian(self.state, 0, 2)
        return J
    

if __name__ == "__main__":
    print(rby.Model_A().robot_joint_names)
    rby1_dyn = RBY1Dyn()
    # joint_positions = np.deg2rad([0.0, 0.0] +
    #             [0.0, 45.0, -90.0, 45.0, 0.0, 0.0] +
    #             [0.0, -15.0, 0.0, -120.0, 0.0, 70.0, 0.0] +
    #             [0.0, 15.0, 0.0, -120.0, 0.0, 70.0, 0.0])
    joint_positions = np.deg2rad([0.0]*24)
    fk_results = rby1_dyn.get_fk(joint_positions)
    for link_name, transform in fk_results.items():
        print(f"Link: {link_name}")
        print(transform)

    J = rby1_dyn.get_jacobian()
    print("Jacobian:\n", J.T)