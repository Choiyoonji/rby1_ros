import logging
import rby1_sdk as rby
import numpy as np
import time

ADDRESS = '192.168.0.83:50051'
MODEL = 'a'

REFERENCE_LINK = 'base'
CONTROLLED_LINK = 'link_right_arm_6'

JOINT_INDEX = range(8, 15)  # 로봇의 관절 인덱스 설정 (예: 8~14번 관절)

TOOL_OFFSET = [0.0, 0.0, -0.15]  # 툴 오프셋 설정 (예: 그리퍼 길이)

T_TOOL_OFFSET = np.eye(4)
T_TOOL_OFFSET[0:3, 3] = TOOL_OFFSET

def robot_connect(robot):
    robot.connect()
    print(robot.get_robot_info())   
    
    if not robot.is_connected():
        print("Robot is not connected")
        exit(1)

    servo_pattern = "^(?!head_).*" 
    if not robot.is_power_on(servo_pattern):
        rv = robot.power_on(servo_pattern)
            
    if not robot.is_servo_on(servo_pattern):
        rv = robot.servo_on(servo_pattern)
        if not rv:
            print("Failed to servo on")
            exit(1)

    cm_state = robot.get_control_manager_state().state
    if cm_state in [
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ]:
        logging.warning(f"Control Manager is in Fault state: {cm_state.name}. Attempting reset...")
        if not robot.reset_fault_control_manager():
            logging.critical("Failed to reset Control Manager. Exiting program.")
            exit(1)
        logging.info("Control Manager reset successfully.")
    if not robot.enable_control_manager(unlimited_mode_enabled=True):
        logging.critical("Failed to enable Control Manager. Exiting program.")
        exit(1)
    logging.info("Control Manager successfully enabled. (Unlimited Mode: enabled)")
    
    return robot

def get_position_with_tool_offset(robot):
    robot_state = robot.get_state()
    # print("Robot State:", robot_state)
    
    joint_positions = [robot_state.position[i] for i in JOINT_INDEX]
    print("Joint Positions:", joint_positions)
    
    dyn = robot.get_dynamics()
    state = dyn.make_state([REFERENCE_LINK, CONTROLLED_LINK], robot.model().robot_joint_names)
    
    state.set_q(robot_state.position.copy())
    dyn.compute_forward_kinematics(state)
    
    T_link = dyn.compute_transformation(state, 0, 1)
    T_with_tool = T_link @ T_TOOL_OFFSET
    print("Transformation with Tool Offset:\n", T_with_tool)
    
    return T_with_tool[:3, 3]

def main():
    position_list = []
    
    robot = rby.create_robot(ADDRESS, MODEL)
    
    print("Initializing robot connection...")
    robot = robot_connect(robot)
    robot.disconnect()
    
    print("Robot disconnected.")
    
    while True:
        user_input = input(">> 기록하려면 Enter, 종료하려면 'q': ").strip().lower()
        
        if user_input == 'q':
            break
        
        try:
            robot = robot_connect(robot)
            position = get_position_with_tool_offset(robot)
            print("End-Effector Position with Tool Offset:", position)
            position_list.append(position)
            robot.disconnect()
        
        except KeyboardInterrupt:
            print("Exiting program.")
            return
    
    xs, ys, zs = zip(*position_list)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    
    print("\n" + "="*40)
    print(" 📊 결과 분석 (단위: m 또는 mm, 설정에 따름)")
    print("="*40)
    print(f"총 기록된 포인트 수: {len(position_list)}개")
    print("-" * 40)
    print(f"X 범위: {min_x:.4f} ~ {max_x:.4f} \t(변화량: {max_x - min_x:.4f})")
    print(f"Y 범위: {min_y:.4f} ~ {max_y:.4f} \t(변화량: {max_y - min_y:.4f})")
    print(f"Z 범위: {min_z:.4f} ~ {max_z:.4f} \t(변화량: {max_z - min_z:.4f})")
    print("="*40)
        
if __name__ == "__main__":
    main()