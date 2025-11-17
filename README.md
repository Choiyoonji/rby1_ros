# rby1_ros


## Check the UPC and external PC connection before running the code.
## You can check the connection by ping command.


## In UPC terminal, run the following commands:
0. set usb permission and latency timer and source ros2 workspace

usb0
usb1
lt0
lt1

sb
si

1. robot start 
ros2 run rby1_ros rby1_control_only_right

2. master arm start
ros2 run rby1_ros master_arm_bridge


## In external PC terminal, run the following commands:
### You can change the data directory in launch file.
### default: ~/rby1_data/digit_test

3. main start
ros2 run rby1_ros main_node

4. recoder set
ros2 launch rby1_ros record_digit.launch.py

ros2 run rby1_ros rby1_data_node_only_right


## HDF5 data check code

cd ~/ros2_ws/src/rby1_ros/rby1_ros/cam_utils
python rby1_check.py


## Synchronize and visualize the data

cd ~/ros2_ws/src/rby1_ros/rby1_ros/cam_utils
python sync_data_only_right_interpolation.py
python sync_viewer_only_right.py


## Gripper open/close value
- Open: 0.0
- Close: 1.0


## ERROR troubleshooting

2025-11-17 22:13:23.725 [RTPS_TRANSPORT_SHM Error] Failed init_port fastrtps_port9923: open_and_lock_file failed -> Function open_port_internal

Solution: Terminate all the ROS2 nodes and re-run the code. If the error persists, reboot the pc.
