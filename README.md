# rby1_ros

1. robot start
ros2 run rby1_ros rby1_control

2. meta start
ros2 run rby1_ros meta_node

3. main start
ros2 run rby1_ros main_node

4. recoder set
ros2 run rby1_ros tick_publisher --ros-args -p task:="test_task_001" -p base_dir:="~/rby1_data"

ros2 run rby1_ros realsense_record_node --ros-args -p shm_name:="right_wrist_D405" -p camera_model:="D405" -p serial_number:="\"218622272411\""

ros2 run rby1_ros realsense_record_node --ros-args -p shm_name:="left_wrist_D405" -p camera_model:="D405" -p serial_number:="\"218622278157\""

ros2 run rby1_ros realsense_record_node --ros-args -p shm_name:="external_D435I" -p camera_model:="D435I" -p serial_number:="\"233522076898\""

ros2 run rby1_ros digit_record_node --ros-args -p shm_name:="left_hand_digit1" -p serial_number:="\"D20612\""

ros2 run rby1_ros digit_record_node --ros-args -p shm_name:="left_hand_digit2" -p serial_number:="\"D20669\""

ros2 run rby1_ros zed_record_node

ros2 run rby1_ros rby1_data_node

5. meta-zed image send start
ros2 run rby1_ros zed_img_sender