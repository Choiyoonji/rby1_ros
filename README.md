# rby1_ros

> RBY1 로봇 조작 및 데이터 수집을 위한 ROS2 패키지

---

## ✔ Before You Start

### 1. UPC ↔ External PC 연결 확인

코드를 실행하기 전에 **UPC와 외부 PC가 정상적으로 통신**하는지 확인하세요.

```bash
ping <UPc_IP>
ping <external_pc_IP>
```

### 2. USB Permission & Latency 설정

UPC 터미널에서 먼저 USB 권한과 latency timer를 설정합니다.

```bash
usb0
usb1
lt0
lt1

sb   # source bashrc
si   # source install (ros2 workspace)
```

---

# 🚀 How to Run

## ▶ On the **UPC Terminal**

### **1) Robot Control Node 실행**

```bash
ros2 run rby1_ros rby1_control_only_right
```

### **2) Master Arm Bridge 실행**

```bash
ros2 run rby1_ros master_arm_bridge
```

---

## ▶ On the **External PC Terminal**

> 데이터 저장 경로는 launch 파일에서 변경 가능
> 기본 경로: `~/rby1_data/digit_test`

### **3) Main Node 실행**

```bash
ros2 run rby1_ros main_node
```

### **4) Recorder 실행**

```bash
ros2 launch rby1_ros record_digit.launch.py
```

또는 직접:

```bash
ros2 run rby1_ros rby1_data_node_only_right
```

---

# 📦 Checking Recorded HDF5 Data

아래 스크립트로 데이터 구조 및 타임스탬프 확인 가능:

```bash
cd ~/ros2_ws/src/rby1_ros/rby1_ros/cam_utils
python rby1_check.py
```

---

# 🔄 Synchronize & Visualize

촬영된 영상·센서·로봇 데이터를 싱크하고 시각화하려면:

```bash
cd ~/ros2_ws/src/rby1_ros/rby1_ros/cam_utils

# Sync (interpolation or nearest-frame based)
python sync_data_only_right_interpolation.py

# Viewer (4cam + robot state overlay)
python sync_viewer_only_right.py
```

---

# ✋ Gripper Command Values

| Action    | Value |
| --------- | ----- |
| **Open**  | `0.0` |
| **Close** | `1.0` |

---

# ❗ Troubleshooting

### **RTPS Shared Memory Error**

```
[RTPS_TRANSPORT_SHM Error] Failed init_port fastrtps_port9923:
open_and_lock_file failed -> Function open_port_internal
```

**Solution**

1. 모든 ROS2 노드를 종료하고 다시 실행
2. 문제가 지속되면 **PC 재부팅**
