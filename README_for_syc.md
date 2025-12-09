# rby1_ros

> **RBY1 Robot Manipulation Package for VLA (Vision-Language-Action)**
>
> 이 패키지는 Rainbow Robotics RBY1 로봇을 제어하기 위한 ROS2 기반 시스템입니다. VLM(Vision-Language Model) 학습 및 추론을 위해 **시각 데이터(Camera)와 행동(Action)을 연결**하고, 로봇을 정밀하게 제어하는 인터페이스를 제공합니다.

---

## 📂 File Description (Context for VLM)

VLM/VLA 시스템 구성을 위해 첨부된 각 파일이 담당하는 역할은 다음과 같습니다.

### 1. Node & Interface (Core)
* **`action_node.py` (VLM Agent / Interface)**
    * **Role:** VLM과 연결하는 인터페이스 노드입니다.
    * **Function:** Wrist/External 카메라 이미지를 수신하여 모델에 전달(또는 시각화)하고, 사용자나 모델의 입력을 받아 **Action(End-Effector 이동, 그리퍼 조작)** 명령을 발행할 수 있습니다.
* **`rby1_impedance_control_command.py` (Robot Driver)**
    * **Role:** **UPC(로봇 내부 PC)**에서 실행되며, 물리적 로봇 하드웨어와 직접 통신하는 드라이버입니다.
    * **Function:** 상위 노드에서 받은 명령을 임피던스 제어(Impedance Control) 신호로 변환하여 관절을 구동하고, 현재 로봇 상태(Joint/EE Pose/Torque)를 ROS 토픽으로 발행합니다.
* **`qos_profiles.py`**
    * **Role:** 통신 품질(QoS) 설정 파일입니다.
    * **Note:** 대용량 이미지 스트리밍은 지연 최소화를 위해 `Best Effort`로, 중요 제어 명령은 패킷 손실 방지를 위해 `Reliable`로 설정하여 VLM 시스템의 안정성을 보장합니다.

### 2. Motion Planning & Math
* **`ee_move_class.py` (Trajectory Generator)**
    * **Role:** VLM이 생성한 목표 좌표로 로봇을 부드럽게 이동시키기 위한 경로 생성기입니다.
    * **Function:** 급격한 움직임을 방지하기 위해 Spline 곡선을 사용하여 위치(Position)와 회전(Rotation) 궤적을 계산합니다.
* **`utils.py` (Math Helper)**
    * **Role:** 3D 공간 연산을 위한 수학 라이브러리입니다. (Quaternion 연산, SE(3) 변환 등 포함)
* **`get_bounding_box.py` (Workspace Calibration)**
    * **Role:** 로봇의 작업 영역(Workspace)을 측정하는 도구입니다.
    * **Function:** 로봇을 수동으로 움직이며 좌표를 기록하여, VLM이 안전하게 작업할 수 있는 최소/최대 범위(Bounding Box)를 계산합니다.

### 3. State Management
* **`main_status.py`**: 로봇의 현재 상태(Current State) 및 시스템 전반의 플래그(Ready, Move, Stop 등)를 관리합니다.
* **`control_status.py`**: 로봇에게 보낼 목표 상태(Desired State)와 제어 모드를 정의합니다.
* **`rby1_status.py`**: 로봇 하드웨어의 Low-level 센서 데이터(관절 전류, 토크, 리미트 등)를 정의합니다.

---

## 🎮 Action Node Controls (OpenCV Window)

`action_node` 실행 시 나타나는 **Camera View 창**에서 사용할 수 있는 키보드 명령어입니다. 데이터 수집이나 테스트 시 사용됩니다.

| Key | Function | Description |
| :---: | :--- | :--- |
| **`q`** | **Quit** | 프로그램 및 노드를 종료합니다. |
| **`i`** | **Image Request** | 현재 카메라(Wrist/External) 이미지를 요청/캡처합니다. |
| **`o`** | **Open** | 오른쪽 그리퍼를 엽니다. |
| **`c`** | **Close** | 오른쪽 그리퍼를 닫습니다. |
| **`w`** | **Move +X** | End-Effector +0.10m (전진) |
| **`s`** | **Move -X** | End-Effector -0.10m (후진) |
| **`a`** | **Move +Y** | End-Effector +0.10m (좌측 이동) |
| **`d`** | **Move -Y** | End-Effector -0.10m (우측 이동) |
| **`+`** | **Move +Z** | End-Effector +0.10m (상승) |
| **`-`** | **Move -Z** | End-Effector -0.10m (하강) |
| **`l`** | **Rotate +X** | Global X축 기준 +30도 회전 |
| **`j`** | **Rotate -X** | Global X축 기준 -30도 회전 |

---

## ✔ Before You Start

### 1. UPC ↔ External PC 연결 확인

코드를 실행하기 전에 **UPC와 외부 PC가 정상적으로 통신**하는지 확인하세요.

```bash
ping <UPC_IP>
ping <external_pc_IP>
```
### 2. USB Permission & Latency 설정

UPC 터미널에서 먼저 USB 권한과 latency timer를 설정합니다.
(.bashrc에 alias로 등록된 명령어)
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
ros2 run rby1_ros rby1_control_command
```

## ▶ On the **External PC Terminal**

### **2) Main Node 실행**

```bash
ros2 run rby1_ros main_node_command
```

### **3) Action Node 실행**

```bash
ros2 run rby1_ros action_node
```

# 👣 Step-by-Step Workflow

전체 시스템을 안전하게 구동하는 단계별 순서입니다.

1. **작업 영역 설정**
   - `get_bounding_box.py` 코드를 실행하여 로봇을 직접 움직여 작업영역을 확인합니다.
   - `ee_move_class.py` 코드의 Move_ee 클래스에 upper/lower bound를 설정합니다.

1. **Ready Pose 이동 (Main Node)**
   - `main_node` 실행 후 로봇이 초기화되고 **Ready Pose**로 이동하는지 확인합니다.
   - 이때 로봇의 LED가 초록색으로 변경됩니다.

2. **Image 수신 확인 (Action Node)**
   - `action_node` 실행 시 열리는 OpenCV 창에서 카메라 영상이 정상적으로 들어오는지 확인합니다.

3. **Move 모드 활성화 (Main Node)**
   - Main Node에서 로봇 제어가 가능하도록 `move` 상태를 `True`로 변경합니다.
   - 이 상태에서는 움직임 명령이 들어가므로 주의해야 합니다.

4. **Action 명령 전달 (Action Node)**
   - Action Node의 OpenCV 창이 활성화된 상태에서 키보드를 입력하여 로봇을 조작하거나 명령을 보냅니다.

5. **Move 모드 비활성화 (Main Node)**
   - 작업이 끝나면 안전을 위해 Main Node에서 `move` 상태를 `False`로 변경합니다.

6. **Stop 명령 전송 (Main Node)**
   - Main Node를 통해 `Stop` 명령을 보내 로봇을 정지 상태로 만듭니다.
   - 이때 로봇의 LED가 파란색을 변경됩니다.

7. **프로그램 종료**
   - 터미널에서 `Ctrl+C`를 입력하거나 Action Node 창에서 `q`를 눌러 종료합니다.