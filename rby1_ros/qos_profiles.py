"""Shared QoS profiles for rby1_ros nodes.

Place common QoSProfile objects here so nodes can import them and stay consistent.
"""
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

# 1) 제어/설정성 토픽: 마지막 값 유지(라치) + 신뢰모드
qos_ctrl_latched = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,  # 중요: 퍼블리셔도 동일해야 과거 샘플 수신
)

# alias used in some modules
qos_latched = qos_ctrl_latched

# 2) tick(1kHz): 지터 흡수용 소규모 버퍼 + 신뢰모드
qos_tick = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,  # 10~20 권장
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# 3) state: 최신값만 필요 → 깊이 1, 신뢰모드
qos_state_latest = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# 4) command: 주기 제어 입력
qos_cmd = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,  # 5~20 권장
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# 5) image/sensor: 고대역폭 스트리밍 (카메라, 라이다 등)
# 패킷 손실 시 재전송하지 않고 무시함(Best Effort) -> 지연(Latency) 최소화 및 대역폭 절약
# Rviz2 등에서 볼 때도 'Unreliable' 설정과 호환됨
qos_image_stream = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1, # 항상 최신 프레임 1장만 유지
    reliability=QoSReliabilityPolicy.BEST_EFFORT, 
    durability=QoSDurabilityPolicy.VOLATILE,
)

__all__ = [
    "qos_ctrl_latched",
    "qos_latched",
    "qos_tick",
    "qos_state_latest",
    "qos_cmd",
    "qos_image_stream",
]