import json
import math
import time

import cv2
import mediapipe as mp
import websocket  # 웹소켓 클라이언트

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# mode 확정 시스템을 위한 변수들 (캘리브레이션 시스템 초기화 아래에 추가)
mode_confirmation_count = 0  # 같은 모드가 연속으로 감지된 횟수
last_detected_mode = None  # 마지막으로 감지된 모드
last_confirmed_mode = None  # 마지막으로 확정된 모드 (3회 연속)
last_sent_mode = None  # 마지막으로 웹소켓으로 전송한 모드
MODE_CONFIRMATION_THRESHOLD = 3  # 모드 확정을 위한 임계값
mode5_counter = 0
MODE5_CONFIRM_FRAMES = 100  # 약 1초 (30fps 기준)


# 지수이동평균 스무딩 클래스
class ExponentialMovingAverage:
    def __init__(
        self, alpha=0.1
    ):  # Note: The default alpha in the class definition is not what's used for the instance.
        self.alpha = alpha
        self.last_value = None

    def smooth(self, value):
        if self.last_value is None:
            self.last_value = value
            return value
        smoothed_value = self.alpha * value + (1 - self.alpha) * self.last_value
        self.last_value = smoothed_value
        return smoothed_value

    def reset(self):
        self.last_value = None


# 전역 스무딩 객체 생성 (카메라 열기 전에 추가)
distance_smoother = ExponentialMovingAverage(
    alpha=0.2
)  # 변경: alpha 값을 0.05에서 0.03으로 수정
angle_smoother = ExponentialMovingAverage(alpha=0.1)
thumb_angle_smoother = ExponentialMovingAverage(alpha=0.1)  # 엄지 각도 스무딩
# 이전 모드 상태를 추적하기 위한 변수
prev_mode = None


def check_hand_orientation(hand_landmarks):
    """손목의 방향을 확인하여 팔이 수직인지 판단"""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    # 손목에서 중지 MCP로의 벡터 (손의 주축)
    hand_vector_y = middle_mcp.y - wrist.y

    # y값이 양수면 손이 아래쪽을 향함 (정상), 음수면 위쪽을 향함 (팔을 든 상태)
    # 임계값: -0.05 (약간의 여유를 둠)
    is_arm_raised = hand_vector_y < -0.05

    return is_arm_raised


def is_thumb_extended(hand_landmarks, handedness):
    mcp = hand_landmarks.landmark[2]
    tip = hand_landmarks.landmark[4]
    wrist = hand_landmarks.landmark[0]
    index_mcp = hand_landmarks.landmark[5]
    middle_mcp = hand_landmarks.landmark[9]
    ring_mcp = hand_landmarks.landmark[13]
    pinky_mcp = hand_landmarks.landmark[17]

    # 손바닥 중심 계산 (MCP 4개 평균)
    palm_cx = (index_mcp.x + middle_mcp.x + ring_mcp.x + pinky_mcp.x) / 4
    palm_cy = (index_mcp.y + middle_mcp.y + ring_mcp.y + pinky_mcp.y) / 4

    # TIP ~ 손바닥 중심 거리
    dist_tip_palm = math.hypot(tip.x - palm_cx, tip.y - palm_cy)
    # MCP ~ TIP 거리
    dist_mcp_tip = math.hypot(tip.x - mcp.x, tip.y - mcp.y)

    # 엄지 각도
    angle = calculate_angle(
        (mcp.x, mcp.y),
        (hand_landmarks.landmark[3].x, hand_landmarks.landmark[3].y),
        (tip.x, tip.y),
    )

    # TIP이 손바닥 중심에 가까우면 접힘 (더 엄격한 기준)
    if dist_tip_palm < dist_mcp_tip * 0.8:
        return False  # 접힘

    # 엄지와 검지 PIP 사이 거리 체크 (mode2에서 false positive 방지)
    index_pip = hand_landmarks.landmark[6]
    thumb_index_distance = math.hypot(tip.x - index_pip.x, tip.y - index_pip.y)
    palm_width = math.hypot(index_mcp.x - pinky_mcp.x, index_mcp.y - pinky_mcp.y)

    # 엄지가 검지에 너무 가까우면 접힌 것으로 판단 (더 엄격한 기준)
    if thumb_index_distance < palm_width * 0.8:
        return False

    # 기존 각도 및 위치 조건 (더 엄격한 각도 기준)
    if angle > 145 and angle < 180:  # 기존 155도에서 160도로 상향
        if handedness == "Right":
            return tip.x > mcp.x and tip.x > hand_landmarks.landmark[1].x
        else:
            return tip.x < mcp.x and tip.x < hand_landmarks.landmark[1].x
    return False


# 각도 계산 함수 (세 점으로 이루어진 각도)
def calculate_angle(a, b, c):
    """a, b, c는 (x, y, z) 튜플. b는 각도의 꼭짓점"""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    cosine_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (
        math.sqrt(ba[0] ** 2 + ba[1] ** 2) * math.sqrt(bc[0] ** 2 + bc[1] ** 2) + 1e-6
    )
    angle = math.acos(cosine_angle)
    return math.degrees(angle)


# 3D 각도 계산 함수 (Z축 포함)
def calculate_angle_3d(a, b, c):
    """a, b, c는 (x, y, z) 튜플. b는 각도의 꼭짓점 - 3D 벡터 계산"""
    # 벡터 계산
    ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])

    # 내적 계산
    dot_product = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]

    # 벡터 크기 계산
    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)

    # 코사인 값 계산 (0으로 나누기 방지)
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc + 1e-6)

    # 코사인 값을 [-1, 1] 범위로 클램핑 (부동소수점 오차 방지)
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    # 각도 계산 (라디안 → 도)
    angle = math.acos(cosine_angle)
    return math.degrees(angle)


def calculate_thumb_spread_angle(hand_landmarks, handedness):
    """엄지 관절 각도 계산 (Index finger와 동일한 방식으로 안정화)"""
    # 엄지 관절 계산: CMC(1) → MCP(2) → IP(3) → TIP(4)
    # Index finger와 동일한 3점 각도 계산 방식 사용

    # 방법 1: 엄지 MCP 관절 각도 (MCP-IP-TIP)
    mcp = hand_landmarks.landmark[2]  # 엄지 MCP (중간 관절)
    ip = hand_landmarks.landmark[3]  # 엄지 IP (끝 관절)
    tip = hand_landmarks.landmark[4]  # 엄지 끝

    # 기본 관절 각도 계산 (MCP-IP-TIP)
    joint_angle = calculate_angle_3d(
        (mcp.x, mcp.y, mcp.z), (ip.x, ip.y, ip.z), (tip.x, tip.y, tip.z)
    )

    # 방법 2: 엄지와 검지 MCP 사이의 상대적 위치로 펼침 정도 판단
    index_mcp = hand_landmarks.landmark[5]  # 검지 MCP

    # 엄지 끝과 검지 MCP 사이의 거리 (정규화)
    thumb_index_distance = math.sqrt(
        (tip.x - index_mcp.x) ** 2
        + (tip.y - index_mcp.y) ** 2
        + (tip.z - index_mcp.z) ** 2
    )

    # 손 크기 기준 거리 (손목-검지MCP)
    wrist = hand_landmarks.landmark[0]
    hand_size = math.sqrt(
        (index_mcp.x - wrist.x) ** 2
        + (index_mcp.y - wrist.y) ** 2
        + (index_mcp.z - wrist.z) ** 2
    )

    # 정규화된 거리 (0~2 범위)
    normalized_spread = thumb_index_distance / (hand_size + 1e-6)

    # 방법 3: 손의 방향성을 고려한 엄지 위치 보정
    # 손바닥 평면의 법선 벡터 계산
    middle_mcp = hand_landmarks.landmark[9]
    ring_mcp = hand_landmarks.landmark[13]

    # 손목 → 중지MCP 벡터 (손의 주축)
    hand_axis_x = middle_mcp.x - wrist.x
    hand_axis_y = middle_mcp.y - wrist.y
    hand_axis_z = middle_mcp.z - wrist.z

    # 검지MCP → 약지MCP 벡터 (손의 너비)
    width_axis_x = ring_mcp.x - index_mcp.x
    width_axis_y = ring_mcp.y - index_mcp.y
    width_axis_z = ring_mcp.z - index_mcp.z

    # 외적으로 손바닥 법선 벡터 계산
    normal_x = hand_axis_y * width_axis_z - hand_axis_z * width_axis_y
    normal_y = hand_axis_z * width_axis_x - hand_axis_x * width_axis_z
    normal_z = hand_axis_x * width_axis_y - hand_axis_y * width_axis_x

    # 법선 벡터 정규화
    normal_magnitude = math.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    if normal_magnitude > 1e-6:
        normal_x /= normal_magnitude
        normal_y /= normal_magnitude
        normal_z /= normal_magnitude

    # 엄지 끝에서 손바닥 중심으로의 벡터
    palm_center_x = (index_mcp.x + middle_mcp.x + ring_mcp.x) / 3
    palm_center_y = (index_mcp.y + middle_mcp.y + ring_mcp.y) / 3
    palm_center_z = (index_mcp.z + middle_mcp.z + ring_mcp.z) / 3

    thumb_to_palm_x = tip.x - palm_center_x
    thumb_to_palm_y = tip.y - palm_center_y
    thumb_to_palm_z = tip.z - palm_center_z

    # 엄지가 손바닥 평면 위쪽(+)인지 아래쪽(-)인지 판단
    plane_position = (
        thumb_to_palm_x * normal_x
        + thumb_to_palm_y * normal_y
        + thumb_to_palm_z * normal_z
    )

    # === 최종 각도 계산 ===

    # Case 1: 완전히 펼쳐진 상태 (거리 기반)
    if normalized_spread > 1.2:
        return +joint_angle  # 양수 (펼침)

    # Case 2: 손바닥 아래로 숨긴 상태 (평면 위치 기반)
    elif plane_position < -0.01 and normalized_spread < 0.8:
        return -(joint_angle * 1.5)  # 강한 음수 (아래로 숨김)

    # Case 3: 손바닥 옆에 붙인 상태
    elif normalized_spread < 0.8 and abs(plane_position) < 0.01:
        return -(joint_angle * 0.3)  # 약한 음수 (옆에 붙임)

    # Case 4: 중간 상태들
    elif normalized_spread > 0.8:
        if plane_position > 0:
            return +(joint_angle * 0.7)  # 약간 펼침
        else:
            return -(joint_angle * 0.6)  # 약간 접음
    else:
        return -(joint_angle * 0.8)  # 일반 접음

    return joint_angle


def calculate_thumb_relative_to_palm(hand_landmarks):
    """손바닥 평면을 기준으로 엄지의 상대적 위치 계산 (양수: 바깥쪽, 음수: 손바닥 아래)"""
    # 손바닥 주요 4개 점 (MCP들)
    index_mcp = hand_landmarks.landmark[5]  # 검지 MCP
    middle_mcp = hand_landmarks.landmark[9]  # 중지 MCP
    ring_mcp = hand_landmarks.landmark[13]  # 약지 MCP
    pinky_mcp = hand_landmarks.landmark[17]  # 소지 MCP

    # 손바닥 중심점 계산
    palm_center_x = (index_mcp.x + middle_mcp.x + ring_mcp.x + pinky_mcp.x) / 4
    palm_center_y = (index_mcp.y + middle_mcp.y + ring_mcp.y + pinky_mcp.y) / 4
    palm_center_z = (index_mcp.z + middle_mcp.z + ring_mcp.z + pinky_mcp.z) / 4

    # 손목과 손바닥 중심을 잇는 벡터 (손의 주축)
    wrist = hand_landmarks.landmark[0]
    hand_direction_x = palm_center_x - wrist.x
    hand_direction_y = palm_center_y - wrist.y
    hand_direction_z = palm_center_z - wrist.z

    # 엄지 끝점
    thumb_tip = hand_landmarks.landmark[4]

    # 손바닥 중심에서 엄지 끝으로의 벡터
    thumb_vector_x = thumb_tip.x - palm_center_x
    thumb_vector_y = thumb_tip.y - palm_center_y
    thumb_vector_z = thumb_tip.z - palm_center_z

    # 손바닥 평면의 법선 벡터 계산 (검지→약지 × 손목→중심)
    palm_width_x = ring_mcp.x - index_mcp.x
    palm_width_y = ring_mcp.y - index_mcp.y
    palm_width_z = ring_mcp.z - index_mcp.z

    # 외적으로 법선 벡터 계산
    normal_x = palm_width_y * hand_direction_z - palm_width_z * hand_direction_y
    normal_y = palm_width_z * hand_direction_x - palm_width_x * hand_direction_z
    normal_z = palm_width_x * hand_direction_y - palm_width_y * hand_direction_x

    # 엄지 벡터와 법선 벡터의 내적 (양수면 바깥쪽, 음수면 안쪽)
    dot_product = (
        thumb_vector_x * normal_x
        + thumb_vector_y * normal_y
        + thumb_vector_z * normal_z
    )

    return dot_product


# 손가락이 펴졌는지 판단 함수
def finger_angle(hand_landmarks, mcp_id, pip_id, tip_id):
    mcp = hand_landmarks.landmark[mcp_id]
    pip = hand_landmarks.landmark[pip_id]
    tip = hand_landmarks.landmark[tip_id]
    return calculate_angle((mcp.x, mcp.y), (pip.x, pip.y), (tip.x, tip.y))


# 손가락별 각도 임계값 (160도 이상이면 펴짐)
ANGLE_THRESHOLD = 150


# 캘리브레이션 시스템
class CalibrationSystem:
    def __init__(self):
        self.state = "ready"  # ready, mode1_collect, mode2_collect
        self.mode1_values = []
        self.mode2_values = []
        self.mode1_min = None
        self.mode1_max = None
        self.mode2_min = None
        self.mode2_max = None
        self.collection_count = 0
        self.target_samples = 60  # 2초간 수집 (30fps 기준)

        # 성능 최적화를 위한 캐싱 변수들
        self.mode1_range_10_90 = None
        self.mode1_offset_10 = None
        self.mode2_range_10_90 = None
        self.mode2_offset_10 = None

    def get_remaining_time(self):
        """남은 시간을 초 단위로 반환"""
        remaining_frames = self.target_samples - self.collection_count
        remaining_seconds = remaining_frames / 30.0  # 30fps 기준
        return max(0, remaining_seconds)

    def start_mode1_calibration(self):
        """Mode1 캘리브레이션 시작"""
        self.state = "mode1_collect"
        self.mode1_values = []
        self.collection_count = 0

    def start_mode2_calibration(self):
        """Mode2 캘리브레이션 시작"""
        self.state = "mode2_collect"
        self.mode2_values = []
        self.collection_count = 0

    def collect_sample(self, mode, distance_value):
        """샘플 수집"""
        if self.state == "mode1_collect" and mode == "mode1":
            self.mode1_values.append(distance_value)
            self.collection_count += 1

            if self.collection_count >= self.target_samples:
                self.mode1_min = min(self.mode1_values)
                self.mode1_max = max(self.mode1_values)
                # 캐싱 값 미리 계산
                self._update_mode1_cache()
                self.state = "ready"  # mode1 완료 후 ready 상태로

        elif self.state == "mode2_collect" and mode == "mode2":
            self.mode2_values.append(distance_value)
            self.collection_count += 1

            if self.collection_count >= self.target_samples:
                self.mode2_min = min(self.mode2_values)
                self.mode2_max = max(self.mode2_values)
                # 캐싱 값 미리 계산
                self._update_mode2_cache()
                self.state = "ready"  # mode2 완료 후 ready 상태로

    def _update_mode1_cache(self):
        """Mode1 캐싱 값 업데이트"""
        if self.mode1_min is not None and self.mode1_max is not None:
            range_val = self.mode1_max - self.mode1_min
            if range_val > 1e-6:
                # 전체 범위의 15~85% 구간만 사용 (70% 스팬)
                self.mode1_range_10_90 = range_val * 0.7  # 이전 0.8에서 변경
                self.mode1_offset_10 = (
                    self.mode1_min + range_val * 0.15
                )  # 이전 0.1에서 변경
            else:
                # 유효하지 않은 범위이면 캐시를 None으로 설정
                self.mode1_range_10_90 = None
                self.mode1_offset_10 = None
        else:
            # min/max가 설정되지 않았으면 캐시를 None으로 설정
            self.mode1_range_10_90 = None
            self.mode1_offset_10 = None

    def _update_mode2_cache(self):
        """Mode2 캐싱 값 업데이트"""
        if self.mode2_min is not None and self.mode2_max is not None:
            range_val = self.mode2_max - self.mode2_min
            if range_val > 1e-6:
                # 전체 범위의 15~85% 구간만 사용 (70% 스팬)
                self.mode2_range_10_90 = range_val * 0.7  # 이전 0.8에서 변경
                self.mode2_offset_10 = (
                    self.mode2_min + range_val * 0.15
                )  # 이전 0.1에서 변경
            else:
                # 유효하지 않은 범위이면 캐시를 None으로 설정
                self.mode2_range_10_90 = None
                self.mode2_offset_10 = None
        else:
            # min/max가 설정되지 않았으면 캐시를 None으로 설정
            self.mode2_range_10_90 = None
            self.mode2_offset_10 = None

    def is_calibrated(self):
        """두 모드 모두 캘리브레이션 완료되었는지 확인"""
        return (
            self.mode1_min is not None
            and self.mode1_max is not None
            and self.mode2_min is not None
            and self.mode2_max is not None
        )

    def get_percentage(self, mode, distance_value):
        """캘리브레이션된 값으로 퍼센트 반환 (최대 120%까지 허용)"""
        if (
            mode == "mode1"
            and self.mode1_range_10_90 is not None
            and self.mode1_offset_10 is not None
        ):
            normalized = (
                distance_value - self.mode1_offset_10
            ) / self.mode1_range_10_90
            return max(0, min(120, normalized * 100))  # 최대 120%까지 허용
        elif (
            mode == "mode2"
            and self.mode2_range_10_90 is not None
            and self.mode2_offset_10 is not None
        ):
            normalized = (
                distance_value - self.mode2_offset_10
            ) / self.mode2_range_10_90
            return max(0, min(120, normalized * 100))  # 최대 120%까지 허용
        return None

    def set_defaults(self):
        """기본값 설정 (대략적인 값)"""
        self.mode1_min = 0.3
        self.mode1_max = 1.5
        self.mode2_min = 0.1
        self.mode2_max = 0.8
        # 캐싱 값 업데이트
        self._update_mode1_cache()
        self._update_mode2_cache()
        self.state = "ready"

    def reset(self):
        """캘리브레이션 리셋"""
        self.state = "ready"
        self.mode1_values = []
        self.mode2_values = []
        self.mode1_min = None
        self.mode1_max = None
        self.mode2_min = None
        self.mode2_max = None
        self.collection_count = 0
        # 캐싱 값 초기화
        self.mode1_range_10_90 = None
        self.mode1_offset_10 = None
        self.mode2_range_10_90 = None
        self.mode2_offset_10 = None


# 캘리브레이션 시스템 초기화
calibration = CalibrationSystem()

# 카메라 열기
cap = cv2.VideoCapture(0)
# 성능 최적화를 위해 해상도 설정 HD
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 모드 상태 변수 추가
mode = None

# WebSocket 서버 연결
ws = None
try:
    # Add a timeout to the connection attempt (e.g., 2 seconds)
    ws = websocket.create_connection("ws://192.168.0.210:5678", timeout=2)
    print("WebSocket connected.")
except websocket.WebSocketTimeoutException:
    print("WebSocket connection timed out after 2 seconds.")
    print("Continuing without WebSocket connection...")
    ws = None
except ConnectionRefusedError:
    print(
        "WebSocket connection refused by server. Check if the server is running and accessible."
    )
    print("Continuing without WebSocket connection...")
    ws = None
except Exception as e:  # Catch other potential exceptions during connection attempt
    print(f"WebSocket connection attempt failed: {e}")
    print("Continuing without WebSocket connection...")
    ws = None
last_send_time = time.time() - 1

# 메인 루프 시작 전 웹소켓 전송 타이밍 변수 초기화
last_send_time = time.time()

with mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # 이미지 처리
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # 결과 다시 BGR로
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 랜드마크 처리
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 손 방향 확인 (팔을 들었는지 체크)
                is_arm_raised = check_hand_orientation(hand_landmarks)

                # 각 손가락 각도 계산
                angles = {
                    "Thumb": finger_angle(hand_landmarks, 2, 3, 4),
                    "Index": finger_angle(hand_landmarks, 5, 6, 8),
                    "Middle": finger_angle(hand_landmarks, 9, 10, 12),
                    "Ring": finger_angle(hand_landmarks, 13, 14, 16),
                    "Pinky": finger_angle(hand_landmarks, 17, 18, 20),
                }

                # 검지 각도 스무딩 처리
                raw_index_angle = angles["Index"]
                smoothed_index_angle = angle_smoother.smooth(raw_index_angle)

                # 검지 PIP(6), 중지 PIP(10) 좌표 (명시적 랜드마크 이름 사용)
                index_pip = hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_PIP
                ]
                middle_pip = hand_landmarks.landmark[
                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP
                ]

                # 이미지 좌표계로 변환 (픽셀 단위)
                h, w, _ = image.shape
                x1_pip, y1_pip = int(index_pip.x * w), int(index_pip.y * h)
                x2_pip, y2_pip = int(middle_pip.x * w), int(middle_pip.y * h)
                pip_distance = math.sqrt(
                    (x1_pip - x2_pip) ** 2 + (y1_pip - y2_pip) ** 2
                )

                # 기준 거리: 손목(0) ~ 검지 MCP(5)
                index_mcp = hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_MCP
                ]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                x1_base, y1_base = int(index_mcp.x * w), int(index_mcp.y * h)
                x2_base, y2_base = int(wrist.x * w), int(wrist.y * h)
                base_dist_pixel = math.sqrt(
                    (x1_base - x2_base) ** 2 + (y1_base - y2_base) ** 2
                )

                # 모드별 거리 계산
                norm_dist = -1.0  # 초기값
                raw_norm_dist = -1.0  # 원본 거리값 저장용

                if base_dist_pixel > 1e-6:
                    if mode == "mode2":
                        # 검지 PIP(6), 중지 PIP(10)
                        index_pip = hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_PIP
                        ]
                        middle_pip = hand_landmarks.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_PIP
                        ]
                        x1_pip, y1_pip = int(index_pip.x * w), int(index_pip.y * h)
                        x2_pip, y2_pip = int(middle_pip.x * w), int(middle_pip.y * h)
                        pip_distance = math.sqrt(
                            (x1_pip - x2_pip) ** 2 + (y1_pip - y2_pip) ** 2
                        )
                        raw_norm_dist = pip_distance / base_dist_pixel
                        norm_dist = distance_smoother.smooth(
                            raw_norm_dist
                        )  # 스무딩 적용
                        dist_label = "Norm Dist (indexPIP-middlePIP)"

                    elif mode == "mode1":
                        # 검지 PIP(6), 엄지 IP(3)
                        index_pip = hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_PIP
                        ]
                        thumb_ip = hand_landmarks.landmark[
                            mp_hands.HandLandmark.THUMB_IP
                        ]
                        x1_pip, y1_pip = int(index_pip.x * w), int(index_pip.y * h)
                        x2_pip, y2_pip = int(thumb_ip.x * w), int(thumb_ip.y * h)
                        pip_distance = math.sqrt(
                            (x1_pip - x2_pip) ** 2 + (y1_pip - y2_pip) ** 2
                        )
                        raw_norm_dist = pip_distance / base_dist_pixel
                        norm_dist = distance_smoother.smooth(
                            raw_norm_dist
                        )  # 스무딩 적용
                        dist_label = "Norm Dist (indexPIP-thumbIP)"

                    else:
                        dist_label = "Norm Dist (wrist-idxMCP)"

                # 각도 기반 펴짐/굽힘 판단 (스무딩된 각도 사용)
                fingers = {
                    finger: (angle > ANGLE_THRESHOLD)
                    for finger, angle in angles.items()
                }

                # 검지는 스무딩된 각도로 다시 판단
                fingers["Index"] = smoothed_index_angle > ANGLE_THRESHOLD

                # 엄지손가락은 별도 판별 (handedness 정보 활용)
                handedness = None
                thumb_angle = 0
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx].classification[0].label
                if handedness:
                    fingers["Thumb"] = is_thumb_extended(hand_landmarks, handedness)
                    # 새로운 엄지 각도 계산: 검지와의 벡터 각도 기준
                    raw_thumb_angle = calculate_thumb_spread_angle(
                        hand_landmarks, handedness
                    )
                    # 엄지 각도 스무딩 처리
                    thumb_angle = thumb_angle_smoother.smooth(raw_thumb_angle)

                # 모드 판별 및 상태 유지 (팔을 든 상태에서는 모드 변경 방지)
                if not is_arm_raised:
                    # mode5 조건: 엄지만 굽힘, 나머지 손가락은 모두 펴짐
                    if (
                        not fingers["Thumb"]
                        and fingers["Index"]
                        and fingers["Middle"]
                        and fingers["Ring"]
                        and fingers["Pinky"]
                    ):
                        mode5_counter += 1
                        if mode5_counter >= MODE5_CONFIRM_FRAMES:
                            current_mode = "mode5"
                        else:
                            current_mode = None  # 아직 확정 아님
                    else:
                        mode5_counter = 0  # 모드5 카운터 리셋

                        # mode0 조건: 손가락 5개 모두 펴짐
                        if (
                            fingers["Thumb"]
                            and fingers["Index"]
                            and fingers["Middle"]
                            and fingers["Ring"]
                            and fingers["Pinky"]
                        ):
                            current_mode = "mode0"

                        # 기존 mode1, mode2 판별 로직...
                        elif mode == "mode1":
                            if (
                                fingers["Index"]
                                and fingers["Middle"]
                                and not fingers["Ring"]
                                and not fingers["Pinky"]
                            ):
                                current_mode = "mode2"
                            else:
                                current_mode = "mode1"

                        elif mode == "mode2":
                            if (
                                fingers["Index"]
                                and not fingers["Middle"]
                                and not fingers["Ring"]
                                and not fingers["Pinky"]
                            ):
                                current_mode = "mode1"
                            else:
                                current_mode = "mode2"

                        else:
                            # mode2 조건: 검지와 중지만 펴짐, 나머지는 굽힘 (엄지 무관)
                            if (
                                fingers["Index"]
                                and fingers["Middle"]
                                and not fingers["Ring"]
                                and not fingers["Pinky"]
                            ):
                                current_mode = "mode2"
                            # mode1 조건: 검지만 펴짐, 나머지는 굽힘 (엄지 무관)
                            elif (
                                fingers["Index"]
                                and not fingers["Middle"]
                                and not fingers["Ring"]
                                and not fingers["Pinky"]
                            ):
                                current_mode = "mode1"
                            else:
                                current_mode = None

                    # === 새로 추가: Mode 확정 시스템 ===
                    if current_mode == last_detected_mode:
                        # 같은 모드가 연속으로 감지됨
                        mode_confirmation_count += 1
                        print(
                            f"Mode confirmation: {current_mode} x{mode_confirmation_count}"
                        )
                    else:
                        # 다른 모드가 감지됨 - 카운트 리셋
                        mode_confirmation_count = 1
                        last_detected_mode = current_mode
                        print(f"Mode changed to: {current_mode} (count reset)")

                    # 모드 확정 조건: 3회 연속 동일한 모드 감지
                    if (
                        mode_confirmation_count >= MODE_CONFIRMATION_THRESHOLD
                        and current_mode != last_confirmed_mode
                    ):
                        last_confirmed_mode = current_mode
                        if current_mode:
                            mode = current_mode  # 실제 mode 변수 업데이트
                            print(f"Mode CONFIRMED: {mode} (3회 연속 감지)")

                            # 모드 변경 시 거리 스무딩 초기화
                            if prev_mode != mode:
                                distance_smoother.reset()
                                thumb_angle_smoother.reset()
                        else:
                            mode = None
                    elif mode_confirmation_count >= MODE_CONFIRMATION_THRESHOLD:
                        # 이미 확정된 모드가 계속 유지되는 경우
                        mode = last_confirmed_mode

                # 이전 모드 상태 업데이트
                prev_mode = mode

                # 결과 출력
                y0 = 30
                for finger_name, is_extended in fingers.items():
                    text = f"{finger_name}: {'1' if is_extended else '0'}"
                    cv2.putText(
                        image,
                        text,
                        (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    y0 += 25

                # Index Finger status classification
                if smoothed_index_angle <= 81:
                    index_status = "Fully Bent"
                    index_color = (0, 0, 255)  # Red
                elif 82 <= smoothed_index_angle <= 114:
                    index_status = "Half Bent"
                    index_color = (0, 165, 255)  # Orange
                else:
                    index_status = "Extended"
                    index_color = (0, 255, 0)  # Green

                cv2.putText(
                    image,
                    f"Index Angle: {int(smoothed_index_angle)} ({index_status})",
                    (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    index_color,
                    2,
                )
                y0 += 25

                # Thumb status classification
                thumb_status = None
                thumb_color = None

                if 140 <= thumb_angle <= 160:
                    thumb_status = "Fully Extended"
                    thumb_color = (0, 255, 0)  # Green
                elif 90 <= thumb_angle <= 122:
                    thumb_status = "Default"
                    thumb_color = (255, 255, 0)  # Yellow
                elif (-144 <= thumb_angle <= -140) or (-90 <= thumb_angle <= -25):
                    thumb_status = "Touching Palm"
                    thumb_color = (255, 0, 0)  # Red
                elif -130 <= thumb_angle <= -111:
                    thumb_status = "Under Palm"
                    thumb_color = (0, 0, 255)  # Blue

                # 정의된 상태가 있을 때는 상태와 함께 표시
                if thumb_status is not None:
                    cv2.putText(
                        image,
                        f"Thumb Angle: {thumb_angle:+.1f} ({thumb_status})",
                        (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        thumb_color,
                        2,
                    )
                else:
                    # 정의되지 않은 범위에서는 각도만 표시 (회색)
                    cv2.putText(
                        image,
                        f"Thumb Angle: {thumb_angle:+.1f}",
                        (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (128, 128, 128),  # Gray
                        2,
                    )
                y0 += 25

                # Finger Distance 또는 퍼센트 표시 (각 모드별로 개별 적용)
                if mode and norm_dist > 0:
                    if mode == "mode1":
                        # Mode1에서만 Mode1 캘리브레이션 상태 확인
                        if (
                            calibration.mode1_min is not None
                            and calibration.mode1_max is not None
                        ):
                            # Mode1이 캘리브레이션된 경우 퍼센트 표시
                            percentage = calibration.get_percentage("mode1", norm_dist)
                            if percentage is not None:
                                cv2.putText(
                                    image,
                                    f"Mode1 Distance: {percentage:.1f}%",
                                    (10, y0),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 255, 255),
                                    2,
                                )
                            else:
                                cv2.putText(
                                    image,
                                    f"Mode1 Distance: {norm_dist:.3f}",
                                    (10, y0),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 0, 255),
                                    2,
                                )
                        else:
                            # Mode1이 캘리브레이션되지 않은 경우 원래 숫자 표시
                            cv2.putText(
                                image,
                                f"Mode1 Distance: {norm_dist:.3f}",
                                (10, y0),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 0, 255),
                                2,
                            )

                    elif mode == "mode2":
                        # Mode2에서만 Mode2 캘리브레이션 상태 확인
                        if (
                            calibration.mode2_min is not None
                            and calibration.mode2_max is not None
                        ):
                            # Mode2가 캘리브레이션된 경우 퍼센트 표시
                            percentage = calibration.get_percentage("mode2", norm_dist)
                            if percentage is not None:
                                cv2.putText(
                                    image,
                                    f"Mode2 Distance: {percentage:.1f}%",
                                    (10, y0),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 255, 255),
                                    2,
                                )
                            else:
                                cv2.putText(
                                    image,
                                    f"Mode2 Distance: {norm_dist:.3f}",
                                    (10, y0),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 0, 255),
                                    2,
                                )
                        else:
                            # Mode2가 캘리브레이션되지 않은 경우 원래 숫자 표시
                            cv2.putText(
                                image,
                                f"Mode2 Distance: {norm_dist:.3f}",
                                (10, y0),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 0, 255),
                                2,
                            )
                else:
                    cv2.putText(
                        image,
                        "Distance: N/A",
                        (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 255),
                        2,
                    )
                y0 += 25

                # 모드 출력
                if mode == "mode0":
                    cv2.putText(
                        image,
                        "MODE 0",
                        (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 255),
                        3,
                    )
                elif mode == "mode1":
                    cv2.putText(
                        image,
                        "MODE 1",
                        (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        3,
                    )
                elif mode == "mode2":
                    cv2.putText(
                        image,
                        "MODE 2",
                        (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        3,
                    )
                elif mode == "mode5":  # 새로 추가
                    cv2.putText(
                        image,
                        "MODE 5",
                        (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),  # 빨간색으로 표시
                        3,
                    )

                # 손 랜드마크 시각화
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # 캘리브레이션 상태 및 안내 메시지 표시
        y_cal = image.shape[0] - 100

        if calibration.state == "ready":
            # 개별 모드 캘리브레이션 상태 표시
            mode1_status = (
                "✓"
                if (
                    calibration.mode1_min is not None
                    and calibration.mode1_max is not None
                )
                else "✗"
            )
            mode2_status = (
                "✓"
                if (
                    calibration.mode2_min is not None
                    and calibration.mode2_max is not None
                )
                else "✗"
            )

            # 각 모드별 캘리브레이션 상태 및 버튼
            cv2.putText(
                image,
                f"Mode1 Calibration {mode1_status} - Press '1' to calibrate",
                (10, y_cal + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if mode1_status == "✓" else (255, 255, 255),
                2,
            )
            cv2.putText(
                image,
                f"Mode2 Calibration {mode2_status} - Press '2' to calibrate",
                (10, y_cal + 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if mode2_status == "✓" else (255, 255, 255),
                2,
            )

        elif calibration.state == "mode1_collect":
            cv2.putText(
                image,
                "MODE1 Calibration in progress...",
                (10, y_cal),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image,
                "Move index finger from closest to farthest",
                (10, y_cal + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        elif calibration.state == "mode2_collect":
            cv2.putText(
                image,
                "MODE2 Calibration in progress...",
                (10, y_cal),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image,
                "Move index-middle from closest to farthest",
                (10, y_cal + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # 화면에 표시
        cv2.imshow("Hand Tracking", image)

        # 창에 포커스 유지를 위해 창 속성 설정
        cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_TOPMOST, 1)

        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF

        # 디버그: 키 입력 감지 확인
        if key != 255:  # 키가 눌렸을 때만 출력
            print(
                f"Key pressed: {key} (char: {chr(key) if 32 <= key <= 126 else 'N/A'})"
            )

        if key == 27:  # ESC 키
            break
        elif key == ord("1"):  # Mode1 캘리브레이션 시작
            calibration.start_mode1_calibration()
            print("Mode1 calibration started with key '1'")  # 디버그 출력
        elif key == ord("2"):  # Mode2 캘리브레이션 시작
            calibration.start_mode2_calibration()
            print("Mode2 calibration started with key '2'")  # 디버그 출력
            print("Mode2 calibration started with key '2'")  # 디버그 출력
        elif key == ord("0"):  # 'r' 키로 웹소켓 재연결
            if ws is None:
                try:
                    ws = websocket.create_connection(
                        "ws://192.168.0.213:5678", timeout=2
                    )
                    print("WebSocket reconnected.")
                except Exception as e:
                    print(f"WebSocket reconnection failed: {e}")
                    ws = None
            else:
                print("WebSocket is already connected.")

        # 웹소켓 전송 (초당 20회, 루프마다 체크)
        # === 수정된 웹소켓 전송 로직: mode5는 1초 이상 유지 시에만 전송 ===
        # 모든 모드에서 바로 웹소켓 전송
        if (
            (mode and norm_dist > 0)
            or (mode == "mode0")
            or (mode == "mode5" and mode5_counter >= MODE5_CONFIRM_FRAMES)
        ):
            current_time = time.time()
            # mode 변경 체크: 확정된 모드가 마지막 전송 모드와 다를 때만 전송
            mode_changed = (
                last_confirmed_mode != last_sent_mode
                and last_confirmed_mode is not None
            )

            # mode0과 mode5는 모드 변경 시에만 전송
            if mode == "mode0" and mode_changed:
                payload = {
                    "m": 0,
                    "d": 0,
                    "a": int(smoothed_index_angle)
                    if "smoothed_index_angle" in locals()
                    else 0,
                }
                try:
                    if ws is not None:
                        ws.send(json.dumps(payload))
                        last_sent_mode = mode
                        print(f"Mode change sent: {mode}")
                except Exception as e:
                    print(f"WebSocket send error: {e}")
                    ws = None

            elif (
                mode == "mode5"
                and mode5_counter >= MODE5_CONFIRM_FRAMES
                and mode_changed
            ):
                payload = {
                    "m": 5,
                    "d": 0,
                    "a": int(smoothed_index_angle)
                    if "smoothed_index_angle" in locals()
                    else 0,
                }
                try:
                    if ws is not None:
                        ws.send(json.dumps(payload))
                        last_sent_mode = mode
                        print(f"Mode change sent: {mode}")
                except Exception as e:
                    print(f"WebSocket send error: {e}")
                    ws = None

            # === 수정된 웹소켓 전송 로직 ===
            elif mode in ["mode1", "mode2"] and current_time - last_send_time >= 0.05:
                # 검지 상태 분류 (3단계) - IB 필드로 숫자 코드
                if smoothed_index_angle <= 81:
                    index_status_code = 1  # Fully Bent
                elif 82 <= smoothed_index_angle <= 114:
                    index_status_code = 2  # Half Bent
                else:
                    index_status_code = 3  # Extended

                # 엄지 상태 분류 (4단계) - FB 필드로 숫자 코드
                if 140 <= thumb_angle <= 160:
                    thumb_status_code = 1  # Fully Extended
                elif 90 <= thumb_angle <= 122:
                    thumb_status_code = 2  # Default
                elif (-144 <= thumb_angle <= -140) or (-90 <= thumb_angle <= -25):
                    thumb_status_code = 3  # Touching Palm
                elif -130 <= thumb_angle <= -111:
                    thumb_status_code = 4  # Under Palm
                else:
                    thumb_status_code = 0  # Undefined

                if mode_changed or last_sent_mode is None:
                    # 모드 변경 시에만 m 값 포함하여 전송
                    # 캘리브레이션 없이 원본 거리값 사용 (0-100 스케일링)
                    distance_value = int(norm_dist * 100)

                    m_val = 1 if mode == "mode1" else 2

                    payload = {
                        "m": m_val,  # 모드 변경 시에만 m 값 포함
                        "d": distance_value,
                        "IB": index_status_code,
                        "FB": thumb_status_code,
                    }

                    try:
                        if ws is not None:
                            ws.send(json.dumps(payload))
                            last_sent_mode = mode
                            print(
                                f"Mode change sent: {mode} with distance: {distance_value}, IB: {index_status_code}, FB: {thumb_status_code}"
                            )
                    except Exception as e:
                        print(f"WebSocket send error: {e}")
                        ws = None
                else:
                    # === 실시간 값 전송 (m 값 제외) ===
                    # 캘리브레이션 없이 원본 거리값 사용 (0-100 스케일링)
                    distance_value = int(norm_dist * 100)

                    payload = {
                        "d": distance_value,
                        "IB": index_status_code,
                        "FB": thumb_status_code,
                    }

                    try:
                        if ws is not None:
                            ws.send(json.dumps(payload))
                            print(
                                f"Data update sent: distance={distance_value}, IB={index_status_code}, FB={thumb_status_code}"
                            )
                    except Exception as e:
                        print(f"WebSocket send error: {e}")
                        ws = None

                last_send_time = current_time


cap.release()
cv2.destroyAllWindows()
