import cv2
import mediapipe as mp
import math

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# 지수이동평균 스무딩 클래스
class ExponentialMovingAverage:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.smoothed_value = None

    def smooth(self, value):
        if self.smoothed_value is None:
            self.smoothed_value = value
        else:
            self.smoothed_value = (
                self.alpha * value + (1 - self.alpha) * self.smoothed_value
            )
        return self.smoothed_value

    def reset(self):
        """스무딩 값 초기화"""
        self.smoothed_value = None


# 전역 스무딩 객체 생성 (카메라 열기 전에 추가)
distance_smoother = ExponentialMovingAverage(alpha=0.1)
angle_smoother = ExponentialMovingAverage(alpha=0.1)
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

    # TIP이 손바닥 중심에 가까우면 접힘
    if dist_tip_palm < dist_mcp_tip * 0.7:
        return False  # 접힘

    # 기존 각도 및 위치 조건
    if angle > 155:
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


# 손가락이 펴졌는지 판단 함수
def finger_angle(hand_landmarks, mcp_id, pip_id, tip_id):
    mcp = hand_landmarks.landmark[mcp_id]
    pip = hand_landmarks.landmark[pip_id]
    tip = hand_landmarks.landmark[tip_id]
    return calculate_angle((mcp.x, mcp.y), (pip.x, pip.y), (tip.x, tip.y))


# 손가락별 각도 임계값 (160도 이상이면 펴짐)
ANGLE_THRESHOLD = 160

# 카메라 열기
cap = cv2.VideoCapture(0)

# 모드 상태 변수 추가
mode = None

with mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                index_mcp = hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_MCP
                ]
                x1_base, y1_base = int(wrist.x * w), int(wrist.y * h)
                x2_base, y2_base = int(index_mcp.x * w), int(index_mcp.y * h)
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
                        # 검지 TIP(8), 엄지 TIP(4)
                        index_tip = hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_TIP
                        ]
                        thumb_tip = hand_landmarks.landmark[
                            mp_hands.HandLandmark.THUMB_TIP
                        ]
                        x1_tip, y1_tip = int(index_tip.x * w), int(index_tip.y * h)
                        x2_tip, y2_tip = int(thumb_tip.x * w), int(thumb_tip.y * h)
                        tip_distance = math.sqrt(
                            (x1_tip - x2_tip) ** 2 + (y1_tip - y2_tip) ** 2
                        )
                        raw_norm_dist = tip_distance / base_dist_pixel
                        norm_dist = distance_smoother.smooth(
                            raw_norm_dist
                        )  # 스무딩 적용
                        dist_label = "Norm Dist (indexTIP-thumbTIP)"

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
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx].classification[0].label
                if handedness:
                    fingers["Thumb"] = is_thumb_extended(hand_landmarks, handedness)

                # 모드 판별 및 상태 유지 (팔을 든 상태에서는 모드 변경 방지)
                if not is_arm_raised:  # 팔을 들지 않은 상태에서만 모드 판별
                    if mode == "mode1":
                        # mode1 상태 유지, 단 mode2 조건이 되면 mode2로 전환
                        if (
                            fingers["Index"]
                            and fingers["Middle"]
                            and not fingers["Thumb"]
                            and not fingers["Ring"]
                            and not fingers["Pinky"]
                        ):
                            mode = "mode2"
                            # 모드 변경 시 거리 스무딩 초기화
                            if prev_mode != mode:
                                distance_smoother.reset()

                    elif mode == "mode2":
                        # mode2 상태 유지, 단 mode1 조건이 되면 mode1로 전환
                        if (
                            fingers["Thumb"]
                            and fingers["Index"]
                            and not fingers["Middle"]
                            and not fingers["Ring"]
                            and not fingers["Pinky"]
                        ):
                            mode = "mode1"
                            # 모드 변경 시 거리 스무딩 초기화
                            if prev_mode != mode:
                                distance_smoother.reset()

                    else:
                        # mode2 조건: 검지와 중지만 펴짐, 나머지는 굽힘
                        if (
                            fingers["Index"]
                            and fingers["Middle"]
                            and not fingers["Thumb"]
                            and not fingers["Ring"]
                            and not fingers["Pinky"]
                        ):
                            mode = "mode2"
                            # 모드 변경 시 거리 스무딩 초기화
                            if prev_mode != mode:
                                distance_smoother.reset()

                        # mode1 조건: 엄지와 검지만 펴짐, 나머지는 굽힘
                        elif (
                            fingers["Thumb"]
                            and fingers["Index"]
                            and not fingers["Middle"]
                            and not fingers["Ring"]
                            and not fingers["Pinky"]
                        ):
                            mode = "mode1"
                            # 모드 변경 시 거리 스무딩 초기화
                            if prev_mode != mode:
                                distance_smoother.reset()
                        else:
                            mode = None

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

                cv2.putText(
                    image,
                    f"Index Angle: {int(smoothed_index_angle)}",
                    (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )
                y0 += 25

                cv2.putText(
                    image,
                    f"Finger Distance: {norm_dist:.3f}",
                    (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2,
                )
                y0 += 25

                # 모드 출력
                if mode == "mode1":
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

                # 손 랜드마크 시각화
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # 화면에 표시
        cv2.imshow("Hand Tracking", image)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키
            break

cap.release()
cv2.destroyAllWindows()
