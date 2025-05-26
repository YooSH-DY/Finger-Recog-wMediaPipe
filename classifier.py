import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pickle
import hashlib
import time
from datetime import datetime
import glob

# 캐시 디렉토리 설정
CACHE_DIR = os.path.join("/Users/yoosehyeok/Documents/RecordingData", "cache")
REFERENCE_CACHE_FILE = os.path.join(CACHE_DIR, "reference_data_cache.pkl")

# 캐시 디렉토리 생성
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"파일 불러오기 성공: {file_path}")
        print(f"데이터 크기: {data.shape}")
        return data
    except Exception as e:
        print(f"파일 로드 오류 ({file_path}): {e}")
        return None


def extract_features(data):
    features = {}

    # DOT 센서 데이터 찾기
    dot_acc_cols = [col for col in data.columns if "DOT_Acc" in col]
    dot_gyro_cols = [col for col in data.columns if "DOT_Gyro" in col]
    dot_euler_cols = [col for col in data.columns if "DOT_Euler" in col]

    # 자이로스코프 특징 추출
    if dot_gyro_cols and len(dot_gyro_cols) >= 3:
        gyro_x = data[dot_gyro_cols[0]].values
        gyro_y = data[dot_gyro_cols[1]].values
        gyro_z = data[dot_gyro_cols[2]].values

        # Z축 자이로스코프 관련 특징 (회전 동작에 중요)
        features["dot_gyro_z_sum"] = np.sum(gyro_z)
        features["dot_gyro_z_cumsum"] = np.max(np.abs(np.cumsum(gyro_z)))
        features["dot_gyro_z_std"] = np.std(gyro_z)
        features["dot_gyro_z_max"] = np.max(np.abs(gyro_z))

        # X축, Y축 자이로스코프 관련 특징
        features["dot_gyro_x_sum"] = np.sum(gyro_x)
        features["dot_gyro_y_sum"] = np.sum(gyro_y)
        features["dot_gyro_x_max"] = np.max(np.abs(gyro_x))
        features["dot_gyro_y_max"] = np.max(np.abs(gyro_y))

    # 오일러 특징 추출
    if dot_euler_cols and len(dot_euler_cols) >= 3:
        roll = data[dot_euler_cols[0]].values
        pitch = data[dot_euler_cols[1]].values
        yaw = data[dot_euler_cols[2]].values

        # 전체 각도 변화량 (처음과 끝 비교)
        features["dot_roll_change"] = np.max(roll) - np.min(roll)
        features["dot_pitch_change"] = np.max(pitch) - np.min(pitch)
        features["dot_yaw_change"] = np.max(yaw) - np.min(yaw)

        # 끝과 처음의 각도 차이 (최종 자세 변화)
        features["dot_roll_end_diff"] = roll[-1] - roll[0]
        features["dot_pitch_end_diff"] = pitch[-1] - pitch[0]
        features["dot_yaw_end_diff"] = yaw[-1] - yaw[0]

        # 각도 변화의 표준편차 (움직임의 안정성)
        features["dot_roll_std"] = np.std(roll)
        features["dot_pitch_std"] = np.std(pitch)
        features["dot_yaw_std"] = np.std(yaw)

    # 가속도 특징 추출
    if dot_acc_cols and len(dot_acc_cols) >= 3:
        acc_x = data[dot_acc_cols[0]].values
        acc_y = data[dot_acc_cols[1]].values
        acc_z = data[dot_acc_cols[2]].values

        # 전체 가속도 변화량
        features["dot_acc_x_change"] = np.max(acc_x) - np.min(acc_x)
        features["dot_acc_y_change"] = np.max(acc_y) - np.min(acc_y)
        features["dot_acc_z_change"] = np.max(acc_z) - np.min(acc_z)

        # 가속도 합 (움직임의 총량)
        features["dot_acc_magnitude"] = np.mean(np.sqrt(acc_x**2 + acc_y**2 + acc_z**2))

    return features


def extract_time_series(data):
    # DOT 센서 데이터 찾기
    dot_acc_cols = [col for col in data.columns if "DOT_Acc" in col]
    dot_gyro_cols = [col for col in data.columns if "DOT_Gyro" in col]
    dot_euler_cols = [col for col in data.columns if "DOT_Euler" in col]

    time_series = {}

    # 자이로스코프 데이터 - 패턴 중심 추출
    if dot_gyro_cols and len(dot_gyro_cols) >= 3:
        gyro_x = data[dot_gyro_cols[0]].values
        gyro_y = data[dot_gyro_cols[1]].values
        gyro_z = data[dot_gyro_cols[2]].values

        # 원본 데이터
        time_series["gyro"] = np.column_stack([gyro_x, gyro_y, gyro_z])

        # 1. 스무딩 적용 - 노이즈 감소 및 일관성 증가
        window = 5  # 스무딩 윈도우 크기
        gyro_x_smooth = np.convolve(gyro_x, np.ones(window) / window, mode="valid")
        gyro_y_smooth = np.convolve(gyro_y, np.ones(window) / window, mode="valid")
        gyro_z_smooth = np.convolve(gyro_z, np.ones(window) / window, mode="valid")
        time_series["gyro_smooth"] = np.column_stack(
            [gyro_x_smooth, gyro_y_smooth, gyro_z_smooth]
        )

        # 2. 특징적 패턴 추출 (미분값)
        gyro_x_diff = np.diff(gyro_x, prepend=gyro_x[0])
        gyro_y_diff = np.diff(gyro_y, prepend=gyro_y[0])
        gyro_z_diff = np.diff(gyro_z, prepend=gyro_z[0])
        time_series["gyro_diff"] = np.column_stack(
            [gyro_x_diff, gyro_y_diff, gyro_z_diff]
        )

        # 3. 정규화된 패턴 형태 (크기에 무관)
        gyro_magnitude = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        gyro_magnitude[gyro_magnitude == 0] = 1  # 0으로 나누기 방지
        gyro_x_norm = gyro_x / gyro_magnitude
        gyro_y_norm = gyro_y / gyro_magnitude
        gyro_z_norm = gyro_z / gyro_magnitude
        time_series["gyro_pattern"] = np.column_stack(
            [gyro_x_norm, gyro_y_norm, gyro_z_norm]
        )

    # 가속도 데이터 - 상대적 변화 중심으로 추출
    if dot_acc_cols and len(dot_acc_cols) >= 3:
        acc_x = data[dot_acc_cols[0]].values
        acc_y = data[dot_acc_cols[1]].values
        acc_z = data[dot_acc_cols[2]].values

        # 1. 첫 샘플 대비 상대적 변화 (위치 무관성)
        acc_x_rel = acc_x - acc_x[0]
        acc_y_rel = acc_y - acc_y[0]
        acc_z_rel = acc_z - acc_z[0]
        time_series["acc_relative"] = np.column_stack([acc_x_rel, acc_y_rel, acc_z_rel])

        # 2. 방향 패턴 (크기와 위치에 무관)
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        acc_magnitude[acc_magnitude == 0] = 1  # 0으로 나누기 방지
        acc_x_norm = acc_x / acc_magnitude
        acc_y_norm = acc_y / acc_magnitude
        acc_z_norm = acc_z / acc_magnitude
        time_series["acc_direction"] = np.column_stack(
            [acc_x_norm, acc_y_norm, acc_z_norm]
        )

    # 오일러 각도 데이터 - 각도 변화 패턴 중심
    if dot_euler_cols and len(dot_euler_cols) >= 3:
        roll = data[dot_euler_cols[0]].values
        pitch = data[dot_euler_cols[1]].values
        yaw = data[dot_euler_cols[2]].values

        # 1. 첫 샘플 대비 상대적 변화
        roll_rel = roll - roll[0]
        pitch_rel = pitch - pitch[0]
        yaw_rel = yaw - yaw[0]
        time_series["euler_relative"] = np.column_stack([roll_rel, pitch_rel, yaw_rel])

        # 2. 각도 변화량 (미분값)
        roll_diff = np.diff(roll, prepend=roll[0])
        pitch_diff = np.diff(pitch, prepend=pitch[0])
        yaw_diff = np.diff(yaw, prepend=yaw[0])
        time_series["euler_diff"] = np.column_stack([roll_diff, pitch_diff, yaw_diff])

        # 3. 롤과 피치 패턴 (요우 제외 - 방향성 문제)
        time_series["roll_pitch"] = np.column_stack([roll, pitch])

    # E9/EB 센서가 모두 존재할 때의 차이 시계열 및 손등(E9) 시계열 추가
    if "E9_DOT_Euler_Roll" in data.columns and "EB_DOT_Euler_Roll" in data.columns:
        e9_roll = data["E9_DOT_Euler_Roll"].values
        e9_pitch = data["E9_DOT_Euler_Pitch"].values
        eb_roll = data["EB_DOT_Euler_Roll"].values
        eb_pitch = data["EB_DOT_Euler_Pitch"].values
        # 센서간 롤/피치 차이
        diff_roll = e9_roll - eb_roll
        diff_pitch = e9_pitch - eb_pitch
        time_series["diff_roll_pitch"] = np.column_stack([diff_roll, diff_pitch])
        # 손등(E9) 롤/피치 시계열
        time_series["E9_roll_pitch"] = np.column_stack([e9_roll, e9_pitch])

    # 테스트 데이터의 다운샘플링 (패턴 일관성 향상)
    for key in list(time_series.keys()):
        if key != "length" and len(time_series[key]) > 20:
            # 30 포인트로 다운샘플링 (성능 향상)
            indices = np.linspace(0, len(time_series[key]) - 1, 20).astype(int)
            time_series[key] = time_series[key][indices]

    # 데이터 길이 정보도 추가
    time_series["length"] = len(data)

    return time_series


# 참조 데이터 캐시를 관리하는 함수들
def get_reference_files_hash(folder_path):
    reference_files = []

    # Gun 폴더 파일 경로만 수집
    gun_folder = os.path.join(folder_path, "Gun")
    if os.path.exists(gun_folder):
        for file in os.listdir(gun_folder):
            if file.endswith(".csv"):
                reference_files.append(os.path.join(gun_folder, file))

    # V 폴더 파일 경로만 수집
    v_folder = os.path.join(folder_path, "V")
    if os.path.exists(v_folder):
        for file in os.listdir(v_folder):
            if file.endswith(".csv"):
                reference_files.append(os.path.join(v_folder, file))

    # 기존 폴더 구조도 체크 (폴더 1-7)
    for motion_id in range(1, 8):
        motion_path = os.path.join(folder_path, str(motion_id))
        if os.path.exists(motion_path):
            for file in os.listdir(motion_path):
                if file.endswith("_merged.csv"):
                    reference_files.append(os.path.join(motion_path, file))

    # 파일들의 마지막 수정 시간으로 해시 생성
    hash_data = []
    for file in sorted(reference_files):
        if os.path.exists(file):
            mtime = os.path.getmtime(file)
            size = os.path.getsize(file)
            hash_data.append(f"{file}:{mtime}:{size}")

    # 해시 생성
    hasher = hashlib.md5()
    hasher.update("".join(hash_data).encode())
    return hasher.hexdigest()


def load_cached_reference_data():
    if os.path.exists(REFERENCE_CACHE_FILE):
        try:
            with open(REFERENCE_CACHE_FILE, "rb") as f:
                cache_data = pickle.load(f)
                print(
                    f"캐시된 참조 데이터 로드됨: {len(cache_data['reference_data'])}개 동작 유형"
                )
                print(f"캐시 생성 시간: {cache_data['timestamp']}")
                return cache_data["reference_data"], cache_data["file_hash"]
        except Exception as e:
            print(f"캐시 로드 오류: {e}")

    return None, None


def save_reference_data_cache(reference_data, file_hash):
    try:
        cache_data = {
            "reference_data": reference_data,
            "file_hash": file_hash,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(REFERENCE_CACHE_FILE, "wb") as f:
            pickle.dump(cache_data, f)

        print(f"참조 데이터 캐시 저장 완료: {len(reference_data)}개 동작 유형")
        return True
    except Exception as e:
        print(f"캐시 저장 오류: {e}")
        return False


def collect_reference_data(folder_path):
    # 참조 데이터 파일의 현재 상태 해시 계산
    current_hash = get_reference_files_hash(folder_path)

    # 캐시된 데이터 확인
    cached_data, cached_hash = load_cached_reference_data()

    # 캐시가 유효하면 사용
    if cached_data and cached_hash == current_hash:
        print("참조 데이터 캐시가 최신 상태입니다. 캐시를 사용합니다.")
        return cached_data

    print("참조 데이터 캐시가 없거나 최신 상태가 아닙니다. 새로 계산합니다...")

    reference_data = {}

    # Gun 폴더 처리 - 동작 1로 분류
    gun_folder = os.path.join(folder_path, "Gun")
    motion_id = 1  # Gun 폴더의 파일은 모두 동작 1로 분류
    reference_data[motion_id] = []
    if os.path.exists(gun_folder):
        print(f"\n== Gun 폴더 참조 데이터 수집 중 (동작 {motion_id}) ==")

        # 세션 번호별로 E9와 EB 파일 페어링
        paired_files = find_paired_files(gun_folder)

        if paired_files:
            print(f"페어링된 파일 발견: {len(paired_files)}개 세션")
            for session_num, files in paired_files.items():
                e9_file = files["E9"]
                eb_file = files["EB"]
                print(f"세션 {session_num} 처리 중:")
                print(f"  - E9 파일: {os.path.basename(e9_file)}")
                print(f"  - EB 파일: {os.path.basename(eb_file)}")

                # 개별 파일 처리
                e9_data = load_data(e9_file)
                eb_data = load_data(eb_file)

                # 페어 정보 저장 (두 센서 데이터를 함께 분석)
                if e9_data is not None and eb_data is not None:
                    # 페어 데이터 처리 (두 센서 데이터 결합)
                    try:
                        # 센서 컬럼 이름에 E9, EB 접두어 추가
                        e9_prefixed_cols = {
                            col: f"E9_{col}"
                            for col in e9_data.columns
                            if not col.startswith("E9_")
                        }
                        eb_prefixed_cols = {
                            col: f"EB_{col}"
                            for col in eb_data.columns
                            if not col.startswith("EB_")
                        }

                        e9_data = e9_data.rename(columns=e9_prefixed_cols)
                        eb_data = eb_data.rename(columns=eb_prefixed_cols)

                        # 동일한 길이로 리샘플링 (더 짧은 쪽에 맞춤)
                        min_length = min(len(e9_data), len(eb_data))
                        e9_indices = np.linspace(
                            0, len(e9_data) - 1, min_length
                        ).astype(int)
                        eb_indices = np.linspace(
                            0, len(eb_data) - 1, min_length
                        ).astype(int)

                        e9_data_resampled = e9_data.iloc[e9_indices]
                        eb_data_resampled = eb_data.iloc[eb_indices]

                        # 두 데이터 결합 (인덱스 리셋 후 결합)
                        e9_data_resampled.reset_index(drop=True, inplace=True)
                        eb_data_resampled.reset_index(drop=True, inplace=True)
                        paired_data = pd.concat(
                            [e9_data_resampled, eb_data_resampled], axis=1
                        )

                        # 결합된 데이터에서 시계열 특징 추출
                        time_series_paired = extract_time_series(paired_data)
                        time_series_paired["file_path"] = f"{e9_file}+{eb_file}"
                        time_series_paired["sensor_type"] = "PAIRED"
                        time_series_paired["session_num"] = session_num
                        reference_data[motion_id].append(time_series_paired)
                        print(f"  - 페어링된 데이터 생성 성공 (길이: {min_length})")
                    except Exception as e:
                        print(f"  - 페어링된 데이터 생성 실패: {e}")
                        # 페어링 실패 시 개별 센서 데이터 사용
                        if e9_data is not None:
                            time_series_e9 = extract_time_series(e9_data)
                            time_series_e9["file_path"] = e9_file
                            time_series_e9["sensor_type"] = "E9"
                            time_series_e9["session_num"] = session_num
                            reference_data[motion_id].append(time_series_e9)
                            print(f"  - E9 데이터만 사용")
        else:
            # 페어링 실패시 기존 방식으로 처리
            csv_files = []
            for session_num in range(1, 21):
                # "Session N.csv"로 끝나는 파일만 정확히 매칭
                session_pattern_files = [
                    f
                    for f in os.listdir(gun_folder)
                    if f.endswith(f"Session {session_num}.csv")
                ]
                csv_files.extend(session_pattern_files)

            for file in csv_files:
                file_path = os.path.join(gun_folder, file)
                print(f"파일 처리 중: {file}")

                data = load_data(file_path)
                if data is not None:
                    # 시계열 데이터 추출
                    time_series = extract_time_series(data)
                    time_series["file_path"] = file_path

                    # 센서 타입 저장
                    if "E9" in file:
                        time_series["sensor_type"] = "E9"
                    elif "EB" in file:
                        time_series["sensor_type"] = "EB"
                    else:
                        time_series["sensor_type"] = "UNKNOWN"

                    reference_data[motion_id].append(time_series)

    # V 폴더 처리 - 동작 2로 분류
    v_folder = os.path.join(folder_path, "V")
    motion_id = 2  # V 폴더의 파일은 모두 동작 2로 분류
    reference_data[motion_id] = []
    if os.path.exists(v_folder):
        print(f"\n== V 폴더 참조 데이터 수집 중 (동작 {motion_id}) ==")

        # 세션 번호별로 E9와 EB 파일 페어링
        paired_files = find_paired_files(v_folder)

        if paired_files:
            print(f"페어링된 파일 발견: {len(paired_files)}개 세션")
            for session_num, files in paired_files.items():
                e9_file = files["E9"]
                eb_file = files["EB"]
                print(f"세션 {session_num} 처리 중:")
                print(f"  - E9 파일: {os.path.basename(e9_file)}")
                print(f"  - EB 파일: {os.path.basename(eb_file)}")

                # 개별 파일 처리
                e9_data = load_data(e9_file)
                eb_data = load_data(eb_file)

                # 페어 정보 저장 (두 센서 데이터를 함께 분석)
                if e9_data is not None and eb_data is not None:
                    # 페어 데이터 처리 (두 센서 데이터 결합)
                    try:
                        # 센서 컬럼 이름에 E9, EB 접두어 추가
                        e9_prefixed_cols = {
                            col: f"E9_{col}"
                            for col in e9_data.columns
                            if not col.startswith("E9_")
                        }
                        eb_prefixed_cols = {
                            col: f"EB_{col}"
                            for col in eb_data.columns
                            if not col.startswith("EB_")
                        }

                        e9_data = e9_data.rename(columns=e9_prefixed_cols)
                        eb_data = eb_data.rename(columns=eb_prefixed_cols)

                        # 동일한 길이로 리샘플링 (더 짧은 쪽에 맞춤)
                        min_length = min(len(e9_data), len(eb_data))
                        e9_indices = np.linspace(
                            0, len(e9_data) - 1, min_length
                        ).astype(int)
                        eb_indices = np.linspace(
                            0, len(eb_data) - 1, min_length
                        ).astype(int)

                        e9_data_resampled = e9_data.iloc[e9_indices]
                        eb_data_resampled = eb_data.iloc[eb_indices]

                        # 두 데이터 결합 (인덱스 리셋 후 결합)
                        e9_data_resampled.reset_index(drop=True, inplace=True)
                        eb_data_resampled.reset_index(drop=True, inplace=True)
                        paired_data = pd.concat(
                            [e9_data_resampled, eb_data_resampled], axis=1
                        )

                        # 결합된 데이터에서 시계열 특징 추출
                        time_series_paired = extract_time_series(paired_data)
                        time_series_paired["file_path"] = f"{e9_file}+{eb_file}"
                        time_series_paired["sensor_type"] = "PAIRED"
                        time_series_paired["session_num"] = session_num
                        reference_data[motion_id].append(time_series_paired)
                        print(f"  - 페어링된 데이터 생성 성공 (길이: {min_length})")
                    except Exception as e:
                        print(f"  - 페어링된 데이터 생성 실패: {e}")
                        # 페어링 실패 시 개별 센서 데이터 사용
                        if e9_data is not None:
                            time_series_e9 = extract_time_series(e9_data)
                            time_series_e9["file_path"] = e9_file
                            time_series_e9["sensor_type"] = "E9"
                            time_series_e9["session_num"] = session_num
                            reference_data[motion_id].append(time_series_e9)
                            print(f"  - E9 데이터만 사용")
        else:
            # 페어링 실패시 기존 방식으로 처리
            csv_files = [f for f in os.listdir(v_folder) if f.endswith(".csv")]
            for file in csv_files:
                file_path = os.path.join(v_folder, file)
                print(f"파일 처리 중: {file}")

                data = load_data(file_path)
                if data is not None:
                    # 시계열 데이터 추출
                    time_series = extract_time_series(data)
                    time_series["file_path"] = file_path

                    # 센서 타입 저장
                    if "E9" in file:
                        time_series["sensor_type"] = "E9"
                    elif "EB" in file:
                        time_series["sensor_type"] = "EB"
                    else:
                        time_series["sensor_type"] = "UNKNOWN"

                    reference_data[motion_id].append(time_series)

    # 기존 1~7 폴더 구조도 유지 (추가 동작 유형)
    for motion_id in range(3, 8):  # 3부터 시작 (1, 2는 이미 사용됨)
        motion_path = os.path.join(folder_path, str(motion_id))
        reference_data[motion_id] = []

        # 해당 동작의 폴더가 있는 경우
        if os.path.exists(motion_path):
            print(f"\n== 동작 {motion_id} 참조 데이터 수집 중 ==")

            csv_files = [
                f for f in os.listdir(motion_path) if f.endswith("_merged.csv")
            ]
            for file in csv_files:
                file_path = os.path.join(motion_path, file)
                print(f"파일 처리 중: {file}")

                data = load_data(file_path)
                if data is not None:
                    # 시계열 데이터 추출
                    time_series = extract_time_series(data)
                    time_series["file_path"] = file_path  # 파일 경로 저장
                    reference_data[motion_id].append(time_series)

    # 참조 데이터가 비어있는 동작 제외
    empty_motions = [k for k, v in reference_data.items() if not v]
    for k in empty_motions:
        del reference_data[k]

    print(f"\n수집된 참조 데이터: {len(reference_data)}개 동작 유형")
    for motion_id, data_list in reference_data.items():
        print(f"동작 {motion_id}: {len(data_list)}개 샘플")

    # 계산된 참조 데이터 캐시 저장
    save_reference_data_cache(reference_data, current_hash)

    return reference_data


# DTW 거리 계산 캐싱을 위한 변수
distance_cache = {}


def dtw_distance(ts1, ts2):
    # 작은 시계열은 캐싱의 오버헤드가 이득보다 클 수 있음
    if ts1.shape[0] <= 10 or ts2.shape[0] <= 10:
        try:
            distance, _ = fastdtw(ts1, ts2, dist=euclidean, radius=3)
            return distance
        except Exception as e:
            print(f"DTW 거리 계산 오류: {e}")
            return float("inf")  # 오류 발생 시 무한대 거리 반환

    # 해시 키 생성
    try:
        key = (hash(ts1.tobytes()), hash(ts2.tobytes()))

        # 캐시에서 결과 확인
        if key in distance_cache:
            return distance_cache[key]

        # 계산 및 캐시
        distance, _ = fastdtw(ts1, ts2, dist=euclidean, radius=3)

        # 캐시 크기 제한 (너무 커지면 초기화)
        if len(distance_cache) > 1000:
            distance_cache.clear()

        distance_cache[key] = distance
        return distance

    except Exception as e:
        print(f"DTW 거리 계산 오류: {e}")
        return float("inf")  # 오류 발생 시 무한대 거리 반환


def normalize_time_series(ts):
    mean = np.mean(ts, axis=0)
    std = np.std(ts, axis=0)
    std[std == 0] = 1.0  # 0으로 나누기 방지
    return (ts - mean) / std


# 페어 파일 처리 함수 (새로 추가 필요)
def load_paired_data(eb_file_path, e9_file_path):
    eb_data = pd.read_csv(eb_file_path)
    e9_data = pd.read_csv(e9_file_path)

    # 타임스탬프 일치 또는 데이터 정렬
    # 두 데이터를 적절히 병합

    # 센서 데이터 구분을 위해 컬럼 이름 변경
    eb_data = eb_data.rename(
        columns={
            "DOT_Acc_X": "EB_DOT_Acc_X",
            "DOT_Acc_Y": "EB_DOT_Acc_Y",
            # 나머지 컬럼들...
        }
    )

    e9_data = e9_data.rename(
        columns={
            "DOT_Acc_X": "E9_DOT_Acc_X",
            "DOT_Acc_Y": "E9_DOT_Acc_Y",
            # 나머지 컬럼들...
        }
    )

    # 데이터 병합
    merged_data = pd.merge(eb_data, e9_data, on="Timestamp", how="outer")
    return merged_data


def classify_with_dtw(
    test_time_series, reference_data, use_normalized=True, weigh_by_type=True
):
    min_distances = {}

    # 사용할 특징 선택 (중요도 높은 것만)
    selected_types = [
        "gyro_pattern",
        "euler_relative",
        "roll_pitch",
        "gyro_smooth",
        "diff_roll_pitch",
        "E9_roll_pitch",
    ]

    # 시계열 유형별 가중치 - 각 동작 고유의 특성을 더 잘 반영하도록 조정
    type_weights = {
        "diff_roll_pitch": 2.0,
        "E9_roll_pitch": 1.0,
        "roll_pitch": 1.2,
        "euler_relative": 1.1,
        "gyro_pattern": 0.5,
        "gyro_smooth": 0.6,
        "acc_direction": 0.6,
        "euler_diff": 0.6,
        "gyro_diff": 0.6,
        "gyro": 0.6,
        "acc_relative": 0.5,
    }

    # 동작별 패널티 가중치 조정
    motion_weights = {
        1: 1.0,
        2: 1.0,
        3: 0.9,
        4: 0.8,
        5: 0.9,
        6: 0.9,
        7: 1.1,
    }

    # 신뢰도 정보 저장
    confidence_scores = {}
    pattern_matches = {}

    test_length = test_time_series.get("length", 0)
    print(f"테스트 데이터 길이: {test_length}")

    for motion_id, reference_list in reference_data.items():
        distances = []
        pattern_match_scores = []

        # 빠른 필터링: 길이가 너무 다른 참조는 건너뛰기
        for ref_idx, ref_ts in enumerate(reference_list):
            ref_length = ref_ts.get("length", 0)
            # 길이 차이가 매우 크면 상세 계산 건너뛰기 (성능 향상)
            if abs(test_length - ref_length) > test_length * 1.5:
                continue

            type_distances = {}

            # 사용 가능한 시계열 유형에 대해 계산 (선택적 계산으로 성능 향상)
            for ts_type in selected_types:  # 중요 특징만 사용
                if ts_type in test_time_series and ts_type in ref_ts:
                    test_data = test_time_series[ts_type]
                    ref_data = ref_ts[ts_type]

                    # 데이터 정규화
                    if use_normalized:
                        test_data = normalize_time_series(test_data)
                        ref_data = normalize_time_series(ref_data)

                    # DTW 거리 계산 - 캐싱된 결과 사용
                    dist = dtw_distance(test_data, ref_data)
                    type_distances[ts_type] = dist

            # 가중 평균 계산
            if type_distances:
                # 패턴 유사성 점수 (특징적 패턴 유형에 대한 일치도)
                pattern_types = ["gyro_pattern", "acc_direction", "euler_relative"]
                pattern_score = 0
                if any(t in type_distances for t in pattern_types):
                    pattern_dists = [
                        type_distances[t] for t in pattern_types if t in type_distances
                    ]
                    pattern_score = 1.0 / (1.0 + np.mean(pattern_dists))
                pattern_match_scores.append(pattern_score)

                if weigh_by_type:
                    weighted_sum = sum(
                        type_distances[t] * type_weights[t] for t in type_distances
                    )
                    weight_sum = sum(type_weights[t] for t in type_distances)
                    avg_dist = weighted_sum / weight_sum
                else:
                    avg_dist = sum(type_distances.values()) / len(type_distances)

                distances.append(avg_dist)

        # 이 동작 유형의 최소 거리
        if distances:
            min_idx = np.argmin(distances)
            min_distances[motion_id] = distances[min_idx]

            # 패턴 매칭 점수
            if pattern_match_scores:
                pattern_matches[motion_id] = max(pattern_match_scores)
                # 거리와 패턴 매칭 점수를 결합한 신뢰도 점수
                confidence = pattern_matches[motion_id] / (
                    1 + np.log1p(min_distances[motion_id])
                )
                confidence_scores[motion_id] = confidence

    # 최소 거리를 가진 동작 유형 선택
    if min_distances:
        # 가중치 적용
        for motion_id in list(min_distances.keys()):
            if motion_id in motion_weights:
                print(f"동작 {motion_id}에 가중치 {motion_weights[motion_id]} 적용")
                min_distances[motion_id] *= motion_weights[motion_id]
                # 신뢰도 점수도 조정
                if motion_id in confidence_scores:
                    confidence_scores[motion_id] /= motion_weights[motion_id]

        # 각 동작 유형별 DTW 거리와 신뢰도 출력
        print("\n== 각 동작 유형별 분석 ==")
        for motion_id in sorted(min_distances.keys(), key=lambda k: min_distances[k]):
            confidence = confidence_scores.get(motion_id, 0)
            pattern = pattern_matches.get(motion_id, 0)
            print(
                f"동작 {motion_id}: 거리={min_distances[motion_id]:.4f}, 패턴={pattern:.4f}, 신뢰도={confidence:.4f}"
            )

        # 신뢰도 기반 선택 (거리가 비슷한 경우 패턴 매칭 점수가 높은 것 선택)
        if confidence_scores:
            best_confidence = max(confidence_scores.items(), key=lambda x: x[1])
            best_distance = min(min_distances.items(), key=lambda x: x[1])

            # 거리 차이가 20% 이내면 신뢰도가 높은 것 선택
            if min_distances[best_confidence[0]] < best_distance[1] * 1.2:
                best_motion_id = best_confidence[0]
                print(
                    f"신뢰도 기반 선택: 동작 {best_motion_id} (신뢰도: {best_confidence[1]:.4f})"
                )
            else:
                best_motion_id = best_distance[0]
                print(
                    f"거리 기반 선택: 동작 {best_motion_id} (거리: {best_distance[1]:.4f})"
                )

            return best_motion_id, f"동작 {best_motion_id}"
        else:
            # 기존 방식 (최소 거리)
            best_motion_id = min(min_distances, key=min_distances.get)
            return best_motion_id, f"동작 {best_motion_id}"

    # 참조 데이터가 없는 경우
    print("주의: 참조 데이터가 없습니다. 기본값 반환")
    return 1, "기본 형태 (참조 데이터 없음)"


# E9 EB 페어링 함수
def find_paired_files(folder_path):
    paired_files = {}

    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    for session_num in range(1, 21):
        session_str = f"Session {session_num}"

        e9_files = [f for f in all_files if "E9" in f and session_str in f]

        eb_files = [f for f in all_files if "EB" in f and session_str in f]

        if e9_files and eb_files:
            paired_files[session_num] = {
                "E9": os.path.join(folder_path, e9_files[0]),
                "EB": os.path.join(folder_path, eb_files[0]),
            }

    print(f"찾은 페어 파일 수: {len(paired_files)}개 세션")
    return paired_files


def check_paired_cache():
    if os.path.exists(REFERENCE_CACHE_FILE):
        try:
            with open(REFERENCE_CACHE_FILE, "rb") as f:
                cache_data = pickle.load(f)
                reference_data = cache_data["reference_data"]
                timestamp = cache_data.get("timestamp", "알 수 없음")

                print(f"\n===== 캐시 파일 정보 =====")
                print(f"생성 시간: {timestamp}")
                print(f"파일 경로: {REFERENCE_CACHE_FILE}")

                paired_count = 0
                unpaired_count = 0

                for motion_id, samples in reference_data.items():
                    paired_sessions = 0
                    unpaired_sessions = 0

                    for sample in samples:
                        if sample.get("sensor_type") == "PAIRED" or "+" in sample.get(
                            "file_path", ""
                        ):
                            paired_sessions += 1
                        else:
                            unpaired_sessions += 1

                    print(f"\n동작 {motion_id} 분석:")
                    print(f"  - 총 샘플 수: {len(samples)}")
                    print(f"  - 페어링된 샘플 수: {paired_sessions}")
                    print(f"  - 페어링되지 않은 샘플 수: {unpaired_sessions}")

                    paired_count += paired_sessions
                    unpaired_count += unpaired_sessions

                    # 세션 번호별 상세 정보
                    session_info = {}
                    for sample in samples:
                        session_num = sample.get("session_num", "알 수 없음")
                        sensor_type = sample.get("sensor_type", "알 수 없음")

                        if session_num not in session_info:
                            session_info[session_num] = []
                        session_info[session_num].append(sensor_type)

                    print("\n  세션별 상세 정보:")
                    for session_num, sensor_types in sorted(session_info.items()):
                        if session_num != "알 수 없음":
                            print(
                                f"    - 세션 {session_num}: {', '.join(sensor_types)}"
                            )

                print(f"\n===== 전체 요약 =====")
                print(f"총 동작 유형: {len(reference_data)}개")
                print(f"총 페어링된 샘플 수: {paired_count}개")
                print(f"총 페어링되지 않은 샘플 수: {unpaired_count}개")

                if paired_count > 0 and unpaired_count == 0:
                    print("\n현재 캐시는 모두 페어링된 샘플로 구성되어 있습니다. ✓")
                elif paired_count > 0:
                    print("\n현재 캐시는 일부 페어링된 샘플을 포함하고 있습니다.")
                    print(
                        "모든 샘플이 페어링되도록 캐시를 재생성하려면 캐시 파일을 삭제하세요."
                    )
                else:
                    print("\n현재 캐시는 페어링된 샘플을 포함하고 있지 않습니다.")
                    print(
                        "E9와 EB 센서 데이터를 페어링하려면 캐시 파일을 삭제하고 다시 실행하세요."
                    )

                return True

        except Exception as e:
            print(f"캐시 파일 확인 중 오류 발생: {e}")
    else:
        print(f"캐시 파일이 존재하지 않습니다: {REFERENCE_CACHE_FILE}")

    return False


def delete_cache():
    if os.path.exists(REFERENCE_CACHE_FILE):
        try:
            os.remove(REFERENCE_CACHE_FILE)
            print(f"캐시 파일이 성공적으로 삭제되었습니다: {REFERENCE_CACHE_FILE}")
            return True
        except Exception as e:
            print(f"캐시 파일 삭제 중 오류 발생: {e}")
    else:
        print(f"삭제할 캐시 파일이 존재하지 않습니다: {REFERENCE_CACHE_FILE}")

    return False


def inspect_cache_structure():
    if os.path.exists(REFERENCE_CACHE_FILE):
        try:
            with open(REFERENCE_CACHE_FILE, "rb") as f:
                cache_data = pickle.load(f)
                reference_data = cache_data["reference_data"]
                timestamp = cache_data.get("timestamp", "알 수 없음")
                file_hash = cache_data.get("file_hash", "알 수 없음")

                print(f"\n===== 캐시 파일 기본 정보 =====")
                print(f"파일 경로: {REFERENCE_CACHE_FILE}")
                print(f"생성 시간: {timestamp}")
                print(f"파일 해시: {file_hash}")

                print(f"\n===== 캐시 데이터 구조 =====")
                print(f"캐시 최상위 키: {list(cache_data.keys())}")
                print(f"동작 유형 수: {len(reference_data)} 개")
                print(f"동작 ID 목록: {list(reference_data.keys())}")

                # 첫 번째 동작의 첫 번째 샘플 상세 출력
                if reference_data:
                    first_motion_id = list(reference_data.keys())[0]
                    first_motion_samples = reference_data[first_motion_id]

                    print(f"\n===== 동작 {first_motion_id} 데이터 구조 =====")
                    print(f"샘플 수: {len(first_motion_samples)}")

                    if first_motion_samples:
                        first_sample = first_motion_samples[0]
                        print(f"\n----- 샘플 데이터 구조 -----")

                        # 메타데이터 출력
                        metadata_keys = [
                            k
                            for k in first_sample.keys()
                            if isinstance(first_sample[k], (str, int, float))
                        ]
                        print(f"메타데이터 키: {metadata_keys}")

                        print("\n메타데이터 예시:")
                        for key in metadata_keys:
                            print(f"  - {key}: {first_sample[key]}")

                        # 시계열 데이터 출력
                        timeseries_keys = [
                            k
                            for k in first_sample.keys()
                            if isinstance(first_sample[k], np.ndarray)
                        ]
                        print(f"\n시계열 데이터 키: {timeseries_keys}")

                        print("\n시계열 데이터 형태:")
                        for key in timeseries_keys:
                            if key in first_sample:
                                array_shape = first_sample[key].shape
                                array_dtype = first_sample[key].dtype
                                print(
                                    f"  - {key}: 형태={array_shape}, 타입={array_dtype}"
                                )

                                # 시계열 데이터의 처음 일부 값 출력
                                if len(first_sample[key]) > 0:
                                    if len(array_shape) > 1 and array_shape[1] > 0:
                                        print(f"    샘플 데이터(처음 3개 행): ")
                                        max_rows = min(3, array_shape[0])
                                        for i in range(max_rows):
                                            print(f"      {first_sample[key][i]}")
                                    else:
                                        max_items = min(5, len(first_sample[key]))
                                        print(
                                            f"    샘플 데이터(처음 {max_items}개): {first_sample[key][:max_items]}"
                                        )

                print("\n===== 전체 샘플 요약 =====")
                for motion_id, samples in reference_data.items():
                    print(f"동작 {motion_id}: {len(samples)}개 샘플")

                    # 각 센서 타입별 개수 집계
                    sensor_counts = {}
                    session_nums = set()

                    for sample in samples:
                        sensor_type = sample.get("sensor_type", "알 수 없음")
                        sensor_counts[sensor_type] = (
                            sensor_counts.get(sensor_type, 0) + 1
                        )

                        # 세션 번호가 있으면 수집
                        if "session_num" in sample:
                            session_nums.add(sample["session_num"])

                    print(f"  - 센서 타입별: {sensor_counts}")
                    if session_nums:
                        print(f"  - 세션 번호: {sorted(session_nums)}")

                return True
        except Exception as e:
            print(f"캐시 파일 검사 중 오류 발생: {e}")
    else:
        print(f"캐시 파일이 존재하지 않습니다: {REFERENCE_CACHE_FILE}")

    return False


if __name__ == "__main__":
    import argparse
    import re

    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="손동작 분류기")
    parser.add_argument("--file", type=str, help="분류할 특정 파일 경로")
    parser.add_argument(
        "--pair", action="store_true", help="E9/EB 파일 페어 분류 모드 사용"
    )
    args = parser.parse_args()

    folder_path = "/Users/yoosehyeok/Documents/RecordingData"

    # 1. 참조 데이터 수집
    print("== 참조 데이터 수집 중... ==")
    reference_data = collect_reference_data(folder_path)

    # 참조 데이터 부재 시 사용자에게 안내
    if not reference_data:
        print("\n주의: 참조 데이터를 찾을 수 없습니다!")
        print("각 동작별로 다음 폴더 구조가 필요합니다:")
        print("  /Documents/RecordingData/1/  (동작 1 폴더)")
        print("  /Documents/RecordingData/2/  (동작 2 폴더)")
        print("  ...등")
        print("또는 session1_*.csv, session2_*.csv 등의 파일이 필요합니다.")
        exit(1)

    # 특정 파일만 분류하는 모드 (--file 인자가 제공된 경우)
    if args.file:
        file_path = args.file
        if os.path.exists(file_path):
            print(f"\n===== 자동 분류 시작: {os.path.basename(file_path)} =====")
            data = load_data(file_path)

            if data is not None:
                # 동작 분류
                test_time_series = extract_time_series(data)
                motion_id, motion_desc = classify_with_dtw(
                    test_time_series, reference_data
                )

                print(f"\n========== 분류 결과 ==========")
                print(f"파일: {os.path.basename(file_path)}")
                print(f"분류된 동작: {motion_id} ({motion_desc})")
                print("================================\n")

                # 특성값 출력
                features = extract_features(data)

                # 중요 특징 그룹화하여 일부만 표시
                print("== 주요 특징값 ==")
                important_features = [
                    "dot_gyro_z_sum",
                    "dot_gyro_z_cumsum",
                    "dot_pitch_change",
                    "dot_pitch_end_diff",
                    "dot_roll_change",
                    "dot_roll_end_diff",
                    "dot_yaw_change",
                    "dot_yaw_end_diff",
                ]

                for feature in important_features:
                    if feature in features:
                        print(f"{feature}: {features[feature]:.4f}")
            else:
                print(f"파일을 로드할 수 없습니다: {file_path}")

            print("\n========== 분류 완료 ==========")
            exit(0)  # 특정 파일 분류 후 종료

    # E9/EB 파일 페어 처리 모드
    elif args.pair or True:  # 기본적으로 페어 모드 사용
        print("\n== E9/EB 페어 테스트 파일 분류 시작 ==")

        # 현재 디렉토리에서 DOT(E9)와 DOT(EB) 파일을 찾아 세션별로 페어링
        all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

        # 세션 번호별로 E9와 EB 파일 매칭
        paired_files = {}
        for file in all_files:
            if "DOT(E9)" in file or "DOT(EB)" in file:
                # 세션 번호 추출 (Session X 패턴)
                session_match = re.search(r"Session\s+(\d+)", file)
                if session_match:
                    session_num = int(session_match.group(1))
                    if session_num not in paired_files:
                        paired_files[session_num] = {}

                    if "DOT(E9)" in file:
                        paired_files[session_num]["E9"] = os.path.join(
                            folder_path, file
                        )
                    elif "DOT(EB)" in file:
                        paired_files[session_num]["EB"] = os.path.join(
                            folder_path, file
                        )

        # 페어링된 파일들 처리
        if paired_files:
            print(f"페어링된 테스트 파일: {len(paired_files)}개 세션")

            for session_num, files in paired_files.items():
                # E9와 EB 파일이 모두 있는지 확인
                if "E9" in files and "EB" in files:
                    e9_file = files["E9"]
                    eb_file = files["EB"]

                    print(f"\n===== 세션 {session_num} 페어 파일 처리 중 =====")
                    print(f"E9 파일: {os.path.basename(e9_file)}")
                    print(f"EB 파일: {os.path.basename(eb_file)}")

                    # 개별 파일 로드
                    e9_data = load_data(e9_file)
                    eb_data = load_data(eb_file)

                    if e9_data is not None and eb_data is not None:
                        try:
                            # 센서 컬럼 이름에 E9, EB 접두어 추가
                            e9_prefixed_cols = {
                                col: f"E9_{col}"
                                for col in e9_data.columns
                                if not col.startswith("E9_")
                            }
                            eb_prefixed_cols = {
                                col: f"EB_{col}"
                                for col in eb_data.columns
                                if not col.startswith("EB_")
                            }

                            e9_data = e9_data.rename(columns=e9_prefixed_cols)
                            eb_data = eb_data.rename(columns=eb_prefixed_cols)

                            # 타임스탬프 기반으로 데이터 정렬
                            # 동일한 길이로 리샘플링 (더 짧은 쪽에 맞춤)
                            min_length = min(len(e9_data), len(eb_data))
                            e9_indices = np.linspace(
                                0, len(e9_data) - 1, min_length
                            ).astype(int)
                            eb_indices = np.linspace(
                                0, len(eb_data) - 1, min_length
                            ).astype(int)

                            e9_data_resampled = e9_data.iloc[e9_indices]
                            eb_data_resampled = eb_data.iloc[eb_indices]

                            # 두 데이터 결합 (인덱스 리셋 후 결합)
                            e9_data_resampled.reset_index(drop=True, inplace=True)
                            eb_data_resampled.reset_index(drop=True, inplace=True)
                            paired_data = pd.concat(
                                [e9_data_resampled, eb_data_resampled], axis=1
                            )

                            print(
                                f"페어링된 데이터 생성 성공 (샘플 수: {len(paired_data)})"
                            )

                            # 결합된 데이터에서 특징 추출 및 분류
                            test_time_series = extract_time_series(paired_data)
                            test_time_series["sensor_type"] = "PAIRED"
                            motion_id, motion_desc = classify_with_dtw(
                                test_time_series, reference_data
                            )

                            print(f"\n========== 분류 결과 ==========")
                            print(f"세션 {session_num} (E9+EB 페어)")
                            print(f"분류된 동작: {motion_id} ({motion_desc})")
                            print("================================\n")

                            # 특성값 출력 - 양쪽 센서 기준으로 표시
                            # print("== E9 센서 주요 특징값 ==")
                            e9_features = extract_features(e9_data)
                            important_features = [
                                "dot_gyro_z_sum",
                                "dot_gyro_z_cumsum",
                                "dot_pitch_change",
                                "dot_pitch_end_diff",
                                "dot_roll_change",
                                "dot_roll_end_diff",
                                "dot_yaw_change",
                                "dot_yaw_end_diff",
                            ]

                            # for feature in important_features:
                            #     if feature in e9_features:
                            #         print(f"{feature}: {e9_features[feature]:.4f}")

                            # print("\n== EB 센서 주요 특징값 ==")
                            eb_features = extract_features(eb_data)
                            # for feature in important_features:
                            #     if feature in eb_features:
                            #         print(f"{feature}: {eb_features[feature]:.4f}")

                        except Exception as e:
                            print(f"페어 파일 처리 오류: {e}")
                            # 페어링 실패 시 개별 처리
                            print("개별 센서 파일로 분류를 시도합니다...")

                            # E9 파일 분류
                            if e9_data is not None:
                                e9_time_series = extract_time_series(e9_data)
                                e9_time_series["sensor_type"] = "E9"
                                e9_motion_id, e9_motion_desc = classify_with_dtw(
                                    e9_time_series, reference_data
                                )
                                print(
                                    f"\nE9 센서 분류 결과: 동작 {e9_motion_id} ({e9_motion_desc})"
                                )

                            # EB 파일 분류
                            if eb_data is not None:
                                eb_time_series = extract_time_series(eb_data)
                                eb_time_series["sensor_type"] = "EB"
                                eb_motion_id, eb_motion_desc = classify_with_dtw(
                                    eb_time_series, reference_data
                                )
                                print(
                                    f"EB 센서 분류 결과: 동작 {eb_motion_id} ({eb_motion_desc})"
                                )
                    else:
                        print(f"파일을 로드할 수 없습니다: {e9_file} 또는 {eb_file}")
                else:
                    print(f"세션 {session_num}의 E9/EB 파일 중 일부가 누락되었습니다.")
        else:
            print("페어링 가능한 테스트 파일(DOT(E9)/DOT(EB))을 찾을 수 없습니다.")
            print("일반 테스트 파일 모드로 전환합니다.")

            # 기존 테스트 파일 분류 로직으로 진행
            test_files = []

            # test*.csv 파일 먼저 찾기
            test_csv_files = glob.glob(os.path.join(folder_path, "test*.csv"))
            if test_csv_files:
                test_files.extend(test_csv_files)

            # 지정된 테스트 파일이 없으면 session*.csv 파일 처리
            if not test_files:
                all_session_files = glob.glob(
                    os.path.join(folder_path, "session*_merged.csv")
                )
                # 참조 데이터에 사용된 파일 확인을 위한 세트 생성
                reference_files = set()
                for motion_files in reference_data.values():
                    for motion_file in motion_files:
                        if "file_path" in motion_file:
                            reference_files.add(motion_file["file_path"])

                # 참조 데이터에 포함되지 않은 세션 파일만 테스트 파일로 사용
                for file_path in all_session_files:
                    if file_path not in reference_files:
                        test_files.append(file_path)

            # 일반 테스트 파일 처리 로직...
            if test_files:
                print(f"\n일반 테스트 파일: {len(test_files)}개")

                # 파일 처리 로직 (기존 코드와 동일)
                # ...
            else:
                print("테스트 파일을 찾을 수 없습니다.")

    print("\n========== 분류 완료 ==========")
