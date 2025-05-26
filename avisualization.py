import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

# 스타일 설정
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "AppleGothic" if os.name != "nt" else "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (12, 9)


def load_sensor_file(file_path):
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"파일 로드 성공: {file_path}, 행 수: {len(df)}")

            # Timestamp 처리
            if "Timestamp" in df.columns:
                if (
                    isinstance(df["Timestamp"].iloc[0], str)
                    and "-" in df["Timestamp"].iloc[0]
                ):
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                    df["Timestamp"] = df["Timestamp"].apply(
                        lambda x: x.timestamp() * 1000
                    )
                else:
                    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")

                # 상대적 시간 계산
                df["Relative_Time"] = df["Timestamp"] - df["Timestamp"].min()
            else:
                print(f"경고: {file_path}에 Timestamp 열이 없습니다.")

            return df
        except Exception as e:
            print(f"파일 로드 오류 ({file_path}): {e}")
            return None
    else:
        print(f"파일을 찾을 수 없음: {file_path}")
        return None


def visualize_sensor_data(df, sensor_type, output_path):
    if df is None or len(df) == 0:
        print(f"{sensor_type} 데이터가 없습니다.")
        return

    # 컬럼명 확인 및 매핑
    acc_columns = []
    gyro_columns = []

    if sensor_type == "DOT":
        if "DOT_Acc_X" in df.columns:
            acc_columns = ["DOT_Acc_X", "DOT_Acc_Y", "DOT_Acc_Z"]
            gyro_columns = ["DOT_Gyro_X", "DOT_Gyro_Y", "DOT_Gyro_Z"]
        elif "Acc_X" in df.columns:  # 접두사 없는 경우
            acc_columns = ["Acc_X", "Acc_Y", "Acc_Z"]
            gyro_columns = ["Gyro_X", "Gyro_Y", "Gyro_Z"]
    else:  # Watch
        if "Watch_Acc_X" in df.columns:
            acc_columns = ["Watch_Acc_X", "Watch_Acc_Y", "Watch_Acc_Z"]
            gyro_columns = ["Watch_Gyro_X", "Watch_Gyro_Y", "Watch_Gyro_Z"]
        elif "Acc_X" in df.columns:  # 접두사 없는 경우
            acc_columns = ["Acc_X", "Acc_Y", "Acc_Z"]
            gyro_columns = ["Gyro_X", "Gyro_Y", "Gyro_Z"]

    if not acc_columns or not gyro_columns:
        print(f"{sensor_type} 센서 데이터를 찾을 수 없습니다.")
        return

    # 색상 설정
    colors = {
        "X": "red",  # X축: 빨강
        "Y": "green",  # Y축: 초록
        "Z": "blue",  # Z축: 파랑
    }

    # 그래프 생성 (3x2 레이아웃)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig.suptitle(f"{sensor_type} 센서 데이터 시각화", fontsize=16)

    # 각 축에 대한 가속도계 데이터 시각화 (왼쪽 열)
    for i, axis in enumerate(["X", "Y", "Z"]):
        axes[i, 0].plot(
            df["Relative_Time"],
            df[acc_columns[i]],
            linewidth=1.5,
            color=colors[axis],
            label=f"{axis}축",
        )
        axes[i, 0].set_title(f"가속도계 {axis}축", fontsize=14)
        axes[i, 0].set_ylabel(f"가속도 (m/s²)", fontsize=12)
        axes[i, 0].grid(True)
        axes[i, 0].legend()
        if i == 2:  # 마지막 행에만 x축 레이블 추가
            axes[i, 0].set_xlabel("상대 시간 (ms)", fontsize=12)

    # 각 축에 대한 자이로스코프 데이터 시각화 (오른쪽 열)
    for i, axis in enumerate(["X", "Y", "Z"]):
        axes[i, 1].plot(
            df["Relative_Time"],
            df[gyro_columns[i]],
            linewidth=1.5,
            color=colors[axis],
            label=f"{axis}축",
        )
        axes[i, 1].set_title(f"자이로스코프 {axis}축", fontsize=14)
        axes[i, 1].set_ylabel(f"각속도 (rad/s)", fontsize=12)
        axes[i, 1].grid(True)
        axes[i, 1].legend()
        if i == 2:  # 마지막 행에만 x축 레이블 추가
            axes[i, 1].set_xlabel("상대 시간 (ms)", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # 제목 공간 확보
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"{sensor_type} 센서 데이터 시각화 저장 완료: {output_path}")


def visualize_specific_files():
    # 현재 스크립트 경로
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "vis")
    os.makedirs(output_dir, exist_ok=True)

    # 시각화할 파일 경로 지정
    watch_file = os.path.join(base_dir, "session2_watch.csv")
    dot_file = os.path.join(base_dir, "session2_dot.csv")

    # Watch 파일 시각화
    if os.path.exists(watch_file):
        print(f"\n워치 센서 파일 시각화 시작: {watch_file}")
        watch_df = load_sensor_file(watch_file)
        output_path = os.path.join(output_dir, "session1_watch_sensors.png")
        visualize_sensor_data(watch_df, "Watch", output_path)
    else:
        print(f"워치 파일을 찾을 수 없음: {watch_file}")

    # DOT 파일 시각화
    if os.path.exists(dot_file):
        print(f"\nDOT 센서 파일 시각화 시작: {dot_file}")
        dot_df = load_sensor_file(dot_file)
        output_path = os.path.join(output_dir, "session1_dot_sensors.png")
        visualize_sensor_data(dot_df, "DOT", output_path)
    else:
        print(f"DOT 파일을 찾을 수 없음: {dot_file}")

    print("\n시각화 작업이 완료되었습니다.")


# 메인 실행 부분
if __name__ == "__main__":
    # 사용자 입력 없이 특정 파일 시각화
    visualize_specific_files()
