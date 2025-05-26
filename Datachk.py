import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from datetime import datetime


def analyze_sampling_frequency(csv_file):
    print(f"파일 '{csv_file}' 분석 중...")

    # 파일 존재 확인
    if not os.path.isfile(csv_file):
        print(f"오류: '{csv_file}' 파일을 찾을 수 없습니다.")
        return

    try:
        # 타임스탬프 열만 로드 (문자열로 읽기)
        df = pd.read_csv(csv_file, usecols=["Timestamp"])

        # 타임스탬프 형식 확인 및 변환
        print("\n타임스탬프 변환 전 첫 5개 샘플:")
        print(df.head())

        # 문자열로 타임스탬프 직접 처리
        timestamps = []
        for ts in df["Timestamp"]:
            try:
                dt = pd.to_datetime(ts)
                timestamps.append(dt)
            except:
                print(f"변환 실패한 타임스탬프: {ts}")
                timestamps.append(None)

        df["Timestamp_dt"] = timestamps
        df = df.dropna(subset=["Timestamp_dt"])

        # 타임스탬프를 마이크로초 단위로 변환
        df["Timestamp_us"] = df["Timestamp_dt"].apply(lambda x: x.timestamp() * 1000000)

        # 정렬
        df = df.sort_values("Timestamp_us")

        # 타임스탬프 간 차이 계산 (마이크로초)
        df["Diff_us"] = df["Timestamp_us"].diff()
        df = df.dropna(subset=["Diff_us"])

        # 마이크로초를 밀리초로 변환
        df["Diff_ms"] = df["Diff_us"] / 1000

        # 기본 통계량 계산
        avg_diff = df["Diff_ms"].mean()
        median_diff = df["Diff_ms"].median()
        std_diff = df["Diff_ms"].std()
        min_diff = df["Diff_ms"].min()
        max_diff = df["Diff_ms"].max()

        # 0ms 간격이 있는지 확인
        zero_intervals = (df["Diff_ms"] < 0.001).sum()
        if zero_intervals > 0:
            print(
                f"⚠️ 경고: {zero_intervals}개의 샘플이 동일한 타임스탬프를 가지고 있습니다."
            )

        # 음수 간격이 있는지 확인
        neg_intervals = (df["Diff_ms"] < 0).sum()
        if neg_intervals > 0:
            print(
                f"⚠️ 경고: {neg_intervals}개의 샘플이 음수 시간 간격을 가지고 있습니다. 정렬에 문제가 있을 수 있습니다."
            )

        # 실제 샘플링 주파수 계산 (Hz = 샘플/초)
        if avg_diff > 0:
            actual_frequency = 1000 / avg_diff
            median_frequency = 1000 / median_diff
        else:
            # 시간 간격이 0이면 전체 데이터 길이로 계산
            first_ts = df["Timestamp_dt"].min()
            last_ts = df["Timestamp_dt"].max()
            duration_seconds = (last_ts - first_ts).total_seconds()
            if duration_seconds > 0:
                actual_frequency = len(df) / duration_seconds
                median_frequency = actual_frequency  # 시간 간격이 0일 때는 같은 값 사용
            else:
                actual_frequency = float("nan")
                median_frequency = float("nan")

        # 결과 출력
        print("\n==== 샘플링 주파수 분석 결과 ====")
        print(f"총 샘플 수: {len(df)}")
        print(f"첫 타임스탬프: {df['Timestamp_dt'].min()}")
        print(f"마지막 타임스탬프: {df['Timestamp_dt'].max()}")
        print(
            f"전체 녹화 시간: {(df['Timestamp_dt'].max() - df['Timestamp_dt'].min()).total_seconds():.2f}초"
        )

        print(
            f"\n평균 시간 간격: {avg_diff:.4f} ms (주파수: {actual_frequency:.2f} Hz)"
        )
        print(
            f"중간값 시간 간격: {median_diff:.4f} ms (주파수: {median_frequency:.2f} Hz)"
        )
        print(f"시간 간격 표준편차: {std_diff:.4f} ms")
        print(f"최소/최대 시간 간격: {min_diff:.4f} ms / {max_diff:.4f} ms")

        # 계산된 샘플링 주파수로 주파수 범위 결정
        if not np.isnan(actual_frequency):
            # 가장 가까운 일반적인 샘플링 주파수 찾기
            common_frequencies = [
                10,
                20,
                25,
                30,
                50,
                60,
                100,
                120,
                144,
                200,
                240,
                250,
                500,
                1000,
            ]
            closest_freq = min(
                common_frequencies, key=lambda x: abs(x - actual_frequency)
            )

            # 오차 계산
            error_percent = abs(closest_freq - actual_frequency) / closest_freq * 100

            print(
                f"\n가장 가까운 표준 샘플링 주파수: {closest_freq} Hz (오차: {error_percent:.2f}%)"
            )
            if error_percent <= 5:
                print(
                    f"✓ 데이터는 {closest_freq}Hz에 매우 가깝게 샘플링된 것으로 보입니다."
                )
            elif error_percent <= 15:
                print(
                    f"△ 데이터는 대략 {closest_freq}Hz로 샘플링되었을 가능성이 있습니다."
                )
            else:
                print(f"✗ 데이터는 표준 샘플링 주파수와 맞지 않습니다.")

        # 초당 샘플 수 분석
        df["second_group"] = df["Timestamp_dt"].dt.floor("1s")
        samples_per_second = df.groupby("second_group").size()

        print(f"\n평균 초당 샘플 수: {samples_per_second.mean():.2f}")
        print(
            f"초당 샘플 수 범위: {samples_per_second.min()} ~ {samples_per_second.max()}"
        )
        print(f"초당 샘플 수 표준편차: {samples_per_second.std():.2f}")

        # 시각화
        plt.figure(figsize=(12, 10))

        # 1. 시간 간격 히스토그램
        plt.subplot(2, 2, 1)
        plt.hist(df["Diff_ms"], bins=30, color="skyblue", edgecolor="black", alpha=0.7)
        plt.axvline(
            avg_diff,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"평균: {avg_diff:.4f}ms ({actual_frequency:.2f}Hz)",
        )
        plt.xlabel("시간 간격 (ms)")
        plt.ylabel("빈도")
        plt.title("샘플 간 시간 간격 분포")
        plt.legend()

        # 2. 시간에 따른 간격 변화
        plt.subplot(2, 2, 2)
        plt.plot(df["Timestamp_dt"], df["Diff_ms"], "o-", markersize=2, alpha=0.5)
        plt.axhline(avg_diff, color="red", linestyle="dashed", linewidth=2)
        plt.xlabel("시간")
        plt.ylabel("시간 간격 (ms)")
        plt.title("시간에 따른 샘플링 간격 변화")

        # 3. 초당 샘플 수 변화
        plt.subplot(2, 2, 3)
        seconds = range(len(samples_per_second))
        plt.bar(seconds, samples_per_second, alpha=0.7)
        plt.axhline(
            samples_per_second.mean(),
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"평균: {samples_per_second.mean():.2f} 샘플/초",
        )
        plt.xlabel("시간 구간 (초)")
        plt.ylabel("샘플 수")
        plt.title("초당 샘플 수")
        plt.legend()

        # 4. 누적 샘플 수
        plt.subplot(2, 2, 4)
        df["sample_count"] = range(1, len(df) + 1)
        plt.plot(df["Timestamp_dt"], df["sample_count"])
        plt.xlabel("시간")
        plt.ylabel("누적 샘플 수")
        plt.title("누적 샘플 수 증가")

        plt.tight_layout()
        plt.show()

        # 결론
        print("\n==== 결론 ====")
        if not np.isnan(actual_frequency):
            if abs(actual_frequency - closest_freq) / closest_freq <= 0.05:
                print(f"✓ 데이터는 약 {closest_freq}Hz로 수집된 것으로 판단됩니다.")
            else:
                print(f"데이터는 약 {actual_frequency:.2f}Hz로 수집되었습니다.")
        else:
            print(
                "샘플링 주파수를 결정할 수 없습니다. 타임스탬프 데이터에 문제가 있을 수 있습니다."
            )

        return actual_frequency

    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    analyze_sampling_frequency("session1.csv")
