import os
import glob
import re
import shutil


def move_session_files_to_folders():
    # 현재 디렉토리의 모든 세션 파일 찾기
    session_files = glob.glob("session*_merged.csv")

    # 파일명에서 세션 번호를 추출하기 위한 정규식
    session_pattern = re.compile(r"session(\d+)_")

    # 1-7 폴더가 없으면 생성
    for folder_num in range(1, 8):
        folder_path = str(folder_num)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"폴더 생성됨: {folder_path}")

    # 각 폴더의 파일 정보 수집 및 출력
    print("\n== 현재 각 폴더 내 파일 현황 ==")
    folder_last_numbers = {}
    for folder_num in range(1, 8):
        folder_path = str(folder_num)
        folder_files = glob.glob(os.path.join(folder_path, "session*_merged.csv"))

        # 세션 번호 추출 및 정렬
        session_nums = []
        for file_path in folder_files:
            match = session_pattern.search(os.path.basename(file_path))
            if match:
                num = int(match.group(1))
                session_nums.append(num)

        # 세션 번호 정렬
        session_nums.sort()

        # 최대 세션 번호 저장
        max_num = max(session_nums) if session_nums else 0
        folder_last_numbers[folder_num] = max_num

        # 폴더 내 정렬된 파일 목록 출력
        print(f"\n폴더 {folder_num} 파일 목록 (총 {len(session_nums)}개):")
        if session_nums:
            print(f"세션 번호: {', '.join(map(str, session_nums))}")
            print(f"최대 세션 번호: {max_num}")
        else:
            print("파일 없음")

    # 현재 폴더의 세션 파일 정렬
    current_files = []
    for file_path in session_files:
        match = session_pattern.search(file_path)
        if match:
            session_num = int(match.group(1))
            current_files.append((file_path, session_num))

    # 세션 번호 기준 정렬
    current_files.sort(key=lambda x: x[1])

    print("\n== 현재 디렉토리 파일 목록 ==")
    for file_path, session_num in current_files:
        print(f"- {os.path.basename(file_path)} (세션 {session_num})")

    # 이동할 파일 목록 생성
    files_to_move = []
    for file_path, session_num in current_files:
        motion_type = session_num % 7  # 동작 타입 결정 (1-7)
        if motion_type == 0:
            motion_type = 7

        # 대상 폴더의 마지막 세션 번호에서 1 증가
        new_session_num = folder_last_numbers[motion_type] + 1
        folder_last_numbers[motion_type] = new_session_num

        # 새 파일 경로
        new_file_name = os.path.basename(file_path).replace(
            f"session{session_num}_", f"session{new_session_num}_"
        )
        new_file_path = os.path.join(str(motion_type), new_file_name)

        files_to_move.append((file_path, new_file_path, motion_type, new_session_num))

    # 이동할 파일 정보 출력
    print(f"\n== 이동할 파일: {len(files_to_move)}개 ==")

    # 세션 번호 순서로 출력
    for file_path, new_path, motion_type, new_num in sorted(
        files_to_move, key=lambda x: int(session_pattern.search(x[0]).group(1))
    ):
        print(f"- {os.path.basename(file_path)} → {new_path}")
        print(
            f"  (세션 {session_pattern.search(file_path).group(1)} → 동작 타입 {motion_type}, 새 세션 번호: {new_num})"
        )

    # 사용자 확인 없이 즉시 파일 이동 및 이름 변경
    print("\n== 파일 이동 진행 중... ==")
    for file_path, new_path, motion_type, new_num in files_to_move:
        try:
            shutil.move(file_path, new_path)
            print(f"이동 성공: {os.path.basename(file_path)} → {new_path}")
        except Exception as e:
            print(f"이동 실패: {os.path.basename(file_path)} - {str(e)}")

    print("\n파일 이동 및 이름 변경이 완료되었습니다.")

    # 최종 폴더 상태 출력
    print("\n== 작업 후 폴더 상태 ==")
    for folder_num in range(1, 8):
        folder_path = str(folder_num)
        folder_files = glob.glob(os.path.join(folder_path, "session*_merged.csv"))

        # 세션 번호 추출 및 정렬
        session_nums = []
        for file_path in folder_files:
            match = session_pattern.search(os.path.basename(file_path))
            if match:
                num = int(match.group(1))
                session_nums.append(num)

        # 세션 번호 정렬
        session_nums.sort()

        # 폴더 내 정렬된 파일 목록 출력
        print(f"\n폴더 {folder_num} 최종 파일 목록 (총 {len(session_nums)}개):")
        if session_nums:
            print(f"세션 번호: {', '.join(map(str, session_nums))}")
        else:
            print("파일 없음")


if __name__ == "__main__":
    print("세션 파일을 동작 타입별 폴더로 이동합니다...")
    move_session_files_to_folders()
