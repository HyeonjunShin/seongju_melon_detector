"""Collect RGB/depth PNG pairs: python training/data_collection.py."""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.gemini336 import Gemini336

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "gemini336_settings.json"


def setup_save_directory(base_path="/media/uon/Samsung USB/melon_pack"):
    """시작 시 USB 저장 위치를 확인하고 rgb/depth 폴더를 생성합니다."""
    print(f"\n[저장 위치 확인 중]: {base_path}")

    if not os.path.exists(base_path):
        print(f"❌ [오류] 저장 경로가 존재하지 않습니다: {base_path}")
        print("USB 마운트 상태를 확인하거나 경로를 올바르게 입력했는지 검토하세요.")
        sys.exit(1)

    rgb_dir = os.path.join(base_path, "rgb")
    depth_dir = os.path.join(base_path, "depth")

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    print(f"✅ [준비 완료] 이미지 저장 폴더:")
    print(f"  - RGB   : {rgb_dir}")
    print(f"  - Depth : {depth_dir}\n")
    return rgb_dir, depth_dir


def main():
    # 1. 저장 경로 확인 및 생성
    rgb_dir, depth_dir = setup_save_directory("/media/uon/jsh/melon_data")

    camera = Gemini336(
        color_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
        depth_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1),
        settings_path=SETTINGS_PATH,
    )

    # 데이터 수집 관련 주요 변수 및 임계값
    is_recording = False
    save_count = 0
    prev_gray = None

    # [설정 1] 선명도 태깅 임계값 (라플라시안 분산)
    BLUR_THRESHOLD = 100.0

    # [설정 2] 카메라 이동 임계값 (중복 프레임 저장 방지)
    MOVEMENT_THRESHOLD = 8.0

    try:
        print("==================================================")
        print("  [SPACEBAR] : 자동 데이터 수집 시작 / 일시정지")
        print("  [q]        : 종료")
        print("  [수집 정책] : 모션 블러 포함, 화면 변화 감지 시 자동 저장")
        print(f"  [변화량 기준] : Frame Diff >= {MOVEMENT_THRESHOLD}")
        print("==================================================\n")

        while True:
            frame = camera.get_frame()
            if frame is None:
                continue
            color_frame = frame.get_color_frame()
            depth_frame = frame.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            timestamp_us = depth_frame.get_global_timestamp_us()
            color = np.frombuffer(color_frame.get_data(), dtype=np.uint8).reshape(
                IMAGE_HEIGHT, IMAGE_WIDTH, 3
            )
            depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

            # 1. 계산용 그레이스케일 및 시각화용 이미지 준비
            current_gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            view = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            # 2. 선명도 측정 (라플라시안 분산)
            sharpness = cv2.Laplacian(current_gray, cv2.CV_64F).var()
            is_sharp = sharpness >= BLUR_THRESHOLD
            blur_tag = "sharp" if is_sharp else "blur"

            # 3. 이전 프레임 대비 변화량 계산 (Mean Absolute Difference)
            motion_score = 0.0
            if prev_gray is not None:
                frame_diff = cv2.absdiff(current_gray, prev_gray)
                motion_score = float(np.mean(frame_diff))

            # 4. 수집 조건 충족 시 저장 (녹화 활성화 & 카메라 이동 발생)
            if is_recording and (motion_score >= MOVEMENT_THRESHOLD):
                file_stem = f"melon_{timestamp_us}_{blur_tag}"

                rgb_path = os.path.join(rgb_dir, f"{file_stem}.png")
                depth_png_path = os.path.join(depth_dir, f"{file_stem}.png")

                # UI 글씨를 그리기 전의 원본 RGB와 uint16 depth를 저장합니다.
                cv2.imwrite(rgb_path, view)
                cv2.imwrite(depth_png_path, depth)

                save_count += 1
                print(
                    f"📸 [{save_count}] 저장 ({blur_tag.upper()}) | Sharpness: {sharpness:.1f} | Diff: {motion_score:.1f} -> {file_stem}"
                )

                # 기준 프레임 갱신
                prev_gray = current_gray.copy()

            elif prev_gray is None:
                prev_gray = current_gray.copy()

            # 5. UI 가이드 표시
            status_text = (
                f"REC | Saved: {save_count} | Sharp: {sharpness:.0f} ({blur_tag.upper()}) | Diff: {motion_score:.1f}"
                if is_recording
                else f"PAUSED | Saved: {save_count}"
            )
            status_color = (0, 255, 0) if is_recording else (255, 255, 0)

            cv2.putText(view, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            cv2.imshow("color", view)
            key = cv2.waitKey(1)

            # [SPACEBAR] : 수집 시작 / 일시정지 토글
            if key == 32:
                is_recording = not is_recording
                state_str = "시작" if is_recording else "일시정지"
                print(f"\n▶️ [자동 수집 {state_str}]")

            elif key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[알림] 사용자에 의해 Ctrl + C가 입력되었습니다.")

    finally:
        camera.stop_camera()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
