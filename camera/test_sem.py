import os
import signal
import sys
import time
from buffer import CameraShm
import cv2
import numpy as np
import posix_ipc

# 현재 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 세마포어 및 공유 메모리 이름 설정 (camera_runner.py와 일치)
SEM_NAME = "/camera_sem"
SHM_NAME = "camera_shm"

COLOR_SHAPE = (720, 1280, 3)
DEPTH_SHAPE = (720, 1280, 1)

is_running = True


def signal_handler(signum, frame):
    global is_running
    print("\n🛑 [Test Consumer] 종료 신호를 수신했습니다...")
    is_running = False


def main():
    global is_running

    # OS 종료 신호 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("⏳ Shared Memory 및 Semaphore 연결 시도 중...")

    # 1. 공유 메모리 연결 (Consumer 역: is_owner=False)
    try:
        camera_shm = CameraShm(
            shm_name=SHM_NAME,
            is_owner=False,
            color_shape=COLOR_SHAPE,
            depth_shape=DEPTH_SHAPE,
        )
    except FileNotFoundError:
        print(f"❌ [Error] Shared Memory ({SHM_NAME})를 찾을 수 없습니다.")
        print("💡 camera_runner.py가 먼저 실행되어 있는지 확인해 주세요.")
        return

    # 2. POSIX 세마포어 연결
    try:
        camera_sem = posix_ipc.Semaphore(SEM_NAME)
    except posix_ipc.ExistError:
        print(f"❌ [Error] Semaphore ({SEM_NAME})를 찾을 수 없습니다.")
        camera_shm.close()
        return

    window_name = "Orbbec Gemini336 - Live Preview (Color & Depth)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("🚀 [Test Consumer] 구동 성공! 카메라 데이터 대기 중...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" ▶ 'Q' 또는 'ESC'를 누르거나 Ctrl+C를 누르면 종료됩니다.")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    real_frame_count = 0
    start_time = time.time()
    last_sequence_id = 0

    try:
        while is_running:
            # 💡 카메라 프로세스 신호 대기 (CPU 점유율 0%)
            try:
                camera_sem.acquire(timeout=1.0)
            except posix_ipc.BusyError:
                continue

            # 💡 [제품 조건 1] 카메라 프로세스/하드웨어 정상 동작 가드
            if not camera_shm.status:
                continue

            # 💡 [제품 조건 2] sequence_id 기반 최신 프레임 갱신 여부 검증
            current_seq = camera_shm.sequence_id
            if current_seq <= last_sequence_id:
                # 세마포어 백로그 또는 중복 접근 시 무시 (46 FPS 오버카운팅 방지)
                continue

            # 공유 메모리에서 최신 프레임 수신 및 시퀀스 최신화
            frame = camera_shm.read()
            last_sequence_id = current_seq

            color_data = frame.color
            depth_data = frame.depth

            # (720, 1280) Reshaping
            color_img = color_data.reshape((720, 1280, 3))
            depth_img = depth_data.reshape((720, 1280))

            # 1. Color (RGB -> BGR)
            color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

            # 2-1. 상단 Depth (전체 스케일: 0 ~ 10m)
            depth_top_filtered = np.where(
                (depth_img > 0) & (depth_img < 10000), depth_img, 0
            )
            depth_top_norm = cv2.normalize(
                depth_top_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_top_colored = cv2.applyColorMap(depth_top_norm, cv2.COLORMAP_JET)
            depth_top_colored[depth_top_filtered == 0] = [0, 0, 0]

            # 2-2. 하단 Depth (근거리 서보 정밀 스케일: 0 ~ 3m)
            depth_bot_filtered = np.where(
                (depth_img > 0) & (depth_img < 3000), depth_img, 0
            )
            depth_bot_norm = cv2.normalize(
                depth_bot_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_bot_colored = cv2.applyColorMap(depth_bot_norm, cv2.COLORMAP_JET)
            depth_bot_colored[depth_bot_filtered == 0] = [0, 0, 0]

            # 실제 유효 FPS 계산 (새 프레임 수신 건수 기반)
            real_frame_count += 1
            elapsed_time = time.time() - start_time
            fps = real_frame_count / elapsed_time if elapsed_time > 0 else 0

            # 텍스트 오버레이 박스 적용 (SEQ 정보 표시 추가)
            color_bgr_top = color_bgr.copy()
            cv2.rectangle(color_bgr_top, (10, 10), (580, 60), (0, 0, 0), -1)
            cv2.putText(
                color_bgr_top,
                f"COLOR (1280x720) | FPS: {fps:.1f}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.rectangle(depth_top_colored, (10, 10), (750, 60), (0, 0, 0), -1)
            cv2.putText(
                depth_top_colored,
                f"DEPTH FULL (10m) | SEQ: {current_seq}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            color_bgr_bot = color_bgr.copy()
            cv2.rectangle(color_bgr_bot, (10, 10), (580, 60), (0, 0, 0), -1)
            cv2.putText(
                color_bgr_bot,
                f"COLOR (1280x720) | FPS: {fps:.1f}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.rectangle(depth_bot_colored, (10, 10), (750, 60), (0, 0, 0), -1)
            cv2.putText(
                depth_bot_colored,
                f"DEPTH NEAR (3m) | SEQ: {current_seq}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 2x2 화면 병합
            top_row = np.hstack((color_bgr_top, depth_top_colored))
            bottom_row = np.hstack((color_bgr_bot, depth_bot_colored))
            grid_2x2 = np.vstack((top_row, bottom_row))

            cv2.imshow(window_name, grid_2x2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\n[알림] 사용자가 시각화 종료를 요청했습니다.")
                break

    except KeyboardInterrupt:
        print("\n[알림] Ctrl + C 입력으로 중단되었습니다.")

    finally:
        print("🛑 리소스 해제 중...")
        cv2.destroyAllWindows()

        # Consumer는 close()만 호출하고 unlink()는 수행하지 않습니다.
        camera_shm.close()
        try:
            camera_sem.close()
        except Exception:
            pass

        print("✅ [Test Consumer] 프로세스가 정상적으로 종료되었습니다.")


if __name__ == "__main__":
    main()
