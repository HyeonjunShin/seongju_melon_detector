import time
import signal
import posix_ipc

from common.shm_camera import (
    CameraShm,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    COLOR_SHAPE as SHM_COLOR_SHAPE,
    DEPTH_SHAPE as SHM_DEPTH_SHAPE,
)
from camera.gemini import Gemini

SEM_NAME = "/camera_sem"
SHM_NAME = "camera_shm"

COLOR_SHAPE = (CAMERA_WIDTH, CAMERA_HEIGHT, 3)
DEPTH_SHAPE = (CAMERA_WIDTH, CAMERA_HEIGHT, 1)

CONFIG_PATH = "./config/gemini355_settings.json"

is_running = True


def signal_handler(signum, frame):
    global is_running

    print("\n[Camera Runner] 종료 신호 수신. " "정리 작업을 시작합니다.")

    is_running = False


def open_camera_ipc():
    try:
        camera_shm = CameraShm(
            shm_name=SHM_NAME,
            is_owner=False,
            color_shape=SHM_COLOR_SHAPE,
            depth_shape=SHM_DEPTH_SHAPE,
        )

    except FileNotFoundError as exc:
        raise RuntimeError(
            "Shared Memory가 없습니다. main.py를 먼저 실행하세요."
        ) from exc

    try:
        camera_sem = posix_ipc.Semaphore(SEM_NAME)

    except posix_ipc.ExistentialError as exc:
        camera_shm.close()
        raise RuntimeError("Semaphore가 없습니다. main.py를 먼저 실행하세요.") from exc

    return camera_shm, camera_sem


def main():
    global is_running
    is_running = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        camera_shm, sem = open_camera_ipc()

    except RuntimeError as exc:
        print(f"[Camera Runner] {exc}")
        return

    try:
        camera_shm.status = False

        print("[Camera Runner] Shared Memory와 " "Semaphore 연결 완료")

        with Gemini(
            color_shape=COLOR_SHAPE,
            depth_shape=DEPTH_SHAPE,
            config_path=CONFIG_PATH,
        ) as camera:

            camera_shm.intrinsics = camera.get_color_calibration()
            print("[Camera Runner] Gemini Pipeline 시작 완료")

            previous_timestamp = None
            written_count = 0
            last_stats_time = time.perf_counter()

            while is_running:

                # 새 RGB-D 프레임이 들어올 때까지 대기
                frame = camera.wait_for_new_frame(
                    previous_timestamp=previous_timestamp,
                    timeout=0.1,
                )

                if frame is None:
                    # 데이터가 없다고 매번 카메라를
                    # 비정상 상태로 처리하지 않음.
                    continue

                color, depth, timestamp_color, timestamp_depth = frame

                # 이미 처리한 프레임이면 건너뛰기
                if timestamp_color == previous_timestamp:
                    continue

                # Shape / 데이터 타입 확인 (초기 디버깅용)
                # print(color.shape, color.dtype)
                # print(depth.shape, depth.dtype)

                # Shared Memory 기록
                camera_shm.write(
                    timestamp_color,
                    timestamp_depth,
                    color,
                    depth,
                )

                # 기록 성공 후 이전 Timestamp 업데이트
                previous_timestamp = timestamp_color

                # 최초 프레임 기록 이후 데이터 준비 완료
                camera_shm.status = True

                written_count += 1

                # 소비자 프로세스 알림
                if sem.value == 0:
                    sem.release()

                # 1초마다 실제 Shared Memory 쓰기 FPS 표시
                now = time.perf_counter()

                elapsed = now - last_stats_time

                if elapsed >= 1.0:
                    write_fps = written_count / elapsed

                    print(
                        f"[Camera Runner] "
                        f"SHM Write FPS: {write_fps:.2f} | "
                        f"Callback Frames: {camera.received_count}"
                    )

                    written_count = 0
                    last_stats_time = now

    except Exception as e:
        print(f"[Camera Runner] 실행 중 오류: {e}")
        raise

    finally:
        print("[Camera Runner] 리소스 정리 시작")

        try:
            camera_shm.status = False
        except Exception as exc:
            print(f"[Warning] Status update failed: {exc}")

        try:
            camera_shm.close()
        except Exception as exc:
            print(f"[Warning] SHM close failed: {exc}")

        try:
            sem.close()
        except Exception as exc:
            print(f"[Warning] Semaphore close failed: {exc}")

        print("[Camera Runner] 종료 완료")


if __name__ == "__main__":
    main()
