import os
import time
import signal
import posix_ipc
from common.shm_camera import CameraShm
from camera.gemini336 import Gemini336

SEM_NAME = "/camera_sem"
SHM_NAME = "camera_shm"
COLOR_SHAPE = (1280, 720, 3)
DEPTH_SHAPE = (1280, 720, 1)

is_running = True


def signal_handler(signum, frame):
    """OS 종료 신호(SIGINT, SIGTERM) 수신 시 루프 정지"""
    global is_running
    print("\n🛑 [Camera Runner] 종료 신호를 수신했습니다. 정리 작업을 진행합니다...")
    is_running = False


def open_camera_ipc() -> tuple[CameraShm, posix_ipc.Semaphore]:
    try:
        camera_shm = CameraShm(
            shm_name=SHM_NAME,
            is_owner=False,
            color_shape=COLOR_SHAPE,
            depth_shape=DEPTH_SHAPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("main.py를 먼저 실행하세요") from exc

    try:
        camera_sem = posix_ipc.Semaphore(SEM_NAME)
    except posix_ipc.ExistentialError as exc:
        camera_shm.close()
        raise RuntimeError("main.py를 먼저 실행하세요") from exc

    return camera_shm, camera_sem


def main():
    global is_running
    is_running = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {0})
        except Exception:
            pass

    try:
        camera_shm, sem = open_camera_ipc()
    except RuntimeError as exc:
        print(f"❌ [Camera Runner] {exc}")
        return

    camera = None
    try:
        camera_shm.status = False
        camera = Gemini336(
            color_shape=COLOR_SHAPE,
            depth_shape=DEPTH_SHAPE,
            settings_path="./config/gemini336_settings.json",
        )
        print("🚀 [Camera Runner] 독립 프로세스 구동 시작 (SHM & Semaphore 연결 완료)")

        while is_running:
            frame = camera.get_frame()
            if frame is None:
                camera_shm.status = False
                time.sleep(0.001)
                continue

            color = frame.get_color_frame()
            depth = frame.get_depth_frame()
            if color is None or depth is None:
                camera_shm.status = False
                time.sleep(0.001)
                continue

            timestamp = depth.get_global_timestamp_us()
            color_data = color.get_data()
            depth_data = depth.get_data()

            # Shared Memory에 프레임 쓰기
            camera_shm.write(timestamp, color_data, depth_data)

            # One pending wake-up is enough: consumers always read the latest
            # snapshot. This channel has a single producer.
            if sem.value == 0:
                sem.release()

            time.sleep(0.001)

    except Exception as e:
        print(f"❌ [Camera Runner] 실행 중 예외 발생: {e}")

    finally:
        print("🛑 [Camera Runner] 리소스 연결 해제 중...")
        camera_shm.status = False

        try:
            if camera is not None:
                camera.stop_camera()
        except Exception as exc:
            print(f"❌ [Camera Runner] 카메라 정지 중 예외 발생: {exc}")
        finally:
            try:
                camera_shm.close()
            finally:
                sem.close()

        print("✅ [Camera Runner] 프로세스 및 자원 해제 완료")


if __name__ == "__main__":
    main()
