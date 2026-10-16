import os
import sys
import time
import signal
import posix_ipc
from common.shm_camera import CameraShm
from camera.gemini336 import Gemini336

SEM_NAME = "/camera_sem"
SHM_NAME = "camera_shm"

is_running = True


def signal_handler(signum, frame):
    """OS 종료 신호(SIGINT, SIGTERM) 수신 시 루프 정지"""
    global is_running
    print("\n🛑 [Camera Runner] 종료 신호를 수신했습니다. 정리 작업을 진행합니다...")
    is_running = False


def main():
    global is_running
    sequence_id = 0

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {0})
        except Exception:
            pass

    # 1. POSIX Named Semaphore 생성 및 초기화 (Owner)
    try:
        try:
            old_sem = posix_ipc.Semaphore(SEM_NAME)
            old_sem.close()
            old_sem.unlink()
        except Exception:
            pass

        sem = posix_ipc.Semaphore(SEM_NAME, flags=posix_ipc.O_CREAT, initial_value=0)
    except Exception as e:
        print(f"❌ [Camera Runner] 세마포어 생성 실패: {e}")
        sys.exit(1)

    # 2. Shared Memory 생성 (is_owner=True)
    color_shape = (1280, 720, 3)
    depth_shape = (1280, 720, 1)

    camera_shm = CameraShm(
        shm_name=SHM_NAME,
        is_owner=True,
        color_shape=color_shape,
        depth_shape=depth_shape,
    )

    settings_path = "./config/gemini336_settings.json"
    camera = Gemini336(
        color_shape=color_shape, depth_shape=depth_shape, settings_path=settings_path
    )

    print("🚀 [Camera Runner] 독립 프로세스 구동 시작 (SHM & Semaphore 생성 완료)")

    try:
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

            camera_shm.status = True

            timestamp = depth.get_global_timestamp_us()
            color_data = color.get_data()
            depth_data = depth.get_data()

            # Shared Memory에 프레임 쓰기
            camera_shm.write(timestamp, color_data, depth_data)

            # 💡 [수정 완료] @property setter 형태 대입 구문으로 변경
            sequence_id += 1
            camera_shm.sequence_id = sequence_id

            # 세마포어 신호(Signal) 전송 -> 수신부 프로세스 깨우기
            sem.release()

            time.sleep(0.001)

    except Exception as e:
        print(f"❌ [Camera Runner] 실행 중 예외 발생: {e}")

    finally:
        print("🛑 [Camera Runner] 리소스 및 메모리 완전히 삭제 중...")
        camera_shm.status = False

        if hasattr(camera, "stop_camera"):
            camera.stop_camera()
        elif hasattr(camera, "pipeline") and camera.pipeline:
            try:
                camera.pipeline.stop()
            except Exception:
                pass

        camera_shm.close()
        try:
            sem.close()
            sem.unlink()
        except Exception:
            pass

        print("✅ [Camera Runner] 프로세스 및 자원 해제 완료")


if __name__ == "__main__":
    main()
