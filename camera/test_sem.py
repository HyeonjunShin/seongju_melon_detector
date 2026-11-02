import signal
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import posix_ipc

# Support both python -m camera.test_sem and direct script execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.shm_camera import (
    CameraShm, CAMERA_WIDTH, CAMERA_HEIGHT, COLOR_SHAPE, DEPTH_SHAPE,
)

# 세마포어 및 공유 메모리 이름 설정 (camera_runner.py와 일치)
SEM_NAME = "/camera_sem"
SHM_NAME = "camera_shm"

is_running = True


def signal_handler(signum, frame):
    global is_running
    print("\n🛑 [Test Consumer] 종료 신호를 수신했습니다...")
    is_running = False


def render_preview(frame, fps, scale=0.5):
    """Downsample before color mapping; retain RGB/full-depth/near-depth layout."""
    height, width = frame.color.shape[:2]
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    color = frame.color
    depth = frame.depth.reshape(height, width)
    if size != (width, height):
        color = cv2.resize(color, size, interpolation=cv2.INTER_AREA)
        # Nearest-neighbor preserves invalid depth and surface boundaries.
        depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    width, height = size
    grid = np.empty((height * 2, width * 2, 3), np.uint8)
    for row, limit, label in ((0, 10., "FULL"), (1, 3., "NEAR")):
        left = grid[row * height:(row + 1) * height, :width]
        right = grid[row * height:(row + 1) * height, width:]
        left[:] = color
        filtered = np.where((depth > 0) & (depth < limit), depth, 0)
        normalized = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        right[:] = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        right[filtered == 0] = 0
        font_scale = .9 * scale
        for image, text, ink in (
            (left, f"COLOR ({frame.color.shape[1]}x{frame.color.shape[0]}) | FPS: {fps:.1f}", (0, 255, 0)),
            (right, f"DEPTH {label} ({limit:g}m) | SEQ: {frame.sequence_id}", (255, 255, 255)),
        ):
            cv2.rectangle(image, (0, 0), (width, max(25, round(60 * scale))), (0, 0, 0), -1)
            cv2.putText(image, text, (max(5, round(20 * scale)), max(18, round(45 * scale))),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, ink,
                        max(1, round(2 * scale)), cv2.LINE_AA)
    return grid


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RGB-D preview with per-stage timing")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Tile scale: default .5 gives an 848x530 grid; 1 gives 1696x1060")
    args = parser.parse_args()
    if not 0 < args.scale <= 1:
        parser.error("--scale must be in (0, 1]")
    global is_running
    is_running = True

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
        print("💡 main.py 실행 후 camera_runner.py를 실행해 주세요.")
        return

    except RuntimeError as exc:
        print(f"❌ {exc}")
        return

    # 2. POSIX 세마포어 연결
    try:
        camera_sem = posix_ipc.Semaphore(SEM_NAME)
    except posix_ipc.ExistentialError:
        print(f"❌ [Error] Semaphore ({SEM_NAME})를 찾을 수 없습니다.")
        camera_shm.close()
        return

    window_name = "Gemini - Live Preview (Color & Depth)"

    print("🚀 [Test Consumer] 구동 성공! 카메라 데이터 대기 중...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" ▶ 'Q' 또는 'ESC'를 누르거나 Ctrl+C를 누르면 종료됩니다.")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    real_frame_count = 0
    start_time = time.perf_counter()
    fps = 0.
    timings = dict(wait=0., shm=0., render=0., gui=0.)
    skipped = 0
    last_sequence_id = 0

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, round(CAMERA_WIDTH * args.scale) * 2,
                         round(CAMERA_HEIGHT * args.scale) * 2)
        while is_running:
            waiting = time.perf_counter()
            try:
                camera_sem.acquire(timeout=0.1)
            except posix_ipc.BusyError:
                timings['wait'] += (time.perf_counter() - waiting) * 1000
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

            reading = time.perf_counter()
            timings['wait'] += (reading - waiting) * 1000
            frame = camera_shm.read(last_sequence_id=last_sequence_id)
            rendering = time.perf_counter()
            timings['shm'] += (rendering - reading) * 1000
            if frame is None:
                continue
            if last_sequence_id:
                skipped += max(0, frame.sequence_id - last_sequence_id - 1)
            last_sequence_id = frame.sequence_id
            grid_2x2 = render_preview(frame, fps, args.scale)
            displaying = time.perf_counter()
            timings['render'] += (displaying - rendering) * 1000
            cv2.imshow(window_name, grid_2x2)
            key = cv2.waitKey(1) & 0xFF
            now = time.perf_counter()
            timings['gui'] += (now - displaying) * 1000
            real_frame_count += 1
            elapsed = now - start_time
            if elapsed >= 1.:
                fps = real_frame_count / elapsed
                detail = ' | '.join(f'{name}={value / real_frame_count:.2f}ms'
                                    for name, value in timings.items())
                print(f'[Preview] FPS={fps:.1f} | {detail} | skipped={skipped}', flush=True)
                real_frame_count = 0
                start_time = now
                timings = dict.fromkeys(timings, 0.)
                skipped = 0

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
