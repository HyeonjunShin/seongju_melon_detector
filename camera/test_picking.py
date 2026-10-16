import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import posix_ipc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera.gemini336 import D, K
from common.shm_camera import CameraShm
from common.shm_flange import FlangeShm
from common.shm_target import TargetShm

SEM_NAME = "/camera_sem"
SHM_NAME = "camera_shm"
FLANGE_SHM_NAME = "shm_flange"
TARGET_SHM_NAME = "shm_target"

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
COLOR_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
DEPTH_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 1)
PICK_PATCH_SIZE = 15
# Camera pose expressed in the flange frame; translation is given in meters.
T_FLANGE_CAMERA = np.array(
    [
        [0.93969262, 0.0, 0.34202014, 0.07141],
        [0.0, 1.0, 0.0, -0.011],
        [-0.34202014, 0.0, 0.93969262, 0.05256],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

is_running = True


def signal_handler(signum, frame):
    global is_running
    print("\n[Picking Preview] 종료 신호를 수신했습니다...")
    is_running = False


def pixel_to_camera_xyz(u, v, depth_img):
    """Return the camera-frame XYZ in meters for a depth-image pixel."""
    half_patch = PICK_PATCH_SIZE // 2
    height, width = depth_img.shape
    patch = depth_img[
        max(0, v - half_patch) : min(height, v + half_patch + 1),
        max(0, u - half_patch) : min(width, u + half_patch + 1),
    ]
    valid_depths = patch[patch > 0]
    if valid_depths.size == 0:
        return None

    z_m = float(np.median(valid_depths)) / 1000.0

    pixel = np.array([[[u, v]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(pixel, K, D, P=K)[0, 0]
    x_m = (float(undistorted[0]) - K[0, 2]) * z_m / K[0, 0]
    y_m = (float(undistorted[1]) - K[1, 2]) * z_m / K[1, 1]
    return np.array([x_m, y_m, z_m], dtype=np.float64)


def camera_to_base_xyz(camera_xyz_m, base_from_flange):
    """Transform camera XYZ to robot-base XYZ, preserving meters."""
    point_camera_m = np.append(camera_xyz_m, 1.0)
    point_base_m = base_from_flange @ T_FLANGE_CAMERA @ point_camera_m
    return point_base_m[:3]


def main():
    global is_running
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("⏳ camera_runner.py의 Shared Memory 및 Semaphore 연결 중...")
    try:
        camera_shm = CameraShm(
            shm_name=SHM_NAME,
            is_owner=False,
            color_shape=COLOR_SHAPE,
            depth_shape=DEPTH_SHAPE,
        )
    except FileNotFoundError:
        print("❌ Shared Memory를 찾을 수 없습니다. camera_runner.py를 먼저 실행하세요.")
        return

    try:
        camera_sem = posix_ipc.Semaphore(SEM_NAME)
    except posix_ipc.ExistError:
        print("❌ Semaphore를 찾을 수 없습니다. camera_runner.py를 먼저 실행하세요.")
        camera_shm.close()
        return

    try:
        flange_shm = FlangeShm(shm_name=FLANGE_SHM_NAME, is_owner=False)
    except FileNotFoundError:
        print(f"❌ Flange Shared Memory ({FLANGE_SHM_NAME})를 찾을 수 없습니다.")
        camera_sem.close()
        camera_shm.close()
        return

    try:
        target_shm = TargetShm(shm_name=TARGET_SHM_NAME, is_owner=False)
    except FileNotFoundError:
        print(f"❌ Target Shared Memory ({TARGET_SHM_NAME})를 찾을 수 없습니다.")
        flange_shm.close()
        camera_sem.close()
        camera_shm.close()
        return

    target_shm.status = True
    window_name = "Gemini336 - Depth Pixel Picking (XYZ)"
    latest = {"depth": None, "timestamp": None, "selection": None}
    last_sequence_id = 0

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        depth_img = latest["depth"]
        if depth_img is None:
            return

        # The displayed 2x2 preview is half the size of the source montage.
        panel_width = IMAGE_WIDTH // 2
        panel_height = IMAGE_HEIGHT // 2
        if x < panel_width or x >= panel_width * 2:
            print("깊이 화면(오른쪽 위/아래)을 클릭해 주세요.")
            return

        u = min((x - panel_width) * 2 + 1, IMAGE_WIDTH - 1)
        if y < panel_height:
            v = y * 2 + 1
        elif y < panel_height * 2:
            v = (y - panel_height) * 2 + 1
        else:
            return
        v = min(v, IMAGE_HEIGHT - 1)

        xyz = pixel_to_camera_xyz(u, v, depth_img)
        if xyz is None:
            latest["selection"] = (u, v, None, None)
            target_shm.write(
                timestamp=latest["timestamp"],
                detected=False,
                score=0.0,
            )
            print(f"픽셀 ({u}, {v}): 주변에 유효한 depth 값이 없습니다.")
            return

        if not flange_shm.status:
            latest["selection"] = (u, v, xyz, None)
            target_shm.write(
                timestamp=latest["timestamp"],
                detected=False,
                score=0.0,
            )
            print("❌ Flange Shared Memory가 비활성 상태라 target을 전송하지 않았습니다.")
            return

        flange_pose = flange_shm.read(latest["timestamp"])
        if flange_pose.matched_ts <= 0 or not np.isfinite(flange_pose.TF).all():
            latest["selection"] = (u, v, xyz, None)
            target_shm.write(
                timestamp=latest["timestamp"],
                detected=False,
                score=0.0,
            )
            print("❌ 유효한 flange TF를 찾지 못해 target을 전송하지 않았습니다.")
            return

        base_xyz_m = camera_to_base_xyz(xyz, flange_pose.TF.copy())
        # A picked point has no target orientation, so retain the current flange orientation.
        target_tf = flange_pose.TF.copy()
        target_tf[:3, 3] = base_xyz_m
        target_shm.write(
            timestamp=latest["timestamp"],
            detected=True,
            score=1.0,
            TF=target_tf,
        )

        latest["selection"] = (u, v, xyz, base_xyz_m)
        print(
            f"픽셀 ({u}, {v}) -> Camera XYZ: "
            f"X={xyz[0]:.4f} m, Y={xyz[1]:.4f} m, Z={xyz[2]:.4f} m"
        )
        print(
            f"Robot Base XYZ -> X={base_xyz_m[0]:.4f} m, "
            f"Y={base_xyz_m[1]:.4f} m, Z={base_xyz_m[2]:.4f} m "
            f"(flange Δt={flange_pose.time_diff_us} µs)"
        )

    frame_count = 0
    start_time = time.time()

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, on_mouse)
        print(
            "🚀 연결 완료. 오른쪽 Depth 화면을 클릭하면 "
            "카메라/Robot Base 좌표를 계산해 target Shared Memory로 전송합니다."
        )
        print("종료: Q 또는 ESC")

        while is_running:
            try:
                camera_sem.acquire(timeout=1.0)
            except posix_ipc.BusyError:
                continue

            if not camera_shm.status:
                continue

            current_seq = camera_shm.sequence_id
            if current_seq <= last_sequence_id:
                continue

            frame = camera_shm.read()
            last_sequence_id = current_seq

            color_img = frame.color.reshape(COLOR_SHAPE).copy()
            depth_img = frame.depth.reshape(DEPTH_SHAPE).reshape(
                IMAGE_HEIGHT, IMAGE_WIDTH
            ).copy()
            latest["depth"] = depth_img
            latest["timestamp"] = frame.timestamp

            color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

            depth_full = np.where(
                (depth_img > 0) & (depth_img < 10000), depth_img, 0
            )
            depth_full_norm = cv2.normalize(
                depth_full, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_full_colored = cv2.applyColorMap(
                depth_full_norm, cv2.COLORMAP_JET
            )
            depth_full_colored[depth_full == 0] = [0, 0, 0]

            depth_near = np.where(
                (depth_img > 0) & (depth_img < 3000), depth_img, 0
            )
            depth_near_norm = cv2.normalize(
                depth_near, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_near_colored = cv2.applyColorMap(
                depth_near_norm, cv2.COLORMAP_JET
            )
            depth_near_colored[depth_near == 0] = [0, 0, 0]

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                color_bgr,
                f"COLOR (1280x720) | FPS: {fps:.1f}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                depth_full_colored,
                f"DEPTH FULL (10m) | SEQ: {current_seq}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                depth_near_colored,
                f"DEPTH NEAR (3m) | SEQ: {current_seq}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            selection = latest["selection"]
            if selection is not None:
                u, v, xyz, base_xyz_m = selection
                for depth_view in (depth_full_colored, depth_near_colored):
                    cv2.drawMarker(
                        depth_view,
                        (u, v),
                        (255, 255, 255),
                        cv2.MARKER_CROSS,
                        24,
                        2,
                    )
                if xyz is not None:
                    cv2.putText(
                        depth_full_colored,
                        f"XYZ: {xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f} m",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                if base_xyz_m is not None:
                    cv2.putText(
                        depth_full_colored,
                        "BASE XYZ: "
                        f"{base_xyz_m[0]:.3f}, {base_xyz_m[1]:.3f}, "
                        f"{base_xyz_m[2]:.3f} m",
                        (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            top_row = np.hstack((color_bgr, depth_full_colored))
            bottom_row = np.hstack((color_bgr.copy(), depth_near_colored))
            grid = np.vstack((top_row, bottom_row))
            preview = cv2.resize(
                grid,
                (IMAGE_WIDTH, IMAGE_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        print("\n[알림] Ctrl+C 입력으로 중단되었습니다.")
    finally:
        cv2.destroyAllWindows()
        target_shm.status = False
        target_shm.close()
        flange_shm.close()
        camera_shm.close()
        camera_sem.close()
        print("✅ Shared Memory 및 Semaphore 연결을 해제했습니다.")


if __name__ == "__main__":
    main()
