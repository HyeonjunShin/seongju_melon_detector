import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import posix_ipc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.shm_camera import (
    CameraShm,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    COLOR_SHAPE,
    DEPTH_SHAPE,
)
from common.shm_flange import FlangeShm
from common.shm_target import TargetShm

SEM_NAME = "/camera_sem"
SHM_NAME = "camera_shm"
FLANGE_SHM_NAME = "flange_shm"
TARGET_SHM_NAME = "target_shm"

IMAGE_WIDTH = CAMERA_WIDTH
IMAGE_HEIGHT = CAMERA_HEIGHT
# Filled from the running camera's calibration before accepting any clicks.
K = D = None
DEPTH_PREVIEW_WIDTH = 320
DEPTH_PREVIEW_HEIGHT = round(DEPTH_PREVIEW_WIDTH * IMAGE_HEIGHT / IMAGE_WIDTH)
DEPTH_PREVIEW_X = IMAGE_WIDTH - DEPTH_PREVIEW_WIDTH - 16
DEPTH_PREVIEW_Y = 16
PICK_PATCH_SIZE = 30
NORMAL_PATCH_SIZE = 31
NORMAL_FILTERS_ENABLED = False  # Disable depth-band and plane-quality filtering.
NORMAL_DEPTH_BAND_M = 0.03  # Exclude points far from the local surface depth.
NORMAL_MAX_RMS_M = 0.004  # Maximum plane-fit RMS residual (4 mm).
NORMAL_MAX_THICKNESS_RATIO = 0.35
NORMAL_MIN_POINTS = 9
TARGET_Z_OFFSET_M = 0.0  # Offset along robot-base +Z, after coordinate conversion.
# Camera pose expressed in the flange frame; translation is given in meters.
T_FLANGE_CAMERA = np.array(
    [
        [0.0, -0.819152, 0.573576, 0.09445],
        [1.0, 0.0, -0.0, -0.0449],
        [0.0, 0.573576, 0.819152, 0.06003],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

is_running = True


def signal_handler(signum, frame):
    global is_running
    print("\n[Picking Preview] 종료 신호를 수신했습니다...")
    is_running = False


def pixel_to_camera_xyz(u, v, depth_img, camera_matrix=None, distortion=None):
    """Return the camera-frame XYZ in meters for a depth-image pixel."""
    camera_matrix = K if camera_matrix is None else camera_matrix
    distortion = D if distortion is None else distortion
    half_patch = PICK_PATCH_SIZE // 2
    height, width = depth_img.shape
    if not (0 <= u < width and 0 <= v < height):
        return None
    patch = depth_img[
        max(0, v - half_patch) : min(height, v + half_patch + 1),
        max(0, u - half_patch) : min(width, u + half_patch + 1),
    ]
    valid_depths = patch[(patch > 0) & np.isfinite(patch)]
    if valid_depths.size == 0:
        return None

    z_m = float(np.median(valid_depths))

    pixel = np.array([[[u, v]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(pixel, camera_matrix, distortion, P=camera_matrix)[0, 0]
    x_m = (float(undistorted[0]) - camera_matrix[0, 2]) * z_m / camera_matrix[0, 0]
    y_m = (float(undistorted[1]) - camera_matrix[1, 2]) * z_m / camera_matrix[1, 1]
    return np.array([x_m, y_m, z_m], dtype=np.float64)

def camera_to_base_xyz(camera_xyz_m, base_from_flange):
    """Transform camera XYZ to robot-base XYZ, preserving meters."""
    point_camera_m = np.append(camera_xyz_m, 1.0)
    point_base_m = base_from_flange @ T_FLANGE_CAMERA @ point_camera_m
    return point_base_m[:3]


def estimate_surface_normal(u, v, depth_img, camera_matrix=None, distortion=None, diagnostics=None):
    """Fit a local plane; return its camera-frame unit normal into the surface."""
    camera_matrix = K if camera_matrix is None else camera_matrix
    distortion = D if distortion is None else distortion
    valid_count = kept_count = 0
    rms_m = thickness_ratio = None
    coverage = 0.0
    depth_summary = "n/a"

    def reject(reason):
        if diagnostics is not None:
            diagnostics.update(reason=reason, valid=valid_count, kept=kept_count,
                               coverage=coverage, rms_m=rms_m, thickness_ratio=thickness_ratio,
                               filters_enabled=NORMAL_FILTERS_ENABLED,
                               depth_summary=depth_summary)
            return None
        rms_text = "n/a" if rms_m is None else f"{rms_m:.6f}"
        ratio_text = "n/a" if thickness_ratio is None else f"{thickness_ratio:.3f}"
        print(
            f"[Surface Normal] pixel=({u}, {v}) reason={reason} "
            f"window={NORMAL_PATCH_SIZE}x{NORMAL_PATCH_SIZE} "
            f"valid={valid_count} kept={kept_count} "
            f"coverage={coverage:.1%} depth_m(p10/median/p90)={depth_summary} "
            f"rms_m={rms_text} (max={NORMAL_MAX_RMS_M:.3f}) "
            f"thickness_ratio={ratio_text} (max={NORMAL_MAX_THICKNESS_RATIO:.3f})"
        )
        return None

    height, width = depth_img.shape
    if not (0 <= u < width and 0 <= v < height):
        return reject("pixel_out_of_bounds")
    half = NORMAL_PATCH_SIZE // 2
    x0, x1 = max(0, u - half), min(width, u + half + 1)
    y0, y1 = max(0, v - half), min(height, v + half + 1)
    depth_m = depth_img[y0:y1, x0:x1].astype(np.float64)
    valid = (depth_m > 0) & np.isfinite(depth_m)
    valid_count = int(np.count_nonzero(valid))
    coverage = valid_count / depth_m.size
    if valid_count:
        depth_summary = "/".join(
            f"{value:.4f}" for value in np.percentile(depth_m[valid], [10, 50, 90])
        )
    kept_count = valid_count
    minimum_points = NORMAL_MIN_POINTS if NORMAL_FILTERS_ENABLED else 3
    if valid_count < minimum_points:
        return reject("insufficient_depth")
    if NORMAL_FILTERS_ENABLED:
        # Prefer the clicked surface over a background occupying most of the patch.
        center = depth_img[
            max(0, v - 1) : min(height, v + 2), max(0, u - 1) : min(width, u + 2)
        ]
        center_valid = center[(center > 0) & np.isfinite(center)]
        reference_m = (
            float(np.median(center_valid))
            if center_valid.size
            else float(np.median(depth_m[valid]))
        )
        valid &= np.abs(depth_m - reference_m) <= NORMAL_DEPTH_BAND_M
        kept_count = int(np.count_nonzero(valid))
        if kept_count < minimum_points:
            return reject("insufficient_depth_after_filter")
    rows, cols = np.mgrid[y0:y1, x0:x1]
    pixels = np.column_stack((cols[valid], rows[valid])).astype(np.float64)
    rays = cv2.undistortPoints(pixels.reshape(-1, 1, 2), camera_matrix, distortion).reshape(-1, 2)
    z = depth_m[valid]
    points = np.column_stack((rays * z[:, None], z))
    centroid = points.mean(axis=0)
    _, singular, vt = np.linalg.svd(points - centroid, full_matrices=False)
    # Reject lines, tiny patches and clouds without a well-defined plane.
    rms_m = float(singular[2] / np.sqrt(len(points)))
    if singular[1] < 1e-6:
        return reject("degenerate_patch")
    thickness_ratio = float(singular[2] / singular[1])
    if NORMAL_FILTERS_ENABLED and thickness_ratio > NORMAL_MAX_THICKNESS_RATIO:
        return reject("thick_cloud")
    if NORMAL_FILTERS_ENABLED and rms_m > NORMAL_MAX_RMS_M:
        return reject("plane_rms_too_large")
    normal = vt[-1]
    # SVD has an arbitrary sign: choose the direction away from the camera.
    facing = float(normal @ centroid)
    if abs(facing) < 1e-8:
        return reject("ambiguous_normal_direction")
    if diagnostics is not None:
        diagnostics.update(reason="ok", valid=valid_count, kept=kept_count,
                           coverage=coverage, rms_m=rms_m, thickness_ratio=thickness_ratio,
                               filters_enabled=NORMAL_FILTERS_ENABLED,
                           depth_summary=depth_summary)
    return normal if facing > 0 else -normal

def preview_to_pixel(x, y):
    """Map full-size Color or the top-right Depth inset to a source pixel."""
    if not (0 <= x < IMAGE_WIDTH and 0 <= y < IMAGE_HEIGHT):
        return None
    if (
        DEPTH_PREVIEW_X <= x < DEPTH_PREVIEW_X + DEPTH_PREVIEW_WIDTH
        and DEPTH_PREVIEW_Y <= y < DEPTH_PREVIEW_Y + DEPTH_PREVIEW_HEIGHT
    ):
        return (
            int((x - DEPTH_PREVIEW_X + 0.5) * IMAGE_WIDTH / DEPTH_PREVIEW_WIDTH),
            int((y - DEPTH_PREVIEW_Y + 0.5) * IMAGE_HEIGHT / DEPTH_PREVIEW_HEIGHT),
        )
    return x, y


def render_preview(color_bgr, depth_colored, selection, fps):
    """Draw Color with a Depth inset and a label beside the selected source point."""
    preview = color_bgr.copy()
    depth_small = cv2.resize(depth_colored, (DEPTH_PREVIEW_WIDTH, DEPTH_PREVIEW_HEIGHT))
    if selection is not None:
        u, v = selection[:2]
        cv2.drawMarker(
            depth_small,
            (
                u * DEPTH_PREVIEW_WIDTH // IMAGE_WIDTH,
                v * DEPTH_PREVIEW_HEIGHT // IMAGE_HEIGHT,
            ),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            12,
            1,
        )
    x0, y0 = DEPTH_PREVIEW_X, DEPTH_PREVIEW_Y
    preview[y0 : y0 + DEPTH_PREVIEW_HEIGHT, x0 : x0 + DEPTH_PREVIEW_WIDTH] = depth_small
    cv2.rectangle(
        preview,
        (x0, y0),
        (x0 + DEPTH_PREVIEW_WIDTH, y0 + DEPTH_PREVIEW_HEIGHT),
        (255, 255, 255),
        1,
    )
    cv2.putText(
        preview,
        "DEPTH (10m)",
        (x0 + 8, y0 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        f"COLOR | FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    if selection is None:
        return preview

    u, v, camera_xyz, base_xyz, target_xyz = selection
    lines = [f"Pixel (u, v): {u}, {v}"]
    for label, xyz in (
        ("Camera XYZ", camera_xyz),
        ("Robot Base XYZ", base_xyz),
        ("Target XYZ", target_xyz),
    ):
        if xyz is None:
            lines.append(f"{label}: unavailable")
        else:
            lines.append(f"{label} (m): " + ", ".join(f"{value:.4f}" for value in xyz))

    font, scale, spacing, padding = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 23, 10
    box_width = (
        max(cv2.getTextSize(line, font, scale, 1)[0][0] for line in lines) + padding * 2
    )
    box_height = len(lines) * spacing + padding
    # Place the label beside the point, flipping at edges and avoiding the Depth inset.
    left = u + 18 if u + 18 + box_width < IMAGE_WIDTH else u - 18 - box_width
    left = max(0, min(left, IMAGE_WIDTH - box_width - 1))
    top = v + 18 if v + 18 + box_height < IMAGE_HEIGHT else v - 18 - box_height
    top = max(0, min(top, IMAGE_HEIGHT - box_height - 1))
    if (
        left + box_width >= x0
        and left < x0 + DEPTH_PREVIEW_WIDTH
        and top < y0 + DEPTH_PREVIEW_HEIGHT
    ):
        top = y0 + DEPTH_PREVIEW_HEIGHT + 12
    cv2.line(
        preview,
        (u, v),
        (max(left, min(u, left + box_width)), max(top, min(v, top + box_height))),
        (0, 255, 255),
        1,
    )
    cv2.rectangle(
        preview, (left, top), (left + box_width, top + box_height), (25, 25, 25), -1
    )
    for index, line in enumerate(lines):
        cv2.putText(
            preview,
            line,
            (left + padding, top + (index + 1) * spacing),
            font,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.drawMarker(preview, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
    return preview


def build_target_tf(camera_xyz_m, base_from_flange, normal_camera):
    """Align flange +Z with the inward normal, using camera axes as reference."""
    normal = np.asarray(normal_camera, dtype=np.float64)
    if (
        normal.shape != (3,)
        or not np.isfinite(normal).all()
        or np.linalg.norm(normal) < 1e-12
    ):
        raise ValueError("A finite, nonzero camera-frame normal is required")
    normal = normal / np.linalg.norm(normal)
    # Shortest rotation from camera +Z to the normal, without extra axial spin.
    axis = np.cross([0.0, 0.0, 1.0], normal)
    sine = np.linalg.norm(axis)
    if sine < 1e-12:
        rotation_camera = np.eye(3) if normal[2] >= 0 else np.diag([1.0, -1.0, -1.0])
    else:
        angle = np.arctan2(sine, normal[2])
        rotation_camera = cv2.Rodrigues(axis * (angle / sine))[0]
    target_tf = base_from_flange.copy()
    target_tf[:3, :3] = (
        base_from_flange[:3, :3] @ T_FLANGE_CAMERA[:3, :3] @ rotation_camera
    )
    target_tf[:3, 3] = camera_to_base_xyz(camera_xyz_m, base_from_flange)
    target_tf[2, 3] += TARGET_Z_OFFSET_M
    return target_tf


def main():
    global is_running, K, D
    is_running = True
    K = D = None
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
        print(
            "❌ Shared Memory를 찾을 수 없습니다. main.py 실행 후 camera_runner.py를 실행하세요."
        )
        return
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return

    try:
        camera_sem = posix_ipc.Semaphore(SEM_NAME)
    except posix_ipc.ExistentialError:
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
    window_name = "Gemini - Pixel Picking (Surface Normal)"
    latest = {"depth": None, "timestamp": None, "selection": None}
    last_sequence_id = 0

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        depth_img = latest["depth"]
        if depth_img is None or K is None:
            return

        pixel = preview_to_pixel(x, y)
        if pixel is None:
            return
        u, v = pixel

        xyz = pixel_to_camera_xyz(u, v, depth_img)
        if xyz is None:
            latest["selection"] = (u, v, None, None, None)
            target_shm.write(
                timestamp=latest["timestamp"],
                detected=False,
                score=0.0,
            )
            print(f"픽셀 ({u}, {v}): 주변에 유효한 depth 값이 없습니다.")
            return

        normal_camera = estimate_surface_normal(u, v, depth_img)
        if normal_camera is None:
            latest["selection"] = (u, v, xyz, None, None)
            target_shm.write(timestamp=latest["timestamp"], detected=False, score=0.0)
            print(
                f"픽셀 ({u}, {v}): 표면 법선을 안정적으로 추정할 수 없어 target을 전송하지 않았습니다."
            )
            return

        if not flange_shm.status:
            latest["selection"] = (u, v, xyz, None, None)
            target_shm.write(
                timestamp=latest["timestamp"],
                detected=False,
                score=0.0,
            )
            print(
                "❌ Flange Shared Memory가 비활성 상태라 target을 전송하지 않았습니다."
            )
            return

        flange_pose = flange_shm.read(latest["timestamp"])
        if flange_pose.matched_ts <= 0 or not np.isfinite(flange_pose.TF).all():
            latest["selection"] = (u, v, xyz, None, None)
            target_shm.write(
                timestamp=latest["timestamp"],
                detected=False,
                score=0.0,
            )
            print("❌ 유효한 flange TF를 찾지 못해 target을 전송하지 않았습니다.")
            return

        base_xyz_m = camera_to_base_xyz(xyz, flange_pose.TF.copy())
        target_tf = build_target_tf(xyz, flange_pose.TF, normal_camera)
        target_shm.write(
            timestamp=latest["timestamp"],
            detected=True,
            score=1.0,
            TF=target_tf,
        )

        latest["selection"] = (u, v, xyz, base_xyz_m, target_tf[:3, 3].copy())
        print(
            f"픽셀 ({u}, {v}) -> Camera XYZ: "
            f"X={xyz[0]:.4f} m, Y={xyz[1]:.4f} m, Z={xyz[2]:.4f} m"
        )
        print(
            f"Robot Base XYZ -> X={base_xyz_m[0]:.4f} m, "
            f"Y={base_xyz_m[1]:.4f} m, Z={base_xyz_m[2]:.4f} m "
            f"(flange Δt={flange_pose.time_diff_us} µs)"
        )
        print(
            f"Target (Base Z offset {TARGET_Z_OFFSET_M:+.4f}m) -> X={target_tf[0, 3]:.4f} m, "
            f"Y={target_tf[1, 3]:.4f} m, Z={target_tf[2, 3]:.4f} m "
            "(카메라 기준 표면 법선에 +Z 정렬, target SHM 전송 완료)"
        )
        tilt_deg = np.rad2deg(np.arccos(np.clip(normal_camera[2], -1.0, 1.0)))
        print(
            f"Camera inward normal: {normal_camera}, camera +Z 기준 기울기: {tilt_deg:.1f}°"
        )

    frame_count = 0
    start_time = time.time()

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, on_mouse)
        print(
            "🚀 연결 완료. RGB/Depth 화면을 클릭하면 "
            "표면 법선에 +Z를 정렬하고 Robot Base 좌표계의 target TF를 SHM으로 전송합니다."
        )
        print("종료: Q 또는 ESC")

        while is_running:
            try:
                camera_sem.acquire(timeout=1.0)
            except posix_ipc.BusyError:
                # Keep the window responsive while the producer is stopped.
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

            frame = camera_shm.read(last_sequence_id=last_sequence_id)
            if frame is None:
                continue
            matrix, distortion = camera_shm.intrinsics
            if not (
                np.isfinite(matrix).all()
                and np.isfinite(distortion).all()
                and matrix[0, 0] > 0
                and matrix[1, 1] > 0
            ):
                print("카메라 내부 파라미터 대기 중: camera_runner.py를 확인하세요.")
                continue
            K, D = matrix, distortion
            last_sequence_id = frame.sequence_id

            color_img = frame.color.reshape(COLOR_SHAPE)
            depth_img = frame.depth.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
            latest["depth"] = depth_img
            latest["timestamp"] = frame.timestamp

            color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

            depth_full = np.where((depth_img > 0) & (depth_img < 10.0), depth_img, 0)
            depth_full_norm = cv2.normalize(
                depth_full, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_full_colored = cv2.applyColorMap(depth_full_norm, cv2.COLORMAP_JET)
            depth_full_colored[depth_full == 0] = [0, 0, 0]

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            preview = render_preview(
                color_bgr, depth_full_colored, latest["selection"], fps
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
