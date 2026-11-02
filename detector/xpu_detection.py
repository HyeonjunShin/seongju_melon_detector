import sys
import time
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.shm_detection import DetectionShm, OBJECT_DTYPE, MAX_OBJECTS
from common.shm_flange import FlangeShm
from melon_pose_nmsfree import Detector as TensorRTModel

DEFAULT_ENGINE = (
    Path(__file__).resolve().parents[1]
    / "melon_pose_nmsfree/models/rtx5060_laptop_fp16.engine"
)


def depth_patch_stats(u, v, depth):
    """Use the same window as picking; expose depth availability without guessing."""
    from camera.test_picking import PICK_PATCH_SIZE

    h, w = depth.shape
    if not (0 <= u < w and 0 <= v < h):
        return dict(valid=0, total=0, median_m=None)
    half = PICK_PATCH_SIZE // 2
    patch = depth[
        max(0, v - half) : min(h, v + half + 1), max(0, u - half) : min(w, u + half + 1)
    ]
    valid = patch[(patch > 0) & np.isfinite(patch)]
    return dict(
        valid=int(valid.size),
        total=int(patch.size),
        median_m=float(np.median(valid)) if valid.size else None,
    )


def build_base_detections(
    predictions, depth, matrix, distortion, base_from_flange, diagnostics=None
):
    """Picking Body position/+Z normal; +X follows Body-to-Knot in the tangent plane."""
    from camera.test_picking import (
        pixel_to_camera_xyz,
        estimate_surface_normal,
        build_target_tf,
        T_FLANGE_CAMERA,
    )

    if not (
        np.isfinite(matrix).all()
        and np.isfinite(distortion).all()
        and matrix[0, 0] > 0
        and matrix[1, 1] > 0
        and np.isfinite(base_from_flange).all()
    ):
        return np.empty(0, dtype=OBJECT_DTYPE)
    depth = depth.reshape(depth.shape[:2])
    objects = []
    for score, bbox, kpts, visibility in sorted(
        predictions, key=lambda p: p[0], reverse=True
    ):
        detail = dict(bbox=np.asarray(bbox).copy(), reason="invalid_prediction")
        if diagnostics is not None:
            diagnostics.append(detail)
        if (
            not np.isfinite(score)
            or not np.isfinite(bbox).all()
            or not np.isfinite(kpts).all()
            or not np.isfinite(visibility).all()
        ):
            continue
        if np.any(visibility < 0.5):
            detail["reason"] = "keypoint_visibility"
            continue
        u, v = np.rint(kpts[0]).astype(int)
        if diagnostics is not None:
            detail["body_depth"] = depth_patch_stats(u, v, depth)
        xyz = pixel_to_camera_xyz(u, v, depth, matrix, distortion)
        if xyz is None:
            detail["reason"] = "no_body_depth"
            continue
        normal_info = {} if diagnostics is not None else None
        normal = estimate_surface_normal(
            u, v, depth, matrix, distortion, diagnostics=normal_info
        )
        if normal_info is not None:
            detail["normal"] = normal_info
        if normal is None:
            detail["reason"] = "normal_" + (
                normal_info["reason"] if normal_info is not None else "rejected"
            )
            continue
        z_axis = normal / np.linalg.norm(normal)
        # Lift the IMAGE direction into the Body tangent plane. Knot depth is
        # neither sampled nor assumed: +X lies in both the Body tangent plane
        # and the viewing plane through the Body/Knot camera rays.
        knot_ray_xy = cv2.undistortPoints(
            np.asarray(kpts[1], dtype=np.float64).reshape(1, 1, 2), matrix, distortion
        ).reshape(2)
        body_ray = xyz / xyz[2]
        knot_ray = np.r_[knot_ray_xy, 1.0]
        viewing_normal = np.cross(body_ray, knot_ray)
        x_axis = np.cross(viewing_normal, z_axis)
        x_length = np.linalg.norm(x_axis)
        if not np.isfinite(x_length) or x_length < 1e-9:
            detail["reason"] = "undefined_body_knot_axis"
            continue
        x_axis /= x_length
        projected_direction = x_axis[:2] - body_ray[:2] * x_axis[2]
        alignment = np.dot(projected_direction, knot_ray[:2] - body_ray[:2])
        if not np.isfinite(alignment) or abs(alignment) < 1e-12:
            detail["reason"] = "undefined_body_knot_axis"
            continue
        if alignment < 0:
            x_axis = -x_axis
        y_axis = np.cross(z_axis, x_axis)
        rotation_camera = np.column_stack((x_axis, y_axis, z_axis))
        transform = build_target_tf(xyz, base_from_flange, normal)
        transform[:3, :3] = (
            base_from_flange[:3, :3] @ T_FLANGE_CAMERA[:3, :3] @ rotation_camera
        )
        detail["reason"] = "ok"
        objects.append((score, bbox, transform))
        if len(objects) == MAX_OBJECTS:
            break
    return np.array(objects, dtype=OBJECT_DTYPE)


class Detector:
    """TensorRT adapter; RGB input, original-pixel outputs, Body then Knot."""

    def __init__(self, model_path=DEFAULT_ENGINE, conf_threshold=0.3):
        if not 0 <= conf_threshold <= 1:
            raise ValueError("confidence must be in [0, 1]")
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.engine = TensorRTModel(self.model_path)
        minimum = self.engine.metadata["confidence"]
        if conf_threshold < minimum:
            raise ValueError(
                f"confidence must be >= engine's fixed threshold {minimum}"
            )
        print(f"TensorRT 모델 로드 완료: {self.model_path}")

    def detect_all(self, color_img):
        """Return all (score, bbox, keypoints, visibility) above the threshold."""
        objects = self.engine(cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
        return [
            (
                float(obj["score"]),
                np.asarray(obj["bbox"], dtype=np.float32),
                np.asarray([obj["body"], obj["knot"]], dtype=np.float32),
                np.asarray(obj["visibility_probability"], dtype=np.float32),
            )
            for obj in objects
            if obj["score"] >= self.conf_threshold
        ]

    def detect(self, color_img):
        """Return (score, bbox[4], keypoints[2,2], visibility probabilities[2])."""
        objects = self.engine(cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
        if not objects:
            return 0.0, None, None, None
        best = max(objects, key=lambda obj: obj["score"])
        if best["score"] < self.conf_threshold:
            return 0.0, None, None, None
        return (
            float(best["score"]),
            np.asarray(best["bbox"], dtype=np.float32),
            np.asarray([best["body"], best["knot"]], dtype=np.float32),
            np.asarray(best["visibility_probability"], dtype=np.float32),
        )


def draw_detection(color_img, score, bbox, kpts, vis):
    """Render original-image pixel coordinates without resizing or unpadding."""
    return draw_detections(color_img, [(score, bbox, kpts, vis)])


def draw_detections(color_img, predictions):
    """Render every detection on one BGR image."""
    display = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    for prediction in predictions:
        _draw_detection(display, *prediction)
    return display


def format_pose_diagnostic(detail):
    parts = [detail["reason"], f"bbox={np.asarray(detail['bbox']).tolist()}"]
    for name in ("body",):
        patch = detail.get(name + "_depth")
        if patch is not None:
            parts.append(f"{name}={patch}")
    if "normal" in detail:
        parts.append(f"normal={detail['normal']}")
    return " | ".join(parts)


def draw_pose_diagnostics(display, diagnostics):
    """Annotate rejected poses beside their detection boxes."""
    h, w = display.shape[:2]
    for detail in diagnostics:
        if detail["reason"] == "ok" or not np.isfinite(detail["bbox"]).all():
            continue
        x = max(0, min(int(detail["bbox"][0]), max(0, w - 410)))
        y = max(18, min(int(detail["bbox"][3]) + 18, h - 26))
        messages = ["Pose: " + detail["reason"]]
        patches = []
        for name in ("body",):
            patch = detail.get(name + "_depth")
            if patch is not None:
                z = "--" if patch["median_m"] is None else f"{patch['median_m']:.3f}m"
                patches.append(f"{name}: {z} ({patch['valid']}/{patch['total']})")
        if patches:
            messages.append(" | ".join(patches))
        for row, message in enumerate(messages):
            for color, thickness in (((0, 0, 0), 3), ((0, 200, 255), 1)):
                cv2.putText(
                    display,
                    message,
                    (x, y + row * 17),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
    return display


def draw_base_poses(
    display, objects, matrix, distortion, base_from_camera, axis_length=0.05
):
    """Overlay the published base-frame poses; local RGB axes are 5 cm long."""
    if base_from_camera is None or not len(objects):
        return display
    camera_from_base = np.linalg.inv(base_from_camera)
    local_points = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [axis_length, 0.0, 0.0, 1.0],
            [0.0, axis_length, 0.0, 1.0],
            [0.0, 0.0, axis_length, 1.0],
        ]
    )
    height, width = display.shape[:2]
    for index, obj in enumerate(objects, 1):
        camera_from_object = camera_from_base @ obj["TF"]
        points = (camera_from_object @ local_points.T).T[:, :3]
        if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
            continue
        pixels, _ = cv2.projectPoints(
            points, np.zeros(3), np.zeros(3), matrix, distortion
        )
        pixels = pixels.reshape(-1, 2)
        if not np.isfinite(pixels).all() or np.max(np.abs(pixels)) > 1e6:
            continue
        pixels = np.rint(pixels).astype(int)
        origin = tuple(pixels[0])
        if not (0 <= origin[0] < width and 0 <= origin[1] < height):
            continue
        for endpoint, color, label in zip(
            pixels[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0)), "XYZ"
        ):
            cv2.arrowedLine(display, origin, tuple(endpoint), color, 2, cv2.LINE_AA)
            cv2.putText(
                display,
                label,
                tuple(endpoint + [4, -4]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.circle(display, origin, 4, (255, 255, 255), -1)
        xyz = obj["TF"][:3, 3]
        text = f"P{index} Base(m): {xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}"
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        label_origin = (
            max(0, min(origin[0] + 12, width - text_width - 4)),
            max(16, min(origin[1] + 25, height - 6)),
        )
        cv2.putText(
            display,
            text,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            text,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return display


def _draw_detection(display, score, bbox, kpts, vis):
    if bbox is None or kpts is None or vis is None:
        return display
    box = np.asarray(bbox, dtype=np.float32).reshape(4)
    points = np.asarray(kpts, dtype=np.float32).reshape(2, 2)
    visibility = np.asarray(vis, dtype=np.float32).reshape(2)
    if not all(np.isfinite(value).all() for value in (box, points, visibility)):
        return display
    height, width = display.shape[:2]
    x1, x2 = np.clip(box[[0, 2]], 0, width - 1).astype(int)
    y1, y2 = np.clip(box[[1, 3]], 0, height - 1).astype(int)
    if x2 <= x1 or y2 <= y1:
        return display
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 140, 255), 2)
    cv2.putText(
        display,
        f"Melon {score:.2f}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 140, 255),
        2,
        cv2.LINE_AA,
    )
    points = np.clip(points, [0, 0], [width - 1, height - 1]).astype(int)
    if np.all(visibility >= 0.5):
        cv2.line(
            display, tuple(points[0]), tuple(points[1]), (0, 140, 255), 2, cv2.LINE_AA
        )
    for point, label, color, probability in zip(
        points, ("Body", "Knot"), ((255, 0, 0), (0, 0, 255)), visibility
    ):
        if probability >= 0.5:
            px, py = map(int, point)
            cv2.circle(display, (px, py), 5, color, -1, cv2.LINE_AA)
            cv2.putText(
                display,
                f"{label} {probability:.2f}",
                (px + 7, py - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return display


def pixel_to_3d_point(u, v, depth_img, K, DIST_COEFFS, patch_size=15):
    """픽셀 좌표 (u, v) 부근의 미터 단위 Depth 패치를 추출하여 카메라 3D 좌표 [X, Y, Z] (m 단위) 계산"""
    h, w = depth_img.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return None

    u_idx, v_idx = int(round(u)), int(round(v))
    half_p = patch_size // 2

    v_min, v_max = max(0, v_idx - half_p), min(h, v_idx + half_p + 1)
    u_min, u_max = max(0, u_idx - half_p), min(w, u_idx + half_p + 1)

    depth_patch = depth_img[v_min:v_max, u_min:u_max]
    valid_depths = depth_patch[depth_patch > 0]

    if len(valid_depths) == 0:
        return None

    z_m = float(np.median(valid_depths))

    # 렌즈 왜곡 보정 적용
    pixel_pt = np.array([[[u, v]]], dtype=np.float32)
    undistorted_pt = cv2.undistortPoints(pixel_pt, K, DIST_COEFFS, P=K)
    u_undist, v_undist = undistorted_pt[0][0]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (u_undist - cx) * z_m / fx
    y_cam = (v_undist - cy) * z_m / fy
    z_cam = z_m

    return np.array([x_cam, y_cam, z_cam], dtype=np.float64)


def compute_pose_with_kpts(depth_img, kpts, K, DIST_COEFFS, vis_thresh=0.5):
    """Body Keypoint를 3D Centroid(로봇 그리퍼 목표지점)로 삼고,
    Body -> Knot 방향 벡터를 기반으로 회전 행렬(Rotation Matrix / Quaternion)을 생성합니다.

    kpts: [[knot_x, knot_y, knot_vis_logit], [body_x, body_y, body_vis_logit]]
    """
    knot_kpt, body_kpt = kpts[0], kpts[1]

    # Visibility Sigmoid 계산
    knot_vis = 1.0 / (1.0 + np.exp(-knot_kpt[2])) if knot_kpt[2] < 10 else knot_kpt[2]
    body_vis = 1.0 / (1.0 + np.exp(-body_kpt[2])) if body_kpt[2] < 10 else body_kpt[2]

    default_rotation = np.array(
        [0.0, 0.0, 0.0, 1.0], dtype=np.float64
    )  # [qx, qy, qz, qw]

    # 1. Body 3D Centroid 계산
    if body_vis < vis_thresh:
        return None, None

    body_3d = pixel_to_3d_point(body_kpt[0], body_kpt[1], depth_img, K, DIST_COEFFS)
    if body_3d is None:
        return None, None

    centroid = body_3d  # Body가 로봇 제어 중심점 (Centroid)

    # 2. Knot 3D Point 계산 및 Orientation(회전 행렬) 구성
    if knot_vis < vis_thresh:
        # 매듭이 가려진 경우 기본 회전값([0, 0, 0, 1]) 사용
        return centroid, default_rotation

    knot_3d = pixel_to_3d_point(knot_kpt[0], knot_kpt[1], depth_img, K, DIST_COEFFS)
    if knot_3d is None:
        return centroid, default_rotation

    # 3. Body -> Knot 방향 벡터 생성
    dir_vec = knot_3d - body_3d
    vec_len = np.linalg.norm(dir_vec)

    if vec_len < 1e-6:
        return centroid, default_rotation

    # Z축: Body -> Knot 단위 방향 벡터
    z_axis = dir_vec / vec_len

    # X축: Z축과 카메라 광학축([0, 0, 1])의 외적
    cam_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.cross(z_axis, cam_z)
    x_norm = np.linalg.norm(x_axis)

    if x_norm < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis = x_axis / x_norm

    # Y축: Z축과 X축의 외적 (오른손 좌표계 완결)
    y_axis = np.cross(z_axis, x_axis)

    # 4. 회전 행렬 R = [x_axis, y_axis, z_axis] 및 Quaternion 변환
    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
    r = R.from_matrix(rot_matrix)
    quaternion = r.as_quat()  # SciPy 순서: [qx, qy, qz, qw]

    return centroid, quaternion


is_running = True


def main():
    import argparse
    import signal
    import posix_ipc
    import torch
    from common.shm_camera import CameraShm, COLOR_SHAPE, DEPTH_SHAPE
    from camera.camera_runner import SHM_NAME, SEM_NAME

    parser = argparse.ArgumentParser(
        description="TensorRT multi-object melon detection"
    )
    parser.add_argument(
        "--engine", "--model", dest="model", type=Path, default=DEFAULT_ENGINE
    )
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Measure processing without OpenCV window rendering",
    )
    args = parser.parse_args()
    # The small CPU input copy is slower with this PC's default 20 threads.
    # Match the serving CLI's measured thread count.
    torch.set_num_threads(4)

    global is_running
    is_running = True

    def signal_handler(signum, frame):
        global is_running
        is_running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    detector = Detector(args.model, conf_threshold=args.confidence)

    try:
        camera_shm = CameraShm(
            shm_name=SHM_NAME,
            is_owner=False,
            color_shape=COLOR_SHAPE,
            depth_shape=DEPTH_SHAPE,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(
            f"카메라 SHM 연결 실패: {exc}. main.py 실행 후 camera_runner.py를 실행하세요."
        )
        return
    try:
        camera_sem = posix_ipc.Semaphore(SEM_NAME)
    except posix_ipc.ExistentialError:
        camera_shm.close()
        print(f"Semaphore ({SEM_NAME})가 없습니다. main.py를 먼저 실행하세요.")
        return

    try:
        flange_shm = FlangeShm(shm_name="flange_shm", is_owner=False)
    except FileNotFoundError:
        camera_shm.close()
        camera_sem.close()
        print("Flange SHM 연결 실패: main.py와 로봇 수신 프로세스를 확인하세요.")
        return
    try:
        detection_shm = DetectionShm(is_owner=False)
    except (FileNotFoundError, RuntimeError) as exc:
        flange_shm.close()
        camera_shm.close()
        camera_sem.close()
        print(f"Detection SHM 연결 실패: {exc}. main.py를 먼저 실행하세요.")
        return

    window_name = "TensorRT Melon Detection"
    last_sequence_id = 0
    fps = None
    frame_count = 0
    stats_started = time.perf_counter()
    stage_ms = dict(wait=0.0, shm=0.0, infer=0.0, draw=0.0, gui=0.0)
    skipped = 0
    pose_counts = Counter()
    pose_samples = {}
    try:
        if not args.no_display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while is_running:
            wait_started = time.perf_counter()
            try:
                camera_sem.acquire(timeout=0.1)
            except posix_ipc.BusyError:
                detection_shm.write(0, last_sequence_id, [], status=False)
                stage_ms["wait"] += (time.perf_counter() - wait_started) * 1000
                if not args.no_display and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            read_started = time.perf_counter()
            stage_ms["wait"] += (read_started - wait_started) * 1000
            frame = camera_shm.read(last_sequence_id=last_sequence_id)
            inference_started = time.perf_counter()
            stage_ms["shm"] += (inference_started - read_started) * 1000
            if frame is None:
                continue
            if last_sequence_id:
                skipped += max(0, frame.sequence_id - last_sequence_id - 1)
            last_sequence_id = frame.sequence_id
            predictions = detector.detect_all(frame.color)
            matrix, distortion = camera_shm.intrinsics
            objects = np.empty(0, dtype=OBJECT_DTYPE)
            pose_ready = False
            diagnostics = []
            pose_gate = (
                "camera_inactive"
                if not camera_shm.status
                else "flange_inactive" if not flange_shm.status else "invalid_flange"
            )
            base_from_camera = None
            if camera_shm.status and flange_shm.status:
                flange_pose = flange_shm.read(frame.timestamp)
                if flange_pose.matched_ts > 0 and np.isfinite(flange_pose.TF).all():
                    pose_ready = bool(
                        np.isfinite(matrix).all()
                        and np.isfinite(distortion).all()
                        and matrix[0, 0] > 0
                        and matrix[1, 1] > 0
                    )
                    pose_gate = "ready" if pose_ready else "invalid_calibration"
                    if pose_ready:
                        from camera.test_picking import T_FLANGE_CAMERA

                        base_from_flange = flange_pose.TF.copy()
                        base_from_camera = base_from_flange @ T_FLANGE_CAMERA
                        objects = build_base_detections(
                            predictions,
                            frame.depth,
                            matrix,
                            distortion,
                            base_from_flange,
                            diagnostics=diagnostics,
                        )
            pose_counts.update(item["reason"] for item in diagnostics)
            for item in diagnostics:
                if item["reason"] != "ok":
                    pose_samples[item["reason"]] = item
            if not pose_ready:
                pose_counts[pose_gate] += 1
            detection_shm.write(
                frame.timestamp,
                frame.sequence_id,
                objects,
                status=pose_ready,
                latency_ms=(time.perf_counter() - inference_started) * 1000,
            )
            # Includes inference, pose geometry and publishing robot-base results.
            inference_ms = (time.perf_counter() - inference_started) * 1000
            stage_ms["infer"] += inference_ms
            draw_started = time.perf_counter()
            key = -1
            if not args.no_display:
                display = draw_detections(frame.color, predictions)
                draw_pose_diagnostics(display, diagnostics)
                draw_base_poses(display, objects, matrix, distortion, base_from_camera)
                fps_text = "--" if fps is None else f"{fps:.1f}"
                cv2.rectangle(display, (8, 8), (530, 94), (20, 20, 20), -1)
                for row, text in enumerate(
                    (
                        f"Process FPS: {fps_text}",
                        f"Detection + pose: {inference_ms:.2f} ms",
                        (
                            f"Poses: {len(objects)}/8 | XYZ axes: R/G/B (5cm)"
                            if pose_ready
                            else f"Pose unavailable: {pose_gate}"
                        ),
                    )
                ):
                    cv2.putText(
                        display,
                        text,
                        (18, 32 + row * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                gui_started = time.perf_counter()
                stage_ms["draw"] += (gui_started - draw_started) * 1000
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                stage_ms["gui"] += (time.perf_counter() - gui_started) * 1000
            frame_count += 1
            now = time.perf_counter()
            elapsed = now - stats_started
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                timings = " | ".join(
                    f"{name}={value / frame_count:.2f}ms"
                    for name, value in stage_ms.items()
                )
                print(
                    f"[Detection] FPS={fps:.1f} | {timings} | skipped={skipped}"
                    + " | pose="
                    + (
                        ", ".join(
                            f"{key}:{value}" for key, value in pose_counts.items()
                        )
                        or "no_detections"
                    ),
                    flush=True,
                )
                for item in pose_samples.values():
                    print("[Pose Detail] " + format_pose_diagnostic(item), flush=True)
                pose_samples.clear()
                frame_count = 0
                stats_started = now
                stage_ms = dict.fromkeys(stage_ms, 0.0)
                skipped = 0
                pose_counts.clear()
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()
        camera_shm.close()
        camera_sem.close()
        detection_shm.write(0, last_sequence_id, [], status=False)
        detection_shm.close()
        flange_shm.close()
        print("검출 프로세스 리소스 정리 완료")


if __name__ == "__main__":
    main()
