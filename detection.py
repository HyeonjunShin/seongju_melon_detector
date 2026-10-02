import os
import ctypes
import numpy as np
import cv2
import openvino as ov
from multiprocessing import shared_memory
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R


class BufferHeader(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_bool),
        ("index", ctypes.c_int64),
        ("latency", ctypes.c_double),
    ]


@dataclass
class TrackObj:
    timestamp: int
    detected: bool
    score: float
    bbox: np.ndarray  # [x1, y1, x2, y2]
    centroid: np.ndarray  # [X, Y, Z] (3D m 단위, Body Keypoint 중심)
    rotation: np.ndarray  # [qx, qy, qz, qw] (Body -> Knot 방향 쿼터니언)


class DetectorBuffer:
    def __init__(
        self,
        shm_name: str,
        is_owner: bool = False,
    ):
        self.shm_name = shm_name
        self.is_owner = is_owner

        # Header: status(bool_ 1 byte -> 8 byte alignment) + write_index(int64 8 bytes)
        self.header_bytes = 8 + 8

        # Slot Layout (Bytes)
        self.timestamp_bytes = np.dtype(np.uint64).itemsize  # 8
        self.detected_bytes = np.dtype(np.bool_).itemsize  # 1
        self.detected_pad_bytes = 7  # 8byte alignment 패딩
        self.score_bytes = np.dtype(np.float64).itemsize  # 8
        self.bbox_bytes = np.dtype(np.float64).itemsize * 4  # 32
        self.centroid_bytes = np.dtype(np.float64).itemsize * 3  # 24
        self.rotation_bytes = np.dtype(np.float64).itemsize * 4  # 32

        self.slot_bytes = (
            self.timestamp_bytes
            + self.detected_bytes
            + self.detected_pad_bytes
            + self.score_bytes
            + self.bbox_bytes
            + self.centroid_bytes
            + self.rotation_bytes
        )  # Total: 136 bytes

        self.total_bytes = self.header_bytes + (2 * self.slot_bytes)

        if self.is_owner:
            try:
                old_shm = shared_memory.SharedMemory(name=self.shm_name)
                old_shm.close()
                old_shm.unlink()
            except FileNotFoundError:
                pass
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name,
                create=True,
                size=self.total_bytes,
            )
        else:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)

        # Header Mapping
        self.status_arr = np.ndarray(
            (1,), dtype=np.bool_, buffer=self.shm.buf, offset=0
        )
        self.write_index_arr = np.ndarray(
            (1,), dtype=np.int64, buffer=self.shm.buf, offset=8
        )

        # Double Buffer Slots Mapping
        self.slots = []
        for i in range(2):
            slot_offset = self.header_bytes + (i * self.slot_bytes)

            timestamp_arr = np.ndarray(
                (1,), dtype=np.uint64, buffer=self.shm.buf, offset=slot_offset
            )

            offset_curr = slot_offset + self.timestamp_bytes
            detected_arr = np.ndarray(
                (1,), dtype=np.bool_, buffer=self.shm.buf, offset=offset_curr
            )

            offset_curr += self.detected_bytes + self.detected_pad_bytes
            score_arr = np.ndarray(
                (1,), dtype=np.float64, buffer=self.shm.buf, offset=offset_curr
            )

            offset_curr += self.score_bytes
            bbox_arr = np.ndarray(
                (4,), dtype=np.float64, buffer=self.shm.buf, offset=offset_curr
            )

            offset_curr += self.bbox_bytes
            centroid_arr = np.ndarray(
                (3,), dtype=np.float64, buffer=self.shm.buf, offset=offset_curr
            )

            offset_curr += self.centroid_bytes
            rotation_arr = np.ndarray(
                (4,), dtype=np.float64, buffer=self.shm.buf, offset=offset_curr
            )

            self.slots.append(
                {
                    "timestamp": timestamp_arr,
                    "detected": detected_arr,
                    "score": score_arr,
                    "bbox": bbox_arr,
                    "centroid": centroid_arr,
                    "rotation": rotation_arr,
                }
            )

    def write(
        self,
        timestamp: int,
        detected: bool,
        score: float = 0.0,
        bbox: np.ndarray = None,
        centroid: np.ndarray = None,
        rotation: np.ndarray = None,
    ):
        if bbox is None:
            bbox = np.zeros((4,), dtype=np.float64)
        if centroid is None:
            centroid = np.zeros((3,), dtype=np.float64)
        if rotation is None:
            rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        current_idx = int(self.write_index_arr[0])
        next_idx = 1 - current_idx
        target_slot = self.slots[next_idx]

        target_slot["timestamp"][0] = timestamp
        target_slot["detected"][0] = detected
        target_slot["score"][0] = score

        np.copyto(target_slot["bbox"], bbox.reshape(-1))
        np.copyto(target_slot["centroid"], centroid.reshape(-1))
        np.copyto(target_slot["rotation"], rotation.reshape(-1))

        self.write_index_arr[0] = next_idx

    def get_latest_detection(self) -> TrackObj:
        current_idx = int(self.write_index_arr[0])
        slot = self.slots[current_idx]

        return TrackObj(
            timestamp=int(slot["timestamp"][0]),
            detected=bool(slot["detected"][0]),
            score=float(slot["score"][0]),
            bbox=slot["bbox"].copy(),
            centroid=slot["centroid"].copy(),
            rotation=slot["rotation"].copy(),
        )

    def set_status(self, is_good: bool):
        self.status_arr[0] = is_good

    def get_status(self) -> bool:
        return bool(self.status_arr[0])

    def close(self):
        del self.slots
        del self.status_arr
        del self.write_index_arr

        import gc

        gc.collect()

        self.shm.close()
        if self.is_owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


class Detector:
    def __init__(
        self, model_path: str = "./model_int8.xml", conf_threshold: float = 0.25
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold

        core = ov.Core()
        device_name = "GPU" if "GPU" in core.available_devices else "CPU"
        print(f"📦 OpenVINO Pose INT8 모델 로드 중... [디바이스: {device_name}]")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(1, "Not found the model file.", self.model_path)

        ov_model = core.read_model(self.model_path)
        self.compiled_model = core.compile_model(ov_model, device_name)

        # PrePostProcessor가 전처리를 완료한 포즈 모델의 출력 바인딩
        self.output_scores = self.compiled_model.output(0)  # [1, 5040, 1]
        self.output_bboxes = self.compiled_model.output(1)  # [1, 5040, 4]
        self.output_kpts = self.compiled_model.output(2)  # [1, 5040, 2, 3]

    def detect(self, color_img):
        """color_img: (720, 1280, 3) Raw Camera Frame (u8, NHWC)
        PrePostProcessor가 내장되어 있으므로 원본 1280x720 프레임을 바로 주입합니다.
        """
        input_tensor = color_img[None, ...]  # [1, 720, 1280, 3]
        results = self.compiled_model({0: input_tensor})

        scores = results[self.output_scores][0].squeeze(-1)  # [5040]
        bboxes = results[self.output_bboxes][0]  # [5040, 4]
        kpts = results[self.output_kpts][0]  # [5040, 2, 3]

        # NMS-Free 방식: 가장 점수가 높은 Top-1 앵커 선택
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        if best_score < self.conf_threshold:
            return 0.0, None, None

        best_bbox = bboxes[best_idx]  # [x1, y1, x2, y2]
        best_kpt = kpts[
            best_idx
        ]  # [[knot_x, knot_y, knot_vis_logit], [body_x, body_y, body_vis_logit]]

        return float(best_score), best_bbox, best_kpt


def pixel_to_3d_point(u, v, depth_img, K, DIST_COEFFS, patch_size=15):
    """픽셀 좌표 (u, v) 부근의 Depth 패치를 추출하여 카메라 3D 좌표 [X, Y, Z] (m 단위) 계산"""
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
    if z_m > 10.0:  # mm -> m 변환
        z_m /= 1000.0

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
