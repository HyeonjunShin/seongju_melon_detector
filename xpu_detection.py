import os
import ctypes
import numpy as np
import cv2
import openvino as ov
import torch
from multiprocessing import shared_memory
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

from training.model import DecodeModule, Model


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


class DetectionEMA:
    def __init__(self, alpha=0.3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the range (0, 1]")
        self.alpha = alpha
        self.bbox = None
        self.keypoints = None

    def update(self, score, bbox, keypoints, visibility):
        if bbox is None or keypoints is None:
            self.bbox = None
            self.keypoints = None
            return score, bbox, keypoints, visibility

        bbox = np.asarray(bbox, dtype=np.float32)
        keypoints = np.asarray(keypoints, dtype=np.float32)
        if self.bbox is None:
            self.bbox = bbox.copy()
            self.keypoints = keypoints.copy()
        else:
            self.bbox = self.alpha * bbox + (1.0 - self.alpha) * self.bbox
            self.keypoints = (
                self.alpha * keypoints + (1.0 - self.alpha) * self.keypoints
            )
        return score, self.bbox.copy(), self.keypoints.copy(), visibility


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
        self.output_scores_o2m = self.compiled_model.output(0)  # [1, 5040, 1]
        self.output_bboxes_o2m = self.compiled_model.output(1)  # [1, 5040, 4]
        self.output_kpts_o2m = self.compiled_model.output(2)  # [1, 5040, 2, 2]
        self.output_vis_o2m = self.compiled_model.output(3)  # [1, 5040, 2]

        self.output_scores_o2o = self.compiled_model.output(4)  # [1, 5040, 1]
        self.output_bboxes_o2o = self.compiled_model.output(5)  # [1, 5040, 4]
        self.output_kpts_o2o = self.compiled_model.output(6)  # [1, 5040, 2, 2]
        self.output_vis_o2o = self.compiled_model.output(7)  # [1, 5040, 2]

    def detect(self, color_img):
        """color_img: (720, 1280, 3) Raw Camera Frame (u8, NHWC)
        PrePostProcessor가 내장되어 있으므로 원본 1280x720 프레임을 바로 주입합니다.
        """
        input_tensor = color_img[None, ...]  # [1, 720, 1280, 3]
        results = self.compiled_model({0: input_tensor})

        # scores = results[self.output_scores][0].squeeze(-1)  # [5040]
        # bboxes = results[self.output_bboxes][0]  # [5040, 4]
        # kpts = results[self.output_kpts][0]  # [5040, 2, 2]
        # vis = results[self.output_vis][0]  # [5040, 2, 2]

        # scores = results[self.output_scores_o2m][0].squeeze(-1)  # [5040]
        # bboxes = results[self.output_bboxes_o2m][0]  # [5040, 4]
        # kpts = results[self.output_kpts_o2m][0]  # [5040, 2, 2]
        # vis = results[self.output_vis_o2m][0]  # [5040, 2, 2]

        scores = results[self.output_scores_o2o][0].squeeze(-1)  # [5040]
        bboxes = results[self.output_bboxes_o2o][0]  # [5040, 4]
        kpts = results[self.output_kpts_o2o][0]  # [5040, 2, 2]
        vis = results[self.output_vis_o2o][0]  # [5040, 2]

        # NMS-Free 방식: 가장 점수가 높은 Top-1 앵커 선택
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        if best_score < self.conf_threshold:
            return 0.0, None, None, None

        best_bbox = bboxes[best_idx]  # [x1, y1, x2, y2]
        best_kpt = kpts[best_idx]  # [[body_x, body_y], [knot_x, knot_y]]
        best_vis = vis[best_idx]  # [body_visibility, knot_visibility]

        return float(best_score), best_bbox, best_kpt, best_vis


def prepare_pytorch_input(color_img, device):
    resized = cv2.resize(color_img, (640, 360), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return (
        torch.from_numpy(padded)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )


class PyTorchDetector:
    def __init__(self, checkpoint, device=None, conf_threshold=0.25):
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(1, "Not found the checkpoint file.", checkpoint)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.conf_threshold = conf_threshold
        self.model = Model(num_classes=1, reg_max=16, num_kpts=2).to(self.device)
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if "model" in state:
            state = state["model"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()
        self.decoder = DecodeModule(reg_max=16, num_kpts=2).to(self.device).eval()
        print(f"📦 PyTorch FP32 모델 로드 완료: {checkpoint} [{self.device}]")

    @torch.inference_mode()
    def detect(self, color_img):
        _, one2one = self.model(prepare_pytorch_input(color_img, self.device))
        cls_logits, reg_pred, kpt_pred = one2one
        bboxes, keypoints, vis_logits = self.decoder(reg_pred, kpt_pred)
        scores = cls_logits[0, :, 0].sigmoid()
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])

        if best_score < self.conf_threshold:
            return 0.0, None, None, None

        return (
            best_score,
            bboxes[0, best_idx].cpu().numpy(),
            keypoints[0, best_idx].cpu().numpy(),
            vis_logits[0, best_idx].sigmoid().cpu().numpy(),
        )


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


is_running = True


def main():
    import argparse
    import signal
    import posix_ipc
    from common.shm_camera import CameraShm
    from camera.gemini336 import color_shape, depth_shape
    from camera.camera_runner import SHM_NAME, SEM_NAME

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("openvino", "pytorch"), default="openvino")
    parser.add_argument("--checkpoint", default="runs/dual_v2/best.pt")
    parser.add_argument("--model", default="model_int8.xml")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    args = parser.parse_args()

    global is_running

    def signal_handler(signum, frame):
        global is_running
        print("\n🛑 [Detection Process] 종료 신호를 수신했습니다...")
        is_running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        camera_shm = CameraShm(
            shm_name=SHM_NAME,
            is_owner=False,
            color_shape=color_shape,
            depth_shape=depth_shape,
        )
    except FileNotFoundError:
        print(
            f"❌ [Error] Shared Memory ({SHM_NAME})를 찾을 수 없습니다. camera_runner_2.py를 먼저 실행하세요."
        )
        return

    try:
        camera_sem = posix_ipc.Semaphore(SEM_NAME)
    except posix_ipc.ExistError:
        print(f"❌ [Error] Semaphore ({SEM_NAME})를 찾을 수 없습니다.")
        camera_shm.close()
        return

    if args.backend == "pytorch":
        detector = PyTorchDetector(args.checkpoint, conf_threshold=args.confidence)
    else:
        detector = Detector(args.model, conf_threshold=args.confidence)
    detection_ema = DetectionEMA(args.ema_alpha)

    window_name = f"{args.backend.title()} Melon Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_sequence_id = 0
    try:
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
            last_sequence_id = current_seq

            current_frame = camera_shm.read()

            timestamp = current_frame.timestamp
            # CameraShm uses (width, height, channels); inference expects HWC.
            color = current_frame.color.reshape((720, 1280, 3)).copy()
            depth = current_frame.depth.reshape((720, 1280)).copy()

            score, bbox, kpts, vis = detector.detect(color)
            score, bbox, kpts, vis = detection_ema.update(score, bbox, kpts, vis)

            # The model input is 640x384: original camera pixels were halved,
            # then 12 pixels of vertical padding were added.
            display = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            if bbox is not None and kpts is not None and vis is not None:
                box = np.asarray(bbox, dtype=np.float32).reshape(4)
                points = np.asarray(kpts, dtype=np.float32).reshape(-1, 2)
                visibility = np.asarray(vis, dtype=np.float32).reshape(-1)
                if (
                    np.isfinite(box).all()
                    and np.isfinite(points).all()
                    and np.isfinite(visibility).all()
                    and len(points) >= 2
                    and len(visibility) >= 2
                ):
                    x1, y1, x2, y2 = box
                    x1 = int(np.clip(x1 * 2.0, 0, 1279))
                    x2 = int(np.clip(x2 * 2.0, 0, 1279))
                    y1 = int(np.clip((y1 - 12.0) * 2.0, 0, 719))
                    y2 = int(np.clip((y2 - 12.0) * 2.0, 0, 719))
                    if x2 > x1 and y2 > y1:
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

                        rendered_points = []
                        for point in points[:2]:
                            px = int(np.clip(point[0] * 2.0, 0, 1279))
                            py = int(np.clip((point[1] - 12.0) * 2.0, 0, 719))
                            rendered_points.append((px, py))

                        if np.all(visibility[:2] >= 0.5):
                            cv2.line(
                                display,
                                rendered_points[0],
                                rendered_points[1],
                                (0, 140, 255),
                                2,
                                cv2.LINE_AA,
                            )
                        for point, label, color_bgr, probability in zip(
                            rendered_points,
                            ("Body", "Knot"),
                            ((255, 0, 0), (0, 0, 255)),
                            visibility[:2],
                        ):
                            if probability >= 0.5:
                                cv2.circle(
                                    display, point, 5, color_bgr, -1, cv2.LINE_AA
                                )
                                cv2.putText(
                                    display,
                                    f"{label} {probability:.2f}",
                                    (point[0] + 7, point[1] - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color_bgr,
                                    1,
                                    cv2.LINE_AA,
                                )

            cv2.imshow(window_name, display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        print("\n[알림] Ctrl + C 입력으로 중단되었습니다.")

    finally:
        print("🛑 리소스 해제 중...")
        cv2.destroyAllWindows()
        camera_shm.close()
        try:
            camera_sem.close()
        except Exception:
            pass
        print("✅ [Detection Process] 정상 종료되었습니다.")


if __name__ == "__main__":
    main()
