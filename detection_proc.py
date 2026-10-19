import os
import sys
import time
import signal
import posix_ipc
import numpy as np
import cv2
import torch
import torchvision.ops

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common.shm_camera import CameraShm
from training.model import Model, CombinedModel

SEM_NAME = "/camera_sem"
SHM_NAME = "camera_shm"

COLOR_SHAPE = (1280, 720, 3)
DEPTH_SHAPE = (1280, 720, 1)

# 모델 해상도 및 패딩 파라미터
MODEL_W = 640
MODEL_H = 384
NET_IMG_H = 360  # Letterbox 유효 영역 높이
PAD_TOP = 12.0  # 상단 패딩 크기

CONF_THRESH = 0.9  # 러프한 오검출 방지를 위해 0.35로 조정
VIS_THRESH = 0.5
IOU_NMS_THRESH = 0.45

# BGR 색상 정의
COLOR_PRED_BOX = (0, 140, 255)  # 주황색 BBox
COLOR_KNOT = (0, 0, 255)  # 빨간색 (Knot - 꼭지)
COLOR_BODY = (255, 0, 0)  # 파란색 (Body - 몸통/중심)

is_running = True


class DetectionSmoother:
    """실시간 검출 결과의 프레임 간 지터링(덜덜 떨림)을 부드럽게 안정화하는 지수이동평균(EMA) 필터"""

    def __init__(self, alpha=0.65):
        self.alpha = alpha  # 최신 프레임 반영 비율 (0.6~0.7)
        self.prev_box = None
        self.prev_kpt = None
        self.lost_count = 0
        self.max_lost = 5

    def update(self, box, kpt):
        box_arr = np.array(box, dtype=np.float32)
        kpt_arr = np.array(kpt, dtype=np.float32)

        if self.prev_box is None or self.lost_count >= self.max_lost:
            self.prev_box = box_arr
            self.prev_kpt = kpt_arr
            self.lost_count = 0
            return box_arr, kpt_arr

        # EMA 평활화 적용
        smooth_box = self.alpha * box_arr + (1.0 - self.alpha) * self.prev_box
        smooth_kpt = self.alpha * kpt_arr + (1.0 - self.alpha) * self.prev_kpt

        self.prev_box = smooth_box
        self.prev_kpt = smooth_kpt
        self.lost_count = 0

        return smooth_box, smooth_kpt

    def reset(self):
        self.lost_count += 1
        if self.lost_count >= self.max_lost:
            self.prev_box = None
            self.prev_kpt = None


def draw_refined_box(
    img, x1, y1, x2, y2, color, label="Melon", score=0.0, corner_len=16
):
    """모서리가 강조된 정교한 타겟팅 BBox 및 헤더 라벨 렌더링"""
    # 1. 얇은 외곽 사각형
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # 2. 4개 모서리 코너 브래킷 (2px 굵기)
    thick = 2
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thick, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thick, cv2.LINE_AA)

    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thick, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thick, cv2.LINE_AA)

    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thick, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thick, cv2.LINE_AA)

    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thick, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thick, cv2.LINE_AA)

    # 3. 텍스트 라벨 박스 (반투명 어두운 배경)
    text = f"{label} {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    bx1 = x1
    by1 = max(y1 - th - 8, 4)
    bx2 = x1 + tw + 10
    by2 = max(y1, th + 12)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)
    cv2.putText(
        img,
        text,
        (bx1 + 5, by2 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_keypoint(img, pt, color, name, vis_score):
    """정밀 중심점을 확인할 수 있는 이중 원 키포인트 렌더링"""
    cx, cy = int(pt[0]), int(pt[1])
    # 외곽 흰색 링 + 내부 코어 원
    cv2.circle(img, (cx, cy), 6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, color, -1, cv2.LINE_AA)

    # 텍스트 라벨 (가독성을 위한 그림자 효과)
    txt = f"{name} {vis_score:.2f}"
    tx, ty = cx + 8, cy - 4
    cv2.putText(
        img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA
    )
    cv2.putText(
        img,
        txt,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def signal_handler(signum, frame):
    global is_running
    print("\n🛑 [Detection Process] 종료 신호를 수신했습니다...")
    is_running = False


def load_model(model_path="./runs/train/best_fixed.pt", device="cuda"):
    print(f"🚀 [Detection Process] AI CombinedModel 로딩 중... ({device})")
    base_model = Model(num_classes=1, reg_max=16, num_kpts=2)
    model = CombinedModel(base_model, reg_max=16, num_kpts=2).to(device)

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        if "model_state_dict" in ckpt:
            model.base_model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"✅ 모델 체크포인트 로드 완료: {model_path}")
    else:
        print(
            f"⚠️ 경고: 체크포인트 파일({model_path})이 없습니다. 가중치 없이 실행됩니다."
        )

    model.eval()
    return model


@torch.no_grad()
def main():
    global is_running

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path="./runs/train/best_fixed.pt", device=device)

    print("⏳ Shared Memory 및 Semaphore 연결 시도 중...")

    try:
        camera_shm = CameraShm(
            shm_name=SHM_NAME,
            is_owner=False,
            color_shape=COLOR_SHAPE,
            depth_shape=DEPTH_SHAPE,
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

    window_name = "Orbbec Gemini336 - Real-time Melon Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("🚀 [Detection Process] 구동 시작! 정확한 스케일 변환이 적용되었습니다.")

    real_frame_count = 0
    start_time = time.time()
    last_sequence_id = 0

    # smoother = DetectionSmoother(alpha=0.85)
    smoother = DetectionSmoother(alpha=1.0)

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

            frame = camera_shm.read()
            last_sequence_id = current_seq

            # (1280, 720, 3) -> (720, 1280, 3) HWC RGB
            color_img = frame.color.reshape((720, 1280, 3))
            orig_h, orig_w, _ = color_img.shape

            # 💡 1. Datasets.py 방식과 100% 일치하는 이미지 전처리
            # 1280x720 -> 640x360 리사이즈 후 상하 12px 패딩 추가 => (384, 640)
            img_resized = cv2.resize(color_img, (640, 360))
            img_padded = cv2.copyMakeBorder(
                img_resized,
                int(PAD_TOP),
                int(PAD_TOP),
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

            # Tensor 전처리 [1, 3, 384, 640]
            img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
            input_tensor = img_tensor.unsqueeze(0).to(device)

            # 💡 2. CombinedModel 추론 (scores, bboxes, kpts, vis_probs)
            scores, decoded_bboxes, decoded_kpts, vis_probs = model(input_tensor)

            scores_b = scores[0, :, 0]  # [5040]
            boxes_b = decoded_bboxes[0]  # [5040, 4]
            kpts_b = decoded_kpts[0]  # [5040, 2, 2]
            vis_b = vis_probs[0]  # [5040, 2]

            img_draw = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

            # 💡 3. NMS 기반 객체 필터링 (다중 앵커 노이즈 제거)
            conf_mask = scores_b >= CONF_THRESH
            scale_x = orig_w / float(MODEL_W)  # 2.0
            scale_y = orig_h / float(NET_IMG_H)  # 2.0

            if conf_mask.any():
                cand_indices = torch.nonzero(conf_mask, as_tuple=True)[0]
                cand_boxes = boxes_b[cand_indices]
                cand_scores = scores_b[cand_indices]

                # NMS 수행
                keep = torchvision.ops.nms(cand_boxes, cand_scores, IOU_NMS_THRESH)
                top_idx = cand_indices[keep[0]].item()
                score = scores_b[top_idx].item()

                p_box = boxes_b[top_idx].cpu().numpy()  # [4]
                p_kpt = kpts_b[top_idx].cpu().numpy()  # [2, 2] (Body, Knot)
                p_vis = vis_b[top_idx].cpu().numpy()  # [2]

                # 💡 4. 원본 1280x720 좌표계 복원
                raw_x1 = p_box[0] * scale_x
                raw_y1 = (p_box[1] - PAD_TOP) * scale_y
                raw_x2 = p_box[2] * scale_x
                raw_y2 = (p_box[3] - PAD_TOP) * scale_y

                raw_box = [raw_x1, raw_y1, raw_x2, raw_y2]

                raw_kpt = [
                    [p_kpt[0, 0] * scale_x, (p_kpt[0, 1] - PAD_TOP) * scale_y],  # Body
                    [p_kpt[1, 0] * scale_x, (p_kpt[1, 1] - PAD_TOP) * scale_y],  # Knot
                ]

                # 💡 5. 프레임 간 떨림(Jittering) 완화 (EMA Smoother)
                smooth_box, smooth_kpt = smoother.update(raw_box, raw_kpt)

                # 이미지 경계 클램핑
                px1 = int(np.clip(smooth_box[0], 0, orig_w - 1))
                py1 = int(np.clip(smooth_box[1], 0, orig_h - 1))
                px2 = int(np.clip(smooth_box[2], 0, orig_w - 1))
                py2 = int(np.clip(smooth_box[3], 0, orig_h - 1))

                pk_body = (
                    int(np.clip(smooth_kpt[0, 0], 0, orig_w - 1)),
                    int(np.clip(smooth_kpt[0, 1], 0, orig_h - 1)),
                )
                pk_knot = (
                    int(np.clip(smooth_kpt[1, 0], 0, orig_w - 1)),
                    int(np.clip(smooth_kpt[1, 1], 0, orig_h - 1)),
                )

                pk_body_vis = p_vis[0] >= VIS_THRESH
                pk_knot_vis = p_vis[1] >= VIS_THRESH

                # 💡 6. 세련된 타겟팅 BBox 및 키포인트 렌더링
                draw_refined_box(
                    img_draw,
                    px1,
                    py1,
                    px2,
                    py2,
                    COLOR_PRED_BOX,
                    label="Melon",
                    score=score,
                )

                # Body -> Knot 연결선
                if pk_body_vis and pk_knot_vis:
                    cv2.line(img_draw, pk_body, pk_knot, COLOR_PRED_BOX, 2, cv2.LINE_AA)

                # Body 키포인트 (파란색)
                if pk_body_vis:
                    draw_keypoint(img_draw, pk_body, COLOR_BODY, "Body", p_vis[0])

                # Knot 키포인트 (빨간색)
                if pk_knot_vis:
                    draw_keypoint(img_draw, pk_knot, COLOR_KNOT, "Knot", p_vis[1])

            else:
                smoother.reset()

            # FPS 및 정보 표시
            real_frame_count += 1
            elapsed_time = time.time() - start_time
            fps = real_frame_count / elapsed_time if elapsed_time > 0 else 0

            cv2.rectangle(img_draw, (10, 10), (520, 50), (0, 0, 0), -1)
            cv2.putText(
                img_draw,
                f"FPS: {fps:.1f} | SEQ: {current_seq} | Device: {device.type}",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, img_draw)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
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
