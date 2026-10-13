import os
import time
import tempfile
from pathlib import Path
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from datasets import (
    ChamaePackDataset,
    chamae_pack_collate_fn,
    find_melon_data,
    split_data,
)
from model import DecodeModule, Model

KEYPOINT_DECODE_VERSION = "anchor_stride_v1"
TRAINING_ASSIGNMENT_VERSION = "multi_gt_tal_v3_bgweight"


def save_checkpoint_atomic(checkpoint, path):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def compute_ciou_loss(pred_bboxes, target_bboxes, eps=1e-7):
    """
    매칭된 예측/정답 bbox 쌍의 CIoU loss 계산.
    입력 shape: [..., 4] -> 출력 shape: [...]
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = pred_bboxes.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target_bboxes.unbind(-1)

    w1 = (b1_x2 - b1_x1).clamp(min=eps)
    h1 = (b1_y2 - b1_y1).clamp(min=eps)
    w2 = (b2_x2 - b2_x1).clamp(min=eps)
    h2 = (b2_y2 - b2_y1).clamp(min=eps)

    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union & IoU
    union_area = (w1 * h1) + (w2 * h2) - inter_area + eps
    iou = inter_area / union_area

    # Enclosing Box
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw**2 + ch**2 + eps

    # Center Distance
    b1_xc, b1_yc = (b1_x1 + b1_x2) * 0.5, (b1_y1 + b1_y2) * 0.5
    b2_xc, b2_yc = (b2_x1 + b2_x2) * 0.5, (b2_y1 + b2_y2) * 0.5
    rho2 = (b1_xc - b2_xc) ** 2 + (b1_yc - b2_yc) ** 2

    # Aspect Ratio penalty
    v = (4.0 / (torch.pi**2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - (rho2 / c2 + alpha * v)
    return 1.0 - ciou


def assign_targets_with_tal(
    pred_scores,
    pred_bboxes,
    target_bboxes,
    gt_mask,
    anchor_points,
    alpha=0.5,
    beta=2.0,
    is_o2o=False,
):
    """각 anchor를 유효 GT 하나에 할당한다. 미할당 anchor는 -1."""
    batch_size, num_anchors, _ = pred_scores.shape
    max_objs = target_bboxes.size(1)
    assigned_gt = torch.full(
        (batch_size, num_anchors), -1, dtype=torch.long, device=pred_scores.device
    )
    assigned_iou = pred_scores.new_zeros((batch_size, num_anchors))
    if max_objs == 0:
        return assigned_gt, assigned_iou

    anchors = anchor_points.expand(batch_size, -1, -1)
    ax, ay = anchors[..., 0:1], anchors[..., 1:2]
    gx1 = target_bboxes[:, None, :, 0]
    gy1 = target_bboxes[:, None, :, 1]
    gx2 = target_bboxes[:, None, :, 2]
    gy2 = target_bboxes[:, None, :, 3]
    valid_gt = gt_mask[:, None, :]

    # 앵커 중심점이 GT 박스 내부에 있는지 확인
    in_gt = (ax >= gx1) & (ax <= gx2) & (ay >= gy1) & (ay <= gy2) & valid_gt

    px1, py1, px2, py2 = pred_bboxes.unbind(-1)
    inter_x1 = torch.maximum(px1.unsqueeze(-1), gx1)
    inter_y1 = torch.maximum(py1.unsqueeze(-1), gy1)
    inter_x2 = torch.minimum(px2.unsqueeze(-1), gx2)
    inter_y2 = torch.minimum(py2.unsqueeze(-1), gy2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    gt_area = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    overlaps = inter_area / (pred_area.unsqueeze(-1) + gt_area - inter_area + 1e-7)
    overlaps = overlaps.clamp(min=0.0, max=1.0)

    # TAL 얼라인먼트 스코어 계산 (Classification^alpha * IoU^beta)
    confidence = pred_scores.detach().sigmoid().amax(dim=-1, keepdim=True)
    alignment = confidence.pow(alpha) * overlaps.pow(beta)
    alignment = alignment.masked_fill(~in_gt, -1.0)

    # 1단계: O2M 방식으로 상위 10개(Top-K) 후보군 선별 (공통 과정)
    topk = min(10, num_anchors)
    topk_values, topk_indices = alignment.topk(topk, dim=1)
    candidates = torch.zeros_like(alignment, dtype=torch.bool)
    candidates.scatter_(1, topk_indices, topk_values >= 0)
    candidates &= in_gt

    if is_o2o:
        # ⭐ [Consistent Matching] O2M 후보군(candidates) 내에서만 이분 매칭 수행
        candidate_alignment = alignment.masked_fill(~candidates, -1.0)

        # Batch 루프를 돌며 각 객체당 후보군 중 최고의 앵커 딱 1개씩 선정
        for b in range(batch_size):
            gt_indices = gt_mask[b].nonzero(as_tuple=True)[0]
            if gt_indices.numel() == 0:
                continue

            used_anchors = torch.zeros(
                num_anchors, dtype=torch.bool, device=pred_scores.device
            )

            # 각 GT 객체별로 후보군 중 가장 얼라인먼트 점수가 높은 앵커 선정
            for gt_idx in gt_indices:
                gt_val = candidate_alignment[b, :, gt_idx].clone()
                gt_val[used_anchors] = -1.0  # 이미 다른 GT가 선점한 앵커는 제외

                anchor_idx = gt_val.argmax()
                if gt_val[anchor_idx] >= 0:
                    assigned_gt[b, anchor_idx] = gt_idx
                    assigned_iou[b, anchor_idx] = overlaps[b, anchor_idx, gt_idx]
                    used_anchors[anchor_idx] = True
    else:
        # 기존 전통적인 O2M 다중 할당 방식
        candidate_alignment = alignment.masked_fill(~candidates, -1.0)
        best_alignment, best_gt = candidate_alignment.max(dim=2)
        positive = best_alignment >= 0
        batch_indices, anchor_indices = positive.nonzero(as_tuple=True)
        assigned_gt[batch_indices, anchor_indices] = best_gt[
            batch_indices, anchor_indices
        ]

        matched_iou = overlaps.gather(2, best_gt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        assigned_iou[batch_indices, anchor_indices] = matched_iou[
            batch_indices, anchor_indices
        ]

    return assigned_gt, assigned_iou


def compute_cls_loss_with_tal(
    pred_scores,
    pred_bboxes,
    target_bboxes,
    gt_mask,
    anchor_points,
    alpha=0.5,
    beta=2.0,
    is_o2o=False,
):
    assigned_gt, assigned_iou = assign_targets_with_tal(
        pred_scores,
        pred_bboxes,
        target_bboxes,
        gt_mask,
        anchor_points,
        alpha=alpha,
        beta=beta,
        is_o2o=is_o2o,
    )
    mask_pos = assigned_gt >= 0
    target_scores = torch.zeros_like(pred_scores)
    target_scores[..., 0] = torch.where(
        mask_pos, assigned_iou.clamp(min=0.3, max=1.0), 0.0
    )

    bce = F.binary_cross_entropy_with_logits(
        pred_scores, target_scores, reduction="none"
    ).squeeze(-1)
    pos_count = mask_pos.sum(dim=1)
    neg_mask = ~mask_pos
    pos_loss = (bce * mask_pos).sum(dim=1) / pos_count.clamp(min=1)
    neg_loss = (bce * neg_mask).sum(dim=1) / neg_mask.sum(dim=1).clamp(min=1)
    if is_o2o:
        # O2O 헤드는 중복 박스(배경 점수)를 강하게 억제해야 하므로 1.0 온전히 사용
        neg_weight = torch.ones_like(pos_count, dtype=bce.dtype)
    else:
        # O2M 헤드는 기존대로 배경 손실을 낮춤
        neg_weight = torch.where(
            pos_count > 0, torch.full_like(pos_count, 0.01, dtype=bce.dtype), 1.0
        )
    return (pos_loss + neg_weight * neg_loss).mean(), assigned_gt


def compute_bbox_loss(pred_bboxes, target_bboxes, assigned_gt):
    mask_pos = assigned_gt >= 0
    if not mask_pos.any():
        return pred_bboxes.sum() * 0.0

    batch_indices = torch.arange(
        pred_bboxes.size(0), device=pred_bboxes.device
    ).unsqueeze(1)
    matched_targets = target_bboxes[batch_indices, assigned_gt.clamp(min=0)]
    ciou = compute_ciou_loss(pred_bboxes, matched_targets)
    return (ciou * mask_pos).sum() / mask_pos.sum().clamp(min=1)


def compute_kpt_loss(pred_kpts, decoded_kpts, target_kpts, target_vis, assigned_gt):
    """
    할당된 정답 객체의 Keypoint Smooth L1 및 Visibility BCE Loss 계산
    - pred_kpts:     [B, 5040, 6] -> (dx, dy, vis_logit) x 2
    - decoded_kpts:  [B, 5040, 2, 2] (pixels: x, y)
    - target_kpts:   [B, max_objs, 2, 2] (pixels)
    - target_vis:    [B, max_objs, 2]
    - assigned_gt:   [B, 5040] (-1: negative, otherwise GT index)
    """
    B = pred_kpts.size(0)
    kpt_reshaped = pred_kpts.view(B, pred_kpts.size(1), 2, 3)
    mask_pos = assigned_gt >= 0
    if not mask_pos.any():
        zero = pred_kpts.sum() * 0.0
        return zero, zero

    batch_indices = torch.arange(B, device=pred_kpts.device).unsqueeze(1)
    matched_kpts = target_kpts[batch_indices, assigned_gt.clamp(min=0)]
    matched_vis = target_vis[batch_indices, assigned_gt.clamp(min=0)]
    visible = (matched_vis > 0).to(decoded_kpts.dtype)

    coord_loss = (
        F.smooth_l1_loss(decoded_kpts, matched_kpts, reduction="none").sum(dim=-1)
        / 640.0
    )
    coord_loss = coord_loss * visible * mask_pos.unsqueeze(-1)
    kpt_loss = coord_loss.sum() / (
        (visible * mask_pos.unsqueeze(-1)).sum().clamp(min=1)
    )

    visibility_loss = F.binary_cross_entropy_with_logits(
        kpt_reshaped[..., 2],
        (matched_vis > 0).to(decoded_kpts.dtype),
        reduction="none",
    )
    visibility_loss = visibility_loss * mask_pos.unsqueeze(-1)
    vis_loss = visibility_loss.sum() / (
        mask_pos.sum().clamp(min=1) * target_vis.size(-1)
    )
    return kpt_loss, vis_loss


def compute_ap(recalls, precisions):
    """Precision-Recall 곡선 하 면적(AP) 계산 (All-Point 적분)"""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


@torch.no_grad()
def evaluate_metrics(
    model,
    dataloader,
    decoder,
    device,
    iou_thresh=0.5,
    kpt_dist_thresh=0.10,
    conf_thresh=0.25,  # 신뢰도 임계값 상향
    max_detections=100,
):
    """검증 데이터셋에 대한 BBox mAP@50 및 Keypoint 정확도(PCK@0.10) 평가"""
    model.eval()

    eval_preds = []
    total_gt_boxes = 0

    total_gt_kpts = 0
    evaluated_gt_kpts = 0
    correct_kpts = 0
    total_kpt_dist_px = 0.0

    for batch in dataloader:
        images, targets = batch
        images = images.to(device, non_blocking=True)
        target_bboxes = targets["gt_bboxes"].to(device, non_blocking=True).clone()
        target_kpts = targets["gt_keypoints"].to(device, non_blocking=True).clone()
        target_vis = targets["gt_vis"].to(device, non_blocking=True).clone()
        gt_mask = targets["gt_mask"].to(device, non_blocking=True)
        if gt_mask.ndim == 3 and gt_mask.size(-1) == 1:
            gt_mask = gt_mask.squeeze(-1)

        # 정답 픽셀 복원
        target_bboxes[..., [0, 2]] *= 640.0
        target_bboxes[..., [1, 3]] *= 384.0
        target_kpts[..., 0] *= 640.0
        target_kpts[..., 1] *= 384.0

        # 듀얼 헤드 아웃풋에서 명확하게 O2O(1번째 인덱스)만 언패킹
        o2o_cls, o2o_regs, o2o_kpts = model(images)

        decoded_bboxes = decoder.decode_regs(o2o_regs)
        decoded_kpts, _ = decoder.decode_keypoints(o2o_kpts)
        scores = torch.sigmoid(o2o_cls).squeeze(-1)  # [B, Num_Anchors]

        B = images.size(0)
        for b in range(B):
            # GPU 메모리 누수 방지 및 정확한 인덱싱을 위해 .cpu().numpy() 사용
            gt_indices = gt_mask[b].nonzero(as_tuple=True)[0].cpu().numpy()
            gt_boxes = target_bboxes[b, gt_indices]
            gt_kpts = target_kpts[b, gt_indices]
            gt_vis = target_vis[b, gt_indices]
            total_gt_boxes += len(gt_indices)
            total_gt_kpts += (gt_vis > 0).sum().item()

            candidate_indices = (scores[b] >= conf_thresh).nonzero(as_tuple=True)[0]
            if candidate_indices.numel() > max_detections:
                top_order = scores[b, candidate_indices].topk(max_detections).indices
                candidate_indices = candidate_indices[top_order]
            keep_indices = candidate_indices.cpu().numpy()

            matched_gt = torch.zeros(len(gt_indices), dtype=torch.bool, device=device)
            for pred_idx in keep_indices:
                pred_score = scores[b, pred_idx].item()
                if len(gt_indices) == 0:
                    eval_preds.append((pred_score, False))
                    continue

                ious = box_iou(
                    decoded_bboxes[b, pred_idx].unsqueeze(0), gt_boxes
                ).squeeze(0)
                ious = ious.masked_fill(matched_gt, -1.0)
                best_iou, local_gt_idx = ious.max(dim=0)
                is_tp = best_iou.item() >= iou_thresh
                eval_preds.append((pred_score, is_tp))
                if not is_tp:
                    continue

                matched_gt[local_gt_idx] = True
                gt_box = gt_boxes[local_gt_idx]
                gt_kpt = gt_kpts[local_gt_idx]
                gt_kpt_vis = gt_vis[local_gt_idx]
                pred_kpt = decoded_kpts[b, pred_idx]

                gt_w = (gt_box[2] - gt_box[0]).clamp(min=1.0).item()
                gt_h = (gt_box[3] - gt_box[1]).clamp(min=1.0).item()
                box_scale = max(gt_w, gt_h)
                for k in range(2):
                    if gt_kpt_vis[k] > 0:
                        evaluated_gt_kpts += 1
                        dx = pred_kpt[k, 0].item() - gt_kpt[k, 0].item()
                        dy = pred_kpt[k, 1].item() - gt_kpt[k, 1].item()
                        dist = np.sqrt(dx**2 + dy**2)
                        total_kpt_dist_px += dist
                        if dist / box_scale <= kpt_dist_thresh:
                            correct_kpts += 1

    # 지표 계산 단계
    eval_preds.sort(key=lambda x: x[0], reverse=True)
    tps = np.array([1 if x[1] else 0 for x in eval_preds])
    fps = 1 - tps
    cum_tps = np.cumsum(tps)
    cum_fps = np.cumsum(fps)

    recalls = cum_tps / max(total_gt_boxes, 1)
    precisions = cum_tps / np.maximum(cum_tps + cum_fps, 1e-6)
    bbox_map50 = compute_ap(recalls, precisions) * 100.0

    kpt_accuracy = (correct_kpts / max(total_gt_kpts, 1)) * 100.0
    kpt_mae_px = total_kpt_dist_px / max(evaluated_gt_kpts, 1)

    fitness = (bbox_map50 * 0.5) + (kpt_accuracy * 0.5)

    # ⭐ [핵심 수정] 모든 batch 루프가 끝나고 결과 딕셔너리를 확실하게 리턴하도록 들여쓰기 정렬
    return {
        "bbox_map50": bbox_map50,
        "kpt_acc": kpt_accuracy,
        "kpt_mae_px": kpt_mae_px,
        "fitness": fitness,
        "total_gt_boxes": total_gt_boxes,
        "total_gt_kpts": total_gt_kpts,
    }


def save_validation_visualization(
    model,
    val_dataset,
    decoder,
    device,
    epoch,
    fitness,
    bbox_map,
    kpt_acc,
    num_samples=6,
    save_dir=Path("./runs/visualizations"),
):
    """성능 개선 시 검증 데이터셋 추론 결과를 이미지 그리드로 시각화하여 저장"""
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    vis_images = []
    for idx in range(min(num_samples, len(val_dataset))):
        img_t, _, bbox_t, kpt_t, vis_t = val_dataset[idx]

        gt_box = bbox_t.clone()
        gt_box[..., [0, 2]] *= 640.0
        gt_box[..., [1, 3]] *= 384.0
        gt_kpt = kpt_t.clone()
        gt_kpt[..., 0] *= 640.0
        gt_kpt[..., 1] *= 384.0

        input_t = img_t.unsqueeze(0).to(device)
        with torch.no_grad():
            # 듀얼 헤드 아웃풋에서 명확하게 O2O 결과만 추출
            outputs = model(input_t)
            cls_p, reg_p, kpt_p = model(input_t)

            dec_box = decoder.decode_regs(reg_p)
            dec_kpt, _ = decoder.decode_keypoints(kpt_p)
            scores = torch.sigmoid(cls_p).squeeze(-1)[0]

            # 신뢰도 커트라인을 넘긴 인덱스를 cpu numpy로 안전하게 변환
            keep_indices = (scores >= 0.25).nonzero(as_tuple=True)
            if keep_indices[0].numel() > 100:
                top_order = scores[keep_indices].topk(100).indices
                keep_indices = keep_indices[0][top_order]
            keep_indices = (
                keep_indices[0].cpu().numpy()
                if isinstance(keep_indices, tuple)
                else keep_indices.cpu().numpy()
            )

        img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 1. Ground Truth (초록색)
        for obj_idx in range(len(gt_box)):
            gx1, gy1, gx2, gy2 = map(int, gt_box[obj_idx])
            cv2.rectangle(img_bgr, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            gk0 = (int(gt_kpt[obj_idx, 0, 0]), int(gt_kpt[obj_idx, 0, 1]))
            gk1 = (int(gt_kpt[obj_idx, 1, 0]), int(gt_kpt[obj_idx, 1, 1]))
            if vis_t[obj_idx, 0] > 0 and vis_t[obj_idx, 1] > 0:
                cv2.line(img_bgr, gk0, gk1, (0, 255, 0), 1, cv2.LINE_AA)
            if vis_t[obj_idx, 0] > 0:
                cv2.circle(img_bgr, gk0, 4, (0, 255, 128), -1)
            if vis_t[obj_idx, 1] > 0:
                cv2.circle(img_bgr, gk1, 4, (0, 255, 255), -1)

        # 2. Prediction (주황/빨간색)
        for pred_idx in keep_indices:
            score = scores[pred_idx].item()
            p_box = dec_box[0, pred_idx].cpu().numpy()
            p_kpt = dec_kpt[0, pred_idx].cpu().numpy()
            px1, py1, px2, py2 = map(int, p_box)
            cv2.rectangle(img_bgr, (px1, py1), (px2, py2), (0, 140, 255), 2)
            cv2.putText(
                img_bgr,
                f"Melon {score:.2f}",
                (px1, max(py1 - 6, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 140, 255),
                1,
                cv2.LINE_AA,
            )
            # ⭐ [버그 수정] 키포인트 인덱스(x, y 차원 고정) 정상화
            pk0 = (int(p_kpt[0, 0]), int(p_kpt[0, 1]))
            pk1 = (int(p_kpt[1, 0]), int(p_kpt[1, 1]))
            cv2.line(img_bgr, pk0, pk1, (0, 140, 255), 2, cv2.LINE_AA)
            cv2.circle(img_bgr, pk0, 5, (0, 0, 255), -1)
            cv2.circle(img_bgr, pk1, 5, (255, 0, 0), -1)

        vis_images.append(img_bgr)

    if not vis_images:
        return None, None

    if len(vis_images) >= 6:
        row1 = np.hstack(vis_images[:3])
        row2 = np.hstack(vis_images[3:6])
        grid = np.vstack([row1, row2])
    elif len(vis_images) >= 4:
        row1 = np.hstack(vis_images[:2])
        row2 = np.hstack(vis_images[2:4])
        grid = np.vstack([row1, row2])
    else:
        grid = np.hstack(vis_images)

    banner_h = 40
    banner = np.zeros((banner_h, grid.shape[1], 3), dtype=np.uint8)
    banner_text = (
        f"[Epoch {epoch:03d}] Fitness: {fitness:.2f} | BBox mAP@50: {bbox_map:.2f}% | "
        f"Kpt Acc: {kpt_acc:.2f}%  (Green: Ground Truth, Orange: Prediction)"
    )
    cv2.putText(
        banner,
        banner_text,
        (15, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    final_img = np.vstack([banner, grid])

    epoch_save_path = save_dir / f"val_epoch_{epoch:03d}_fit_{fitness:.1f}.png"
    latest_save_path = Path("./runs/train") / "best_val_vis.png"

    cv2.imwrite(str(epoch_save_path), final_img)
    cv2.imwrite(str(latest_save_path), final_img)

    # ⭐ [핵심 수정] 튜플 분해 에러(TypeError)를 막기 위해 경로 2개를 정상 리턴하도록 정렬
    return epoch_save_path, latest_save_path


def main():
    DATA_DIR = "./data"
    SAVE_DIR = Path("./runs/train")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR = Path("./runs/visualizations")
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    TOTAL_EPOCHS = 2000
    BATCH_SIZE = 16
    LR = 1e-3
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 학습 디바이스: {DEVICE}")

    # 1. 공용 디코더 및 384x640 해상도 anchor 생성
    decoder = DecodeModule().to(DEVICE)
    anchor_points = decoder.anchors.unsqueeze(0) * decoder.stride_tensor.unsqueeze(0)

    # 2. 데이터 분할 및 Dataset 생성
    data_list = find_melon_data(DATA_DIR)
    train_paths, val_paths = split_data(data_list, train_ratio=0.9, seed=42)
    print(
        f"📊 Dataset: Total {len(data_list)} | Train {len(train_paths)} | Val {len(val_paths)}"
    )

    train_dataset = ChamaePackDataset(train_paths, is_train=True)
    val_dataset = ChamaePackDataset(val_paths, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=chamae_pack_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=chamae_pack_collate_fn,
    )

    # 3. 모델 및 옵티마이저 생성
    model = Model(num_classes=1, reg_max=16, num_kpts=2).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-5
    )

    start_epoch = 1
    best_fitness = 0.0

    resume_candidate_last = SAVE_DIR / "last_fixed.pt"
    resume_candidate_best = SAVE_DIR / "best_fixed.pt"

    ckpt = None
    ckpt_path = None
    fallback_checkpoint = None
    model_state = model.state_dict()
    for candidate in (resume_candidate_last, resume_candidate_best):
        if not candidate.exists():
            continue
        print(f"\n🔄 [체크포인트 로드] {candidate} 확인 중...")
        try:
            candidate_ckpt = torch.load(candidate, map_location="cpu")
            if not isinstance(candidate_ckpt, dict) or not isinstance(
                candidate_ckpt.get("model_state_dict"), dict
            ):
                raise ValueError("필수 model_state_dict가 없는 체크포인트입니다.")
            candidate_state = candidate_ckpt["model_state_dict"]
            if candidate_state.keys() != model_state.keys() or any(
                candidate_state[key].shape != model_state[key].shape
                for key in model_state
            ):
                raise ValueError("모델 파라미터 구성이 현재 모델과 일치하지 않습니다.")
        except Exception as e:
            print(f"⚠️ 체크포인트를 사용할 수 없습니다: {candidate} ({e})")
            continue
        versions_match = (
            candidate_ckpt.get("keypoint_decode_version") == KEYPOINT_DECODE_VERSION
            and candidate_ckpt.get("training_assignment_version")
            == TRAINING_ASSIGNMENT_VERSION
        )
        if versions_match:
            ckpt_path = candidate
            break
        fitness = candidate_ckpt.get("fitness", candidate_ckpt.get("best_fitness", 0.0))
        if fallback_checkpoint is None or fitness > fallback_checkpoint[0]:
            fallback_checkpoint = (fitness, candidate)

    if ckpt_path is None and fallback_checkpoint is not None:
        ckpt_path = fallback_checkpoint[1]

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])

    if ckpt is not None:
        print(f"✅ [체크포인트 로드 완료] {ckpt_path}")
        if (
            ckpt.get("keypoint_decode_version") == KEYPOINT_DECODE_VERSION
            and ckpt.get("training_assignment_version") == TRAINING_ASSIGNMENT_VERSION
        ):
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception as e:
                    print(f"⚠️ Optimizer 상태 복원 생략: {e}")
            if "scheduler_state_dict" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except Exception as e:
                    print(f"⚠️ Scheduler 상태 복원 생략: {e}")

            start_epoch = ckpt.get("epoch", 0) + 1
            best_fitness = ckpt.get("fitness", ckpt.get("best_fitness", 0.0))
            print(
                f"✅ 학습 재개 준비 완료 -> 시작 에폭: {start_epoch}/{TOTAL_EPOCHS} | "
                f"현재 최고 Fitness: {best_fitness:.2f}\n"
            )
        else:
            print(
                "⚠️ 학습 기준이 다른 체크포인트입니다. "
                "가중치만 불러오고 optimizer/epoch는 초기화해 새 기준으로 재학습합니다.\n"
            )
    else:
        if resume_candidate_last.exists() or resume_candidate_best.exists():
            print(
                "⚠️ 사용 가능한 체크포인트가 없습니다. "
                "모델과 optimizer를 초기화하고 Epoch 1부터 학습합니다.\n"
            )
        else:
            print(
                "\n✨ 이전 체크포인트가 없습니다. Epoch 1부터 새로 학습을 시작합니다.\n"
            )

    print("🏁 [학습 시작] BBox(CIoU) + Anchor-based Keypoint(Smooth L1) 학습 활성화\n")

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        model.train()
        epoch_cls_loss = 0.0
        epoch_bbox_loss = 0.0
        epoch_kpt_loss = 0.0
        epoch_total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch:03d}/{TOTAL_EPOCHS}]")
        for batch in pbar:
            optimizer.zero_grad()

            input_tensor, targets = batch
            input_tensor = input_tensor.to(DEVICE, non_blocking=True)
            target_bboxes = targets["gt_bboxes"].to(DEVICE, non_blocking=True).clone()
            target_kpts = targets["gt_keypoints"].to(DEVICE, non_blocking=True).clone()
            target_vis = targets["gt_vis"].to(DEVICE, non_blocking=True).clone()
            gt_mask = targets["gt_mask"].to(DEVICE, non_blocking=True)
            if gt_mask.ndim == 3 and gt_mask.size(-1) == 1:
                gt_mask = gt_mask.squeeze(-1)

            # 정답 픽셀 좌표 복원
            target_bboxes[..., [0, 2]] *= 640.0
            target_bboxes[..., [1, 3]] *= 384.0
            target_kpts[..., 0] *= 640.0
            target_kpts[..., 1] *= 384.0

            preds_m, preds_o = model(input_tensor)
            o2m_cls, o2m_regs, o2m_kpts = preds_m
            o2o_cls, o2o_regs, o2o_kpts = preds_o

            decoded_o2m_regs = decoder.decode_regs(o2m_regs)
            decoded_o2o_regs = decoder.decode_regs(o2o_regs)

            # 1. TAL 분류 손실
            loss_o2m_cls, mask_o2m = compute_cls_loss_with_tal(
                o2m_cls,
                decoded_o2m_regs,
                target_bboxes,
                gt_mask,
                anchor_points,
                alpha=0.5,
                beta=2.0,
                is_o2o=False,
            )
            loss_o2o_cls, mask_o2o = compute_cls_loss_with_tal(
                o2o_cls,
                decoded_o2o_regs,
                target_bboxes,
                gt_mask,
                anchor_points,
                alpha=0.5,
                beta=2.0,
                is_o2o=True,
            )
            total_cls_loss = loss_o2m_cls + loss_o2o_cls

            decoded_o2m_kpts, _ = decoder.decode_keypoints(o2m_kpts)
            decoded_o2o_kpts, _ = decoder.decode_keypoints(o2o_kpts)

            # 2. CIoU 회귀 손실
            ciou_loss_o2m = compute_bbox_loss(decoded_o2m_regs, target_bboxes, mask_o2m)
            ciou_loss_o2o = compute_bbox_loss(decoded_o2o_regs, target_bboxes, mask_o2o)
            total_ciou_loss = ciou_loss_o2m + ciou_loss_o2o

            # 3. Keypoint 회귀 및 Visibility 손실
            loss_kpt_o2m, loss_vis_o2m = compute_kpt_loss(
                o2m_kpts,
                decoded_o2m_kpts,
                target_kpts,
                target_vis,
                mask_o2m,
            )
            loss_kpt_o2o, loss_vis_o2o = compute_kpt_loss(
                o2o_kpts,
                decoded_o2o_kpts,
                target_kpts,
                target_vis,
                mask_o2o,
            )
            total_kpt_loss = loss_kpt_o2m + loss_kpt_o2o
            total_vis_loss = loss_vis_o2m + loss_vis_o2o

            # 4. 종합 Loss
            total_loss = (
                (total_cls_loss * 1.0)
                + (total_ciou_loss * 5.0)
                + (total_kpt_loss * 10.0)
                + (total_vis_loss * 1.0)
            )

            total_loss.backward()
            optimizer.step()

            epoch_cls_loss += total_cls_loss.item()
            epoch_bbox_loss += total_ciou_loss.item()
            epoch_kpt_loss += total_kpt_loss.item()
            epoch_total_loss += total_loss.item()

            pbar.set_postfix(
                {
                    "Total": f"{total_loss.item():.2f}",
                    "Cls": f"{total_cls_loss.item():.2f}",
                    "Box": f"{total_ciou_loss.item():.2f}",
                    "Kpt": f"{total_kpt_loss.item():.2f}",
                }
            )

        scheduler.step()
        num_batches = len(train_loader)
        avg_train_cls = epoch_cls_loss / num_batches
        avg_train_bbox = epoch_bbox_loss / num_batches
        avg_train_kpt = epoch_kpt_loss / num_batches
        avg_train_total = epoch_total_loss / num_batches

        # 5. Validation 평가
        metrics = evaluate_metrics(
            model=model,
            dataloader=val_loader,
            decoder=decoder,
            device=DEVICE,
            iou_thresh=0.5,
            kpt_dist_thresh=0.10,
        )

        bbox_map = metrics["bbox_map50"]
        kpt_acc = metrics["kpt_acc"]
        kpt_mae_px = metrics["kpt_mae_px"]
        curr_fitness = metrics["fitness"]

        print(
            f" Epoch [{epoch:03d}] | "
            f"Train [Loss: {avg_train_total:.3f}, Cls: {avg_train_cls:.3f}, Box: {avg_train_bbox:.3f}, Kpt: {avg_train_kpt:.3f}] | "
            f"Val [BBox mAP@50: {bbox_map:.2f}%, Kpt Acc: {kpt_acc:.2f}%, "
            f"Kpt MAE: {kpt_mae_px:.2f}px, Fitness: {curr_fitness:.2f}]"
        )

        # 6. 체크포인트 저장
        save_checkpoint_atomic(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "bbox_map50": bbox_map,
                "kpt_acc": kpt_acc,
                "fitness": curr_fitness,
                "best_fitness": best_fitness,
                "keypoint_decode_version": KEYPOINT_DECODE_VERSION,
                "training_assignment_version": TRAINING_ASSIGNMENT_VERSION,
            },
            SAVE_DIR / "last_fixed.pt",
        )

        # 7. 성능 개선 시 Best 저장 및 추론 시각화 저장
        if curr_fitness > best_fitness:
            best_fitness = curr_fitness
            save_path = SAVE_DIR / "best_fixed.pt"
            save_checkpoint_atomic(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "bbox_map50": bbox_map,
                    "kpt_acc": kpt_acc,
                    "fitness": curr_fitness,
                    "keypoint_decode_version": KEYPOINT_DECODE_VERSION,
                    "training_assignment_version": TRAINING_ASSIGNMENT_VERSION,
                },
                save_path,
            )
            print(
                f"  ⭐ [Best Model 갱신!] {save_path} "
                f"(mAP@50: {bbox_map:.2f}%, Kpt Acc: {kpt_acc:.2f}%, Fitness: {curr_fitness:.2f})"
            )

            vis_epoch_file, vis_latest_file = save_validation_visualization(
                model=model,
                val_dataset=val_dataset,
                decoder=decoder,
                device=DEVICE,
                epoch=epoch,
                fitness=curr_fitness,
                bbox_map=bbox_map,
                kpt_acc=kpt_acc,
                num_samples=6,
                save_dir=VIS_DIR,
            )
            print(
                f"  📸 [추론 시각화 저장 완료] {vis_epoch_file} (최신본: {vis_latest_file})"
            )


if __name__ == "__main__":
    main()
