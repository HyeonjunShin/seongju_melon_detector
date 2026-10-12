import os
import time
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

from datasets_fixed import (
    ChamaePackDataset,
    find_melon_data,
    split_data,
)
from model import PoseModel
from anchors import generate_anchors


def decode_regs(pred_regs, anchor_points, anchor_strides, reg_max=16):
    """
    Distribution Regression 출력을 픽셀 좌표 (xmin, ymin, xmax, ymax)로 디코딩
    - pred_regs:       [B, 5040, 4 * reg_max]
    - anchor_points:   [1, 5040, 2] (cx, cy)
    - anchor_strides:  [1, 5040, 1]
    """
    weight = torch.arange(reg_max, dtype=torch.float32, device=pred_regs.device)
    pred_regs = pred_regs.view(-1, 5040, 4, reg_max)
    pred_regs = torch.softmax(pred_regs, dim=-1)

    dist_pixel = (pred_regs @ weight) * anchor_strides.view(1, -1, 1)

    cx = anchor_points[..., 0]
    cy = anchor_points[..., 1]

    xmin = cx - dist_pixel[..., 0]
    ymin = cy - dist_pixel[..., 1]
    xmax = cx + dist_pixel[..., 2]
    ymax = cy + dist_pixel[..., 3]

    decoded_bboxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    return decoded_bboxes


def decode_keypoints(kpt_preds, decoded_bboxes, num_kpts=2):
    """
    상대 오프셋 키포인트 출력을 절대 픽셀 좌표로 디코딩
    - kpt_preds:      [B, 5040, num_kpts * 3] -> (dx, dy, vis_logit) x num_kpts
    - decoded_bboxes: [B, 5040, 4] -> (xmin, ymin, xmax, ymax)
    반환:
    - abs_kpts:       [B, 5040, num_kpts, 2] (pixels: x, y)
    - vis_logits:     [B, 5040, num_kpts]
    """
    B = kpt_preds.size(0)
    kpt_reshaped = kpt_preds.view(B, -1, num_kpts, 3)
    pred_rel_xy = kpt_reshaped[..., :2]  # [B, 5040, num_kpts, 2]
    vis_logits = kpt_reshaped[..., 2]  # [B, 5040, num_kpts]

    bbox_xc = (
        decoded_bboxes[..., 0:1] + decoded_bboxes[..., 2:3]
    ) * 0.5  # [B, 5040, 1]
    bbox_yc = (
        decoded_bboxes[..., 1:2] + decoded_bboxes[..., 3:4]
    ) * 0.5  # [B, 5040, 1]
    bbox_w = (decoded_bboxes[..., 2:3] - decoded_bboxes[..., 0:1]).clamp(min=1.0)
    bbox_h = (decoded_bboxes[..., 3:4] - decoded_bboxes[..., 1:2]).clamp(min=1.0)

    abs_kpt_x = bbox_xc.unsqueeze(-1) + (pred_rel_xy[..., 0:1] * bbox_w.unsqueeze(-1))
    abs_kpt_y = bbox_yc.unsqueeze(-1) + (pred_rel_xy[..., 1:2] * bbox_h.unsqueeze(-1))
    abs_kpts = torch.cat([abs_kpt_x, abs_kpt_y], dim=-1)  # [B, 5040, num_kpts, 2]

    return abs_kpts, vis_logits


def compute_ciou_loss(pred_bboxes, target_bboxes, eps=1e-7):
    """
    미분 가능한 표준 CIoU Loss 계산
    - pred_bboxes:   [B, 5040, 4]  -> (xmin, ymin, xmax, ymax)
    - target_bboxes: [B, 1, 4]     -> (xmin, ymin, xmax, ymax)
    반환 Shape: [B, 5040] (1.0 - ciou)
    """
    b1 = pred_bboxes.unsqueeze(1)  # [B, 1, 5040, 4]
    b2 = target_bboxes.unsqueeze(2)  # [B, 1, 1, 4]

    b1_x1, b1_y1, b1_x2, b1_y2 = b1[..., 0], b1[..., 1], b1[..., 2], b1[..., 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[..., 0], b2[..., 1], b2[..., 2], b2[..., 3]

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
    ciou_loss = (1.0 - ciou).squeeze(1)  # [B, 5040]
    return ciou_loss


def compute_cls_loss_with_tal(
    pred_scores,
    pred_bboxes,
    target_bboxes,
    anchor_points,
    alpha=0.5,
    beta=2.0,
    is_o2o=False,
):
    """
    개선된 TAL(Task Aligned Assigner) 기반 분류 손실 계산
    - pred_scores:   [B, 5040, 1] (logits)
    - pred_bboxes:   [B, 5040, 4] (pixels)
    - target_bboxes: [B, 1, 4] (pixels)
    - anchor_points: [1, 5040, 2] (pixels)
    """
    batch_size, num_anchors, num_classes = pred_scores.shape

    # 1. In-Box 필터링
    anc_x, anc_y = anchor_points[..., 0], anchor_points[..., 1]
    gt_x1, gt_y1 = target_bboxes[..., 0], target_bboxes[..., 1]
    gt_x2, gt_y2 = target_bboxes[..., 2], target_bboxes[..., 3]

    is_in_box = (
        (anc_x >= gt_x1) & (anc_x <= gt_x2) & (anc_y >= gt_y1) & (anc_y <= gt_y2)
    )

    # 2. IoU 계산 [B, 5040]
    b1_x1, b1_y1, b1_x2, b1_y2 = pred_bboxes.unbind(-1)
    inter_x1 = torch.max(b1_x1, gt_x1)
    inter_y1 = torch.max(b1_y1, gt_y1)
    inter_x2 = torch.min(b1_x2, gt_x2)
    inter_y2 = torch.min(b1_y2, gt_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    w1 = (b1_x2 - b1_x1).clamp(min=0)
    h1 = (b1_y2 - b1_y1).clamp(min=0)
    w2 = (gt_x2 - gt_x1).clamp(min=0)
    h2 = (gt_y2 - gt_y1).clamp(min=0)
    union_area = (w1 * h1) + (w2 * h2) - inter_area + 1e-7
    overlaps = (inter_area / union_area).clamp(min=0.0, max=1.0)

    # 3. Alignment Metric: (Classification Confidence ^ alpha) * (IoU ^ beta)
    pred_probs = pred_scores.detach().sigmoid().max(dim=-1)[0]
    alignment_metrics = (pred_probs**alpha) * (overlaps**beta)

    metrics_masked = alignment_metrics.clone()
    metrics_masked[~is_in_box] = -1e9

    # 4. 양성(Positive) 앵커 선택
    mask_pos = torch.zeros_like(is_in_box, dtype=torch.bool)
    if is_o2o:
        max_indices = metrics_masked.argmax(dim=-1, keepdim=True)
        mask_pos.scatter_(1, max_indices, True)
        mask_pos = mask_pos & is_in_box
    else:
        topk = min(10, num_anchors)
        _, topk_indices = torch.topk(metrics_masked, topk, dim=-1, largest=True)
        mask_pos.scatter_(1, topk_indices, True)
        mask_pos = mask_pos & is_in_box

    # 5. Soft Target 점수 부여 (초기 학습 시에도 양성 앵커가 0으로 죽지 않도록 최소 0.3 보장)
    target_scores = torch.zeros_like(pred_scores, dtype=torch.float)
    if mask_pos.any():
        pos_scores = overlaps[mask_pos].clamp(min=0.3, max=1.0)
        target_scores[mask_pos, 0] = pos_scores

    # 6. BCE Loss (Positive 정규화)
    bce = F.binary_cross_entropy_with_logits(
        pred_scores, target_scores, reduction="none"
    )
    pos_loss = (
        bce[mask_pos].sum()
        if mask_pos.any()
        else torch.tensor(0.0, device=pred_scores.device)
    )
    neg_loss = bce[~mask_pos].sum()
    num_pos = max(mask_pos.sum().item(), 1.0)

    cls_loss = (pos_loss / num_pos) + (
        0.01 * (neg_loss / max((~mask_pos).sum().item(), 1.0))
    )
    return cls_loss, mask_pos


def compute_kpt_loss(pred_kpts, target_kpts, target_vis, target_bboxes, mask_pos):
    """
    양성 앵커에 대한 Keypoint Offset Smooth L1 및 Visibility BCE Loss 계산
    - pred_kpts:     [B, 5040, 6] -> (dx, dy, vis_logit) x 2
    - target_kpts:   [B, 1, 2, 2] (pixels)
    - target_vis:    [B, 1, 2]    (0: invisible, 1 or 2: visible)
    - target_bboxes: [B, 1, 4]    (pixels: x1, y1, x2, y2)
    - mask_pos:      [B, 5040] (bool)
    """
    B = pred_kpts.size(0)
    kpt_reshaped = pred_kpts.view(B, 5040, 2, 3)
    pred_xy = kpt_reshaped[..., :2]  # [B, 5040, 2, 2]
    pred_vis_logits = kpt_reshaped[..., 2]  # [B, 5040, 2]

    # 정답 상대 오프셋 계산 (GT BBox 중심 및 크기 기준 정규화)
    gt_xc = (target_bboxes[..., 0] + target_bboxes[..., 2]) * 0.5  # [B, 1]
    gt_yc = (target_bboxes[..., 1] + target_bboxes[..., 3]) * 0.5  # [B, 1]
    gt_w = (target_bboxes[..., 2] - target_bboxes[..., 0]).clamp(min=1.0)
    gt_h = (target_bboxes[..., 3] - target_bboxes[..., 1]).clamp(min=1.0)

    target_dx = (target_kpts[..., 0] - gt_xc.unsqueeze(-1)) / gt_w.unsqueeze(-1)
    target_dy = (target_kpts[..., 1] - gt_yc.unsqueeze(-1)) / gt_h.unsqueeze(-1)
    target_offsets = torch.stack([target_dx, target_dy], dim=-1)  # [B, 1, 2, 2]

    vis_mask = (target_vis > 0).float()  # [B, 1, 2]

    total_kpt_loss = torch.tensor(0.0, device=pred_kpts.device)
    total_vis_loss = torch.tensor(0.0, device=pred_kpts.device)
    pos_count = 0

    for b in range(B):
        pos_idx = mask_pos[b].nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0:
            continue

        b_pred_xy = pred_xy[b, pos_idx]  # [num_pos, 2, 2]
        b_target_offset = target_offsets[b]  # [1, 2, 2]
        b_vis_mask = vis_mask[b]  # [1, 2]

        # Smooth L1 거리 손실
        l1 = F.smooth_l1_loss(
            b_pred_xy, b_target_offset.expand_as(b_pred_xy), reduction="none"
        )
        l1 = l1.sum(dim=-1)  # [num_pos, 2]

        valid_pts = b_vis_mask.sum().clamp(min=1.0)
        l1_masked = (l1 * b_vis_mask).sum() / (len(pos_idx) * valid_pts)
        total_kpt_loss = total_kpt_loss + l1_masked

        # Visibility 손실
        b_pred_vis = pred_vis_logits[b, pos_idx]
        vis_target = (target_vis[b] > 0).float().expand_as(b_pred_vis)
        vis_loss = F.binary_cross_entropy_with_logits(b_pred_vis, vis_target)
        total_vis_loss = total_vis_loss + vis_loss

        pos_count += 1

    if pos_count > 0:
        return total_kpt_loss / pos_count, total_vis_loss / pos_count
    return torch.tensor(0.0, device=pred_kpts.device), torch.tensor(
        0.0, device=pred_kpts.device
    )


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
    anchor_points,
    anchor_strides,
    device,
    iou_thresh=0.5,
    kpt_dist_thresh=0.10,
):
    """
    검증 데이터셋에 대한 BBox mAP@50 및 Keypoint 정확도(PCK@0.10) 평가
    - model.eval() 상태에서는 one2one_head (cls, regs, kpts) 3개 출력을 직접 반환합니다.
    """
    model.eval()

    eval_preds = []  # (confidence_score, is_tp_bool)
    total_gt_boxes = 0

    total_gt_kpts = 0
    correct_kpts = 0

    for batch in dataloader:
        images = batch[0].to(device, non_blocking=True)
        target_bboxes = batch[2].to(device, non_blocking=True).clone()
        target_kpts = batch[3].to(device, non_blocking=True).clone()
        target_vis = batch[4].to(device, non_blocking=True).clone()

        # 픽셀 복원 (384x640 공간)
        target_bboxes[..., [0, 2]] *= 640.0
        target_bboxes[..., [1, 3]] *= 384.0
        target_kpts[..., 0] *= 640.0
        target_kpts[..., 1] *= 384.0

        # eval 모드에서는 one2one_head의 (cls, regs, kpts)가 반환됨
        o2o_out = model(images)
        if isinstance(o2o_out, tuple) and len(o2o_out) == 2:
            o2o_cls, o2o_regs, o2o_kpts = o2o_out[1]
        else:
            o2o_cls, o2o_regs, o2o_kpts = o2o_out

        decoded_bboxes = decode_regs(o2o_regs, anchor_points, anchor_strides)
        decoded_kpts, _ = decode_keypoints(o2o_kpts, decoded_bboxes)
        scores = torch.sigmoid(o2o_cls).squeeze(-1)  # [B, 5040]

        B = images.size(0)
        for b in range(B):
            gt_box = target_bboxes[b, 0]  # [4]
            gt_kpt = target_kpts[b, 0]  # [2, 2]
            gt_vis = target_vis[b, 0]  # [2]

            total_gt_boxes += 1

            # NMS-Free: O2O 헤드에서 가장 높은 점수의 앵커 선택
            top_idx = scores[b].argmax()
            top_score = scores[b, top_idx].item()
            pred_box = decoded_bboxes[b, top_idx]  # [4]
            pred_kpt = decoded_kpts[b, top_idx]  # [2, 2]

            # 1. BBox IoU 계산
            iou = box_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0)).item()
            is_tp = iou >= iou_thresh
            eval_preds.append((top_score, is_tp))

            # 2. Keypoint 거리 오차 계산 (참외 크기 S = max(w, h) 대비 정규화)
            gt_w = (gt_box[2] - gt_box[0]).clamp(min=1.0).item()
            gt_h = (gt_box[3] - gt_box[1]).clamp(min=1.0).item()
            box_scale = max(gt_w, gt_h)

            for k in range(2):
                if gt_vis[k] > 0:
                    total_gt_kpts += 1
                    dx = pred_kpt[k, 0].item() - gt_kpt[k, 0].item()
                    dy = pred_kpt[k, 1].item() - gt_kpt[k, 1].item()
                    dist = np.sqrt(dx**2 + dy**2)
                    norm_dist = dist / box_scale

                    if norm_dist <= kpt_dist_thresh:
                        correct_kpts += 1

    # BBox mAP@50 계산
    eval_preds.sort(key=lambda x: x[0], reverse=True)
    tps = np.array([1 if x[1] else 0 for x in eval_preds])
    fps = 1 - tps
    cum_tps = np.cumsum(tps)
    cum_fps = np.cumsum(fps)

    recalls = cum_tps / max(total_gt_boxes, 1)
    precisions = cum_tps / np.maximum(cum_tps + cum_fps, 1e-6)
    bbox_map50 = compute_ap(recalls, precisions) * 100.0

    # Keypoint Accuracy (PCK@0.10) 계산
    kpt_accuracy = (correct_kpts / max(total_gt_kpts, 1)) * 100.0

    # 종합 평가지표 (BBox 50% + Keypoint 50%)
    fitness = (bbox_map50 * 0.5) + (kpt_accuracy * 0.5)

    return {
        "bbox_map50": bbox_map50,
        "kpt_acc": kpt_accuracy,
        "fitness": fitness,
        "total_gt_boxes": total_gt_boxes,
        "total_gt_kpts": total_gt_kpts,
    }


def save_validation_visualization(
    model,
    val_dataset,
    anchor_points,
    anchor_strides,
    device,
    epoch,
    fitness,
    bbox_map,
    kpt_acc,
    num_samples=6,
    save_dir=Path("./runs/visualizations"),
):
    """
    성능 개선 시 검증 데이터셋 추론 결과를 고품질 이미지 그리드로 시각화하여 저장
    - Ground Truth: 초록색 BBox 및 키포인트
    - Prediction:   주황색 BBox 및 키포인트 (Knot: 빨강, Body: 파랑)
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    vis_images = []
    # 항상 동일한 고정 샘플을 시각화하여 에폭별 성능 개선 추이를 직관적으로 비교
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
            out = model(input_t)
            if isinstance(out, tuple) and len(out) == 2:
                cls_p, reg_p, kpt_p = out[1]
            else:
                cls_p, reg_p, kpt_p = out

            dec_box = decode_regs(reg_p, anchor_points, anchor_strides)
            dec_kpt, _ = decode_keypoints(kpt_p, dec_box)
            scores = torch.sigmoid(cls_p).squeeze(-1)[0]
            top_idx = scores.argmax().item()

            score = scores[top_idx].item()
            p_box = dec_box[0, top_idx].cpu().numpy()
            p_kpt = dec_kpt[0, top_idx].cpu().numpy()

        img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 1. Ground Truth (초록색)
        gx1, gy1, gx2, gy2 = map(int, gt_box[0])
        cv2.rectangle(img_bgr, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        gk0 = (int(gt_kpt[0, 0, 0]), int(gt_kpt[0, 0, 1]))
        gk1 = (int(gt_kpt[0, 1, 0]), int(gt_kpt[0, 1, 1]))
        if vis_t[0, 0] > 0 and vis_t[0, 1] > 0:
            cv2.line(img_bgr, gk0, gk1, (0, 255, 0), 1, cv2.LINE_AA)
        if vis_t[0, 0] > 0:
            cv2.circle(img_bgr, gk0, 4, (0, 255, 128), -1)
        if vis_t[0, 1] > 0:
            cv2.circle(img_bgr, gk1, 4, (0, 255, 255), -1)

        # 2. Prediction (주황/빨간색)
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
        pk0 = (int(p_kpt[0, 0]), int(p_kpt[0, 1]))
        pk1 = (int(p_kpt[1, 0]), int(p_kpt[1, 1]))
        cv2.line(img_bgr, pk0, pk1, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.circle(img_bgr, pk0, 5, (0, 0, 255), -1)  # Knot: 빨간색
        cv2.circle(img_bgr, pk1, 5, (255, 0, 0), -1)  # Body: 파란색

        vis_images.append(img_bgr)

    # 2행 3열 그리드 생성 (총 6장: 2 x 3)
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

    # 상단 정보 배너 추가 (배경 검은색 바)
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

    # 1. 앵커 생성 (384x640 해상도)
    anchor_points, anchor_strides = generate_anchors(device=DEVICE)

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
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 3. 모델 및 옵티마이저 생성
    model = PoseModel(num_classes=1, reg_max=16, num_kpts=2).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-5
    )

    start_epoch = 1
    best_fitness = 0.0

    # ------------------------------------------------------------------
    # 💡 [Resume 기능]: 이전 학습 .pt 파일 존재 시 자동 재개
    # ------------------------------------------------------------------
    resume_candidate_last = SAVE_DIR / "last_fixed.pt"
    resume_candidate_best = SAVE_DIR / "best_fixed.pt"

    ckpt_path = None
    if resume_candidate_last.exists():
        ckpt_path = resume_candidate_last
    elif resume_candidate_best.exists():
        ckpt_path = resume_candidate_best

    if ckpt_path is not None:
        print(
            f"\n🔄 [체크포인트 로드] {ckpt_path} 발견! 이전 학습 상태를 복원합니다..."
        )
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
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
            f"✅ 학습 재개 준비 완료 -> 시작 에폭: {start_epoch}/{TOTAL_EPOCHS} | 현재 최고 Fitness: {best_fitness:.2f}\n"
        )
    else:
        print("\n✨ 이전 체크포인트가 없습니다. Epoch 1부터 새로 학습을 시작합니다.\n")

    print(
        "🏁 [학습 시작] BBox(CIoU) + Keypoint(Smooth L1) 통합 학습 및 자동 시각화 활성화\n"
    )

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        model.train()
        epoch_cls_loss = 0.0
        epoch_bbox_loss = 0.0
        epoch_kpt_loss = 0.0
        epoch_total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch:03d}/{TOTAL_EPOCHS}]")
        for batch in pbar:
            optimizer.zero_grad()

            input_tensor = batch[0].to(DEVICE, non_blocking=True)
            target_bboxes = batch[2].to(DEVICE, non_blocking=True).clone()
            target_kpts = batch[3].to(DEVICE, non_blocking=True).clone()
            target_vis = batch[4].to(DEVICE, non_blocking=True).clone()

            # 정답 박스 및 키포인트 픽셀 복원 (384x640 공간)
            target_bboxes[..., [0, 2]] *= 640.0
            target_bboxes[..., [1, 3]] *= 384.0
            target_kpts[..., 0] *= 640.0
            target_kpts[..., 1] *= 384.0

            # Forward: model.training일 때 (one2many_head, one2one_head) 반환
            preds_m, preds_o = model(input_tensor)
            o2m_cls, o2m_regs, o2m_kpts = preds_m
            o2o_cls, o2o_regs, o2o_kpts = preds_o

            # DFL 픽셀 디코딩
            decoded_o2m_regs = decode_regs(o2m_regs, anchor_points, anchor_strides)
            decoded_o2o_regs = decode_regs(o2o_regs, anchor_points, anchor_strides)

            # 1. TAL 분류 손실
            loss_o2m_cls, mask_o2m = compute_cls_loss_with_tal(
                o2m_cls,
                decoded_o2m_regs,
                target_bboxes,
                anchor_points,
                alpha=0.5,
                beta=2.0,
                is_o2o=False,
            )
            loss_o2o_cls, mask_o2o = compute_cls_loss_with_tal(
                o2o_cls,
                decoded_o2o_regs,
                target_bboxes,
                anchor_points,
                alpha=0.5,
                beta=2.0,
                is_o2o=True,
            )
            total_cls_loss = loss_o2m_cls + loss_o2o_cls

            # 2. CIoU 회귀 손실
            ciou_loss_o2m_raw = compute_ciou_loss(decoded_o2m_regs, target_bboxes)
            ciou_loss_o2o_raw = compute_ciou_loss(decoded_o2o_regs, target_bboxes)

            num_pos_o2m = mask_o2m.sum()
            num_pos_o2o = mask_o2o.sum()

            ciou_loss_o2m = (
                (ciou_loss_o2m_raw * mask_o2m.float()).sum() / (num_pos_o2m + 1e-7)
                if num_pos_o2m > 0
                else torch.tensor(0.0, device=DEVICE)
            )
            ciou_loss_o2o = (
                (ciou_loss_o2o_raw * mask_o2o.float()).sum() / (num_pos_o2o + 1e-7)
                if num_pos_o2o > 0
                else torch.tensor(0.0, device=DEVICE)
            )
            total_ciou_loss = ciou_loss_o2m + ciou_loss_o2o

            # 3. Keypoint 회귀 및 Visibility 손실
            loss_kpt_o2m, loss_vis_o2m = compute_kpt_loss(
                o2m_kpts, target_kpts, target_vis, target_bboxes, mask_o2m
            )
            loss_kpt_o2o, loss_vis_o2o = compute_kpt_loss(
                o2o_kpts, target_kpts, target_vis, target_bboxes, mask_o2o
            )
            total_kpt_loss = loss_kpt_o2m + loss_kpt_o2o
            total_vis_loss = loss_vis_o2m + loss_vis_o2o

            # 4. 종합 Loss (Cls, BBox, Keypoint 균형 가중치)
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

        # 5. Validation 정밀 Metric 평가 (BBox mAP@50 및 Keypoint 정확도)
        metrics = evaluate_metrics(
            model=model,
            dataloader=val_loader,
            anchor_points=anchor_points,
            anchor_strides=anchor_strides,
            device=DEVICE,
            iou_thresh=0.5,
            kpt_dist_thresh=0.10,
        )

        bbox_map = metrics["bbox_map50"]
        kpt_acc = metrics["kpt_acc"]
        curr_fitness = metrics["fitness"]

        print(
            f" Epoch [{epoch:03d}] | "
            f"Train [Loss: {avg_train_total:.3f}, Cls: {avg_train_cls:.3f}, Box: {avg_train_bbox:.3f}, Kpt: {avg_train_kpt:.3f}] | "
            f"Val [BBox mAP@50: {bbox_map:.2f}%, Kpt Acc: {kpt_acc:.2f}%, Fitness: {curr_fitness:.2f}]"
        )

        # 6. 매 에폭마다 last_fixed.pt 저장 (안전한 중단 및 재개 보장)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "bbox_map50": bbox_map,
                "kpt_acc": kpt_acc,
                "fitness": curr_fitness,
                "best_fitness": best_fitness,
            },
            SAVE_DIR / "last_fixed.pt",
        )

        # 7. 성능 개선(Fitness 갱신) 시 Best Model 저장 및 시각화 저장
        if curr_fitness > best_fitness:
            best_fitness = curr_fitness
            save_path = SAVE_DIR / "best_fixed.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "bbox_map50": bbox_map,
                    "kpt_acc": kpt_acc,
                    "fitness": curr_fitness,
                },
                save_path,
            )
            print(
                f"  ⭐ [Best Model 갱신!] {save_path} "
                f"(mAP@50: {bbox_map:.2f}%, Kpt Acc: {kpt_acc:.2f}%, Fitness: {curr_fitness:.2f})"
            )

            # 💡 [요청 사항 반영]: 성능 개선 시 Validation 추론 결과 시각화 이미지 저장
            vis_epoch_file, vis_latest_file = save_validation_visualization(
                model=model,
                val_dataset=val_dataset,
                anchor_points=anchor_points,
                anchor_strides=anchor_strides,
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
