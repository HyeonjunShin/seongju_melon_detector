import os
import time
import gc
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
    ChamaePackDataset,
    chamae_pack_collate_fn,
    find_melon_data,
    split_data,
)
from model import CombinedPoseModel, DualYOLOv8PoseModel, PoseDecoderModule
from loss import DualYOLOv8PoseLoss
import torch.nn.functional as F


def decode_regs(pred_regs, anchor_points, anchor_strides, reg_max=16):
    """
    pred_regs: [B, 5040, 4 * reg_max]
    anchor_points: [1, 5040, 2]
    anchor_strides: [1, 5040, 1]
    """
    weight = torch.arange(reg_max, dtype=torch.float32, device=pred_regs.device)
    pred_regs = pred_regs.view(-1, 5040, 4, reg_max)
    pred_regs = torch.softmax(pred_regs, dim=-1)

    dist_pixel = (pred_regs @ weight) * anchor_strides.view(1, -1, 1)

    cx = anchor_points[..., 0]
    cy = anchor_points[..., 1]

    xmin = cx - dist_pixel[..., 0]  # cx - Left
    ymin = cy - dist_pixel[..., 1]  # cy - Top
    xmax = cx + dist_pixel[..., 2]  # cx + Right
    ymax = cy + dist_pixel[..., 3]  # cy + Bottom

    decoded_bboxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    return decoded_bboxes


def compute_cls_loss_with_tal(
    pred_scores,
    pred_bboxes,
    target_bboxes,
    anchor_points,
    alpha=1.0,
    beta=6.0,
    is_o2o=False,
):
    """
    TAL 기반의 Classification Soft Label Loss 계산 함수

    pred_scores:   [16, 5040, Num_Classes] -> 모델의 분류 출력 (Logits 상태, Sigmoid 전)
    pred_bboxes:   [16, 5040, 4]           -> 모델의 회귀 출력 (xmin, ymin, xmax, ymax)
    target_bboxes: [16, 1, 4]              -> 정답 박스 픽셀 좌표 (xmin, ymin, xmax, ymax)
    anchor_points: [1, 5040, 2]            -> 앵커 중심점 픽셀 좌표 (cx, cy)
    is_o2o:        True면 One-to-One(단 1개 매칭), False면 One-to-Many(Top-K 매칭)
    """
    batch_size, num_anchors, num_classes = pred_scores.shape

    # 1. In-Box 필터링 (앵커 중심점이 정답 박스 내부에 있는지 체크)
    # anchor_points: [1, 5040, 2] -> anc_x, anc_y 각각 [1, 5040]
    anc_x, anc_y = anchor_points[..., 0], anchor_points[..., 1]

    # target_bboxes: [16, 1, 4] -> gt 각각 [16, 1]
    gt_x1, gt_y1 = target_bboxes[..., 0], target_bboxes[..., 1]
    gt_x2, gt_y2 = target_bboxes[..., 2], target_bboxes[..., 3]

    # (16, 5040) => [batch_size, num_anchors]
    is_in_box = (
        (anc_x >= gt_x1) & (anc_x <= gt_x2) & (anc_y >= gt_y1) & (anc_y <= gt_y2)
    )

    # 2. 예측 박스와 정답 박스 간의 IoU 매트릭스 계산 -> Shape: [16, 5040]
    # (앞서 구현한 브로드캐스팅 수식을 1개 GT 타겟에 맞춰 축소 연산)
    b1_x1, b1_y1, b1_x2, b1_y2 = (
        pred_bboxes[..., 0],
        pred_bboxes[..., 1],
        pred_bboxes[..., 2],
        pred_bboxes[..., 3],
    )
    inter_x1 = torch.max(b1_x1, gt_x1)
    inter_y1 = torch.max(b1_y1, gt_y1)
    inter_x2 = torch.min(b1_x2, gt_x2)
    inter_y2 = torch.min(b1_y2, gt_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
    area2 = (gt_x2 - gt_x1).clamp(min=0) * (gt_y2 - gt_y1).clamp(min=0)
    union_area = area1 + area2 - inter_area + 1e-7
    overlaps = inter_area / union_area

    # 3. Alignment Score (t = P^alpha * IoU^beta) 계산
    # 분류 확률값 추출 (Sigmoid 적용 후 최대값 혹은 첫 번째 클래스 확률 사용)
    # 여기서는 다중 클래스 확장성을 위해 최고 확률값을 임시 사용합니다.
    pred_probs = pred_scores.detach().sigmoid().max(dim=-1)[0]  # [16, 5040]

    alignment_metrics = (pred_probs**alpha) * (overlaps**beta)
    alignment_metrics *= is_in_box.float()  # 박스 외부는 점수 0 처리

    # 4. 모드에 따른 최종 양성(Positive) 앵커 선택 (마스킹)
    mask_pos = torch.zeros_like(is_in_box, dtype=torch.bool)  # [16, 5040]

    if is_o2o:
        # [O2O 모드]: 이미지별로 점수가 가장 높은 단 1개의 앵커만 선택
        max_indices = alignment_metrics.argmax(dim=-1)  # [16]
        for b in range(batch_size):
            if is_in_box[b, max_indices[b]]:  # 유효한 박스 내부일 때만
                mask_pos[b, max_indices[b]] = True
    else:
        # [O2M 모드]: 이미지별로 점수가 높은 상위 10개(Top-K) 앵커 선택
        topk = min(10, num_anchors)
        _, topk_indices = torch.topk(
            alignment_metrics, topk, dim=-1, largest=True
        )  # [16, topk]
        for b in range(batch_size):
            # Top-K 중 실제로 박스 내부에 있는 것만 유효화
            valid_topk = topk_indices[b][is_in_box[b, topk_indices[b]]]
            mask_pos[b, valid_topk] = True

    # 5. 분류용 타겟 텐서(Soft Label) 빌드 -> Shape: [16, 5040, Num_Classes]
    # 기본값은 모두 0(배경)으로 채워진 상태
    target_scores = torch.zeros_like(pred_scores, dtype=torch.float)

    # 선택된 양성 앵커 자리에 정렬 점수(t)를 채워 넣음 (Soft Label 정렬의 핵심)
    # 정량화를 위해 선택된 앵커들의 점수를 정규화하거나 IoU 최고값으로 스케일링하기도 합니다.
    # 여기서는 표준 수식인 '정규화된 alignment_metrics' 또는 'overlaps'를 매핑합니다.
    for b in range(batch_size):
        pos_idx = mask_pos[b]
        if pos_idx.any():
            # 가용한 타겟 클래스 인덱스 (단일 클래스 가정이면 0번 채널)
            # 여기서는 0번 클래스에 타겟 점수 주입 (다중 클래스일 경우 정답 인덱스 레이블이 필요함)
            target_scores[b, pos_idx, 0] = overlaps[b, pos_idx]

    # 6. BCEWithLogitsLoss 계산 (Soft Label 타겟 지원을 위해 직접 수식 계산 혹은 크기 일치 연산)
    # 5,040개 전체에 대해 계산하되, 모델 예측(Logits)과 정답 점수(0~1 사이)를 비교
    cls_loss = F.binary_cross_entropy_with_logits(
        pred_scores, target_scores, reduction="mean"
    )

    return cls_loss, mask_pos


def train_one_epoch(model, dataloader, criterion, decoder, optimizer, scaler, device):
    """1 Epoch 학습 함수"""
    model.train()
    total_loss = 0.0
    loss_dict_sum = {"loss_cls": 0.0, "loss_dfl": 0.0, "loss_kpt": 0.0}

    pbar = tqdm(dataloader, desc="[Train]")
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        optimizer.zero_grad()

        with autocast(device_type=device.type):
            one2many_preds, one2one_preds = model(images)
            loss_m, loss_dict_m = criterion(one2many_preds, targets, decoder)
            loss_o, loss_dict_o = criterion(one2one_preds, targets, decoder)
            loss = loss_m + loss_o

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k in loss_dict_sum.keys():
            loss_dict_sum[k] += loss_dict_o[k]

        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Kpt": f"{loss_dict_o['loss_kpt']:.4f}",
                "DFL": f"{loss_dict_o['loss_dfl']:.4f}",
            }
        )

    avg_loss = total_loss / len(dataloader)
    return avg_loss, {k: v / len(dataloader) for k, v in loss_dict_sum.items()}


@torch.no_grad()
def validate_metric(
    model,
    dataloader,
    decoder,
    device,
    conf_thresh=0.25,
    iou_thresh=0.5,
    kpt_dist_thresh=0.10,
):
    """
    NMS-Free 평가 함수 (CPU 메모리 이관으로 Segfault 예방)
    """
    from torchvision.ops import box_iou

    model.eval()

    total_gt_boxes = 0
    correct_bboxes = 0
    total_gt_kpts = 0
    correct_kpts = 0

    pbar = tqdm(dataloader, desc="[Val Metric (NMS-Free)]")
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        with autocast(device_type=device.type):
            cls_pred, reg_pred, kpt_pred = model(images)
            scores, decoded_bboxes, decoded_kpts = decoder(cls_pred, reg_pred, kpt_pred)

        batch_size = images.size(0)

        for b in range(batch_size):
            gt_mask_b = targets["gt_mask"][b]
            if not gt_mask_b.any():
                continue

            gt_bboxes_b = targets["gt_bboxes"][b][gt_mask_b].cpu()  # [GT_N, 4]
            gt_kpts_b = targets["gt_keypoints"][b][gt_mask_b].cpu()  # [GT_N, 2, 3]

            scores_b = scores[b].squeeze(-1)  # [5040]
            bboxes_b = decoded_bboxes[b]  # [5040, 4]
            kpts_b = decoded_kpts[b]  # [5040, 2, 3]

            max_preds = max(gt_bboxes_b.size(0) * 5, 30)
            top_scores, top_indices = scores_b.topk(min(len(scores_b), max_preds))

            valid_mask = top_scores > conf_thresh
            if not valid_mask.any():
                pred_boxes = bboxes_b[top_indices[:5]].cpu()
                pred_kpts = kpts_b[top_indices[:5]].cpu()
            else:
                pred_boxes = bboxes_b[top_indices[valid_mask]].cpu()
                pred_kpts = kpts_b[top_indices[valid_mask]].cpu()

            # CPU로 연산 이관 (CUDA 메모리 파편화 방지)
            iou_matrix = box_iou(pred_boxes, gt_bboxes_b)

            for gt_idx in range(gt_bboxes_b.size(0)):
                total_gt_boxes += 1
                gt_kpt_curr = gt_kpts_b[gt_idx]

                if iou_matrix.numel() == 0:
                    total_gt_kpts += (gt_kpt_curr[..., 2] > 0).sum().item()
                    continue

                max_iou, pred_idx = iou_matrix[:, gt_idx].max(dim=0)

                if max_iou >= iou_thresh:
                    correct_bboxes += 1

                    pred_kpt = pred_kpts[pred_idx]
                    gt_w = (gt_bboxes_b[gt_idx, 2] - gt_bboxes_b[gt_idx, 0]).clamp(
                        min=1e-6
                    )
                    gt_h = (gt_bboxes_b[gt_idx, 3] - gt_bboxes_b[gt_idx, 1]).clamp(
                        min=1e-6
                    )

                    for k in range(2):
                        if gt_kpt_curr[k, 2] > 0:
                            total_gt_kpts += 1
                            dist_x = abs(pred_kpt[k, 0] - gt_kpt_curr[k, 0]) / gt_w
                            dist_y = abs(pred_kpt[k, 1] - gt_kpt_curr[k, 1]) / gt_h
                            norm_dist = torch.sqrt(dist_x**2 + dist_y**2)

                            if norm_dist <= kpt_dist_thresh:
                                correct_kpts += 1
                else:
                    total_gt_kpts += (gt_kpt_curr[..., 2] > 0).sum().item()

    bbox_accuracy = (correct_bboxes / max(total_gt_boxes, 1)) * 100.0
    kpt_accuracy = (correct_kpts / max(total_gt_kpts, 1)) * 100.0
    total_accuracy = (bbox_accuracy * 0.5) + (kpt_accuracy * 0.5)

    return {
        "total_acc": total_accuracy,
        "bbox_acc": bbox_accuracy,
        "kpt_acc": kpt_accuracy,
    }


def main():
    DATA_DIR = "./data"
    SAVE_DIR = Path("./runs/train")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    EPOCHS = 100
    BATCH_SIZE = 16
    LR = 1e-3
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from anchors import generate_anchors

    # Generate anchors
    anchor_points, anchor_strides = generate_anchors(device=DEVICE)

    print(f"🚀 학습 디바이스: {DEVICE}")

    data_list = find_melon_data(DATA_DIR)
    train_paths, val_paths = split_data(data_list, train_ratio=0.9, seed=42)

    print(
        f"📊 Dataset Status -> Total: {len(data_list)} | Train: {len(train_paths)} | Val: {len(val_paths)}"
    )

    train_dataset = ChamaePackDataset(train_paths, is_train=True)
    val_dataset = ChamaePackDataset(val_paths, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # collate_fn=chamae_pack_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        # collate_fn=chamae_pack_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    model = DualYOLOv8PoseModel(num_classes=1, reg_max=16, num_kpts=2).to(DEVICE)
    weight = torch.arange(16, dtype=torch.float32, device=DEVICE)

    train_iter = iter(train_loader)
    batch = next(train_iter)

    input_tensor = batch[0].to(DEVICE)
    target_classes = batch[1].to(DEVICE)
    target_bboxes = batch[2].to(DEVICE)
    target_kpts = batch[3].to(DEVICE)
    target_vis = batch[4].to(DEVICE)

    target_bboxes[..., [0, 2]] *= 640.0
    target_bboxes[..., [1, 3]] *= 384.0

    o2o, o2m = model(batch[0].to(DEVICE))

    o2o_cls, o2o_regs, o2o_kpts = o2o
    o2m_cls, o2m_regs, o2m_kpts = o2m

    decoded_o2o_regs = decode_regs(o2o_regs, anchor_points, anchor_strides)
    decoded_o2m_regs = decode_regs(o2m_regs, anchor_points, anchor_strides)

    loss_o2m_cls, mask_o2m = compute_cls_loss_with_tal(
        o2m_cls, decoded_o2m_regs, target_bboxes, anchor_points, is_o2o=False
    )

    # 3. O2O 브랜치 분류 손실 계산 (NMS-Free용 핵심 헤드)
    loss_o2o_cls, mask_o2o = compute_cls_loss_with_tal(
        o2o_cls, decoded_o2o_regs, target_bboxes, anchor_points, is_o2o=True
    )

    o2o_regs_pred = o2o_regs.view(-1, 5040, 4, 16)
    o2o_regs_pred = torch.softmax(o2o_regs_pred, -1)

    sign_tensor = torch.tensor([-1.0, -1.0, 1.0, 1.0], device=DEVICE)
    LTRB_pred = anchor_points + sign_tensor * (
        (o2o_regs_pred @ weight) * anchor_strides.view(1, -1, 1)
    )
    LTRB_pred[..., [0, 2]] /= 640.0
    LTRB_pred[..., [1, 3]] /= 384.0

    target = batch[2].to(DEVICE).repeat(1, 5040, 1)

    import sys

    sys.exit()
    decoder = PoseDecoderModule(reg_max=16, num_kpts=2).to(DEVICE)
    criterion = DualYOLOv8PoseLoss(num_kpts=2).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
    scaler = GradScaler("cuda" if DEVICE.type == "cuda" else "cpu")

    best_accuracy = 0.0

    # 기존 가중치가 존재하면 복원하여 재개 가능하도록 처리
    checkpoint_path = SAVE_DIR / "last.pt"
    start_epoch = 1
    if checkpoint_path.exists():
        print(f"🔄 이전 체크포인트 로드: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_accuracy = ckpt.get("accuracy", 0.0)

    print("\n🔥 [Chamae Pose Model 학습 시작 (평가 지표: Accuracy)]")
    for epoch in range(start_epoch, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_dict = train_one_epoch(
            model, train_loader, criterion, decoder, optimizer, scaler, DEVICE
        )

        metrics = validate_metric(
            model,
            val_loader,
            decoder,
            DEVICE,
            conf_thresh=0.25,
            iou_thresh=0.5,
            kpt_dist_thresh=0.10,
        )

        scheduler.step()
        elapsed_time = time.time() - start_time

        print(
            f"Epoch [{epoch:03d}/{EPOCHS:03d}] ({elapsed_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Total Acc: {metrics['total_acc']:.2f}% "
            f"(BBox Acc: {metrics['bbox_acc']:.2f}%, Kpt Acc: {metrics['kpt_acc']:.2f}%) | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accuracy": metrics["total_acc"],
        }
        torch.save(checkpoint, SAVE_DIR / "last.pt")

        if metrics["total_acc"] > best_accuracy:
            best_accuracy = metrics["total_acc"]
            torch.save(checkpoint, SAVE_DIR / "best.pt")

            combined_model = CombinedPoseModel(model).eval()
            torch.save(combined_model.state_dict(), SAVE_DIR / "best_combined.pt")

            print(
                f"  🏆 최고 정확도 경신! (Best Accuracy: {best_accuracy:.2f}%) -> 가중치 저장 완료"
            )

        # 💡 매 Epoch 종료 후 GC 및 CUDA 캐시 강제 해제
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ [학습 완료] 최적 가중치 저장 경로: {SAVE_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
