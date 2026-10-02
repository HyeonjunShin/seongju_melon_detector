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
    ChamaePoseDataset,
    chamae_pose_collate_fn,
    find_melon_data,
    split_data,
)
from model import CombinedPoseModel, DualYOLOv8PoseModel, PoseDecoderModule
from loss import DualYOLOv8PoseLoss


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

    print(f"🚀 학습 디바이스: {DEVICE}")

    data_list = find_melon_data(DATA_DIR)
    train_paths, val_paths = split_data(data_list, train_ratio=0.9, seed=42)

    print(
        f"📊 Dataset Status -> Total: {len(data_list)} | Train: {len(train_paths)} | Val: {len(val_paths)}"
    )

    train_dataset = ChamaePoseDataset(train_paths, is_train=True)
    val_dataset = ChamaePoseDataset(val_paths, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=chamae_pose_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=chamae_pose_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    model = DualYOLOv8PoseModel(num_classes=1, reg_max=16, num_kpts=2).to(DEVICE)
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
