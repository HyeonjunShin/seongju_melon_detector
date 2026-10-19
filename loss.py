"""Loss and one-to-many assignment for the custom DFL detector.

Expected model outputs during training:
    one2many, one2one = model(images)
    one2many = (cls_logits, reg_logits, kpt_logits)

This first-stage loss intentionally trains only the one-to-many detection
branch (classification + DFL + IoU). Keypoint/visibility loss and one-to-one
assignment should be added after box training is verified.

Expected targets from chamae_pack_collate_fn:
    gt_bboxes: [B, M, 4], normalized XYXY in [0, 1] relative to 640x384
    gt_mask:   [B, M, 1] or [B, M], True for valid objects
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from training.model import DecodeModule
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision.ops import nms, box_iou


@torch.no_grad()
def evaluate_detection(
    model,
    val_loader,
    device,
    conf_thres=0.05,
    nms_iou_thres=0.5,
    match_iou_thres=0.5,
):
    """
    가정:
      - targets["gt_bboxes"]: normalized XYXY, shape [B, M, 4]
      - targets["gt_mask"]: 유효 GT 표시, shape [B, M] 또는 [B, M, 1]
      - decoder가 pixel XYXY bbox를 반환
      - cls_pred은 sigmoid 전 logits

    반환:
      precision, recall, F1, mean matched IoU,
      mean best IoU per GT, IoU@0.5 detection rate
    """
    model.eval()

    decoder = (
        DecodeModule(
            reg_max=16,
            num_kpts=2,
            strides=[8, 16, 32],
            h=384,
            w=640,
        )
        .to(device)
        .eval()
    )

    tp = 0
    fp = 0
    fn = 0
    matched_ious = []

    # 각 GT에 대해 가장 높은 IoU를 기록
    best_iou_sum = 0.0
    total_gt = 0
    gt_with_iou50 = 0

    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)

        one2many, _one2one = model(images)
        cls_pred, reg_pred, kpt_pred = one2many

        scores = cls_pred.sigmoid().squeeze(-1)
        pred_boxes, _, _ = decoder(reg_pred, kpt_pred)

        gt_boxes_batch = targets["gt_bboxes"]
        gt_mask_batch = targets["gt_mask"].bool().squeeze(-1)

        for i in range(images.shape[0]):
            # -------------------------
            # Ground-truth boxes
            # -------------------------
            gt_boxes = (
                gt_boxes_batch[i][gt_mask_batch[i]]
                .to(device=device, dtype=torch.float32)
                .clone()
            )

            if gt_boxes.numel() > 0:
                gt_boxes *= gt_boxes.new_tensor([640, 384, 640, 384])

                gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clamp(0, 640)
                gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clamp(0, 384)

                valid_gt = (
                    torch.isfinite(gt_boxes).all(dim=1)
                    & (gt_boxes[:, 2] > gt_boxes[:, 0])
                    & (gt_boxes[:, 3] > gt_boxes[:, 1])
                )
                gt_boxes = gt_boxes[valid_gt]

            # -------------------------
            # Predictions
            # -------------------------
            boxes = pred_boxes[i].float().clone()
            conf = scores[i].float()

            valid_pred = (
                torch.isfinite(boxes).all(dim=1)
                & torch.isfinite(conf)
                & (boxes[:, 2] > boxes[:, 0])
                & (boxes[:, 3] > boxes[:, 1])
            )

            boxes = boxes[valid_pred]
            conf = conf[valid_pred]

            keep = conf >= conf_thres
            boxes = boxes[keep]
            conf = conf[keep]

            if boxes.numel() > 0:
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, 640)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, 384)

                keep_idx = nms(boxes, conf, nms_iou_thres)
                boxes = boxes[keep_idx]
                conf = conf[keep_idx]

            # -------------------------
            # Greedy one-to-one matching
            # -------------------------
            total_gt += len(gt_boxes)

            if len(gt_boxes) == 0:
                # 정답이 없는 이미지의 예측은 false positive
                fp += len(boxes)
                continue

            if len(boxes) == 0:
                # GT가 있지만 검출하지 못함
                fn += len(gt_boxes)
                continue

            ious = box_iou(boxes, gt_boxes)

            # 각 GT의 best IoU: 매칭 여부와 별도로 기록
            gt_best_ious = ious.max(dim=0).values
            best_iou_sum += gt_best_ious.sum().item()
            gt_with_iou50 += (gt_best_ious >= match_iou_thres).sum().item()

            # 높은 confidence 예측부터 GT와 일대일 매칭
            order = conf.argsort(descending=True)
            matched_gt = set()

            for pred_idx in order.tolist():
                gt_idx = int(ious[pred_idx].argmax().item())
                iou_value = ious[pred_idx, gt_idx].item()

                if iou_value >= match_iou_thres and gt_idx not in matched_gt:
                    tp += 1
                    matched_gt.add(gt_idx)
                    matched_ious.append(iou_value)
                else:
                    fp += 1

            fn += len(gt_boxes) - len(matched_gt)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )

    mean_matched_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    mean_best_iou = best_iou_sum / max(total_gt, 1)
    iou50_rate = gt_with_iou50 / max(total_gt, 1)

    results = {
        "num_gt": total_gt,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "mean_matched_IoU": mean_matched_iou,
        "mean_best_IoU_per_GT": mean_best_iou,
        "IoU50_detection_rate": iou50_rate,
    }

    print("\n========== Validation Detection Metrics ==========")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:24s}: {value:.4f}")
        else:
            print(f"{key:24s}: {value}")
    print("==================================================\n")

    return results


def visualize_validation(
    model,
    val_loader,
    device,
    save_path="val_visualization.png",
    num_images=8,
    conf_thres=0.05,
    iou_thres=0.5,
    max_det=10,
):
    model.eval()

    decoder = (
        DecodeModule(
            reg_max=16,
            num_kpts=2,
            strides=[8, 16, 32],
            h=384,
            w=640,
        )
        .to(device)
        .eval()
    )

    shown = 0
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)

            # 학습에 사용한 One-to-Many head의 예측
            one2many, _one2one = model(images)
            cls_pred, reg_pred, kpt_pred = one2many

            # 클래스 confidence
            scores = cls_pred.sigmoid().squeeze(-1)

            # bbox를 픽셀 좌표 XYXY로 디코딩
            pred_boxes, _pred_kpts, _vis_logits = decoder(reg_pred, kpt_pred)

            for i in range(images.shape[0]):
                if shown >= num_images:
                    break

                # 입력 이미지는 RGB, float [0, 1]
                img = images[i].detach().cpu()
                img = img.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy()
                pil_img = Image.fromarray(img)
                draw = ImageDraw.Draw(pil_img)

                # -------------------------
                # 정답 bbox: 초록색
                # -------------------------
                gt_mask = targets["gt_mask"][i].squeeze(-1).bool()
                gt_boxes = targets["gt_bboxes"][i][gt_mask].clone()

                # 정규화 좌표 -> 픽셀 좌표
                if gt_boxes.numel() > 0:
                    scale = gt_boxes.new_tensor([640, 384, 640, 384])
                    gt_boxes *= scale

                    for box in gt_boxes:
                        x1, y1, x2, y2 = box.tolist()
                        draw.rectangle(
                            (x1, y1, x2, y2),
                            outline="lime",
                            width=3,
                        )
                        draw.text(
                            (x1, max(0, y1 - 14)),
                            "GT",
                            fill="lime",
                        )

                # -------------------------
                # 예측 bbox: 빨간색
                # -------------------------
                boxes = pred_boxes[i].clone()
                conf = scores[i].clone()

                # 이미지 경계 안으로 제한
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, 639)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, 383)

                valid = (
                    torch.isfinite(boxes).all(dim=1)
                    & torch.isfinite(conf)
                    & (boxes[:, 2] > boxes[:, 0])
                    & (boxes[:, 3] > boxes[:, 1])
                )

                boxes = boxes[valid]
                conf = conf[valid]

                if boxes.numel() > 0:
                    selected = conf >= conf_thres

                    if selected.any():
                        boxes = boxes[selected]
                        conf = conf[selected]
                    else:
                        # threshold 이상인 예측이 없다면
                        # 진단 목적으로 confidence 상위 5개 표시
                        topk = min(5, conf.numel())
                        idx = conf.topk(topk).indices
                        boxes = boxes[idx]
                        conf = conf[idx]

                    # 겹치는 예측을 줄이기 위한 시각화용 NMS
                    keep = nms(boxes, conf, iou_thres)[:max_det]
                    boxes = boxes[keep]
                    conf = conf[keep]

                    for box, score in zip(boxes, conf):
                        x1, y1, x2, y2 = box.tolist()
                        draw.rectangle(
                            (x1, y1, x2, y2),
                            outline="red",
                            width=2,
                        )
                        draw.text(
                            (x1, min(383, y2 + 2)),
                            f"Pred {score.item():.2f}",
                            fill="red",
                        )

                axes[shown].imshow(pil_img)
                axes[shown].set_title(f"Validation {shown + 1}")
                axes[shown].axis("off")
                shown += 1

            if shown >= num_images:
                break

    # 사용하지 않은 subplot 숨기기
    for j in range(shown, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"시각화 저장 완료: {save_path}")
    print(f"표시한 이미지 수: {shown}")


class OneToManyDetectionLoss(nn.Module):
    """Task-aligned-style assignment + BCE classification + IoU + DFL loss.

    This is a practical simplified assignment, not an exact reproduction of
    YOLOv10's training/assignment implementation.
    """

    def __init__(
        self,
        image_height: int = 384,
        image_width: int = 640,
        strides: Tuple[int, ...] = (8, 16, 32),
        reg_max: int = 16,
        topk: int = 10,
        alpha: float = 0.5,
        beta: float = 6.0,
        box_weight: float = 7.5,
        dfl_weight: float = 1.5,
        cls_weight: float = 0.5,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.strides = strides
        self.reg_max = reg_max
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.box_weight = box_weight
        self.dfl_weight = dfl_weight
        self.cls_weight = cls_weight

        self.register_buffer(
            "project", torch.arange(reg_max, dtype=torch.float32), persistent=False
        )
        anchors, stride_tensor = self._make_anchors(image_height, image_width, strides)
        self.register_buffer("anchors", anchors, persistent=False)  # [N, 2], grid units
        self.register_buffer("stride_tensor", stride_tensor, persistent=False)  # [N, 1]

    @staticmethod
    def _make_anchors(height: int, width: int, strides: Tuple[int, ...]):
        anchor_parts = []
        stride_parts = []
        for stride in strides:
            fh, fw = height // stride, width // stride
            y, x = torch.meshgrid(
                torch.arange(fh, dtype=torch.float32),
                torch.arange(fw, dtype=torch.float32),
                indexing="ij",
            )
            anchor_parts.append(torch.stack((x + 0.5, y + 0.5), dim=-1).reshape(-1, 2))
            stride_parts.append(torch.full((fh * fw, 1), float(stride)))
        return torch.cat(anchor_parts, dim=0), torch.cat(stride_parts, dim=0)

    def decode_boxes(self, reg_logits: torch.Tensor) -> torch.Tensor:
        """Decode [B,N,4*reg_max] DFL logits to pixel XYXY boxes."""
        b, n, _ = reg_logits.shape
        dist = reg_logits.reshape(b, n, 4, self.reg_max).softmax(-1)
        dist = (dist * self.project.to(dtype=dist.dtype)).sum(-1)
        stride = self.stride_tensor.to(dtype=dist.dtype)
        anchor_px = self.anchors.to(dtype=dist.dtype) * stride
        dist_px = dist * stride
        x1y1 = anchor_px.unsqueeze(0) - dist_px[..., :2]
        x2y2 = anchor_px.unsqueeze(0) + dist_px[..., 2:]
        return torch.cat((x1y1, x2y2), dim=-1)

    @staticmethod
    def _pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Pairwise IoU for XYXY boxes: [N,4] x [M,4] -> [N,M]."""
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
        lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
            boxes1[:, 3] - boxes1[:, 1]
        ).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
            boxes2[:, 3] - boxes2[:, 1]
        ).clamp(min=0)
        return inter / (area1[:, None] + area2[None, :] - inter).clamp(min=1e-7)

    @staticmethod
    def _box_iou_aligned(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Aligned IoU for corresponding XYXY boxes."""
        lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
        rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
            boxes1[:, 3] - boxes1[:, 1]
        ).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
            boxes2[:, 3] - boxes2[:, 1]
        ).clamp(min=0)
        return inter / (area1 + area2 - inter).clamp(min=1e-7)

    @torch.no_grad()
    def assign_one_to_many(
        self,
        pred_scores: torch.Tensor,  # [N,1], probabilities
        pred_boxes: torch.Tensor,  # [N,4], pixel XYXY
        gt_boxes: torch.Tensor,  # [M,4], pixel XYXY
    ):
        """Return foreground mask [N] and assigned GT index [N].

        For each GT, select up to topk anchor centers inside its box using a
        task-alignment metric. If multiple GTs select the same anchor, assign
        that anchor to the GT with the largest alignment metric.
        """
        n = pred_boxes.shape[0]
        m = gt_boxes.shape[0]
        device = pred_boxes.device
        fg = torch.zeros(n, dtype=torch.bool, device=device)
        assigned_gt = torch.zeros(n, dtype=torch.long, device=device)
        if m == 0 or n == 0:
            return fg, assigned_gt

        anchors_px = self.anchors.to(device=device) * self.stride_tensor.to(
            device=device
        )
        # [N,M]: anchor center must lie inside each GT box.
        inside_l = anchors_px[:, None, 0] - gt_boxes[None, :, 0]
        inside_t = anchors_px[:, None, 1] - gt_boxes[None, :, 1]
        inside_r = gt_boxes[None, :, 2] - anchors_px[:, None, 0]
        inside_b = gt_boxes[None, :, 3] - anchors_px[:, None, 1]
        inside = (
            torch.stack((inside_l, inside_t, inside_r, inside_b), dim=-1).amin(-1) > 0
        )

        ious = self._pairwise_iou(pred_boxes.detach(), gt_boxes).clamp(min=0)
        cls = pred_scores.detach().expand(-1, m).clamp(min=1e-9, max=1.0)
        align = cls.pow(self.alpha) * ious.pow(self.beta)
        align = align * inside.float()

        candidate = torch.zeros_like(inside)
        k = min(self.topk, n)
        for j in range(m):
            valid_indices = torch.where(inside[:, j])[0]
            if valid_indices.numel() == 0:
                # Small/rotated objects can contain no grid center. Leave this
                # GT unassigned rather than assigning a clearly unrelated point.
                continue
            kk = min(k, valid_indices.numel())
            values = align[valid_indices, j]
            chosen = valid_indices[torch.topk(values, k=kk, largest=True).indices]
            candidate[chosen, j] = True

        # Resolve anchors selected by multiple GTs using maximum alignment.
        candidate_align = align.masked_fill(~candidate, -1.0)
        best_metric, best_gt = candidate_align.max(dim=1)
        fg = candidate.any(dim=1) & (best_metric >= 0)
        assigned_gt[fg] = best_gt[fg]
        return fg, assigned_gt

    def _dfl_loss(
        self, reg_logits_pos: torch.Tensor, target_ltrb: torch.Tensor
    ) -> torch.Tensor:
        """Interpolated cross entropy between neighboring DFL bins."""
        target = target_ltrb.clamp(min=0, max=self.reg_max - 1 - 1e-2)
        left = target.floor().long()
        right = left + 1
        wr = target - left.to(target.dtype)
        wl = 1.0 - wr
        logits = reg_logits_pos.reshape(-1, self.reg_max)
        left_flat = left.reshape(-1)
        right_flat = right.reshape(-1)
        loss_l = F.cross_entropy(logits, left_flat, reduction="none")
        loss_r = F.cross_entropy(logits, right_flat, reduction="none")
        loss = loss_l * wl.reshape(-1) + loss_r * wr.reshape(-1)
        return loss.mean()

    def forward(self, one2many_outputs, targets: Dict[str, torch.Tensor]):
        """Compute first-stage detection losses.

        Args:
            one2many_outputs: (cls_logits [B,N,1], reg_logits [B,N,64], kpt_logits)
            targets: output of chamae_pack_collate_fn.

        Returns a dict containing total, cls, box, and dfl losses.
        """
        cls_logits, reg_logits, _kpt_logits = one2many_outputs
        batch_size, num_anchors, num_classes = cls_logits.shape
        expected_n = self.anchors.shape[0]
        if num_anchors != expected_n:
            raise ValueError(
                f"Model produced {num_anchors} anchors, but loss expects {expected_n}. "
                "Check input size and strides."
            )
        if num_classes != 1:
            raise ValueError("This initial loss implementation assumes num_classes=1.")

        device = cls_logits.device
        dtype = cls_logits.dtype
        anchors = self.anchors.to(device=device, dtype=dtype)
        strides = self.stride_tensor.to(device=device, dtype=dtype)
        pred_boxes = self.decode_boxes(reg_logits)

        gt_norm = targets["gt_bboxes"].to(device=device, dtype=dtype)
        gt_mask = targets["gt_mask"].to(device=device).bool().squeeze(-1)
        scale = gt_norm.new_tensor(
            [self.image_width, self.image_height, self.image_width, self.image_height]
        )
        gt_pixel = gt_norm * scale

        cls_target = torch.zeros_like(cls_logits)
        box_loss_sum = cls_logits.new_zeros(())
        dfl_loss_sum = cls_logits.new_zeros(())
        num_pos = 0

        for b in range(batch_size):
            valid_gt = gt_pixel[b, gt_mask[b]]
            if valid_gt.numel() == 0:
                continue

            fg, gt_idx = self.assign_one_to_many(
                cls_logits[b].sigmoid(), pred_boxes[b].detach(), valid_gt
            )
            if not fg.any():
                continue

            pos_gt = valid_gt[gt_idx[fg]]
            cls_target[b, fg, 0] = 1.0
            num_pos += int(fg.sum().item())

            # IoU loss on positive predictions.
            iou = self._box_iou_aligned(pred_boxes[b, fg], pos_gt)
            box_loss_sum = box_loss_sum + (1.0 - iou).sum()

            # DFL target distances are in each anchor's feature-grid units.
            anchor_pos = anchors[fg]
            stride_pos = strides[fg]
            gt_grid = pos_gt / stride_pos.repeat(1, 4)
            target_ltrb = torch.stack(
                (
                    anchor_pos[:, 0] - gt_grid[:, 0],
                    anchor_pos[:, 1] - gt_grid[:, 1],
                    gt_grid[:, 2] - anchor_pos[:, 0],
                    gt_grid[:, 3] - anchor_pos[:, 1],
                ),
                dim=-1,
            )
            reg_pos = reg_logits[b, fg].reshape(-1, 4, self.reg_max)
            dfl_loss_sum = dfl_loss_sum + self._dfl_loss(reg_pos, target_ltrb) * int(
                fg.sum().item()
            )

        # Normalize positive losses by total positive count. Empty images still
        # contribute negative classification supervision.
        normalizer = max(num_pos, 1)
        cls_loss = (
            F.binary_cross_entropy_with_logits(cls_logits, cls_target, reduction="mean")
            / normalizer
        )
        box_loss = box_loss_sum / normalizer
        dfl_loss = dfl_loss_sum / normalizer
        total = (
            self.cls_weight * cls_loss
            + self.box_weight * box_loss
            + self.dfl_weight * dfl_loss
        )

        return {
            "loss": total,
            "loss_cls": cls_loss.detach(),
            "loss_box": box_loss.detach(),
            "loss_dfl": dfl_loss.detach(),
            "num_pos": torch.tensor(float(num_pos), device=device),
        }


if __name__ == "__main__":
    # Minimal smoke test; does not require loading a dataset.
    from training.model import Model
    from datasets import (
        find_melon_data,
        split_data,
        ChamaePackDataset,
        chamae_pack_collate_fn,
    )
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_list = find_melon_data("./data")
    train_paths, val_paths = split_data(data_list, train_ratio=0.9, seed=42)

    print(
        f"📊 Dataset Status -> Total: {len(data_list)} | Train: {len(train_paths)} | Val: {len(val_paths)}"
    )
    train_dataset = ChamaePackDataset(train_paths, is_train=True)
    val_dataset = ChamaePackDataset(val_paths, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=chamae_pack_collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=chamae_pack_collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    model = Model(num_classes=1, reg_max=16, num_kpts=2).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = OneToManyDetectionLoss().to(device)

    num_epochs = 10

    for e in range(num_epochs):
        # -------------------------
        # 1. Train
        # -------------------------
        model.train()

        train_loss_sum = 0.0
        train_cls_sum = 0.0
        train_box_sum = 0.0
        train_dfl_sum = 0.0
        train_batches = 0

        for input_tensor, target in train_loader:
            input_tensor = input_tensor.to(device, non_blocking=True)

            one2many, _one2one = model(input_tensor)
            losses = criterion(one2many, target)

            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()

            train_loss_sum += losses["loss"].item()
            train_cls_sum += losses["loss_cls"].item()
            train_box_sum += losses["loss_box"].item()
            train_dfl_sum += losses["loss_dfl"].item()
            train_batches += 1

        # -------------------------
        # 2. Validation
        # -------------------------
        model.eval()

        val_loss_sum = 0.0
        val_cls_sum = 0.0
        val_box_sum = 0.0
        val_dfl_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for input_tensor, target in val_loader:
                input_tensor = input_tensor.to(device, non_blocking=True)

                one2many, _one2one = model(input_tensor)
                losses = criterion(one2many, target)

                val_loss_sum += losses["loss"].item()
                val_cls_sum += losses["loss_cls"].item()
                val_box_sum += losses["loss_box"].item()
                val_dfl_sum += losses["loss_dfl"].item()
                val_batches += 1

        # -------------------------
        # 3. Epoch results
        # -------------------------
        train_loss = train_loss_sum / max(train_batches, 1)
        val_loss = val_loss_sum / max(val_batches, 1)

        print(
            f"Epoch [{e + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        print(
            f"  Train: cls={train_cls_sum / max(train_batches, 1):.4f}, "
            f"box={train_box_sum / max(train_batches, 1):.4f}, "
            f"dfl={train_dfl_sum / max(train_batches, 1):.4f}"
        )

        print(
            f"  Val:   cls={val_cls_sum / max(val_batches, 1):.4f}, "
            f"box={val_box_sum / max(val_batches, 1):.4f}, "
            f"dfl={val_dfl_sum / max(val_batches, 1):.4f}"
        )

        # visualize_validation(
        #     model=model,
        #     val_loader=val_loader,
        #     device=device,
        #     save_path="val_visualization.png",
        #     num_images=8,
        #     conf_thres=0.05,
        #     iou_thres=0.5,
        #     max_det=10,
        # )

    metrics = evaluate_detection(
        model=model,
        val_loader=val_loader,
        device=device,
        conf_thres=0.5,
        nms_iou_thres=0.5,
        match_iou_thres=0.5,
    )
