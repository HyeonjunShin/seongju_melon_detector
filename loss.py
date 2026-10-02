import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou


class DualYOLOv8PoseLoss(nn.Module):
    def __init__(
        self,
        reg_max=16,
        num_kpts=2,
        box_weight=7.5,
        cls_weight=0.5,
        dfl_weight=1.5,
        kpt_weight=12.0,
    ):
        super().__init__()
        self.reg_max = reg_max
        self.num_kpts = num_kpts
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight
        self.kpt_weight = kpt_weight

    def calc_single_head_loss(self, preds, targets, decoder_module, is_one2one=False):
        cls_pred, reg_pred, kpt_pred = preds
        batch_size = cls_pred.size(0)

        # 5040개 BBox 복원 [B, 5040, 4]
        _, decoded_bboxes, _ = decoder_module(cls_pred, reg_pred, kpt_pred)

        total_loss_cls = 0.0
        total_loss_box = 0.0
        total_loss_kpt = 0.0
        total_loss_vis = 0.0

        for b in range(batch_size):
            gt_mask_b = targets["gt_mask"][b]
            if not gt_mask_b.any():
                continue

            gt_bboxes_b = targets["gt_bboxes"][b][gt_mask_b]  # [GT_N, 4]
            gt_keypoints_b = targets["gt_keypoints"][b][gt_mask_b]  # [GT_N, 2, 3]

            pred_boxes_b = decoded_bboxes[b]  # [5040, 4]

            # IoU 매칭 계산 [5040, GT_N]
            iou_matrix = box_iou(pred_boxes_b, gt_bboxes_b)

            num_gt = gt_bboxes_b.size(0)
            pos_mask = torch.zeros(5040, dtype=torch.bool, device=pred_boxes_b.device)
            matched_gt_idx = torch.zeros(
                5040, dtype=torch.long, device=pred_boxes_b.device
            )

            if is_one2one:
                # Center Prior + IoU + Score 매칭
                gt_xc = (gt_bboxes_b[:, 0] + gt_bboxes_b[:, 2]) * 0.5
                gt_yc = (gt_bboxes_b[:, 1] + gt_bboxes_b[:, 3]) * 0.5

                # 💡 [수정 완료]: stride_tensor 참조
                anchor_xc = (
                    decoder_module.anchors[:, 0] * decoder_module.stride_tensor[:, 0]
                )
                anchor_yc = (
                    decoder_module.anchors[:, 1] * decoder_module.stride_tensor[:, 0]
                )

                dist = torch.sqrt(
                    (anchor_xc.unsqueeze(-1) - gt_xc.unsqueeze(0)) ** 2
                    + (anchor_yc.unsqueeze(-1) - gt_yc.unsqueeze(0)) ** 2
                )

                cls_prob = torch.sigmoid(cls_pred[b]).repeat(1, num_gt)
                cost = (iou_matrix**0.8) * (cls_prob**0.5) / (dist + 1e-6)

                best_anchor_per_gt = cost.max(dim=0).indices  # [GT_N]
                pos_mask[best_anchor_per_gt] = True
                matched_gt_idx[best_anchor_per_gt] = torch.arange(
                    num_gt, device=pred_boxes_b.device
                )
                max_ious = iou_matrix[
                    best_anchor_per_gt, torch.arange(num_gt, device=pred_boxes_b.device)
                ]
            else:
                # One-to-Many 다중 할당
                max_ious_all, max_idx_all = iou_matrix.max(dim=1)
                pos_mask = max_ious_all > 0.2

                if not pos_mask.any():
                    best_anchor_idx = iou_matrix.max(dim=0).indices
                    pos_mask[best_anchor_idx] = True
                    matched_gt_idx[best_anchor_idx] = torch.arange(
                        num_gt, device=pred_boxes_b.device
                    )
                    max_ious = iou_matrix.max(dim=0).values
                else:
                    matched_gt_idx = max_idx_all
                    max_ious = max_ious_all[pos_mask]

            # 1. Classification Loss
            cls_target = torch.zeros_like(cls_pred[b])
            # IoU 기반 Target 적용
            cls_target[pos_mask, 0] = max_ious.clamp(max=1.0).to(cls_pred.dtype)

            # BCE Loss (reduction="none")
            bce_loss = F.binary_cross_entropy_with_logits(
                cls_pred[b], cls_target, reduction="none"
            )

            # Focal Weight 적용 (gamma=2.0)
            p_t = torch.sigmoid(cls_pred[b])
            p_t = torch.where(cls_target > 0, p_t, 1.0 - p_t)
            focal_weight = (1.0 - p_t) ** 2.0

            loss_cls_raw = bce_loss * focal_weight

            # 💡 [핵심 해결]: Positive와 Negative의 Loss를 각각 분리해서 정규화
            pos_loss = loss_cls_raw[pos_mask].sum()
            neg_loss = loss_cls_raw[~pos_mask].sum()

            # Negative 손실의 과도한 영향을 억제하기 위해 0.05 스케일링 적용
            num_pos = max(pos_mask.sum().item(), 1.0)
            loss_cls = (pos_loss + 0.05 * neg_loss) / num_pos

            # 하드코딩 상수 대신 Positive 앵커 개수에 기반한 정규화 (최소 1개)
            num_pos = max(pos_mask.sum().item(), 1.0)
            loss_cls = loss_cls_raw.sum() / num_pos
            # 2. Keypoint & BBox Loss
            pos_kpt_pred = kpt_pred[b][pos_mask].view(-1, self.num_kpts, 3)
            pos_gt_kpts = (
                gt_keypoints_b[matched_gt_idx[pos_mask]]
                if not is_one2one
                else gt_keypoints_b
            )
            pos_gt_boxes = (
                gt_bboxes_b[matched_gt_idx[pos_mask]] if not is_one2one else gt_bboxes_b
            )

            gt_x1, gt_y1, gt_x2, gt_y2 = pos_gt_boxes.unbind(-1)
            gt_xc = (gt_x1 + gt_x2) * 0.5
            gt_yc = (gt_y1 + gt_y2) * 0.5
            gt_bw = (gt_x2 - gt_x1).clamp(min=1e-6)
            gt_bh = (gt_y2 - gt_y1).clamp(min=1e-6)

            target_dx = (pos_gt_kpts[..., 0] - gt_xc.unsqueeze(-1)) / gt_bw.unsqueeze(
                -1
            )
            target_dy = (pos_gt_kpts[..., 1] - gt_yc.unsqueeze(-1)) / gt_bh.unsqueeze(
                -1
            )
            target_offset = torch.stack([target_dx, target_dy], dim=-1)

            vis_mask = (pos_gt_kpts[..., 2] > 0).float()

            pred_xy = pos_kpt_pred[..., :2]
            loss_kpt = F.smooth_l1_loss(pred_xy, target_offset, reduction="none").sum(
                -1
            )
            loss_kpt = (loss_kpt * vis_mask).sum() / (vis_mask.sum() + 1e-6)

            pred_vis_logit = pos_kpt_pred[..., 2]
            vis_target = (pos_gt_kpts[..., 2] > 0).float()
            loss_vis = F.binary_cross_entropy_with_logits(pred_vis_logit, vis_target)

            loss_box = (1.0 - max_ious).mean()

            total_loss_cls += loss_cls
            total_loss_box += loss_box
            total_loss_kpt += loss_kpt
            total_loss_vis += loss_vis

        loss = (
            self.cls_weight * (total_loss_cls / batch_size)
            + self.box_weight * (total_loss_box / batch_size)
            + self.kpt_weight * (total_loss_kpt / batch_size)
            + 0.5 * (total_loss_vis / batch_size)
        )

        return loss, {
            "loss_cls": (total_loss_cls / batch_size).item(),
            "loss_dfl": (total_loss_box / batch_size).item(),
            "loss_kpt": (total_loss_kpt / batch_size).item(),
        }

    def forward(self, preds, targets, decoder_module):
        if isinstance(preds, tuple) and len(preds) == 2 and isinstance(preds[0], tuple):
            loss_m, dict_m = self.calc_single_head_loss(
                preds[0], targets, decoder_module, is_one2one=False
            )
            loss_o, dict_o = self.calc_single_head_loss(
                preds[1], targets, decoder_module, is_one2one=True
            )
            return loss_m + loss_o, dict_o
        else:
            return self.calc_single_head_loss(
                preds, targets, decoder_module, is_one2one=True
            )
