"""Dual-assignment (one-to-many + one-to-one) loss with keypoints.

기존 loss.py 의 OneToManyDetectionLoss 를 상속해서 anchors / decode_boxes /
IoU 유틸을 그대로 재사용한다. model.py 는 수정하지 않아도 동작한다.

학습 + 검증 + 시각화까지 한 번에:
    python loss_dual.py --data ./data --epochs 10
    (라벨 1개로 동작만 확인하려면 --smoke)

사용법 (loss.py 의 학습 루프에서):
    criterion = DualAssignmentPoseLoss().to(device)
    outputs = model(images)                 # (one2many, one2one)
    losses = criterion(outputs, target)     # target = chamae_pack_collate_fn 출력
    losses["loss"].backward()

targets (datasets.py 기준, 모두 640x384 캔버스 기준 0~1 정규화):
    gt_bboxes    [B, M, 4]     XYXY
    gt_keypoints [B, M, 2, 2]  (body, knot) x (x, y)
    gt_vis       [B, M, 2]     0 / 1 (1 = 보임). 0 인 점은 keypoint 좌표 loss 에서 제외
    gt_mask      [B, M, 1]
"""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, nms

from loss import OneToManyDetectionLoss


class DualAssignmentPoseLoss(OneToManyDetectionLoss):
    """one-to-many(top-10) + one-to-one(top-1) 두 헤드를 함께 학습시키는 loss.

    두 헤드는 같은 매칭 metric (cls^alpha * IoU^beta) 을 쓰고 topk 만 다르다.
    이것이 YOLOv10 의 consistent dual assignment 이다.
    """

    def __init__(
        self,
        image_height: int = 384,
        image_width: int = 640,
        strides: Tuple[int, ...] = (8, 16, 32),
        reg_max: int = 16,
        topk_o2m: int = 10,
        topk_o2o: int = 1,
        alpha: float = 0.5,
        beta: float = 6.0,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        dfl_weight: float = 1.5,
        kpt_weight: float = 10.0,
        vis_weight: float = 1.0,
        o2m_weight: float = 1.0,
        o2o_weight: float = 1.0,
        vis_positive_min: float = 1.0,
    ):
        super().__init__(
            image_height=image_height,
            image_width=image_width,
            strides=strides,
            reg_max=reg_max,
            topk=topk_o2m,
            alpha=alpha,
            beta=beta,
            box_weight=box_weight,
            dfl_weight=dfl_weight,
            cls_weight=cls_weight,
        )
        self.topk_o2o = topk_o2o
        self.kpt_weight = kpt_weight
        self.vis_weight = vis_weight
        self.o2m_weight = o2m_weight
        self.o2o_weight = o2o_weight
        # visibility logit 의 positive 기준. gt_vis 가 0/1 이면 1.0 (기본값).
        self.vis_positive_min = vis_positive_min

    # ------------------------------------------------------------------
    # 타깃 전처리: 정규화 좌표 -> 픽셀 좌표, 이미지 밖으로 나간 박스 정리
    # ------------------------------------------------------------------
    def _prepare_targets(self, targets: Dict[str, torch.Tensor], device):
        w, h = float(self.image_width), float(self.image_height)

        boxes = targets["gt_bboxes"].to(device=device, dtype=torch.float32)
        boxes = boxes * boxes.new_tensor([w, h, w, h])
        boxes = torch.stack(
            (
                boxes[..., 0].clamp(0, w),
                boxes[..., 1].clamp(0, h),
                boxes[..., 2].clamp(0, w),
                boxes[..., 3].clamp(0, h),
            ),
            dim=-1,
        )

        kpts = targets["gt_keypoints"].to(device=device, dtype=torch.float32)
        kpts = kpts * kpts.new_tensor([w, h])

        vis = targets["gt_vis"].to(device=device, dtype=torch.float32)

        mask = targets["gt_mask"].to(device=device).bool()
        if mask.dim() == 3:
            mask = mask[..., 0]
        mask = mask & (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])
        return boxes, kpts, vis, mask

    # ------------------------------------------------------------------
    # 할당: topk 만 바꿔서 one-to-many / one-to-one 에 공통으로 사용
    # ------------------------------------------------------------------
    @torch.no_grad()
    def assign(
        self,
        pred_scores: torch.Tensor,  # [N, 1] sigmoid 확률
        pred_boxes: torch.Tensor,  # [N, 4] 픽셀 XYXY
        gt_boxes: torch.Tensor,  # [M, 4] 픽셀 XYXY
        topk: int,
    ):
        """반환: fg [N], gt_idx [N], score [N] (soft cls target, IoU-aware)."""
        n, m = pred_boxes.shape[0], gt_boxes.shape[0]
        device = pred_boxes.device
        fg = torch.zeros(n, dtype=torch.bool, device=device)
        gt_idx = torch.zeros(n, dtype=torch.long, device=device)
        score = torch.zeros(n, dtype=pred_boxes.dtype, device=device)
        if m == 0 or n == 0:
            return fg, gt_idx, score

        anchors_px = self.anchors.to(device) * self.stride_tensor.to(device)
        ax, ay = anchors_px[:, 0:1], anchors_px[:, 1:2]  # [N, 1]
        inside = (
            (ax > gt_boxes[None, :, 0])
            & (ay > gt_boxes[None, :, 1])
            & (ax < gt_boxes[None, :, 2])
            & (ay < gt_boxes[None, :, 3])
        )  # [N, M]

        ious = self._pairwise_iou(pred_boxes, gt_boxes).clamp(min=0)  # [N, M]
        p = pred_scores.expand(-1, m).clamp(min=1e-9, max=1.0)
        align = p.pow(self.alpha) * ious.pow(self.beta)

        # GT 중심 안에 없는 anchor 는 -1 로 눌러서 topk 후보에서 확실히 제외
        metric = align.masked_fill(~inside, -1.0)
        k = min(topk, n)
        idx = metric.topk(k, dim=0).indices  # [k, M]
        cand = torch.zeros_like(inside).scatter_(
            0, idx, torch.ones_like(idx, dtype=torch.bool)
        )
        cand = cand & inside

        # 한 anchor 가 여러 GT 에 뽑히면 align 이 가장 큰 GT 하나만 남김
        best_gt = align.masked_fill(~cand, -1.0).argmax(dim=1)  # [N]
        cand = cand & F.one_hot(best_gt, m).bool()
        fg = cand.any(dim=1)
        gt_idx = best_gt

        # IoU-aware soft target: GT 별 최대 align 으로 정규화 후 최대 IoU 를 곱함
        pos_align = align * cand
        pos_iou = ious * cand
        norm = (
            pos_align / (pos_align.amax(dim=0, keepdim=True) + 1e-9)
        ) * pos_iou.amax(dim=0, keepdim=True)
        score = norm.gather(1, best_gt[:, None]).squeeze(1) * fg
        return fg, gt_idx, score

    def _dfl_per_anchor(
        self, reg_logits_pos: torch.Tensor, target_ltrb: torch.Tensor
    ) -> torch.Tensor:
        """anchor 별 DFL loss [P]. (4 변의 평균)"""
        target = target_ltrb.clamp(min=0, max=self.reg_max - 1 - 1e-2)
        left = target.floor().long()
        right = left + 1
        wr = target - left.to(target.dtype)
        wl = 1.0 - wr
        logits = reg_logits_pos.reshape(-1, self.reg_max)
        ce_l = F.cross_entropy(logits, left.reshape(-1), reduction="none")
        ce_r = F.cross_entropy(logits, right.reshape(-1), reduction="none")
        loss = ce_l * wl.reshape(-1) + ce_r * wr.reshape(-1)
        return loss.view(-1, 4).mean(dim=-1)

    # ------------------------------------------------------------------
    # 헤드 하나에 대한 loss
    # ------------------------------------------------------------------
    def _branch_loss(self, outputs, tgt, topk: int):
        gt_boxes, gt_kpts, gt_vis, gt_mask = tgt
        cls_logits, reg_logits, kpt_logits = (t.float() for t in outputs)
        batch_size, num_anchors, num_classes = cls_logits.shape
        if num_anchors != self.anchors.shape[0]:
            raise ValueError(
                f"Model produced {num_anchors} anchors, but loss expects "
                f"{self.anchors.shape[0]}."
            )
        if num_classes != 1:
            raise ValueError("This loss assumes num_classes=1.")

        device = cls_logits.device
        anchors = self.anchors.to(device)  # [N, 2] grid units
        strides = self.stride_tensor.to(device)  # [N, 1]
        pred_boxes = self.decode_boxes(reg_logits)  # [B, N, 4] 픽셀
        pred_scores = cls_logits.detach().sigmoid()

        cls_target = torch.zeros_like(cls_logits)
        b_list, a_list, g_list = [], [], []
        for b in range(batch_size):
            valid = torch.where(gt_mask[b])[0]
            if valid.numel() == 0:
                continue
            fg, gt_idx, score = self.assign(
                pred_scores[b], pred_boxes[b].detach(), gt_boxes[b, valid], topk
            )
            a_idx = torch.where(fg)[0]
            if a_idx.numel() == 0:
                continue
            cls_target[b, a_idx, 0] = score[a_idx]
            b_list.append(torch.full_like(a_idx, b))
            a_list.append(a_idx)
            g_list.append(valid[gt_idx[a_idx]])

        target_sum = cls_target.sum().clamp(min=1.0)
        loss_cls = (
            F.binary_cross_entropy_with_logits(cls_logits, cls_target, reduction="sum")
            / target_sum
        )

        zero = cls_logits.sum() * 0.0
        loss_box = loss_dfl = loss_kpt = loss_vis = zero
        num_pos = 0

        if a_list:
            b_idx = torch.cat(b_list)
            a_idx = torch.cat(a_list)
            g_idx = torch.cat(g_list)
            num_pos = int(a_idx.numel())

            weight = cls_target[b_idx, a_idx, 0]  # [P] soft score
            tb = gt_boxes[b_idx, g_idx]  # [P, 4]
            stride_pos = strides[a_idx]  # [P, 1]
            anchor_pos = anchors[a_idx]  # [P, 2]

            # --- box (IoU) ---
            iou = self._box_iou_aligned(pred_boxes[b_idx, a_idx], tb)
            loss_box = ((1.0 - iou) * weight).sum() / target_sum

            # --- DFL ---
            gt_grid = tb / stride_pos.repeat(1, 4)
            target_ltrb = torch.stack(
                (
                    anchor_pos[:, 0] - gt_grid[:, 0],
                    anchor_pos[:, 1] - gt_grid[:, 1],
                    gt_grid[:, 2] - anchor_pos[:, 0],
                    gt_grid[:, 3] - anchor_pos[:, 1],
                ),
                dim=-1,
            )
            dfl = self._dfl_per_anchor(reg_logits[b_idx, a_idx], target_ltrb)
            loss_dfl = (dfl * weight).sum() / target_sum

            # --- keypoint / visibility ---
            nk = kpt_logits.shape[-1] // 3
            kp = kpt_logits[b_idx, a_idx].view(-1, nk, 3)
            anchor_px = anchor_pos * stride_pos  # [P, 2]
            # DecodeModule.decode_keypoints 와 동일한 디코딩
            pred_xy = anchor_px[:, None, :] + kp[..., :2] * stride_pos[:, :, None]

            tk = gt_kpts[b_idx, g_idx]  # [P, nk, 2] 픽셀
            tv = gt_vis[b_idx, g_idx]  # [P, nk]
            w, h = float(self.image_width), float(self.image_height)
            # 증강(회전 등)으로 캔버스 밖으로 나간 점은 라벨 없음으로 취급
            in_img = (
                (tk[..., 0] >= 0)
                & (tk[..., 0] < w)
                & (tk[..., 1] >= 0)
                & (tk[..., 1] < h)
            )
            labeled = (tv > 0) & in_img

            # 박스 크기(sqrt(area))로 정규화한 L1 -> 객체 크기에 둔감
            scale = (
                ((tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1]))
                .clamp(min=1.0)
                .sqrt()[:, None]
            )
            l1 = (pred_xy - tk).abs().sum(dim=-1) / scale  # [P, nk]
            loss_kpt = (l1 * labeled).sum() / labeled.sum().clamp(min=1)

            # vis=0 인 점도 "보이지 않음(0)" 타깃으로 학습해야 하므로 마스크 없이 전부 사용.
            # (좌표 loss 와 달리 visibility 는 0 인 점이 오히려 필요한 negative 다)
            vis_target = ((tv >= self.vis_positive_min) & in_img).float()
            loss_vis = F.binary_cross_entropy_with_logits(kp[..., 2], vis_target)

        total = (
            self.cls_weight * loss_cls
            + self.box_weight * loss_box
            + self.dfl_weight * loss_dfl
            + self.kpt_weight * loss_kpt
            + self.vis_weight * loss_vis
        )
        return {
            "loss": total,
            "cls": loss_cls,
            "box": loss_box,
            "dfl": loss_dfl,
            "kpt": loss_kpt,
            "vis": loss_vis,
            "num_pos": torch.tensor(float(num_pos), device=device),
        }

    def forward(self, model_outputs, targets: Dict[str, torch.Tensor]):
        """model_outputs = model(images) = (one2many, one2one)."""
        one2many, one2one = model_outputs
        device = one2many[0].device
        tgt = self._prepare_targets(targets, device)

        m = self._branch_loss(one2many, tgt, self.topk)
        o = self._branch_loss(one2one, tgt, self.topk_o2o)

        out = {"loss": self.o2m_weight * m["loss"] + self.o2o_weight * o["loss"]}
        for prefix, d in (("o2m", m), ("o2o", o)):
            for key, value in d.items():
                if key != "loss":
                    out[f"{prefix}_{key}"] = value.detach()
        return out


# ----------------------------------------------------------------------
# NMS 없이 one2one 헤드를 평가
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate_one2one(
    model,
    val_loader,
    device,
    conf_thres: float = 0.5,
    max_det: int = 100,
    match_iou_thres: float = 0.5,
    nms_iou_thres: Optional[float] = None,
    image_height: int = 384,
    image_width: int = 640,
    num_kpts: int = 2,
    verbose: bool = True,
):
    """one2one 헤드를 top-k 만으로 평가한다 (기본: NMS 없음).

    nms_iou_thres 를 주면 같은 출력에 NMS 를 적용해서, NMS 유무에 따른 차이로
    one2one 헤드가 정말 중복을 학습하지 않았는지 확인할 수 있다.
    """
    from model import DecodeModule

    model.eval()
    decoder = (
        DecodeModule(
            reg_max=16,
            num_kpts=num_kpts,
            strides=[8, 16, 32],
            h=image_height,
            w=image_width,
        )
        .to(device)
        .eval()
    )
    wh_box = torch.tensor(
        [image_width, image_height, image_width, image_height],
        dtype=torch.float32,
        device=device,
    )
    wh_pt = wh_box[:2]

    tp = fp = fn = 0
    num_pred = 0
    num_images = 0
    matched_ious = []
    kpt_err_sum = [0.0] * num_kpts
    kpt_err_cnt = [0] * num_kpts

    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        _o2m, o2o = model(images)
        cls_pred, reg_pred, kpt_pred = o2o
        scores = cls_pred.sigmoid().squeeze(-1)
        boxes_b, kpts_b, _vis_logits = decoder(reg_pred, kpt_pred)

        gt_mask_b = targets["gt_mask"].bool()
        if gt_mask_b.dim() == 3:
            gt_mask_b = gt_mask_b[..., 0]

        for i in range(images.shape[0]):
            num_images += 1

            # ---- GT ----
            sel = gt_mask_b[i]
            gt_boxes = targets["gt_bboxes"][i][sel].to(device).float() * wh_box
            gt_kpts = targets["gt_keypoints"][i][sel].to(device).float() * wh_pt
            gt_vis = targets["gt_vis"][i][sel].to(device).float()
            if gt_boxes.numel() > 0:
                ok = (gt_boxes[:, 2] > gt_boxes[:, 0]) & (
                    gt_boxes[:, 3] > gt_boxes[:, 1]
                )
                gt_boxes, gt_kpts, gt_vis = gt_boxes[ok], gt_kpts[ok], gt_vis[ok]

            # ---- 예측: score top-k -> threshold -> (선택) NMS ----
            s, order = scores[i].topk(min(max_det, scores.shape[1]))
            keep = s >= conf_thres
            order, conf = order[keep], s[keep]
            boxes = boxes_b[i][order]
            kp = kpts_b[i][order]

            ok = (
                torch.isfinite(boxes).all(dim=1)
                & (boxes[:, 2] > boxes[:, 0])
                & (boxes[:, 3] > boxes[:, 1])
            )
            boxes, kp, conf = boxes[ok], kp[ok], conf[ok]
            if nms_iou_thres is not None and boxes.numel() > 0:
                k = nms(boxes, conf, nms_iou_thres)
                boxes, kp, conf = boxes[k], kp[k], conf[k]

            num_pred += len(boxes)
            if len(gt_boxes) == 0:
                fp += len(boxes)
                continue
            if len(boxes) == 0:
                fn += len(gt_boxes)
                continue

            # ---- 신뢰도 순으로 아직 매칭 안 된 GT 와 1:1 매칭 ----
            ious = box_iou(boxes, gt_boxes)
            matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=device)
            for p in conf.argsort(descending=True).tolist():
                cand = ious[p].masked_fill(matched, -1.0)
                g = int(cand.argmax().item())
                if cand[g].item() >= match_iou_thres:
                    tp += 1
                    matched[g] = True
                    matched_ious.append(cand[g].item())
                    for j in range(num_kpts):
                        if gt_vis[g, j] > 0:
                            err = (kp[p, j] - gt_kpts[g, j]).norm().item()
                            kpt_err_sum[j] += err
                            kpt_err_cnt[j] += 1
                else:
                    fp += 1
            fn += int((~matched).sum().item())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    results = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "mean_matched_IoU": sum(matched_ious) / max(len(matched_ious), 1),
        "preds_per_image": num_pred / max(num_images, 1),
        "kpt_err_px": [
            kpt_err_sum[j] / max(kpt_err_cnt[j], 1) for j in range(num_kpts)
        ],
        "kpt_err_count": kpt_err_cnt,
    }

    if not verbose:
        return results

    tag = (
        "one2one (NMS 없음)"
        if nms_iou_thres is None
        else f"one2one + NMS({nms_iou_thres})"
    )
    print(f"\n===== Validation: {tag} =====")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        elif isinstance(value, list):
            print(f"{key:20s}: " + ", ".join(f"{v:.2f}" for v in value))
        else:
            print(f"{key:20s}: {value}")
    print("=" * 40 + "\n")
    return results


def mean_keypoint_error(metrics) -> float:
    """Body/Knot가 모두 평가된 경우 두 키포인트의 평균 픽셀 오차."""
    if not metrics["kpt_err_count"] or any(
        count == 0 for count in metrics["kpt_err_count"]
    ):
        return float("inf")
    return sum(metrics["kpt_err_px"]) / len(metrics["kpt_err_px"])


# ----------------------------------------------------------------------
# 스모크 테스트: 데이터셋 없이 예시 라벨 1개로 과적합이 되는지 확인
# ----------------------------------------------------------------------
def smoke_test():
    from model import Model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # 원본(1280x720) 픽셀 라벨 -> datasets.py 와 같은 변환(0.5배 + 위아래 12px 패딩) 후 정규화
    def to_norm(x, y):
        return x * 0.5 / 640.0, (y * 0.5 + 12.0) / 384.0

    x1, y1 = to_norm(341.6, 402.3)
    x2, y2 = to_norm(648.7, 720.0)
    k1 = to_norm(484.5, 708.8)
    k2 = to_norm(487.9, 443.6)

    batch = 2
    targets = {
        "gt_bboxes": torch.tensor([[[x1, y1, x2, y2]]] * batch),
        "gt_keypoints": torch.tensor([[[list(k1), list(k2)]]] * batch),
        "gt_vis": torch.tensor([[[1.0, 1.0]]] * batch),
        "gt_mask": torch.ones(batch, 1, 1, dtype=torch.bool),
    }
    images = torch.rand(batch, 3, 384, 640, device=device)

    model = Model(num_classes=1, reg_max=16, num_kpts=2).to(device).train()
    criterion = DualAssignmentPoseLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(40):
        losses = criterion(model(images), targets)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        if step % 5 == 0:
            print(
                f"step {step:2d} loss={losses['loss'].item():.3f} "
                f"o2o[cls={losses['o2o_cls'].item():.3f} box={losses['o2o_box'].item():.3f} "
                f"kpt={losses['o2o_kpt'].item():.3f} vis={losses['o2o_vis'].item():.3f}] "
                f"pos o2m/o2o={int(losses['o2m_num_pos'])}/{int(losses['o2o_num_pos'])}"
            )


# ----------------------------------------------------------------------
# 시각화: one2one 예측 (box + keypoint), 기본은 NMS 없음
# ----------------------------------------------------------------------
KPT_NAMES = ("body", "knot")
KPT_COLORS = ((0, 255, 255), (255, 165, 0))  # body = cyan, knot = orange


@torch.no_grad()
def visualize_one2one(
    model,
    val_loader,
    device,
    save_path: str = "val_one2one.png",
    num_images: int = 8,
    conf_thres: float = 0.3,
    max_det: int = 10,
    nms_iou_thres: Optional[float] = None,
    image_height: int = 384,
    image_width: int = 640,
    num_kpts: int = 2,
    show: bool = False,
):
    """검증 이미지에 GT(초록)와 one2one 예측(빨강)을 그려서 저장한다.

    - GT box: 초록 / GT keypoint: 색 있는 빈 원 (vis=0 이면 회색)
    - 예측 box: 빨강 + confidence / 예측 keypoint: 색 있는 채운 원 + visibility 확률
    - conf_thres 이상인 예측이 없으면 진단용으로 상위 3개를 대신 그린다.
    """
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
    from model import DecodeModule

    if not show:
        plt.switch_backend("Agg")

    model.eval()
    decoder = (
        DecodeModule(
            reg_max=16,
            num_kpts=num_kpts,
            strides=[8, 16, 32],
            h=image_height,
            w=image_width,
        )
        .to(device)
        .eval()
    )
    w, h = image_width, image_height
    box_scale = torch.tensor([w, h, w, h], dtype=torch.float32)
    pt_scale = torch.tensor([w, h], dtype=torch.float32)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:  # Pillow < 10.1 은 size 인자가 없음
        font = ImageFont.load_default()

    cols = 4
    rows = max(1, (num_images + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows))
    axes = axes.flatten()

    shown = 0
    for images, targets in val_loader:
        _o2m, o2o = model(images.to(device, non_blocking=True))
        cls_pred, reg_pred, kpt_pred = o2o
        scores = cls_pred.sigmoid().squeeze(-1)
        boxes_b, kpts_b, vis_logits = decoder(reg_pred, kpt_pred)
        vis_prob = vis_logits.sigmoid()

        for i in range(images.shape[0]):
            if shown >= num_images:
                break

            img = (
                images[i]
                .detach()
                .cpu()
                .clamp(0, 1)
                .permute(1, 2, 0)
                .mul(255)
                .byte()
                .contiguous()
                .numpy()
            )
            pil = Image.fromarray(img)
            draw = ImageDraw.Draw(pil)

            # ---------- GT ----------
            mask = targets["gt_mask"][i].reshape(-1).bool()
            gt_b = targets["gt_bboxes"][i][mask].float() * box_scale
            gt_k = targets["gt_keypoints"][i][mask].float() * pt_scale
            gt_v = targets["gt_vis"][i][mask].float()
            for box, kps, vis in zip(gt_b.tolist(), gt_k.tolist(), gt_v.tolist()):
                draw.rectangle(box, outline="lime", width=3)
                for j, (x, y) in enumerate(kps):
                    if (
                        vis[j] <= 0
                    ):  # vis=0 은 좌표가 placeholder(0,0) 이므로 그리지 않음
                        continue
                    r = 7
                    color = KPT_COLORS[j % len(KPT_COLORS)]
                    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)

            # ---------- 예측 (one2one, top-k 만 사용) ----------
            s, order = scores[i].topk(min(max_det, scores.shape[1]))
            keep = s >= conf_thres
            if not keep.any():
                keep = torch.zeros_like(keep)
                keep[:3] = True
            order, conf = order[keep], s[keep]
            boxes = boxes_b[i][order]
            kp = kpts_b[i][order]
            vp = vis_prob[i][order]

            ok = (
                torch.isfinite(boxes).all(dim=1)
                & torch.isfinite(kp).flatten(1).all(dim=1)
                & (boxes[:, 2] > boxes[:, 0])
                & (boxes[:, 3] > boxes[:, 1])
            )
            boxes, kp, vp, conf = boxes[ok], kp[ok], vp[ok], conf[ok]
            if nms_iou_thres is not None and boxes.numel() > 0:
                k = nms(boxes, conf, nms_iou_thres)
                boxes, kp, vp, conf = boxes[k], kp[k], vp[k], conf[k]

            for box, c, kps, vps in zip(
                boxes.tolist(), conf.tolist(), kp.tolist(), vp.tolist()
            ):
                draw.rectangle(box, outline="red", width=2)
                draw.text(
                    (box[0], min(h - 16, box[3] + 2)), f"{c:.2f}", fill="red", font=font
                )
                for j, (x, y) in enumerate(kps):
                    color = KPT_COLORS[j % len(KPT_COLORS)]
                    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
                    draw.text((x + 6, y - 6), f"{vps[j]:.2f}", fill=color, font=font)

            axes[shown].imshow(pil)
            axes[shown].set_title(f"Validation {shown + 1}  (pred: {len(boxes)})")
            axes[shown].axis("off")
            shown += 1

        if shown >= num_images:
            break

    for j in range(shown, len(axes)):
        axes[j].axis("off")

    legend = (
        "GT box: green | pred box: red | "
        + ", ".join(
            f"{n}: {'cyan' if k == 0 else 'orange'}"
            for k, n in enumerate(KPT_NAMES[:num_kpts])
        )
        + " | hollow circle = GT, filled dot = prediction (number = visibility prob.)"
    )
    fig.suptitle(legend, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"시각화 저장 완료: {save_path} (표시한 이미지 수: {shown})")


def sweep_conf_thres(
    model,
    val_loader,
    device,
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
    match_iou_thres: float = 0.5,
):
    """confidence threshold 별 precision / recall / F1 (NMS 없음).

    IoU 를 반영한 soft target 으로 학습하면 confidence 가 낮게 나오므로,
    threshold 를 내리면 recall 이 얼마나 오르는지 확인하는 용도.
    """
    print("---------- confidence threshold sweep (NMS 없음) ----------")
    print(
        f"{'thr':>5} {'TP':>5} {'FP':>4} {'FN':>4} {'prec':>7} {'recall':>7} "
        f"{'F1':>7} {'IoU':>6} {'kptErr(px)':>12}"
    )
    rows = {}
    for t in thresholds:
        r = evaluate_one2one(
            model,
            val_loader,
            device,
            conf_thres=t,
            match_iou_thres=match_iou_thres,
            verbose=False,
        )
        kerr = "/".join(f"{v:.1f}" for v in r["kpt_err_px"])
        print(
            f"{t:5.2f} {r['TP']:5d} {r['FP']:4d} {r['FN']:4d} {r['precision']:7.4f} "
            f"{r['recall']:7.4f} {r['F1']:7.4f} {r['mean_matched_IoU']:6.3f} {kerr:>12}"
        )
        rows[t] = r
    print("-----------------------------------------------------------\n")
    return rows


def run_final_report(
    model,
    val_loader,
    device,
    out_dir: str = "runs/dual",
    conf_thres: float = 0.5,
    nms_iou_thres: float = 0.5,
):
    """학습 후 검증 리포트: NMS 없음 vs NMS 적용 비교 + 시각화 저장."""
    os.makedirs(out_dir, exist_ok=True)

    r_free = evaluate_one2one(model, val_loader, device, conf_thres=conf_thres)
    r_nms = evaluate_one2one(
        model, val_loader, device, conf_thres=conf_thres, nms_iou_thres=nms_iou_thres
    )

    extra_fp = r_free["FP"] - r_nms["FP"]
    print("---------- NMS-free 확인 ----------")
    print(f"NMS 없음 FP={r_free['FP']} / NMS 적용 FP={r_nms['FP']}  (차이 {extra_fp})")
    print(f"NMS 없음 F1={r_free['F1']:.4f} / NMS 적용 F1={r_nms['F1']:.4f}")
    if extra_fp > 0:
        print("-> 차이가 크면 one2one 헤드가 아직 중복 예측을 내고 있다는 뜻입니다.")
    print("-----------------------------------\n")

    sweep_conf_thres(model, val_loader, device)

    print("(참고) 이미지당 예측 1개만 사용: max_det=1, conf_thres=0.0")
    evaluate_one2one(model, val_loader, device, conf_thres=0.0, max_det=1)

    visualize_one2one(
        model,
        val_loader,
        device,
        save_path=os.path.join(out_dir, "val_one2one.png"),
        conf_thres=min(conf_thres, 0.3),
    )
    return {"one2one": r_free, "one2one_nms": r_nms}


# ----------------------------------------------------------------------
# 학습 파이프라인
# ----------------------------------------------------------------------
def run_epoch(
    model, loader, criterion, device, optimizer=None, grad_clip: float = 10.0
):
    """optimizer 가 있으면 학습, 없으면 검증. loss 항목별 평균을 반환."""
    train = optimizer is not None
    model.train(train)
    sums: Dict[str, float] = {}
    batches = 0
    with torch.set_grad_enabled(train):
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            losses = criterion(model(images), target)

            if train:
                if not torch.isfinite(losses["loss"]):
                    print("경고: loss 가 NaN/Inf 라서 이 배치는 건너뜁니다.")
                    continue
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            batches += 1
    return {key: value / max(batches, 1) for key, value in sums.items()}


def _fmt(stats: Dict[str, float], prefix: str) -> str:
    return (
        f"cls={stats[prefix + '_cls']:.4f}, box={stats[prefix + '_box']:.4f}, "
        f"dfl={stats[prefix + '_dfl']:.4f}, kpt={stats[prefix + '_kpt']:.4f}, "
        f"vis={stats[prefix + '_vis']:.4f}, pos={stats[prefix + '_num_pos']:.1f}"
    )


def main():
    import argparse
    import faulthandler

    # segfault 가 나면 어느 Python 코드에서 죽었는지 스택을 출력해 준다
    faulthandler.enable()

    from torch.utils.data import DataLoader

    from datasets import (
        ChamaePackDataset,
        chamae_pack_collate_fn,
        find_melon_data,
        split_data,
    )
    from model import Model

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="runs/dual")
    parser.add_argument("--conf-thres", type=float, default=0.3)
    parser.add_argument(
        "--vis-every",
        type=int,
        default=0,
        help="N epoch 마다 시각화 저장 (0 이면 학습 종료 후에만)",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="데이터셋 없이 라벨 1개로 동작만 확인"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="학습 없이 저장된 체크포인트로 검증/시각화만 실행",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="--eval-only 에서 사용할 체크포인트 (기본: <out>/best.pt)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="학습을 이어서 할 체크포인트 (ckpt_full.pt 또는 last.pt)",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="가중치만 있는 파일(last.pt)로 재개할 때 이미 끝난 epoch 수",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DataLoader worker 를 epoch 마다 새로 만들지 않고 재사용",
    )
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
        return

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    data_list = find_melon_data(args.data)
    train_paths, val_paths = split_data(data_list, train_ratio=0.9, seed=42)
    print(
        f"📊 Dataset Status -> Total: {len(data_list)} | "
        f"Train: {len(train_paths)} | Val: {len(val_paths)}"
    )

    train_loader = DataLoader(
        ChamaePackDataset(train_paths, is_train=True),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=chamae_pack_collate_fn,
        num_workers=args.workers,
        pin_memory=device == "cuda",
        # persistent_workers=args.persistent_workers and args.workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        ChamaePackDataset(val_paths, is_train=False),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=chamae_pack_collate_fn,
        num_workers=args.workers,
        pin_memory=device == "cuda",
        # persistent_workers=args.persistent_workers and args.workers > 0,
        drop_last=False,
    )

    model = Model(num_classes=1, reg_max=16, num_kpts=2).to(device)

    if args.eval_only:
        ckpt = args.ckpt or os.path.join(args.out, "best.pt")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"체크포인트 로드: {ckpt}")
        run_final_report(
            model, val_loader, device, out_dir=args.out, conf_thres=args.conf_thres
        )
        return

    criterion = DualAssignmentPoseLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_path = os.path.join(args.out, "best.pt")
    last_path = os.path.join(args.out, "last.pt")
    full_path = os.path.join(
        args.out, "ckpt_full.pt"
    )  # 모델+optimizer+scheduler (재개용)

    best_kpt_error = float("inf")
    start_epoch = args.start_epoch
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = state["epoch"]
            best_kpt_error = state.get("best_kpt_error", best_kpt_error)
        else:  # last.pt / best.pt 처럼 가중치만 저장된 파일
            model.load_state_dict(state)
            for _ in range(start_epoch):
                scheduler.step()
        print(f"체크포인트에서 재개: {args.resume} (epoch {start_epoch + 1} 부터)")

    for e in range(start_epoch, args.epochs):
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        va = run_epoch(model, val_loader, criterion, device)
        kpt_metrics = evaluate_one2one(
            model, val_loader, device, conf_thres=0.0, max_det=1, verbose=False
        )
        val_kpt_error = mean_keypoint_error(kpt_metrics)
        scheduler.step()

        print(
            f"Epoch [{e + 1}/{args.epochs}] Train Loss: {tr['loss']:.4f} | "
            f"Val Loss: {va['loss']:.4f} | Val Kpt Error: {val_kpt_error:.2f}px | "
            f"lr: {optimizer.param_groups[0]['lr']:.2e}"
        )
        print(f"  Train o2o: {_fmt(tr, 'o2o')}")
        print(f"  Val   o2o: {_fmt(va, 'o2o')}")
        print(f"  Val   o2m: {_fmt(va, 'o2m')}")

        torch.save(model.state_dict(), last_path)
        if val_kpt_error < best_kpt_error:
            best_kpt_error = val_kpt_error
            torch.save(model.state_dict(), best_path)
            print(
                f"  ✔ best 갱신 (val keypoint error {best_kpt_error:.2f}px) -> {best_path}"
            )

        # 저장 도중 죽어도 파일이 깨지지 않도록 임시 파일에 쓰고 교체
        tmp_path = full_path + ".tmp"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": e + 1,
                "best_kpt_error": best_kpt_error,
            },
            tmp_path,
        )
        os.replace(tmp_path, full_path)

        if args.vis_every > 0 and (e + 1) % args.vis_every == 0:
            visualize_one2one(
                model,
                val_loader,
                device,
                save_path=os.path.join(args.out, f"val_ep{e + 1:03d}.png"),
                conf_thres=min(args.conf_thres, 0.3),
            )

    # ---------- 학습 종료 후: best 가중치로 검증 + 시각화 ----------
    print("\n학습 종료. best 가중치로 최종 검증을 진행합니다.")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    run_final_report(
        model, val_loader, device, out_dir=args.out, conf_thres=args.conf_thres
    )


if __name__ == "__main__":
    main()
