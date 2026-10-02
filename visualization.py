import os
import random
from pathlib import Path

import torch
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import ChamaePoseDataset, find_melon_data, split_data
from model import DualYOLOv8PoseModel, CombinedPoseModel


@torch.no_grad()
def visualize_predictions(
    model_path="./runs/train/best_combined.pt",
    data_dir="./data",
    output_dir="./runs/visualizations",
    num_samples=10,
    conf_thresh=0.25,
    seed=42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [Visualizer] 실행 디바이스: {device}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_model = DualYOLOv8PoseModel(num_classes=1, reg_max=16, num_kpts=2)
    model = CombinedPoseModel(base_model, reg_max=16, num_kpts=2).to(device)

    ckpt = torch.load(model_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.base_model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    data_list = find_melon_data(data_dir)
    _, val_paths = split_data(data_list, train_ratio=0.9, seed=seed)

    random.seed(seed)
    sample_paths = random.sample(val_paths, min(num_samples, len(val_paths)))
    val_dataset = ChamaePoseDataset(sample_paths, is_train=False)

    COLOR_GT_BOX = (0, 255, 0)
    COLOR_GT_KNOT = (255, 0, 255)
    COLOR_GT_BODY = (255, 255, 0)

    COLOR_PRED_BOX = (0, 255, 255)
    COLOR_PRED_KNOT = (255, 255, 0)
    COLOR_PRED_BODY = (255, 0, 0)

    print(f"\n🎨 시각화 작업 시작...")

    for idx in range(len(sample_paths)):
        img_tensor, labels, gt_bboxes, gt_kpts = val_dataset[idx]
        image_path, _ = sample_paths[idx]

        input_tensor = img_tensor.unsqueeze(0).to(device)

        scores, decoded_bboxes, decoded_kpts = model(input_tensor)

        scores_b = scores[0].squeeze(-1)  # [5040]
        bboxes_b = decoded_bboxes[0]  # [5040, 4]
        kpts_b = decoded_kpts[0]  # [5040, 2, 3]

        num_gt = len(gt_bboxes)
        top_k = max(num_gt, 1)

        top_scores, top_indices = scores_b.topk(top_k)

        valid_mask = top_scores >= conf_thresh
        if not valid_mask.any():
            valid_mask[0] = True

        pred_boxes = bboxes_b[top_indices[valid_mask]].cpu().numpy()
        pred_kpts = kpts_b[top_indices[valid_mask]].cpu().numpy()
        pred_scores = top_scores[valid_mask].cpu().numpy()

        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()

        # 1. GT 시각화
        for g_idx in range(len(gt_bboxes)):
            gx1, gy1, gx2, gy2 = map(int, gt_bboxes[g_idx])
            cv2.rectangle(img_np, (gx1, gy1), (gx2, gy2), COLOR_GT_BOX, 1)

            g_knot, g_body = gt_kpts[g_idx][0], gt_kpts[g_idx][1]
            if g_knot[2] > 0 and g_body[2] > 0:
                cv2.line(
                    img_np,
                    (int(g_knot[0]), int(g_knot[1])),
                    (int(g_body[0]), int(g_body[1])),
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )
            if g_knot[2] > 0:
                cv2.circle(
                    img_np, (int(g_knot[0]), int(g_knot[1])), 4, COLOR_GT_KNOT, -1
                )
            if g_body[2] > 0:
                cv2.circle(
                    img_np, (int(g_body[0]), int(g_body[1])), 4, COLOR_GT_BODY, -1
                )

        # 2. Prediction 시각화 (Visibility threshold 필터링 적용)
        for p_idx in range(len(pred_boxes)):
            px1, py1, px2, py2 = map(int, pred_boxes[p_idx])
            score = pred_scores[p_idx]

            cv2.rectangle(img_np, (px1, py1), (px2, py2), COLOR_PRED_BOX, 2)
            cv2.putText(
                img_np,
                f"Melon {score:.2f}",
                (px1, max(py1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_PRED_BOX,
                1,
                cv2.LINE_AA,
            )

            p_knot, p_body = pred_kpts[p_idx][0], pred_kpts[p_idx][1]

            # Visibility Sigmoid 계산
            p_knot_vis = (
                1.0 / (1.0 + np.exp(-p_knot[2])) if p_knot[2] < 10 else p_knot[2]
            )
            p_body_vis = (
                1.0 / (1.0 + np.exp(-p_body[2])) if p_body[2] < 10 else p_body[2]
            )

            VIS_THRESH = 0.5  # Visibility가 50% 이상일 때만 표시

            if p_knot_vis >= VIS_THRESH and p_body_vis >= VIS_THRESH:
                cv2.line(
                    img_np,
                    (int(p_knot[0]), int(p_knot[1])),
                    (int(p_body[0]), int(p_body[1])),
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # 가려진 매듭은 그리지 않고 통과
            if p_knot_vis >= VIS_THRESH:
                cv2.circle(
                    img_np, (int(p_knot[0]), int(p_knot[1])), 5, COLOR_PRED_KNOT, -1
                )

            if p_body_vis >= VIS_THRESH:
                cv2.circle(
                    img_np, (int(p_body[0]), int(p_body[1])), 5, COLOR_PRED_BODY, -1
                )

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(img_np)
        ax.set_title(
            f"Sample {idx + 1:02d} | File: {image_path.name} | GT: {len(gt_bboxes)} / Pred: {len(pred_boxes)}",
            fontsize=12,
            fontweight="bold",
        )
        ax.axis("off")

        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="g", lw=2, label="GT BBox"),
            Line2D([0], [0], color="cyan", lw=2, label="Pred BBox"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="yellow",
                markersize=8,
                label="Pred Knot",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=8,
                label="Pred Body",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right", framealpha=0.8)

        plt.tight_layout()
        save_file = output_path / f"val_result_{idx + 1:02d}_{image_path.stem}.png"
        plt.savefig(save_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f" 📸 시각화 저장 완료: {save_file}")

    print(f"\n✅ 시각화 작업 완료! 결과물이 '{output_dir}' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    visualize_predictions(
        model_path="./runs/train/best_combined.pt",
        data_dir="./data",
        output_dir="./runs/visualizations",
        num_samples=10,
        conf_thresh=0.25,
        seed=42,
    )
