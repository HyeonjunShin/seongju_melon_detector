import os
import json
import random
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision.io import read_image, ImageReadMode
from torchvision import tv_tensors
import torchvision.transforms.v2 as v2
from torchvision.utils import draw_bounding_boxes, draw_keypoints, make_grid, save_image


def find_melon_data(data_dir):
    data_dir = Path(data_dir)
    image_paths = list(data_dir.glob("*.png"))

    ret = []
    for image_path in image_paths:
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            print(
                f"Label file not found for {image_path}: {label_path} does not exist."
            )
        else:
            ret.append((image_path, label_path))

    return ret


def split_data(data_list, train_ratio=0.9, seed=42):
    random.seed(seed)
    random.shuffle(data_list)

    split_idx = int(len(data_list) * train_ratio)
    train_data = data_list[:split_idx]
    test_data = data_list[split_idx:]

    return train_data, test_data


def chamae_pack_collate_fn(batch):
    images, labels_list, bboxes_list, kpts_list, vis_list = zip(*batch)

    batch_size = len(images)
    max_objs = max(1, max(labels.shape[0] for labels in labels_list))

    padded_labels = torch.zeros((batch_size, max_objs, 1), dtype=torch.long)
    padded_bboxes = torch.zeros((batch_size, max_objs, 4), dtype=torch.float32)
    padded_keypoints = torch.zeros((batch_size, max_objs, 2, 2), dtype=torch.float32)
    padded_vis = torch.zeros((batch_size, max_objs, 2), dtype=torch.float32)
    gt_mask = torch.zeros((batch_size, max_objs, 1), dtype=torch.bool)

    for i, (labels, bboxes, kpts, vis) in enumerate(
        zip(labels_list, bboxes_list, kpts_list, vis_list)
    ):
        num_objs = bboxes.shape[0]
        if num_objs == 0:
            continue

        padded_labels[i, :num_objs] = labels
        padded_bboxes[i, :num_objs] = bboxes
        padded_keypoints[i, :num_objs] = kpts
        padded_vis[i, :num_objs] = vis
        gt_mask[i, :num_objs] = True

    targets = {
        "gt_labels": padded_labels,
        "gt_bboxes": padded_bboxes,
        "gt_keypoints": padded_keypoints,
        "gt_vis": padded_vis,
        "gt_mask": gt_mask,
    }

    return torch.stack(images, dim=0), targets


class ChamaePackDataset(Dataset):
    """
    참외 패키징 검출 데이터셋 (좌표 정규화 버그 수정 버전)
    - 원본 해상도: (height=720, width=1280)
    - 전처리 변환:
        1) Resize -> (360, 640)
        2) Pad -> 상하 각 12px 추가 -> 최종 이미지 해상도: (384, 640)
    - 좌표 정규화:
        변환된 캔버스 크기인 width=640.0, height=384.0 기준으로 정확히 0~1 정규화 수행
    - 객체가 없는 이미지는 빈 .txt 라벨 파일로 학습용 negative sample로 표시
    """

    def __init__(self, path_list, is_train=True, height=720, width=1280):
        super().__init__()
        self.height = height
        self.width = width
        self.path_list = path_list
        self.is_train = is_train

        # 변환 후 최종 해상도 (width: 640, height: 384)
        self.target_width = self.width // 2  # 640
        self.target_height = (self.height // 2) + 24  # 360 + 24 = 384

        if is_train:
            self.transform = v2.Compose(
                [
                    v2.Resize(size=(self.height // 2, self.width // 2)),
                    v2.Pad(padding=[0, 12, 0, 12], fill=0),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    v2.RandomApply(
                        [v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.5
                    ),
                    v2.RandomApply([v2.RandomRotation(degrees=(-10, 10))], p=0.3),
                    v2.ToDtype(torch.float32, scale=True),
                ]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.Resize(size=(self.height // 2, self.width // 2)),
                    v2.Pad(padding=[0, 12, 0, 12], fill=0),
                    v2.ToDtype(torch.float32, scale=True),
                ]
            )

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        image_path, label_path = self.path_list[index]
        color = read_image(str(image_path), ImageReadMode.RGB)

        with open(label_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]

        if len(labels) == 0:
            ret_color = self.transform({"image": color})["image"]
            ret_cls = torch.zeros((0, 1), dtype=torch.long)
            ret_bbox = torch.zeros((0, 4), dtype=torch.float32)
            ret_kps = torch.zeros((0, 2, 2), dtype=torch.float32)
            ret_vis = torch.zeros((0, 2), dtype=torch.float32)
            return ret_color, ret_cls, ret_bbox, ret_kps, ret_vis

        cls_list = []
        bbox_list = []
        kp_list = []
        vis_list = []
        for label in labels:
            (
                cls,
                xmin,
                ymin,
                xmax,
                ymax,
                body_x,
                body_y,
                body_vis,
                knot_x,
                knot_y,
                knot_vis,
            ) = map(float, label.strip().split())
            cls_list.append([int(cls)])
            bbox_list.append([xmin, ymin, xmax, ymax])
            kp_list.append(
                [
                    [body_x, body_y],
                    [knot_x, knot_y],
                ]
            )
            vis_list.append(
                [
                    body_vis,
                    knot_vis,
                ]
            )

        cls_tensor = torch.tensor(cls_list, dtype=torch.long)
        bbox_tensor = torch.tensor(bbox_list, dtype=torch.float32)
        kp_tensor = torch.tensor(kp_list, dtype=torch.float32)
        vis_tensor = torch.tensor(vis_list, dtype=torch.float32)

        tv_bboxes = tv_tensors.BoundingBoxes(
            bbox_tensor, format="XYXY", canvas_size=(self.height, self.width)
        )
        tv_kpts = tv_tensors.KeyPoints(
            kp_tensor[..., :2], canvas_size=(self.height, self.width)
        )

        # Image, BBox, Keypoints 동시 Transform 연산
        transformed = self.transform(
            {
                "image": color,
                "boxes": tv_bboxes,
                "keypoints": tv_kpts,
            }
        )

        ret_color = transformed["image"]
        ret_bbox = transformed["boxes"].to_dense().clone()

        ret_bbox[..., [0, 2]] = ret_bbox[..., [0, 2]] / float(self.target_width)
        ret_bbox[..., [1, 3]] = ret_bbox[..., [1, 3]] / float(self.target_height)

        ret_kps = transformed["keypoints"].to_dense().clone()
        in_image = (
            (ret_kps[..., 0] >= 0)
            & (ret_kps[..., 0] < self.target_width)
            & (ret_kps[..., 1] >= 0)
            & (ret_kps[..., 1] < self.target_height)
        )
        ret_vis = vis_tensor * in_image.to(vis_tensor.dtype)
        ret_kps[..., 0] = ret_kps[..., 0] / float(self.target_width)
        ret_kps[..., 1] = ret_kps[..., 1] / float(self.target_height)

        return ret_color, cls_tensor, ret_bbox, ret_kps, ret_vis


if __name__ == "__main__":
    import cv2
    from torch.utils.data import DataLoader

    DATA_DIR = "./test_data"
    data_list = find_melon_data(DATA_DIR)
    # train_paths, val_paths = split_data(data_list, train_ratio=0.9, seed=42)

    # print(
    # f"📊 Total: {len(data_list)} | Train: {len(train_paths)} | Val: {len(val_paths)}"
    # )

    train_dataset = ChamaePackDataset(data_list, is_train=True)
    train_loader = DataLoader(
        train_dataset, 32, True, collate_fn=chamae_pack_collate_fn, drop_last=False
    )

    for color, target in train_loader:
        print(color.shape)
        print(target["gt_labels"])
        print(target["gt_bboxes"])
        print(target["gt_keypoints"])
        print(target["gt_vis"])
        print(target["gt_mask"])

    # val_dataset = ChamaePackDataset(val_paths, is_train=False)
    # img, cls, bbox, kps, vis = val_dataset[0]

    # print("✅ 샘플 0번 검증:")
    # print(" - Image Shape:", img.shape)
    # print(" - Normalized BBox:", bbox)

    # # 픽셀 좌표 복원 검증 (640, 384 곱하기)
    # pixel_bbox = bbox.clone()
    # pixel_bbox[..., [0, 2]] *= 640.0
    # pixel_bbox[..., [1, 3]] *= 384.0
    # print(" - Restored Pixel BBox (on 640x384):", pixel_bbox)

    # # 시각화 저장 검증
    # img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
    # x1, y1, x2, y2 = map(int, pixel_bbox[0])
    # cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imwrite(
    #     "fixed_dataset_verification.png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # )
    # print(
    #     "📸 'fixed_dataset_verification.png' 저장 완료 (박스가 참외에 정확히 맞는지 확인 가능)"
    # )
