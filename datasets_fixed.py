import os
import json
import random
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

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
    images = []
    labels_list = []
    bboxes_list = []
    kpts_list = []

    for img, labels, bboxes, kpts, vis in batch:
        images.append(img)
        labels_list.append(labels)
        bboxes_list.append(bboxes)
        kpts_list.append(kpts)

    batch_size = len(images)
    max_objs = max([len(b) for b in bboxes_list])
    max_objs = max(max_objs, 1)

    padded_labels = torch.zeros((batch_size, max_objs), dtype=torch.long)
    padded_bboxes = torch.zeros((batch_size, max_objs, 4), dtype=torch.float32)
    padded_keypoints = torch.zeros((batch_size, max_objs, 2, 3), dtype=torch.float32)
    gt_mask = torch.zeros((batch_size, max_objs), dtype=torch.bool)

    for i in range(batch_size):
        num_obj = len(bboxes_list[i])
        if num_obj > 0:
            padded_labels[i, :num_obj] = labels_list[i]
            padded_bboxes[i, :num_obj] = bboxes_list[i]
            padded_keypoints[i, :num_obj] = kpts_list[i]
            gt_mask[i, :num_obj] = True

    targets = {
        "gt_labels": padded_labels,
        "gt_bboxes": padded_bboxes,
        "gt_keypoints": padded_keypoints,
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
    """
    def __init__(self, path_list, is_train=True, height=720, width=1280, single_object=True):
        super().__init__()
        self.height = height
        self.width = width
        self.path_list = path_list
        self.is_train = is_train
        self.single_object = single_object

        # 변환 후 최종 해상도 (width: 640, height: 384)
        self.target_width = self.width // 2               # 640
        self.target_height = (self.height // 2) + 24      # 360 + 24 = 384

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
                    v2.RandomRotation(degrees=(-30, 30)),
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
            labels = f.readlines()

        assert len(labels) > 0, f"Label file is empty: {label_path}"

        cls_list = []
        bbox_list = []
        kp_list = []
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
            cls_list.append(int(cls))
            bbox_list.append([xmin, ymin, xmax, ymax])
            kp_list.append(
                [
                    [body_x, body_y, body_vis],
                    [knot_x, knot_y, knot_vis],
                ]
            )

        # 단일 객체 모드: 한 이미지에 참외가 여러 개 있더라도 가장 면적이 큰 메인 참외 1개만 선택
        if self.single_object and len(bbox_list) > 1:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bbox_list]
            main_idx = int(np.argmax(areas))
            cls_list = [cls_list[main_idx]]
            bbox_list = [bbox_list[main_idx]]
            kp_list = [kp_list[main_idx]]

        if len(bbox_list) == 0:
            ret_color = self.transform({"image": color})["image"]
            ret_cls = torch.zeros((1,), dtype=torch.long)
            ret_bbox = torch.zeros((1, 4), dtype=torch.float32)
            ret_kps = torch.zeros((1, 2, 2), dtype=torch.float32)
            ret_vis = torch.zeros((1, 2), dtype=torch.float32)
            return ret_color, ret_cls, ret_bbox, ret_kps, ret_vis

        cls_tensor = torch.tensor(cls_list, dtype=torch.long)
        bbox_tensor = torch.tensor(bbox_list, dtype=torch.float32)
        kp_tensor = torch.tensor(kp_list, dtype=torch.float32)

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
        ret_cls = cls_tensor
        ret_bbox = transformed["boxes"].clone()

        # [버그 수정 완료]: 
        # transformed["boxes"]는 (384, 640) 해상도의 픽셀 좌표이므로
        # 변환 후 크기인 target_width(640)와 target_height(384)로 정규화해야 정확히 0~1 범위가 됩니다.
        ret_bbox[..., [0, 2]] = ret_bbox[..., [0, 2]] / float(self.target_width)
        ret_bbox[..., [1, 3]] = ret_bbox[..., [1, 3]] / float(self.target_height)

        ret_kps = transformed["keypoints"].clone()
        ret_kps[..., 0] = ret_kps[..., 0] / float(self.target_width)
        ret_kps[..., 1] = ret_kps[..., 1] / float(self.target_height)

        ret_vis = kp_tensor[..., 2]

        return ret_color, ret_cls, ret_bbox, ret_kps, ret_vis


if __name__ == "__main__":
    import cv2

    DATA_DIR = "./data"
    data_list = find_melon_data(DATA_DIR)
    train_paths, val_paths = split_data(data_list, train_ratio=0.9, seed=42)

    print(f"📊 Total: {len(data_list)} | Train: {len(train_paths)} | Val: {len(val_paths)}")

    val_dataset = ChamaePackDataset(val_paths, is_train=False)
    img, cls, bbox, kps, vis = val_dataset[0]

    print("✅ 샘플 0번 검증:")
    print(" - Image Shape:", img.shape)
    print(" - Normalized BBox:", bbox)

    # 픽셀 좌표 복원 검증 (640, 384 곱하기)
    pixel_bbox = bbox.clone()
    pixel_bbox[..., [0, 2]] *= 640.0
    pixel_bbox[..., [1, 3]] *= 384.0
    print(" - Restored Pixel BBox (on 640x384):", pixel_bbox)

    # 시각화 저장 검증
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
    x1, y1, x2, y2 = map(int, pixel_bbox[0])
    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("fixed_dataset_verification.png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    print("📸 'fixed_dataset_verification.png' 저장 완료 (박스가 참외에 정확히 맞는지 확인 가능)")
