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


def find_melon_data(data_dir):
    data_dir = Path(data_dir)
    image_paths = list(data_dir.glob("*.png"))

    ret = []
    for image_path in image_paths:
        label_path = image_path.with_suffix(".json")
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


def chamae_pose_collate_fn(batch):
    images = []
    labels_list = []
    bboxes_list = []
    kpts_list = []

    for img, labels, bboxes, kpts in batch:
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


class ChamaePoseDataset(Dataset):
    def __init__(self, path_list, is_train=True, height=720, width=1280):
        super().__init__()
        self.height = height
        self.width = width
        self.path_list = path_list
        self.is_train = is_train

        # (720//2, 1280//2) -> (360, 640) + Pad [top=12, bottom=12] -> (384, 640)
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
            json_data = json.load(f)

        objects = (
            json_data.get("objects", json_data)
            if isinstance(json_data, dict)
            else json_data
        )

        bboxes_list = []
        labels_list = []
        kpts_list = []

        # Keypoint 안전 파싱 함수 (null 및 visibility=0 예외 처리)
        def safe_extract_kpt(kpt_dict):
            if not kpt_dict:
                return [0.0, 0.0, 0.0]
            vis = float(kpt_dict.get("visibility", 0))
            pt = kpt_dict.get("point")
            if vis == 0 or pt is None:
                return [0.0, 0.0, 0.0]
            return [float(pt[0]), float(pt[1]), vis]

        for obj_idx, obj in enumerate(objects):
            bbox = obj.get("bbox", {}) if isinstance(obj, dict) else {}
            xmin = bbox.get("xmin")
            ymin = bbox.get("ymin")
            xmax = bbox.get("xmax")
            ymax = bbox.get("ymax")

            # BBox 좌표 중 하나라도 null(None)인 경우 안내 출력 후 스킵
            if xmin is None or ymin is None or xmax is None or ymax is None:
                print(
                    f"\n⚠️ [null 발견!] 파일: {label_path.name} | Object Index: {obj_idx}"
                )
                continue

            bboxes_list.append([float(xmin), float(ymin), float(xmax), float(ymax)])
            labels_list.append(int(obj.get("label", 0)))

            kpt_map = {k["name"]: k for k in obj.get("keypoints", [])}
            knot_data = safe_extract_kpt(kpt_map.get("knot_center"))
            body_data = safe_extract_kpt(kpt_map.get("body_center"))

            kpts_list.append([knot_data, body_data])

        if len(bboxes_list) > 0:
            bboxes_tensor = torch.tensor(bboxes_list, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_list, dtype=torch.long)
            kpts_tensor = torch.tensor(kpts_list, dtype=torch.float32)

            tv_bboxes = tv_tensors.BoundingBoxes(
                bboxes_tensor, format="XYXY", canvas_size=(self.height, self.width)
            )
            tv_kpts = tv_tensors.KeyPoints(
                kpts_tensor[..., :2], canvas_size=(self.height, self.width)
            )
            vis_tensor = kpts_tensor[..., 2]
        else:
            tv_bboxes = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format="XYXY",
                canvas_size=(self.height, self.width),
            )
            tv_kpts = tv_tensors.KeyPoints(
                torch.zeros((0, 2, 2), dtype=torch.float32),
                canvas_size=(self.height, self.width),
            )
            vis_tensor = torch.zeros((0, 2), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        # Image, BBox, Keypoints 동시 Transform 연산
        transformed = self.transform(
            {
                "image": color,
                "boxes": tv_bboxes,
                "keypoints": tv_kpts,
            }
        )

        color_trans = transformed["image"]
        bboxes_trans = transformed["boxes"]
        kpts_trans = transformed["keypoints"]

        if bboxes_trans.numel() > 0:
            final_kpts = torch.cat([kpts_trans, vis_tensor.unsqueeze(-1)], dim=-1)
        else:
            final_kpts = torch.zeros((0, 2, 3), dtype=torch.float32)
            bboxes_trans = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        return color_trans, labels_tensor, bboxes_trans, final_kpts


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")  # DataLoader 멀티프로세싱 Tkinter 충돌 방지
    import matplotlib.pyplot as plt
    import cv2
    from tqdm import tqdm
    from torchvision.utils import draw_bounding_boxes

    DATA_DIR = "./data"

    # 1. 데이터 수집 및 Split
    data_list = find_melon_data(DATA_DIR)
    train_paths, val_paths = split_data(data_list, train_ratio=0.9, seed=42)

    print(
        f"Total: {len(data_list)} | Train: {len(train_paths)} | Val: {len(val_paths)}"
    )

    # 2. Dataset 생성
    train_dataset = ChamaePoseDataset(train_paths, is_train=True)
    val_dataset = ChamaePoseDataset(val_paths, is_train=False)

    # 3. DataLoader 결합
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=chamae_pose_collate_fn,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=chamae_pose_collate_fn,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    # 4. 데이터셋 전체 순회 및 전수 검증
    print("\n🔍 [데이터 전수 검증 시작]")

    total_images_processed = 0
    total_objects_count = 0
    visualized = False

    for loader_name, loader in [
        ("Train Loader", train_loader),
        ("Val Loader", val_loader),
    ]:
        print(f"\n--- {loader_name} 순회 중 ---")

        for batch_idx, (images, targets) in enumerate(tqdm(loader, desc=loader_name)):
            batch_size = images.shape[0]
            total_images_processed += batch_size

            gt_mask = targets["gt_mask"]
            valid_objs_per_img = gt_mask.sum(dim=-1)
            total_objects_count += valid_objs_per_img.sum().item()

            if torch.isnan(images).any() or torch.isnan(targets["gt_bboxes"]).any():
                print(f"❌ [에러] Batch {batch_idx}에서 NaN 값이 발견되었습니다!")
            if torch.isinf(images).any() or torch.isinf(targets["gt_bboxes"]).any():
                print(f"❌ [에러] Batch {batch_idx}에서 Inf 값이 발견되었습니다!")

            # -----------------------------------------------------------------
            # 🎨 첫 번째 배치의 유효한 샘플 1회 시각화 후 이미지 파일로 저장
            # -----------------------------------------------------------------
            if not visualized and valid_objs_per_img[0] > 0:
                visualized = True
                print("\n⚡ [1회 시각화 샘플 생성 및 저장 중...]")
                img = images[0]
                mask_0 = gt_mask[0]

                valid_bboxes = targets["gt_bboxes"][0][mask_0]
                valid_keypoints = targets["gt_keypoints"][0][mask_0]

                img_uint8 = (img * 255).to(torch.uint8)

                if valid_bboxes.numel() > 0:
                    vis_img = draw_bounding_boxes(
                        img_uint8,
                        boxes=valid_bboxes,
                        labels=["Melon"] * len(valid_bboxes),
                        colors="cyan",
                        width=2,
                    )
                else:
                    vis_img = img_uint8

                vis_np = vis_img.permute(1, 2, 0).cpu().numpy().copy()

                kpt_colors = [(255, 255, 0), (255, 0, 0)]  # Knot: Cyan, Body: Red
                kpt_names = ["Knot", "Body"]

                for obj_idx in range(len(valid_keypoints)):
                    kpts = valid_keypoints[obj_idx]
                    knot_pt, body_pt = kpts[0], kpts[1]

                    if knot_pt[2] > 0 and body_pt[2] > 0:
                        cv2.line(
                            vis_np,
                            (int(knot_pt[0]), int(knot_pt[1])),
                            (int(body_pt[0]), int(body_pt[1])),
                            color=(255, 255, 255),
                            thickness=2,
                        )

                    for k_idx, (x, y, vis) in enumerate(kpts):
                        if vis > 0:
                            px, py = int(x), int(y)
                            cv2.circle(
                                vis_np,
                                (px, py),
                                radius=5,
                                color=kpt_colors[k_idx],
                                thickness=-1,
                            )
                            cv2.putText(
                                vis_np,
                                f"{kpt_names[k_idx]} ({px}, {py})",
                                (px + 6, py - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                kpt_colors[k_idx],
                                1,
                                cv2.LINE_AA,
                            )

                plt.figure(figsize=(10, 6))
                plt.imshow(vis_np)
                plt.title("Chamae Pose Dataset Batch Sample Verification")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig("./sample_verification.png")
                plt.close()
                print(
                    " 📸 시각화 결과가 './sample_verification.png' 파일로 저장되었습니다."
                )

    print("\n✅ [데이터셋 전체 순회 및 전수 검증 완료]")
    print(f" - 총 검증된 이미지 수 : {total_images_processed} 장")
    print(f" - 총 검증된 객체 수   : {total_objects_count} 개")
