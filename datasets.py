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
from torchvision.utils import draw_bounding_boxes


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


class ChamaePackDataset(Dataset):
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
            labels = f.readlines()

        assert not len(labels) == 0, f"Label file is empty: {label_path}"

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

        if len(bbox_list) == 0:
            ret_color = self.transform({"image": color})["image"]
            ret_cls = torch.zeros((0,), dtype=torch.long)
            ret_bbox = torch.zeros((0, 4), dtype=torch.float32)
            ret_kps = torch.zeros((0, 2, 3), dtype=torch.float32)
            ret_vis = torch.zeros((0, 2), dtype=torch.float32)
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
        ret_bbox = transformed["boxes"]
        ret_bbox[..., [0, 2]] = ret_bbox[..., [0, 2]] / self.width
        ret_bbox[..., [1, 3]] = ret_bbox[..., [1, 3]] / self.height

        ret_kps = transformed["keypoints"]
        ret_kps[..., 0] = ret_kps[..., 0] / self.width
        ret_kps[..., 1] = ret_kps[..., 1] / self.height

        ret_vis = kp_tensor[..., 2]

        return ret_color, ret_cls, ret_bbox, ret_kps, ret_vis


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("TkAgg")  # 또는 'QtAgg'
    import matplotlib.pyplot as plt

    import cv2
    from tqdm import tqdm
    from torchvision.utils import (
        draw_bounding_boxes,
        draw_keypoints,
        make_grid,
        save_image,
    )

    DATA_DIR = "./data"

    # 1. 데이터 수집 및 Split
    data_list = find_melon_data(DATA_DIR)
    train_paths, val_paths = split_data(data_list, train_ratio=0.9, seed=42)

    print(
        f"Total: {len(data_list)} | Train: {len(train_paths)} | Val: {len(val_paths)}"
    )

    # 2. Dataset 생성
    train_dataset = ChamaePackDataset(train_paths, is_train=True)
    val_dataset = ChamaePackDataset(val_paths, is_train=False)

    # 3. DataLoader 결합
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        # collate_fn=chamae_pack_collate_fn,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        # collate_fn=chamae_pack_collate_fn,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    KP_COLORS = ["green", "yellow"]

    for batch in train_loader:
        color, cls, bbox, kps, vis = batch
        print(bbox)
        print(kps)
        break
        if color.dtype == torch.float32 and color.max() <= 1.0:
            img_batch_uint8 = (color * 255).type(torch.uint8)
        else:
            img_batch_uint8 = color.type(torch.uint8)

        processed_images = []
        batch_size = color.shape[0]

        for i in range(batch_size):
            img = img_batch_uint8[i]
            boxes = bbox[i]
            classes = cls[i]
            keypoints = kps[i]  # 구조: [Num_Instances, Num_Keypoints, 2]

            if boxes.numel() == 0:
                processed_images.append(img)
                continue

            labels = [str(c.item()) for c in classes]
            result_img = draw_bounding_boxes(
                image=img,
                boxes=boxes,
                labels=labels,
                colors="cyan",
                width=2,
            )

            num_kps_types = keypoints.shape[1]  # 참외당 할당된 키포인트 개수
            for k_idx in range(num_kps_types):
                # 미리 정의한 색상 매핑 (범위 초과 시 빨간색)
                kp_color = KP_COLORS[k_idx] if k_idx < len(KP_COLORS) else "red"

                # 특정 인덱스의 키포인트만 슬라이싱 [N, 1, 2]
                single_kp_type = keypoints[:, k_idx : k_idx + 1, :]

                # 이전 단계 결과물(result_img) 위에 누적해서 그리기
                result_img = draw_keypoints(
                    image=result_img,
                    keypoints=single_kp_type,
                    colors=kp_color,
                    radius=5,
                    width=3,
                )

            processed_images.append(result_img)

        result_batch = torch.stack(processed_images, dim=0)

        grid_img = make_grid(result_batch, nrow=4)

        # 4. GUI 여부와 상관없이 무조건 안전하게 파일로 저장 (matplotlib 경고 우회)
        save_image(grid_img.float() / 255.0, "batch_result_grid.png")
        print("배치 전체 시각화 결과가 'batch_result_grid.png' 파일로 저장되었습니다.")

        # 5. (선택) 만약 GUI 창이 정상 동작한다면 화면에도 띄우기
        try:
            plt.figure(figsize=(12, 8))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.axis("off")
            plt.show()
        except Exception as e:
            print("GUI 창을 띄울 수 없어 저장을 완료한 파일로 결과 확인을 대체합니다.")

        break  # 첫 번째 배치만 확인하고 루프 종료

    # 4. 데이터셋 전체 순회 및 전수 검증
    # print("\n🔍 [데이터 전수 검증 시작]")

    # total_images_processed = 0
    # total_objects_count = 0
    # visualized = False

    # for loader_name, loader in [
    #     ("Train Loader", train_loader),
    #     ("Val Loader", val_loader),
    # ]:
    #     print(f"\n--- {loader_name} 순회 중 ---")

    #     for batch_idx, (images, targets) in enumerate(tqdm(loader, desc=loader_name)):
    #         batch_size = images.shape[0]
    #         total_images_processed += batch_size

    #         gt_mask = targets["gt_mask"]
    #         valid_objs_per_img = gt_mask.sum(dim=-1)
    #         total_objects_count += valid_objs_per_img.sum().item()

    #         if torch.isnan(images).any() or torch.isnan(targets["gt_bboxes"]).any():
    #             print(f"❌ [에러] Batch {batch_idx}에서 NaN 값이 발견되었습니다!")
    #         if torch.isinf(images).any() or torch.isinf(targets["gt_bboxes"]).any():
    #             print(f"❌ [에러] Batch {batch_idx}에서 Inf 값이 발견되었습니다!")

    #         # -----------------------------------------------------------------
    #         # 🎨 첫 번째 배치의 유효한 샘플 1회 시각화 후 이미지 파일로 저장
    #         # -----------------------------------------------------------------
    #         if not visualized and valid_objs_per_img[0] > 0:
    #             visualized = True
    #             print("\n⚡ [1회 시각화 샘플 생성 및 저장 중...]")
    #             img = images[0]
    #             mask_0 = gt_mask[0]

    #             valid_bboxes = targets["gt_bboxes"][0][mask_0]
    #             valid_keypoints = targets["gt_keypoints"][0][mask_0]

    #             img_uint8 = (img * 255).to(torch.uint8)

    #             if valid_bboxes.numel() > 0:
    #                 vis_img = draw_bounding_boxes(
    #                     img_uint8,
    #                     boxes=valid_bboxes,
    #                     labels=["Melon"] * len(valid_bboxes),
    #                     colors="cyan",
    #                     width=2,
    #                 )
    #             else:
    #                 vis_img = img_uint8

    #             vis_np = vis_img.permute(1, 2, 0).cpu().numpy().copy()

    #             kpt_colors = [(255, 255, 0), (255, 0, 0)]  # Knot: Cyan, Body: Red
    #             kpt_names = ["Knot", "Body"]

    #             for obj_idx in range(len(valid_keypoints)):
    #                 kpts = valid_keypoints[obj_idx]
    #                 knot_pt, body_pt = kpts[0], kpts[1]

    #                 if knot_pt[2] > 0 and body_pt[2] > 0:
    #                     cv2.line(
    #                         vis_np,
    #                         (int(knot_pt[0]), int(knot_pt[1])),
    #                         (int(body_pt[0]), int(body_pt[1])),
    #                         color=(255, 255, 255),
    #                         thickness=2,
    #                     )

    #                 for k_idx, (x, y, vis) in enumerate(kpts):
    #                     if vis > 0:
    #                         px, py = int(x), int(y)
    #                         cv2.circle(
    #                             vis_np,
    #                             (px, py),
    #                             radius=5,
    #                             color=kpt_colors[k_idx],
    #                             thickness=-1,
    #                         )
    #                         cv2.putText(
    #                             vis_np,
    #                             f"{kpt_names[k_idx]} ({px}, {py})",
    #                             (px + 6, py - 6),
    #                             cv2.FONT_HERSHEY_SIMPLEX,
    #                             0.4,
    #                             kpt_colors[k_idx],
    #                             1,
    #                             cv2.LINE_AA,
    #                         )

    #             plt.figure(figsize=(10, 6))
    #             plt.imshow(vis_np)
    #             plt.title("Chamae Pose Dataset Batch Sample Verification")
    #             plt.axis("off")
    #             plt.tight_layout()
    #             plt.savefig("./sample_verification.png")
    #             plt.close()
    #             print(
    #                 " 📸 시각화 결과가 './sample_verification.png' 파일로 저장되었습니다."
    #             )

    # print("\n✅ [데이터셋 전체 순회 및 전수 검증 완료]")
    # print(f" - 총 검증된 이미지 수 : {total_images_processed} 장")
    # print(f" - 총 검증된 객체 수   : {total_objects_count} 개")
