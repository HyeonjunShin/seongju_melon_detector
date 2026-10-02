import nncf
import openvino as ov
import torch
from torch.utils.data import DataLoader
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm, PaddingMode

# 구축한 최신 모듈 임포트
from datasets import (
    ChamaePoseDataset,
    chamae_pose_collate_fn,
    find_melon_data,
    split_data,
)
from model import DualYOLOv8PoseModel, CombinedPoseModel

# =====================================================================
# 1. Calibration 데이터셋 준비 (200장)
# =====================================================================
data_dir = "./data"
save_dir = "./runs/quantization"
paired_path = find_melon_data(data_dir)
train_path, _ = split_data(paired_path, train_ratio=0.9, seed=42)

dataset = ChamaePoseDataset(train_path, is_train=False)  # Data Augmentation 비활성화
data_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=chamae_pose_collate_fn,
    num_workers=4,
)

calibration_tensors = []
for images, targets in data_loader:
    # images shape: (1, 3, 384, 640)
    calibration_tensors.append(images)
    if len(calibration_tensors) >= 200:
        break

print(
    f"✅ Calibration 데이터 준비 완료: {len(calibration_tensors)}개 (Shape: {calibration_tensors[0].shape})"
)

# =====================================================================
# 2. PyTorch 모델 로드 및 FP16 OpenVINO IR 변환
# =====================================================================
device = torch.device("cpu")

# CombinedPoseModel (Decoder 포함)
base_model = DualYOLOv8PoseModel(num_classes=1, reg_max=16, num_kpts=2)
model = CombinedPoseModel(base_model, reg_max=16, num_kpts=2).to(device)

checkpoint_path = "./runs/train/best_combined.pt"
ckpt = torch.load(checkpoint_path, map_location=device)
if "model_state_dict" in ckpt:
    model.base_model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)

model.eval()

# OpenVINO Model 변환
dummy_input = torch.randn(1, 3, 384, 640)
ov_model = ov.convert_model(model, example_input=dummy_input)
ov.save_model(ov_model, f"{save_dir}/model_fp16.xml", compress_to_fp16=True)
print("🚀 1단계: FP16 OpenVINO 모델 변환 완료! (model_fp16.xml)")

# =====================================================================
# 3. NNCF INT8 양자화 수행
# =====================================================================
nncf_dataset = nncf.Dataset(calibration_tensors)

print("⚡ 2단계: INT8 NNCF Calibration 양자화 진행 중...")
quantized_model = nncf.quantize(ov_model, nncf_dataset)

# =====================================================================
# 4. OpenVINO PrePostProcessor 적용 (실시간 카메라 u8 NHWC 전처리 내장)
# =====================================================================
ppp = PrePostProcessor(quantized_model)

# 모델 내부 포맷 (NCHW, 384x640, float32, 0~1)
ppp.input().model().set_layout(ov.Layout("NCHW"))

# 입력 카메라 텐서 포맷 (1280x720, u8, NHWC, 0~255)
ppp.input().tensor().set_shape([1, 720, 1280, 3]).set_element_type(
    ov.Type.u8
).set_layout(ov.Layout("NHWC"))

# 전처리 파이프라인
# 1280x720 -> 640x360 리사이즈 후 위아래 12px씩 패딩하여 640x384로 맞춤
ppp.input().preprocess().resize(
    ResizeAlgorithm.RESIZE_LINEAR, 360, 640
).convert_element_type(ov.Type.f32).pad(
    pads_begin=[0, 12, 0, 0],
    pads_end=[0, 12, 0, 0],
    value=[0.0],
    mode=PaddingMode.CONSTANT,
).scale(
    255.0
)

quantized_model = ppp.build()

ov.save_model(quantized_model, f"{save_dir}/model_int8.xml")
print("✅ 3단계: 전처리가 내장된 INT8 모델 양자화 완료! (model_int8.xml)")
