# External components and weight provenance

- PyTorch: https://github.com/pytorch/pytorch/blob/main/LICENSE
- Torchvision: https://github.com/pytorch/vision/blob/main/LICENSE (BSD3-Clause code license)
- Backbone: torchvision.models.resnet18, ResNet18_Weights.IMAGENET1K_V1.
- Weight documentation: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- Official weight URL: https://download.pytorch.org/models/resnet18-f37072fd.pth
- Copied from pre-existing local cache; no new weight download.
- File SHA256: f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec
- Upstream training dataset: ImageNet-1K. Code license is not automatically a license grant for all training data/weight uses; company approval of this exact checkpoint remains unresolved. Do not describe production usage as legally approved.

No Ultralytics package, model source, or YOLO Pose checkpoint is used. Matching, dense heads, losses, preprocessing, training, and inference are implemented here. Generic one-to-many/one-to-one learning concepts are discussed in https://arxiv.org/abs/2405.14458 ; this project is not a reproduction of the full YOLOv10 implementation.

numpy and Pillow are external dependencies with their own notices. Preserve applicable dependency notices when distributing software. No overall product license is selected on the company's behalf.

Integrated preprocessing implements separable bilinear antialias coefficient geometry and
22-bit coefficient quantization consistent with Pillow's resize implementation:
https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c .
FP32 accumulation is tested against Pillow's integer path and may differ by one uint8 level.
