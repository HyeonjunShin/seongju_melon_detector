import torch

width = 640
height = 384
stride = [8, 16, 32]


def generate_anchors(width=640, height=384, stride=[8, 16, 32], device="cpu"):
    feats = [(height // s, width // s) for s in stride]
    print(feats)
    anchor_list = []
    stride_list = []
    for s, (h, w) in zip(stride, feats):
        print(h * w)
        y, x = torch.meshgrid(
            [torch.arange(h) + 0.5, torch.arange(w) + 0.5], indexing="ij"
        )
        pts = torch.stack([x, y], dim=-1).view(-1, 2)
        anchor_list.append(pts * s)
        stride_list.append(torch.tensor([s] * (h * w)))

    anchor_point_tensor = torch.cat(anchor_list, dim=0).view(1, -1, 2)
    anchor_stride_tensor = torch.cat(stride_list, dim=0).view(1, -1, 1)
    return anchor_point_tensor.to(device), anchor_stride_tensor.to(device)


anchor_points, anchor_strides = generate_anchors()
print(anchor_points)
