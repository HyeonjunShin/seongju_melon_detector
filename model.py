import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. 기본 레이어 및 모듈 정의 (Conv, C2f, SPPF)
# =====================================================================


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        # FX Tracer 안정성을 위한 split 사용
        y1, y2 = torch.split(self.cv1(x), self.c, dim=1)
        y = [y1, y2]
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


# =====================================================================
# 2. YOLOv8 Pose Head (Class, BBox, Keypoints 예측)
# =====================================================================


class StandardYOLOv8PoseHead(nn.Module):
    def __init__(
        self, ch_channels=[64, 128, 256], num_classes=1, reg_max=16, num_kpts=2
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.reg_output_dim = 4 * reg_max
        self.num_kpts = num_kpts
        self.kpt_dim = num_kpts * 3  # (dx, dy, visibility) x 2개 점

        self.cls_cvs = nn.ModuleList(
            nn.Sequential(
                Conv(c, c, 3, 1), Conv(c, c, 3, 1), nn.Conv2d(c, self.num_classes, 1)
            )
            for c in ch_channels
        )
        self.reg_cvs = nn.ModuleList(
            nn.Sequential(
                Conv(c, c, 3, 1), Conv(c, c, 3, 1), nn.Conv2d(c, self.reg_output_dim, 1)
            )
            for c in ch_channels
        )
        self.kpt_cvs = nn.ModuleList(
            nn.Sequential(
                Conv(c, c, 3, 1), Conv(c, c, 3, 1), nn.Conv2d(c, self.kpt_dim, 1)
            )
            for c in ch_channels
        )

    def forward(self, feats):
        cls_outputs, reg_outputs, kpt_outputs = [], [], []
        for i, x in enumerate(feats):
            cls_logits = self.cls_cvs[i](x)
            reg_dist = self.reg_cvs[i](x)
            kpt_dist = self.kpt_cvs[i](x)

            b = cls_logits.shape[0]
            cls_outputs.append(cls_logits.reshape(b, self.num_classes, -1))
            reg_outputs.append(reg_dist.reshape(b, self.reg_output_dim, -1))
            kpt_outputs.append(kpt_dist.reshape(b, self.kpt_dim, -1))

        cls_all = torch.cat(cls_outputs, dim=-1).permute(0, 2, 1).contiguous()
        reg_all = torch.cat(reg_outputs, dim=-1).permute(0, 2, 1).contiguous()
        kpt_all = torch.cat(kpt_outputs, dim=-1).permute(0, 2, 1).contiguous()

        return cls_all, reg_all, kpt_all


# =====================================================================
# 3. 메인 Pose 감지 모델 (Backbone + Neck + Dual Pose Head)
# =====================================================================


class DualYOLOv8PoseModel(nn.Module):
    def __init__(self, num_classes=1, reg_max=16, num_kpts=2):
        super().__init__()

        # Backbone
        self.stem = Conv(3, 16, 3, 2)
        self.bl1 = C2f(16, 32, n=1, shortcut=True)
        self.down1 = Conv(32, 64, 3, 2)
        self.bl2 = C2f(64, 64, n=2, shortcut=True)
        self.down2 = Conv(64, 128, 3, 2)
        self.bl3 = C2f(128, 128, n=2, shortcut=True)
        self.down3 = Conv(128, 256, 3, 2)
        self.bl4 = C2f(256, 256, n=1, shortcut=True)
        self.down4 = Conv(256, 256, 3, 2)
        self.sppf = SPPF(256, 256, k=5)

        # PAFPN Neck
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv_p4_fuse = Conv(256 + 256, 256, 1, 1)
        self.c2f_p4_up = C2f(256, 128, n=1, shortcut=False)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv_p3_fuse = Conv(128 + 128, 128, 1, 1)
        self.c2f_p3_up = C2f(128, 64, n=1, shortcut=False)

        self.down_p3 = Conv(64, 64, 3, 2)
        self.cv_p4_down = Conv(64 + 128, 128, 1, 1)
        self.c2f_p4_down = C2f(128, 128, n=1, shortcut=False)

        self.down_p4 = Conv(128, 128, 3, 2)
        self.cv_p5_down = Conv(128 + 256, 256, 1, 1)
        self.c2f_p5_down = C2f(256, 256, n=1, shortcut=False)

        # Dual Assignment Head
        self.one2many_head = StandardYOLOv8PoseHead(
            ch_channels=[64, 128, 256],
            num_classes=num_classes,
            reg_max=reg_max,
            num_kpts=num_kpts,
        )
        self.one2one_head = StandardYOLOv8PoseHead(
            ch_channels=[64, 128, 256],
            num_classes=num_classes,
            reg_max=reg_max,
            num_kpts=num_kpts,
        )

    def forward(self, x):
        # Backbone
        x_stem = self.stem(x)
        x_stage2 = self.bl2(self.down1(self.bl1(x_stem)))
        x_p3_raw = self.bl3(self.down2(x_stage2))
        x_p4_raw = self.bl4(self.down3(x_p3_raw))
        x_p5_raw = self.sppf(self.down4(x_p4_raw))

        # Neck
        p5_up = self.up1(x_p5_raw)
        p4_fuse = self.cv_p4_fuse(torch.cat([p5_up, x_p4_raw], dim=1))
        p4_up_feat = self.c2f_p4_up(p4_fuse)

        p4_up = self.up2(p4_up_feat)
        p3_fuse = self.cv_p3_fuse(torch.cat([p4_up, x_p3_raw], dim=1))
        p3_out = self.c2f_p3_up(p3_fuse)

        p3_down = self.down_p3(p3_out)
        p4_down_fuse = self.cv_p4_down(torch.cat([p3_down, p4_up_feat], dim=1))
        p4_out = self.c2f_p4_down(p4_down_fuse)

        p4_down = self.down_p4(p4_out)
        p5_down_fuse = self.cv_p5_down(torch.cat([p4_down, x_p5_raw], dim=1))
        p5_out = self.c2f_p5_down(p5_down_fuse)

        final_features = [p3_out, p4_out, p5_out]

        if self.training:
            return self.one2many_head(final_features), self.one2one_head(final_features)
        else:
            return self.one2one_head(final_features)


# =====================================================================
# 4. Decoder Module (BBox Center & Size 기준 상대 거리 복원)
# =====================================================================


class PoseDecoderModule(nn.Module):
    def __init__(self, reg_max=16, num_kpts=2, strides=[8, 16, 32]):
        super().__init__()
        self.reg_max = reg_max
        self.num_kpts = num_kpts
        self.strides_list = strides

        # DFL Softmax용 가중치 [16]
        self.proj = nn.Parameter(
            torch.arange(reg_max, dtype=torch.float32), requires_grad=False
        )

        # 5040개 Anchor 및 Stride 사전 생성 (384x640 해상도 기준)
        anchors, stride_tensor = self._generate_anchors(strides, h=384, w=640)
        self.register_buffer("anchors", anchors)  # [5040, 2] (grid_x, grid_y)
        self.register_buffer("stride_tensor", stride_tensor)  # [5040, 1]

    def _generate_anchors(self, strides, h=384, w=640):
        anchor_list = []
        stride_list = []

        for stride in strides:
            grid_h = h // stride
            grid_w = w // stride

            grid_y, grid_x = torch.meshgrid(
                torch.arange(grid_h, dtype=torch.float32),
                torch.arange(grid_w, dtype=torch.float32),
                indexing="ij",
            )
            # Center Grid (+0.5)
            grid = torch.stack([grid_x + 0.5, grid_y + 0.5], dim=-1).view(-1, 2)
            strides_grid = torch.full((grid.size(0), 1), stride, dtype=torch.float32)

            anchor_list.append(grid)
            stride_list.append(strides_grid)

        return torch.cat(anchor_list, dim=0), torch.cat(stride_list, dim=0)

    def forward(self, cls_pred, reg_pred, kpt_pred):
        """
        cls_pred: [B, 5040, 1] (Raw Logit)
        reg_pred: [B, 5040, 64] (4 * reg_max)
        kpt_pred: [B, 5040, 6]  (2 * 3: dx, dy, vis_logit)
        """
        batch_size = cls_pred.size(0)

        # 1. Class Score: Sigmoid 명시적 수행 [B, 5040, 1]
        scores = torch.sigmoid(cls_pred)

        # 2. BBox Distances DFL 계산 [B, 5040, 4]
        reg_dist = reg_pred.view(batch_size, -1, 4, self.reg_max)
        reg_dist = reg_dist.softmax(dim=-1) @ self.proj  # [B, 5040, 4]

        # Distance * Stride -> Real Pixel Distance
        lt = reg_dist[..., :2] * self.stride_tensor
        rb = reg_dist[..., 2:] * self.stride_tensor

        # Anchor Pixel Center [5040, 2]
        anchor_pixel = self.anchors * self.stride_tensor  # [5040, 2]

        # Decode BBoxes (x1, y1, x2, y2)
        x1y1 = anchor_pixel - lt
        x2y2 = anchor_pixel + rb
        decoded_bboxes = torch.cat([x1y1, x2y2], dim=-1)  # [B, 5040, 4]

        # 3. Keypoints Decode (차원 확장 적용)
        # kpt_pred: [B, 5040, 6] -> [B, 5040, 2, 3]
        kpt_reshaped = kpt_pred.view(batch_size, -1, self.num_kpts, 3)

        pred_rel_x = kpt_reshaped[..., 0:1]  # [B, 5040, 2, 1]
        pred_rel_y = kpt_reshaped[..., 1:2]  # [B, 5040, 2, 1]
        pred_vis_logit = kpt_reshaped[..., 2:3]  # [B, 5040, 2, 1]

        # BBox Center & Dimensions
        bbox_xc = (
            decoded_bboxes[..., 0:1] + decoded_bboxes[..., 2:3]
        ) * 0.5  # [B, 5040, 1]
        bbox_yc = (
            decoded_bboxes[..., 1:2] + decoded_bboxes[..., 3:4]
        ) * 0.5  # [B, 5040, 1]
        bbox_w = (decoded_bboxes[..., 2:3] - decoded_bboxes[..., 0:1]).clamp(
            min=1.0
        )  # [B, 5040, 1]
        bbox_h = (decoded_bboxes[..., 3:4] - decoded_bboxes[..., 1:2]).clamp(
            min=1.0
        )  # [B, 5040, 1]

        # 💡 [핵심 해결]: unsqueeze(-1)로 [B, 5040, 1, 1] 형태로 만들어 Broadcasting 허용
        bbox_xc_exp = bbox_xc.unsqueeze(-1)
        bbox_yc_exp = bbox_yc.unsqueeze(-1)
        bbox_w_exp = bbox_w.unsqueeze(-1)
        bbox_h_exp = bbox_h.unsqueeze(-1)

        # Absolute Keypoint XY = BBox Center + (Relative Offset * BBox Width/Height)
        abs_kpt_x = bbox_xc_exp + (pred_rel_x * bbox_w_exp)  # [B, 5040, 2, 1]
        abs_kpt_y = bbox_yc_exp + (pred_rel_y * bbox_h_exp)  # [B, 5040, 2, 1]

        abs_kpt_xy = torch.cat([abs_kpt_x, abs_kpt_y], dim=-1)  # [B, 5040, 2, 2]
        decoded_kpts = torch.cat(
            [abs_kpt_xy, pred_vis_logit], dim=-1
        )  # [B, 5040, 2, 3]

        return scores, decoded_bboxes, decoded_kpts


# =====================================================================
# 5. TensorRT / LiteRT 배포용 Combined Model
# =====================================================================


class CombinedPoseModel(nn.Module):
    def __init__(self, base_model, reg_max=16, num_kpts=2):
        super().__init__()
        self.base_model = base_model
        self.decoder = PoseDecoderModule(reg_max=reg_max, num_kpts=num_kpts)

    def forward(self, x):
        cls_pred, reg_pred, kpt_pred = self.base_model(x)
        return self.decoder(cls_pred, reg_pred, kpt_pred)


# =====================================================================
# 6. 실행 및 출력 Shape 검증
# =====================================================================

if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 384, 640)

    base_model = DualYOLOv8PoseModel(num_classes=1, reg_max=16, num_kpts=2)
    export_model = CombinedPoseModel(base_model).eval()

    with torch.no_grad():
        scores, bboxes, keypoints = export_model(dummy_input)

    print("⚡ [검증 완료] Combined Model 출력 확인:")
    print(" - Scores Shape   :", scores.shape)  # [1, 5040, 1]
    print(" - BBoxes Shape   :", bboxes.shape)  # [1, 5040, 4]
    print(
        " - Keypoints Shape:", keypoints.shape
    )  # [1, 5040, 2, 3] -> (Knot, Body) x (x, y, vis)
