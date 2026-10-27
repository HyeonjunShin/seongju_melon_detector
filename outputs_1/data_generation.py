import os
import json
import cv2
import numpy as np
import albumentations as A
import random

# ==========================================
# 1. 경로 설정 및 초기화
# ==========================================
COLOR_IMAGES_PATH = "color_images"   # 원본 JSON 파일이 있는 폴더
NUKKI_IMAGES_PATH = "nukki_images"   # 누끼 딴 PNG 파일이 있는 폴더

OUTPUT_IMG_PATH = "augmented_dataset/images"
OUTPUT_LBL_PATH = "augmented_dataset/labels"

os.makedirs(OUTPUT_IMG_PATH, exist_ok=True)
os.makedirs(OUTPUT_LBL_PATH, exist_ok=True)

nukki_files = [f for f in os.listdir(NUKKI_IMAGES_PATH) if f.endswith('_nukki.png')]

# 이미지 1장당 생성할 증강 데이터 수
AUG_PER_IMAGE = 150 

print(f"📂 총 {len(nukki_files)}개의 누끼 소스로부터 데이터셋 양산을 시작합니다.")

# ==========================================
# 2. 최신 버전에 호환되는 변환 파이프라인 정의
# ==========================================
transform_pipeline = A.Compose([
    # 이미지 회전 및 크기 미세 변환 (최신 스펙에 맞게 cval -> fill 변경)
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=(-0.15, 0.15),
        rotate_limit=360,
        border_mode=cv2.BORDER_CONSTANT,
        fill=(0, 0, 0, 0), 
        p=1.0
    ),
    # 조명 밝기 및 대비 무작위 변경
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
    # 노란색의 채도와 명도를 변경하여 조명 반사에 대응
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=35, val_shift_limit=35, p=0.8),
    # 비닐 표면의 전등 빛 반사 노이즈 모사 (최신 스펙에 맞게 fill_value -> fill 변경)
    A.CoarseDropout(
        num_holes_range=(1, 3), 
        hole_height_range=(0.02, 0.06), 
        hole_width_range=(0.02, 0.06), 
        fill=255, 
        p=0.4
    ),
], 
# [💡 핵심 교정 1] album 대신 호환성이 검증된 pascal_voc(픽셀 단위 좌표) 포맷 지정
bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
keypoint_params=A.KeypointParams(format='xy')
)

# ==========================================
# 3. 데이터 양산 루프 실행
# ==========================================
global_idx = 0

for nukki_file in nukki_files:
    base_name = nukki_file.replace('_nukki.png', '')
    json_file_path = os.path.join(COLOR_IMAGES_PATH, f"{base_name}.json")
    nukki_file_path = os.path.join(NUKKI_IMAGES_PATH, nukki_file)
    
    if not os.path.exists(json_file_path):
        print(f"⚠️ {json_file_path} 라벨 파일이 없어 건너뜁니다.")
        continue
        
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    if isinstance(json_data, list):
        if len(json_data) == 0:
            continue
        obj = json_data[0] # 리스트 내부 객체 안정적 파싱
    else:
        obj = json_data
        
    bbox = obj["bbox"]
    
    # 누끼 이미지 로드
    nukki_img = cv2.imread(nukki_file_path, cv2.IMREAD_UNCHANGED)
    img_h, img_w = nukki_img.shape[:2]
    
    # 원본 BBox 픽셀 단위 좌표 추출
    xmin_val = bbox["xmin"]
    ymin_val = bbox["ymin"]
    xmax_val = bbox["xmax"]
    ymax_val = bbox["ymax"]
    
    # 만약 기존 데이터가 정규화(0~1)되어 저장되어 있다면 픽셀 단위 크기로 복원
    if xmin_val <= 1.0 and xmax_val <= 1.0:
        xmin_val *= img_w; xmax_val *= img_w
        ymin_val *= img_h; ymax_val *= img_h

    # [💡 핵심 교정 2] pascal_voc 규격에 맞는 정수형 픽셀 단위의 BBox 지정
    # 간혹 픽셀이 소수점이거나 경계선에 걸치지 않도록 방어 코드 추가
    xmin = max(0, min(float(xmin_val), img_w - 1))
    ymin = max(0, min(float(ymin_val), img_h - 1))
    xmax = max(xmin + 1, min(float(xmax_val), img_w))
    ymax = max(ymin + 1, min(float(ymax_val), img_h))
    
    pascal_bbox = [xmin, ymin, xmax, ymax]
    
    # 키포인트 좌표 파싱
    src_kps = []
    for kp in obj["keypoints"]:
        kp_x = kp["point"][0]
        kp_y = kp["point"][1]
        if kp_x <= 1.0 and kp_y <= 1.0:
            kp_x *= img_w
            kp_y *= img_h
        src_kps.append([kp_x, kp_y])

    # 이미지 하나당 데이터 생성 시작
    for _ in range(AUG_PER_IMAGE):
        transformed = transform_pipeline(
            image=nukki_img,
            bboxes=[pascal_bbox],
            category_ids=[0],
            keypoints=src_kps
        )
        
        # 기하학 변환 과정에서 화면 밖으로 완전히 이탈한 물체 필터링
        if len(transformed['bboxes']) == 0 or len(transformed['keypoints']) < 2:
            continue
            
        aug_nukki = transformed['image']
        n_xmin, n_ymin, n_xmax, n_ymax = transformed['bboxes'][0]
        aug_kps = transformed['keypoints']
        
        # 4. 현장 맞춤형 벨트 배경 실시간 생성 (초록색 50%, 검은색/어두운 회색 50%)
        if random.random() < 0.5:
            # 🟢 다양한 초록색 벨트
            random_color = (
                random.randint(10, 40),   # Blue
                random.randint(60, 130),  # Green
                random.randint(10, 50)    # Red
            )
        else:
            # ⚫ 검은색 ~ 어두운 회색 벨트
            gray_val = random.randint(20, 65)
            random_color = (gray_val, gray_val, gray_val)
            
        bg_img = np.full((img_h, img_w, 3), random_color, dtype=np.uint8)
        
        # 표면 질감용 미세 가우시안 노이즈 주입
        noise = np.random.normal(0, 2, bg_img.shape).astype(np.uint8)
        bg_img = cv2.add(bg_img, noise)
            
        # 5. 알파 채널을 활용한 누끼 물체 Copy-Paste 합성
        alpha_channel = aug_nukki[:, :, 3] / 255.0
        alpha_mask = np.dstack([alpha_channel, alpha_channel, alpha_channel])
        fg_rgb = aug_nukki[:, :, :3]
        
        final_img = (fg_rgb * alpha_mask + bg_img * (1.0 - alpha_mask)).astype(np.uint8)
        
        # 6. 최종 YOLO Pose 표준 규격 (.txt)으로 정규화 변환 연산
        box_w = n_xmax - n_xmin
        box_h = n_ymax - n_ymin
        cx = n_xmin + (box_w / 2.0)
        cy = n_ymin + (box_h / 2.0)
        
        yolo_cx = cx / img_w
        yolo_cy = cy / img_h
        yolo_bw = box_w / img_w
        yolo_bh = box_h / img_h
        
        kp_strings = []
        for kp in aug_kps:
            kx = kp[0] / img_w
            ky = kp[1] / img_h
            kp_strings.append(f"{kx:.6f} {ky:.6f} 2")
            
        yolo_line = f"0 {yolo_cx:.6f} {yolo_cy:.6f} {yolo_bw:.6f} {yolo_bh:.6f} " + " ".join(kp_strings)
        
        # 7. 변환된 파일 영구 저장
        file_name = f"aug_ready_{global_idx:05d}"
        cv2.imwrite(os.path.join(OUTPUT_IMG_PATH, f"{file_name}.jpg"), final_img)
        with open(os.path.join(OUTPUT_LBL_PATH, f"{file_name}.txt"), 'w') as f_lbl:
            f_lbl.write(yolo_line)
            
        global_idx += 1

print(f"\n🎉 에러 복구 완료! 총 {global_idx}개의 합성 이미지 및 YOLO Pose 라벨이 정상 양산되었습니다.")
