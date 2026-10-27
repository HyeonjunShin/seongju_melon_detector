import os
import json
import cv2
import numpy as np

# ==========================================
# 1. 경로 설정 및 초기화
# ==========================================
FOLDER_PATH = "color_images"  
OUTPUT_PATH = "nukki_images"
os.makedirs(OUTPUT_PATH, exist_ok=True)

json_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.json')]

if not json_files:
    print(f"❌ '{FOLDER_PATH}' 폴더 안에 .json 파일이 없습니다.")
    exit()

print(f"📂 총 {len(json_files)}개의 라벨링 데이터를 처리합니다.")

# ==========================================
# 2. 루프를 돌며 이미지별 자동 누끼 작업 실행
# ==========================================
for json_file in json_files:
    base_name = os.path.splitext(json_file)[0]
    json_p = os.path.join(FOLDER_PATH, json_file)
    
    img_p = os.path.join(FOLDER_PATH, f"{base_name}.png")
    if not os.path.exists(img_p):
        img_p = os.path.join(FOLDER_PATH, f"{base_name}.jpg")
        if not os.path.exists(img_p):
            print(f"⚠️ {json_file}에 대응하는 이미지 파일이 없어 건너뜁니다.")
            continue
            
    with open(json_p, 'r', encoding='utf-8') as f:
        try:
            json_data = json.load(f)
        except Exception as e:
            print(f"❌ {json_file} 읽기 실패: {e}")
            continue

    img = cv2.imread(img_p)
    if img is None:
        print(f"❌ {img_p} 이미지를 불러올 수 없습니다.")
        continue
        
    # 이미지의 실제 픽셀 크기 실시간 획득 (H: 세로, W: 가로)
    img_h, img_w = img.shape[:2]
    
    gc_mask = np.zeros((img_h, img_w), np.uint8)
    
    if not json_data:
        continue
    
    # JSON 구조가 리스트인 경우 첫 번째 객체 선택, 딕셔너리인 경우 그대로 사용
    obj = json_data[0] if isinstance(json_data, list) else json_data
    bbox = obj["bbox"]
    
    # [💡 핵심 해결책 1] 좌표가 0~1 사이의 정규화된 값인지 판별하여 역산
    xmin_val = bbox["xmin"]
    ymin_val = bbox["ymin"]
    xmax_val = bbox["xmax"]
    ymax_val = bbox["ymax"]
    
    if xmin_val <= 1.0 and xmax_val <= 1.0 and ymin_val <= 1.0 and ymax_val <= 1.0:
        # 0~1 사이 값이라면 실제 이미지 해상도(픽셀)를 곱해줌
        xmin_val *= img_w
        xmax_val *= img_w
        ymin_val *= img_h
        ymax_val *= img_h
    
    # [💡 핵심 해결책 2] 이미지의 실제 픽셀 범위 내부로 강제 제한 (Clipping)
    xmin = max(0, min(int(round(xmin_val)), img_w - 1))
    ymin = max(0, min(int(round(ymin_val)), img_h - 1))
    xmax = max(0, min(int(round(xmax_val)), img_w))
    ymax = max(0, min(int(round(ymax_val)), img_h))
    
    width = xmax - xmin
    height = ymax - ymin
    
    # [💡 핵심 해결책 3] 사각형의 크기가 최소 1픽셀 이상인지 검증
    if width <= 0 or height <= 0:
        print(f"⚠️ {base_name} BBox 크기가 잘못되었습니다 (width: {width}, height: {height}). 건너뜁니다.")
        continue
        
    rect = (xmin, ymin, width, height)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # GrabCut 알고리즘 실행
    try:
        cv2.grabCut(img, gc_mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except Exception as e:
        print(f"❌ {base_name} GrabCut 실패. 에러메시지: {e}")
        print(f"   시도한 BBox 픽셀 좌표: rect={rect}, 이미지 크기: (W={img_w}, H={img_h})")
        continue

    mask2 = np.where((gc_mask == 2) | (gc_mask == 0), 0, 255).astype('uint8')
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

    b, g, r = cv2.split(img)
    rgba_img = cv2.merge([b, g, r, mask2], 4)
    
    output_file_path = os.path.join(OUTPUT_PATH, f"{base_name}_nukki.png")
    cv2.imwrite(output_file_path, rgba_img)
    print(f"✅ 성공: {output_file_path} 저장 완료")

print("\n🎉 에러 방지 코드가 적용된 자동 누끼 처리가 완료되었습니다!")
