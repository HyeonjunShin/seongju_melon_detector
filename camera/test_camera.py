import os
import sys
import time
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 기존 작성해두신 gemini336.py 원본을 그대로 가져옵니다.
from gemini336 import Gemini336

COLOR_SHAPE = (1280, 720, 3)
DEPTH_SHAPE = (1280, 720, 1)
PRESET_PATH = "./gemini336_settings.json"


def test_visualizer():
    print("🎥 Orbbec Gemini336 카메라 시각화 테스트를 시작합니다...")
    
    try:
        # 기존 Gemini336 클래스 생성자 호출
        camera = Gemini336(
            color_shape=COLOR_SHAPE,
            depth_shape=DEPTH_SHAPE,
            settings_path=PRESET_PATH
        )
    except Exception as e:
        print(f"❌ 카메라 객체 생성 중 에러 발생: {e}")
        return

    print("⏳ 카메라 스트리밍 시작 중...")
    time.sleep(1.0)

    # 카메라 연결 정상 여부 확인
    if camera.pipeline is None:
        print("❌ 카메라 파이프라인 연결에 실패했습니다.")
        print("💡 [해상도/포맷 체크] 카메라가 지원하는 해상도인지, 또는 USB 3.0 포트에 연결되어 있는지 확인해 주세요.")
        return

    window_name = "Orbbec Gemini336 - Live Preview (Color & Depth)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 360)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" ▶ 'Q' 또는 'ESC'를 누르면 시각화가 종료됩니다.")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # 기존 gemini336.py의 get_frame() 인자 없이 호출
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            color_frame = frame.get_color_frame()
            depth_frame = frame.get_depth_frame()

            if color_frame is None or depth_frame is None:
                continue

            color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)

            # (1280, 720) 해상도를 H, W 순서인 (720, 1280)으로 reshaping
            color_img = color_data.reshape((720, 1280, 3))
            depth_img = depth_data.reshape((720, 1280))

            # 1. Color (RGB -> BGR)
            color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

            # 2. Depth 시각화
            depth_filtered = np.where((depth_img > 0) & (depth_img < 10000), depth_img, 0)
            depth_norm = cv2.normalize(depth_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            depth_colored[depth_filtered == 0] = [0, 0, 0]

            # FPS 표시
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            sn_str = "Unknown"
            if camera.device:
                try:
                    sn_str = camera.device.get_device_info().get_serial_number()
                except Exception:
                    pass

            cv2.putText(color_bgr, f"COLOR (1280x720) | FPS: {fps:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(depth_colored, f"DEPTH (1280x720) | SN: {sn_str}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            combined_preview = np.hstack((color_bgr, depth_colored))
            cv2.imshow(window_name, combined_preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[알림] 시각화를 종료합니다.")
                break

    except KeyboardInterrupt:
        print("\n[알림] Ctrl + C 입력으로 중단되었습니다.")
    
    finally:
        print("🛑 카메라 파이프라인 및 창 해제 중...")
        cv2.destroyAllWindows()
        if hasattr(camera, 'pipeline') and camera.pipeline:
            try:
                camera.pipeline.stop()
            except Exception:
                pass
        print("✅ 정상적으로 종료되었습니다.")


if __name__ == "__main__":
    test_visualizer()