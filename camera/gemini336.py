from pathlib import Path

from pyorbbecsdk import (
    Pipeline,
    Config,
    Context,
    OBSensorType,
    OBFormat,
    OBPropertyID,
)
import numpy as np

# 카메라 파라미터
fx = 693.3102
fy = 693.4061
cx = 639.6599
cy = 365.0724

K = np.array(
    [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ],
    dtype=np.float64,
)

D = np.array(
    [
        0.00743896747007966,
        -0.05456198751926422,
        0.03670734167098999,
        0.0,
        0.0,
        0.0,
        0.00016195396892726421,
        -0.001005938509479165,
    ],
    dtype=np.float32,
)
color_shape = (1280, 720, 3)
depth_shape = (1280, 720, 1)


check_params = [
    ("color_auto_exposure", OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, "bool"),
    (
        "color_auto_white_balance",
        OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL,
        "bool",
    ),
    (
        "color_backlight_compensation",
        OBPropertyID.OB_PROP_COLOR_BACKLIGHT_COMPENSATION_INT,
        "int",
    ),
    ("color_brightness", OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT, "int"),
    ("color_contrast", OBPropertyID.OB_PROP_COLOR_CONTRAST_INT, "int"),
    ("color_exposure_time", OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, "int"),
    ("color_gain", OBPropertyID.OB_PROP_COLOR_GAIN_INT, "int"),
    ("color_gamma", OBPropertyID.OB_PROP_COLOR_GAMMA_INT, "int"),
    ("color_hue", OBPropertyID.OB_PROP_COLOR_HUE_INT, "int"),
    (
        "color_power_line_frequency",
        OBPropertyID.OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT,
        "int",
    ),
    ("color_saturation", OBPropertyID.OB_PROP_COLOR_SATURATION_INT, "int"),
    ("color_sharpness", OBPropertyID.OB_PROP_COLOR_SHARPNESS_INT, "int"),
    ("color_white_balance", OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT, "int"),
    ("depth_auto_exposure", OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, "bool"),
    ("depth_exposure_time", OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT, "int"),
    ("depth_gain", OBPropertyID.OB_PROP_DEPTH_GAIN_INT, "int"),
    ("laser_state", OBPropertyID.OB_PROP_LASER_CONTROL_INT, "int"),
    ("laser_power_level", OBPropertyID.OB_PROP_LASER_POWER_LEVEL_CONTROL_INT, "int"),
    ("HW_align", OBPropertyID.OB_PROP_DEPTH_ALIGN_HARDWARE_BOOL, "bool"),
]


def print_intrinsics(pipeline):
    if not pipeline:
        return
    try:
        param = pipeline.get_camera_param()

        c_intrin = param.rgb_intrinsic
        print("\n" + "=" * 50)
        print(" [Color Camera Intrinsic Parameters]")
        print(f"  ▶ Resolution : {c_intrin.width} x {c_intrin.height}")
        print(f"  ▶ Focal Length (fx, fy) : ({c_intrin.fx:.4f}, {c_intrin.fy:.4f})")
        print(f"  ▶ Principal Point (cx, cy) : ({c_intrin.cx:.4f}, {c_intrin.cy:.4f})")
        c_dist = param.rgb_distortion
        print(
            f"  ▶ Distortion Model : {c_dist.k1} {c_dist.k2} {c_dist.k3} {c_dist.k4} {c_dist.k5} {c_dist.k6} {c_dist.p1} {c_dist.p2}"
        )

        d_intrin = param.depth_intrinsic
        print("-" * 50)
        print(" [Depth Camera Intrinsic Parameters]")
        print(f"  ▶ Resolution : {d_intrin.width} x {d_intrin.height}")
        print(f"  ▶ Focal Length (fx, fy) : ({d_intrin.fx:.4f}, {d_intrin.fy:.4f})")
        print(f"  ▶ Principal Point (cx, cy) : ({d_intrin.cx:.4f}, {d_intrin.cy:.4f})")
        d_dist = param.depth_distortion
        print(
            f"  ▶ Distortion Model : {d_dist.k1} {d_dist.k2} {d_dist.k3} {d_dist.k4} {d_dist.k5} {d_dist.k6} {d_dist.p1} {d_dist.p2}"
        )
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"내적 파라미터(Intrinsic) 조회 실패: {e}")


def check_parameters(device):
    for label, prop_id, prop_type in check_params:
        try:
            if prop_type == "bool":
                val = device.get_bool_property(prop_id)
            elif prop_type == "float":
                val = device.get_float_property(prop_id)
            else:
                val = device.get_int_property(prop_id)
            print(f" ▶ {label:<30} : {val}")
        except Exception:
            print(f"{label:<30} : [조회 실패 / 지원하지 않는 프로퍼티]")


class Gemini336:
    def __init__(self, color_shape, depth_shape, settings_path):
        self.color_shape = color_shape
        self.depth_shape = depth_shape
        self.settings_path = str(settings_path)

        self.device = None
        self.pipeline = None
        self.is_running = False

        self.ctx = Context()
        self.ctx.set_device_changed_callback(self._on_device_changed)
        self.found_preconn_camera()

    def _on_device_changed(self, removed_list, added_list):
        if removed_list.get_count() > 0:
            print("Deleted the camera")
            serial_number = removed_list.get_device_serial_number_by_index(0)
            self.stop_camera()
            print(
                f"  - {removed_list.get_device_name_by_index(0)} (SN: {serial_number})"
            )

        if added_list.get_count() > 0:
            print("Added the camera:")
            device = added_list.get_device_by_index(0)
            info = device.get_device_info()
            serial_number = info.get_serial_number()
            device_name = info.get_name()

            print(f"{device_name} (SN: {serial_number})")
            try:
                self.pipeline = self.start_camera(device)
                self.device = device
                self.is_running = True
                print(f"[Success connection] {serial_number}")
            except Exception as e:
                print(f"[Error the camera] {serial_number}: {e}")

    def found_preconn_camera(self):
        device_list = self.ctx.query_devices()
        if device_list.get_count() > 0:
            device = device_list.get_device_by_index(0)
            info = device.get_device_info()
            serial_number = info.get_serial_number()

            try:
                self.pipeline = self.start_camera(device)
                self.device = device
                self.is_running = True
                print(f"[Success connection] {serial_number}")
            except Exception as e:
                print(f"[Error the camera] {serial_number}: {e}")

    def start_camera(self, device):
        sn = device.get_device_info().get_serial_number()

        # JSON 프리셋 로드
        if Path(self.settings_path).exists():
            try:
                device.load_preset_from_json_file(self.settings_path)
                print(f"✅ 프리셋 로드 성공: {self.settings_path}")
            except Exception as e:
                print(f"프리셋 로드 실패 ({sn}): {e}")
        else:
            print(f"⚠️ 프리셋 파일 경로 없음: {self.settings_path}")

        pipeline = Pipeline(device)
        config = Config()

        # Color 스트림 설정
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = color_profiles.get_video_stream_profile(
                self.color_shape[0], self.color_shape[1], OBFormat.RGB, 30
            )
            config.enable_stream(color_profile)
        except Exception as e:
            print(f"Color 프로필 설정 실패: {e}")

        # Depth 스트림 설정
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            depth_profile = depth_profiles.get_video_stream_profile(
                self.depth_shape[0], self.depth_shape[1], OBFormat.Y16, 30
            )
            config.enable_stream(depth_profile)
        except Exception as e:
            print(f"Depth 프로필 설정 실패: {e}")

        pipeline.start(config)

        print_intrinsics(pipeline)
        check_parameters(device)

        return pipeline

    def stop_camera(self):
        """안전하게 파이프라인을 정지하고 리소스를 해제합니다."""
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"Pipeline stop 중 예외 발생: {e}")
            self.pipeline = None
        self.device = None
        self.is_running = False

    def get_frame(self, timeout_ms=100):
        if self.pipeline is not None and self.is_running:
            try:
                frame = self.pipeline.wait_for_frames(timeout_ms)
                return frame
            except Exception:
                return None
        return None

    def get_state(self):
        return {
            "is_running": self.is_running,
            "has_device": self.device is not None,
            "has_pipeline": self.pipeline is not None,
        }

    def __del__(self):
        self.stop_camera()
