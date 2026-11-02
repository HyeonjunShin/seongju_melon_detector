import threading
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import cv2

from pyorbbecsdk import (
    Pipeline,
    Config,
    Context,
    OBSensorType,
    OBFormat,
    OBAlignMode,
    OBFrameAggregateOutputMode,
    OBPropertyID,
    FrameSet,
)

CHECK_PARAMS = [
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


@dataclass
class Gemini:
    # 기존 인터페이스 유지: (width, height, channels)
    color_shape: tuple[int, int, int]
    depth_shape: tuple[int, int, int]
    config_path: str

    fps: int = 60

    ctx: Context = field(init=False)
    pipeline: Pipeline = field(init=False)
    config: Config = field(init=False)
    device: object = field(init=False)

    depth_scale: float = field(init=False, default=0.1)

    latest_frame: tuple | None = field(init=False, default=None)

    frame_condition: threading.Condition = field(
        init=False, default_factory=threading.Condition
    )

    stop_event: threading.Event = field(init=False, default_factory=threading.Event)

    _started: bool = field(init=False, default=False)

    # 진단용
    received_count: int = field(init=False, default=0)
    missing_count: int = field(init=False, default=0)
    prev_callback_timestamp_color: int | None = field(init=False, default=None)
    prev_callback_timestamp_depth: int | None = field(init=False, default=None)

    def __post_init__(self):
        self.ctx = Context()
        self.pipeline = Pipeline()
        self.config = Config()
        self.device = self.pipeline.get_device()

        # 프리셋 로드
        if Path(self.config_path).exists():
            try:
                self.device.load_preset_from_json_file(self.config_path)
                print(f"[Success] Preset loaded: {self.config_path}")
            except Exception as e:
                print(f"[Warning] Preset load failed: {e}")
        else:
            print(f"[Warning] Preset not found: {self.config_path}")

        cw, ch, _ = self.color_shape
        dw, dh, _ = self.depth_shape

        # Color 프로파일
        color_profiles = self.pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )

        color_profile = color_profiles.get_video_stream_profile(
            cw, ch, OBFormat.MJPG, self.fps
        )

        # Depth 프로파일
        depth_profiles = self.pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        )

        depth_profile = depth_profiles.get_video_stream_profile(
            dw, dh, OBFormat.Y16, self.fps
        )

        self.config.enable_stream(color_profile)
        self.config.enable_stream(depth_profile)

        # 기존 HW Alignment 유지
        # 선택한 프로파일이 HW D2C를 지원해야 함
        self.config.set_align_mode(OBAlignMode.HW_MODE)

        # 완전한 RGB-D FrameSet만 전달
        self.config.set_frame_aggregate_output_mode(
            OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
        )

        # 프레임 동기화
        self.pipeline.enable_frame_sync()

    def __enter__(self):
        self.stop_event.clear()

        self.pipeline.start(self.config, self.on_new_frame)
        self._started = True

        try:
            self.load_camera_parameters()

            self.show_informations()
            self.show_intrinsics()
            self.check_parameters()

        except Exception:
            self.stop()
            raise

        print("[Gemini] Async callback pipeline started.")
        return self

    def load_camera_parameters(self):
        param = self.pipeline.get_camera_param()

        color_intr = param.rgb_intrinsic
        depth_intr = param.depth_intrinsic

        self.fx_color = color_intr.fx
        self.fy_color = color_intr.fy
        self.cx_color = color_intr.cx
        self.cy_color = color_intr.cy

        self.fx_depth = depth_intr.fx
        self.fy_depth = depth_intr.fy
        self.cx_depth = depth_intr.cx
        self.cy_depth = depth_intr.cy

        color_dist = param.rgb_distortion
        depth_dist = param.depth_distortion

        for name in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"):
            setattr(self, f"{name}_color", getattr(color_dist, name))
            setattr(self, f"{name}_depth", getattr(depth_dist, name))

    def get_color_calibration(self):
        """Return color intrinsics for HW D2C frames in OpenCV coefficient order."""
        matrix = np.array(
            [
                [self.fx_color, 0, self.cx_color],
                [0, self.fy_color, self.cy_color],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        distortion = np.array(
            [
                getattr(self, f"{name}_color")
                for name in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
            ],
            dtype=np.float64,
        )
        return matrix, distortion

    def show_informations(self):
        print("Current depth work mode:", self.device.get_depth_work_mode())

    def show_intrinsics(self):
        print("\n" + "=" * 50)

        for sensor in ("color", "depth"):
            print(f"[{sensor.upper()} Camera Intrinsics]")
            print(f"  fx: {getattr(self, f'fx_{sensor}'):.4f}")
            print(f"  fy: {getattr(self, f'fy_{sensor}'):.4f}")
            print(f"  cx: {getattr(self, f'cx_{sensor}'):.4f}")
            print(f"  cy: {getattr(self, f'cy_{sensor}'):.4f}")

            distortion = [
                getattr(self, f"{name}_{sensor}")
                for name in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")
            ]
            print("  Distortion:", distortion)

        print("=" * 50 + "\n")

    def check_parameters(self):
        for label, prop_id, prop_type in CHECK_PARAMS:
            try:
                if prop_type == "bool":
                    value = self.device.get_bool_property(prop_id)
                elif prop_type == "float":
                    value = self.device.get_float_property(prop_id)
                else:
                    value = self.device.get_int_property(prop_id)

                print(f" ▶ {label:<30}: {value}")

            except Exception:
                print(f" ▶ {label:<30}: [Unavailable]")

    # --------------------------------------------------
    # SDK Callback
    # --------------------------------------------------

    def on_new_frame(self, frame_set: FrameSet):
        if self.stop_event.is_set() or frame_set is None:
            return

        depth_frame = frame_set.get_depth_frame()
        color_frame = frame_set.get_color_frame()

        if depth_frame is None or color_frame is None:
            self.missing_count += 1
            return

        # Color RGB8
        # color = (
        #     np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        #     .reshape(color_frame.get_height(), color_frame.get_width(), 3)
        #     .copy()
        # )
        color = cv2.imdecode(color_frame.get_data(), cv2.IMREAD_COLOR_RGB)

        raw_depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            depth_frame.get_height(), depth_frame.get_width()
        )
        depth = raw_depth.astype(np.float32)

        self.depth_scale = depth_frame.get_depth_scale()
        depth *= self.depth_scale * 0.001
        # Match data_collection.py: saturated/unknown Y16 samples are invalid.
        # depth[raw_depth == np.iinfo(np.uint16).max] = 0

        timestamp_color = color_frame.get_system_timestamp_us()
        timestamp_depth = depth_frame.get_system_timestamp_us()

        self.received_count += 1

        # 프레임 간격 진단
        if self.prev_callback_timestamp_color is not None:
            gap_ms = (timestamp_color - self.prev_callback_timestamp_color) / 1000.0
            if gap_ms > 25:
                print(f"[COLOR CALLBACK GAP] {gap_ms:.3f} ms")

        self.prev_callback_timestamp_color = timestamp_color

        if self.prev_callback_timestamp_depth is not None:
            gap_ms = (timestamp_depth - self.prev_callback_timestamp_depth) / 1000.0
            if gap_ms > 25:
                print(f"[DEPTH CALLBACK GAP] {gap_ms:.3f} ms")

        self.prev_callback_timestamp_depth = timestamp_depth

        # Color / Depth / Timestamp를 한 번에 갱신
        with self.frame_condition:
            self.latest_frame = (color, depth, timestamp_color, timestamp_depth)
            self.frame_condition.notify_all()

    # --------------------------------------------------
    # 최신 프레임 조회
    # --------------------------------------------------

    def wait_for_new_frame(
        self,
        previous_timestamp=None,
        timeout=0.2,
    ):
        """
        새로운 프레임이 들어올 때까지 대기.
        timeout이면 None 반환.

        가장 최신 프레임만 제공하며
        중간 프레임은 건너뛸 수 있음.
        """

        with self.frame_condition:
            is_ready = self.frame_condition.wait_for(
                lambda: (
                    self.stop_event.is_set()
                    or (
                        self.latest_frame is not None
                        and self.latest_frame[2] != previous_timestamp
                    )
                ),
                timeout=timeout,
            )

            if not is_ready or self.stop_event.is_set():
                return None

            return self.latest_frame

    def stop(self):
        if self.stop_event.is_set() and not self._started:
            return

        self.stop_event.set()

        with self.frame_condition:
            self.frame_condition.notify_all()

        if self._started:
            try:
                self.pipeline.stop()
            finally:
                self._started = False

        print(
            f"[Gemini] Stopped. "
            f"Received={self.received_count}, "
            f"Missing={self.missing_count}"
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
