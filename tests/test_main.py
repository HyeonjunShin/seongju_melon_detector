import io
import signal
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import numpy as np

import main
from common.ipc import ProductIpc


class FakeCameraShm:
    shm_name = "camera_shm"
    index = 0
    status = True
    sequence_id = 10
    latency = 4.25

    def read(self):
        return SimpleNamespace(
            timestamp=101,
            color=np.zeros((720, 1280, 3), dtype=np.uint8),
            depth=np.zeros((720, 1280, 1), dtype=np.uint16),
        )


class FakeFlangeShm:
    shm_name = "flange_shm"
    head = 20
    buffer_size = 100
    status = True
    sequence = 20

    def get_ordered_data(self):
        transform = np.eye(4)
        transform[:3, 3] = [1.1, 2.2, 3.3]
        return [
            {"timestamp_us": 202 + i * 2000, "TF": transform}
            for i in range(min(self.sequence, self.buffer_size))
        ], self.sequence


class FakeTargetShm:
    shm_name = "target_shm"
    status = True
    index = 0
    timestamp = 303

    def read(self):
        transform = np.eye(4)
        transform[:3, 3] = [4.4, 5.5, 6.6]
        return SimpleNamespace(
            timestamp=self.timestamp,
            detected=True,
            score=0.875,
            bbox=np.array([10.0, 20.0, 30.0, 40.0]),
            TF=transform,
        )


class FakeIpc:
    def __init__(self):
        self.camera_shm = FakeCameraShm()
        self.flange_shm = FakeFlangeShm()
        self.target_shm = FakeTargetShm()


class MainSupervisorTest(unittest.TestCase):
    def test_print_dashboard_always_clears_previous_frame(self):
        output = io.StringIO()

        with patch.object(main.sys, "stdout", output):
            main.print_dashboard("new frame")

        self.assertEqual(output.getvalue(), "\033[2J\033[Hnew frame\n")

    def test_dashboard_reports_rates_and_payload_summaries(self):
        ipc = FakeIpc()
        _, state = main.build_dashboard(ipc, previous=None, now=10.0)
        ipc.camera_shm.sequence_id += 30
        ipc.flange_shm.sequence += 100
        ipc.target_shm.index = 1
        ipc.target_shm.timestamp = 404

        dashboard, _ = main.build_dashboard(ipc, previous=state, now=11.0)

        self.assertRegex(dashboard, r"camera_shm\s+ACTIVE\s+30\.0 Hz\s+0\.0 s")
        self.assertIn("RGB    1280 × 720 · uint8 · 3 ch", dashboard)
        self.assertIn("Depth  1280 × 720 · uint16 · 1 ch", dashboard)
        self.assertRegex(dashboard, r"flange_shm\s+ACTIVE\s+500\.0 Hz\s+0\.0 s")
        self.assertIn("위치   X 1.100000  Y 2.200000  Z 3.300000 m", dashboard)
        self.assertRegex(dashboard, r"target_shm\s+ACTIVE\s+1\.0 Hz\s+0\.0 s")
        self.assertIn("검출   YES · 신뢰도 0.875", dashboard)
        self.assertIn("위치   X 4.400000  Y 5.500000  Z 6.600000 m", dashboard)

    def test_flange_timing_uses_buffer_timestamps(self):
        ipc = FakeIpc()
        for timestamps, expected in [
            ([1000000, 1001900, 1004000], "평균=2.000 ms  최소=1.900 ms  최대=2.100 ms"),
            ([1000000], "주기: 측정 불가"),
            ([1000000, 1000000], "주기: 측정 불가"),
            ([1000000, 999000], "주기: 측정 불가"),
        ]:
            with self.subTest(timestamps=timestamps):
                data = [{"timestamp_us": ts, "TF": np.eye(4)} for ts in timestamps]
                with patch.object(ipc.flange_shm, "get_ordered_data", return_value=(data, 100)):
                    dashboard, _ = main.build_dashboard(ipc, previous=None, now=10.)
                self.assertIn(expected, dashboard)
                if len(timestamps) == 3:
                    self.assertRegex(dashboard, r"flange_shm\s+ACTIVE\s+500\.0 Hz")

    def test_dashboard_reports_time_since_last_change(self):
        ipc = FakeIpc()
        _, state = main.build_dashboard(ipc, previous=None, now=10.0)

        dashboard, _ = main.build_dashboard(ipc, previous=state, now=12.5)

        self.assertEqual(dashboard.count("2.5 s"), 3)
        self.assertRegex(dashboard, r"camera_shm\s+ACTIVE\s+0\.0 Hz\s+2\.5 s")
        self.assertRegex(dashboard, r"target_shm\s+ACTIVE\s+0\.0 Hz\s+2\.5 s")

    def test_dashboard_displays_complete_transforms_and_camera_buffer_details(self):
        ipc = FakeIpc()
        transform = np.array([[0., 0., 1., .123456789],
                              [0., 1., 0., -.234567891],
                              [-1., 0., 0., .345678912],
                              [0., 0., 0., 1.]])
        sample = ipc.target_shm.read()
        sample.TF = transform
        frame = ipc.camera_shm.read()
        frame.depth[0, :3, 0] = [100, 200, 300]
        with patch.object(ipc.target_shm, "read", return_value=sample), patch.object(
            ipc.camera_shm, "read", return_value=frame
        ):
            dashboard, _ = main.build_dashboard(ipc, previous=None, now=10.)
        target_section = dashboard.split("\nTARGET\n")[1]
        rows = [line.strip().strip("[]").split() for line in target_section.splitlines()
                if line.strip().startswith("[")]
        np.testing.assert_allclose(np.array(rows, dtype=float), transform, atol=1e-9)
        self.assertIn("RPY (deg): roll=0.000  pitch=90.000  yaw=0.000", target_section)
        self.assertIn("+Z (base): 1.000000  0.000000  0.000000", target_section)
        self.assertIn("XYZ (mm): 123.457  -234.568  345.679", target_section)
        self.assertIn("valid=3/921600", dashboard)
        self.assertIn("min=100  median=200  max=300 mm", dashboard)
        self.assertIn("head=20", dashboard)
        self.assertIn("buffer=20/100", dashboard)
        self.assertIn("timestamp=101  seq=10  index=0", dashboard)

    def test_dashboard_handles_empty_depth_and_missing_flange_pose(self):
        ipc = FakeIpc()
        with patch.object(ipc.flange_shm, "get_ordered_data", return_value=(None, 0)):
            dashboard, _ = main.build_dashboard(ipc, previous=None, now=10.)
        flange_section = dashboard.split("\nFLANGE\n")[1].split("\nTARGET\n")[0]
        self.assertIn("TF: no data", flange_section)
        self.assertIn("valid=0/921600", dashboard)
        self.assertIn("min=-  median=-  max=- mm", dashboard)

    def test_pose_display_reports_invalid_rotation_without_hiding_matrix(self):
        transform = np.eye(4)
        transform[0, 0] = np.nan
        text = "\n".join(main.pose_lines(transform))
        self.assertIn("RPY (deg): unavailable (invalid rotation)", text)
        self.assertIn("nan", text)

    def test_pose_display_uses_fixed_xyz_angles_for_combined_rotation(self):
        c = np.sqrt(3.) / 2.
        # 30 degrees around X, followed by 90 degrees around fixed Z.
        transform = np.array([[0., -c, .5, 0.], [1., 0., 0., 0.],
                              [0., .5, c, 0.], [0., 0., 0., 1.]])
        text = "\n".join(main.pose_lines(transform))
        self.assertIn("RPY (deg): roll=30.000  pitch=0.000  yaw=90.000", text)

    def test_run_owns_ipc_until_stop_event_and_then_cleans_up(self):
        suffix = uuid4().hex
        stop_event = threading.Event()
        stop_event.set()

        with tempfile.TemporaryDirectory() as tempdir:
            ipc = ProductIpc(
                camera_shm_name=f"camera_main_{suffix}",
                flange_shm_name=f"flange_main_{suffix}",
                target_shm_name=f"target_main_{suffix}",
                camera_sem_name=f"/camera_main_{suffix}",
                lock_path=str(Path(tempdir) / "supervisor.lock"),
            )
            with patch.object(main, "ProductIpc", return_value=ipc):
                self.assertEqual(main.run(stop_event), 0)

        self.assertIsNone(ipc.camera_shm)
        self.assertIsNone(ipc.flange_shm)
        self.assertIsNone(ipc.target_shm)
        self.assertIsNone(ipc.camera_sem)

    def test_installed_signal_handlers_set_shared_stop_event(self):
        handlers = {}
        stop_event = threading.Event()

        with patch.object(
            main.signal,
            "signal",
            side_effect=lambda signum, handler: handlers.__setitem__(signum, handler),
        ):
            main.install_signal_handlers(stop_event)

        self.assertEqual(set(handlers), {signal.SIGINT, signal.SIGTERM})
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signum=signum):
                stop_event.clear()
                handlers[signum](signum, None)
                self.assertTrue(stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
