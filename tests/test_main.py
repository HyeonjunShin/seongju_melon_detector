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
            timestamp_color=101,
            timestamp_depth=105,
            interval_color_us=16667.,
            interval_depth_us=33334.,
            color=np.zeros((720, 1280, 3), dtype=np.uint8),
            depth=np.zeros((720, 1280, 1), dtype=np.float32),
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
    def test_dashboard_shows_multi_object_detection_poses(self):
        from common.shm_detection import DetectionFrame, OBJECT_DTYPE
        objects = np.zeros(2, dtype=OBJECT_DTYPE)
        objects['TF'] = np.eye(4)
        objects['TF'][0, :3, 3] = [.1, .2, .3]
        objects['score'] = [.9, .8]
        frame = DetectionFrame(True, 123, 7, objects)
        dashboard, _ = main.build_dashboard(FakeIpc(), None, 10., detection=frame)
        section = dashboard.split('\nDETECTIONS (robot base)\n')[1]
        self.assertIn('count=2/8', section)
        self.assertIn('timestamp=123  seq=7', section)
        self.assertIn('P1 score=0.900', section)
        self.assertIn('P2 score=0.800', section)
        self.assertIn('X 0.100000  Y 0.200000  Z 0.300000', section)
        self.assertEqual(section.count('TF (4x4'), 2)

    def test_detection_monitor_waits_for_detector(self):
        dashboard, _ = main.build_dashboard(FakeIpc(), None, 10.)
        self.assertIn('detection_shm: WAITING', dashboard)

    def test_summary_orders_components_and_uses_actual_detector_publications(self):
        from common.shm_detection import DetectionFrame, OBJECT_DTYPE
        frame = DetectionFrame(True, 123, 100, np.zeros(0, dtype=OBJECT_DTYPE),
                               publication_count=10, latency_ms=12.34, interval_us=33333.)
        _, previous = main.build_dashboard(FakeIpc(), None, 10., frame, compact=True)
        frame.publication_count += 24
        frame.sequence_id += 60  # Camera sequence jumps must not inflate detector FPS.
        text, _ = main.build_dashboard(FakeIpc(), previous, 11., frame, compact=True)
        summary = text.split('\nCAMERA\n')[0]
        self.assertLess(summary.index('Robot '), summary.index('Camera '))
        self.assertLess(summary.index('Camera '), summary.index('Detector '))
        self.assertRegex(summary, r'NAME.*STATUS.*RATE.*PERIOD.*TIMESTAMP')
        detector_row = next(line for line in summary.splitlines() if line.startswith('Detector '))
        self.assertIn('24.0 Hz', detector_row)
        self.assertIn('33.333 ms', detector_row)
        self.assertTrue(detector_row.endswith('123'))
        robot_row = next(line for line in summary.splitlines() if line.startswith('Robot '))
        self.assertIn('2.000 ms', robot_row)
        camera_row = next(line for line in summary.splitlines() if line.startswith('Camera '))
        self.assertIn('16.667 ms', camera_row)

    def test_tracker_table_shows_id_velocity_and_observation_state(self):
        from common.shm_detection import DetectionFrame
        from detector.tracker import MultiObjectTracker
        from tests.test_tracker import detections
        rows = MultiObjectTracker(min_hits=1).update(1000000, detections([.1, .2, .3]))
        rows['velocity'][0] = [.1, -.2, 0.]
        sample = DetectionFrame(True, 1000000, 10, rows)
        text, _ = main.build_dashboard(FakeIpc(), None, 10., tracking=sample, compact=True)
        section = text.split('\nTRACKS (robot base)\n')[1]
        self.assertIn('VX(m/s)', section)
        self.assertIn('TRACKED', section)
        self.assertIn('+0.100', section)
        self.assertIn('-0.200', section)
        self.assertIn('sigma', section)

    def test_camera_shows_producer_intervals_instead_of_dashboard_poll_interval(self):
        ipc = FakeIpc()
        _, state = main.build_dashboard(ipc, previous=None, now=10.)
        dashboard, _ = main.build_dashboard(ipc, previous=state, now=11.)
        self.assertIn('Color timestamp=101 us  Δt=16.667 ms', dashboard)
        self.assertIn('Depth timestamp=105 us  Δt=33.334 ms', dashboard)

    def test_flange_buffer_duplicate_timestamps(self):
        ipc = FakeIpc()
        for timestamps, expected in [
            ([], "timestamp 중복: 없음"),
            ([10, 20, 30], "timestamp 중복: 없음"),
            ([10, 20, 10, 20, 10], "timestamp 중복: 10 us × 3회, 20 us × 2회"),
        ]:
            with self.subTest(timestamps=timestamps):
                data = [{"timestamp_us": ts, "TF": np.eye(4)} for ts in timestamps]
                with patch.object(ipc.flange_shm, "get_ordered_data",
                                  return_value=(data or None, len(data))):
                    dashboard, _ = main.build_dashboard(ipc, previous=None, now=10.)
                self.assertIn(expected, dashboard)

    def test_redirected_dashboard_has_no_terminal_escape_codes(self):
        output = io.StringIO()

        with patch.object(main.sys, "stdout", output):
            main.print_dashboard("new frame")

        self.assertEqual(output.getvalue(), "new frame\n")

    def test_terminal_dashboard_styles_status_and_clears_old_content(self):
        output = io.StringIO()
        with patch.object(output, "isatty", return_value=True), patch.object(main.sys, "stdout", output):
            main.print_dashboard("CAMERA\ncamera_shm ACTIVE")
        self.assertTrue(output.getvalue().startswith("\033[2J\033[H"))
        self.assertIn("\033[32mACTIVE", output.getvalue())
        self.assertTrue(output.getvalue().endswith("\033[J\n"))

    def test_flange_pose_is_a_table_in_compact_dashboard(self):
        ipc = FakeIpc()
        text, _ = main.build_dashboard(ipc, None, 10., compact=True)
        section = text.split('\nFLANGE\n')[1].split('\nTARGET\n')[0]
        self.assertIn('X(m)', section)
        self.assertIn('ROLL(°)', section)
        self.assertIn('+1.100000', section)
        self.assertIn('+2.200000', section)
        self.assertIn('+3.300000', section)
        self.assertNotIn('TF (4x4', section)
        with patch.object(ipc.flange_shm, 'get_ordered_data', return_value=(None, 0)):
            text, _ = main.build_dashboard(ipc, None, 10., compact=True)
        section = text.split('\nFLANGE\n')[1].split('\nTARGET\n')[0]
        self.assertIn('X(m)', section)
        self.assertIn('--', section)

    def test_compact_dashboard_keeps_eight_poses_in_a_table(self):
        from common.shm_detection import DetectionFrame, OBJECT_DTYPE
        objects = np.zeros(8, dtype=OBJECT_DTYPE)
        objects['TF'] = np.eye(4)
        objects['TF'][:, :3, 3] = [.1, -.2, .3]
        objects['score'] = .9
        dashboard, _ = main.build_dashboard(
            FakeIpc(), None, 10., DetectionFrame(True, 123, 7, objects), compact=True)
        self.assertIn('ROLL(°)', dashboard)
        self.assertIn('P8  0.900', dashboard)
        self.assertIn('+0.1000  -0.2000  +0.3000', dashboard)
        self.assertNotIn('TF (4x4', dashboard)
        self.assertNotIn('BBox', dashboard)
        self.assertIn('count=8/8', dashboard)
        self.assertNotIn('\033[', main.style_dashboard(dashboard))

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
        self.assertIn("Depth  1280 × 720 · float32 · 1 ch", dashboard)
        self.assertRegex(dashboard, r"flange_shm\s+ACTIVE\s+100\.0 Hz\s+0\.0 s")
        self.assertIn("위치   X 1.100000  Y 2.200000  Z 3.300000 m", dashboard)
        self.assertRegex(dashboard, r"target_shm\s+ACTIVE\s+1\.0 Hz\s+0\.0 s")
        self.assertIn("검출   YES · 신뢰도 0.875", dashboard)
        self.assertIn("위치   X 4.400000  Y 5.500000  Z 6.600000 m", dashboard)

    def test_flange_timing_uses_sequence_delta_and_elapsed_time(self):
        ipc = FakeIpc()
        dashboard, state = main.build_dashboard(ipc, previous=None, now=10.)
        self.assertIn("주기: 측정 불가", dashboard)
        # Buffer timestamps remain 2 ms apart; measure actual writes instead.
        for delta, elapsed, hz, period in [
            (250, .5, 500., "2.000"),
            (1000, 1., 1000., "1.000"),
            (0, 1., 0., None),
            (-10, 1., 0., None),
            (10, 0., 0., None),
        ]:
            with self.subTest(delta=delta, elapsed=elapsed):
                ipc.flange_shm.sequence = state["counters"]["flange"] + delta
                dashboard, _ = main.build_dashboard(ipc, previous=state, now=10. + elapsed)
                self.assertRegex(dashboard, rf"flange_shm\s+ACTIVE\s+{hz:.1f} Hz")
                if period is None:
                    self.assertIn("주기: 측정 불가", dashboard)
                else:
                    self.assertIn(f"평균={period} ms", dashboard)

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
        frame.depth[0, :3, 0] = [.1, .2, .3]
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
        self.assertIn("min=0.1000  median=0.2000  max=0.3000 m", dashboard)
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
        self.assertIn("min=-  median=-  max=- m", dashboard)

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
                detection_shm_name=f"detection_main_{suffix}",
                tracking_shm_name=f"tracking_main_{suffix}",
                camera_sem_name=f"/camera_main_{suffix}",
                lock_path=str(Path(tempdir) / "supervisor.lock"),
            )
            with patch.object(main, "ProductIpc", return_value=ipc):
                self.assertEqual(main.run(stop_event), 0)

        self.assertIsNone(ipc.camera_shm)
        self.assertIsNone(ipc.flange_shm)
        self.assertIsNone(ipc.target_shm)
        self.assertIsNone(ipc.detection_shm)
        self.assertIsNone(ipc.tracking_shm)
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
