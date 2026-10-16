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
    status = True
    sequence = 20

    def get_ordered_data(self):
        transform = np.eye(4)
        transform[:3, 3] = [1.1, 2.2, 3.3]
        return [{"timestamp_us": 202, "TF": transform}], self.sequence


class FakeTargetShm:
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

        self.assertIn("camera_shm  ACTIVE  30.0 Hz", dashboard)
        self.assertIn("RGB uint8(720, 1280, 3)", dashboard)
        self.assertIn("Depth uint16(720, 1280, 1)", dashboard)
        self.assertIn("shm_flange ACTIVE  100.0 Hz", dashboard)
        self.assertIn("XYZ=(1.100, 2.200, 3.300) m", dashboard)
        self.assertIn("shm_target  ACTIVE  observed 1.0 Hz", dashboard)
        self.assertIn("detected=True score=0.875", dashboard)
        self.assertIn("XYZ=(4.400, 5.500, 6.600) m", dashboard)

    def test_dashboard_reports_time_since_last_change(self):
        ipc = FakeIpc()
        _, state = main.build_dashboard(ipc, previous=None, now=10.0)

        dashboard, _ = main.build_dashboard(ipc, previous=state, now=12.5)

        self.assertEqual(dashboard.count("stale=2.5s"), 3)
        self.assertIn("camera_shm  ACTIVE  0.0 Hz", dashboard)
        self.assertIn("shm_target  ACTIVE  observed 0.0 Hz", dashboard)

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
