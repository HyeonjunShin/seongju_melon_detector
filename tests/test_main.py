import signal
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import main
from common.ipc import ProductIpc


class MainSupervisorTest(unittest.TestCase):
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
