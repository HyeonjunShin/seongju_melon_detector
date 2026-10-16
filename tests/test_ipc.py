import tempfile
import unittest
from multiprocessing import shared_memory
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from uuid import uuid4

import posix_ipc

from common.ipc import ProductIpc


class ProductIpcTest(unittest.TestCase):
    def setUp(self):
        suffix = uuid4().hex
        self.names = {
            "camera_shm_name": f"camera_ipc_{suffix}",
            "flange_shm_name": f"flange_ipc_{suffix}",
            "target_shm_name": f"target_ipc_{suffix}",
            "camera_sem_name": f"/camera_sem_{suffix}",
        }
        self.tempdir = tempfile.TemporaryDirectory()
        self.names["lock_path"] = str(Path(self.tempdir.name) / "supervisor.lock")
        self.ipcs = []

    def tearDown(self):
        for ipc in reversed(self.ipcs):
            ipc.close()
        self.tempdir.cleanup()

    def assert_shm_missing(self, name):
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=name, create=False)

    def assert_all_missing(self):
        self.assert_shm_missing(self.names["camera_shm_name"])
        self.assert_shm_missing(self.names["flange_shm_name"])
        self.assert_shm_missing(self.names["target_shm_name"])
        with self.assertRaises(posix_ipc.ExistentialError):
            posix_ipc.Semaphore(self.names["camera_sem_name"])

    def run_worker(self, source):
        subprocess.run([sys.executable, "-c", source], check=True)

    def test_start_creates_resources_and_attached_close_preserves_them(self):
        ipc = ProductIpc(**self.names)
        self.ipcs.append(ipc)
        ipc.start()

        self.assertFalse(ipc.camera_shm.status)
        self.assertFalse(ipc.flange_shm.status)
        self.assertFalse(ipc.target_shm.status)

        self.run_worker(
            f"""
import posix_ipc
from common.shm_camera import CameraShm
from common.shm_flange import FlangeShm
from common.shm_target import TargetShm
camera = CameraShm({self.names['camera_shm_name']!r}, is_owner=False)
flange = FlangeShm({self.names['flange_shm_name']!r}, is_owner=False)
target = TargetShm({self.names['target_shm_name']!r}, is_owner=False)
semaphore = posix_ipc.Semaphore({self.names['camera_sem_name']!r})
camera.close()
flange.close()
target.close()
semaphore.close()
"""
        )

        probe = shared_memory.SharedMemory(
            name=self.names["camera_shm_name"], create=False
        )
        probe.close()

        ipc.close()
        self.assert_all_missing()

    def test_second_supervisor_fails_without_modifying_live_resources(self):
        first = ProductIpc(**self.names)
        second = ProductIpc(**self.names)
        self.ipcs.extend((first, second))
        first.start()
        first.camera_shm.status = True

        with self.assertRaisesRegex(RuntimeError, "이미 실행 중"):
            second.start()

        self.assertTrue(first.camera_shm.status)
        self.run_worker(
            f"""
from common.shm_camera import CameraShm
shm = CameraShm({self.names['camera_shm_name']!r}, is_owner=False)
assert shm.status
shm.close()
"""
        )

    def test_partial_start_failure_removes_resources_created_by_attempt(self):
        ipc = ProductIpc(**self.names)
        self.ipcs.append(ipc)

        with patch("common.ipc.TargetShm", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                ipc.start()

        self.assert_all_missing()

    def test_context_manager_cleans_up_resources(self):
        with ProductIpc(**self.names):
            self.run_worker(
                f"""
from common.shm_target import TargetShm
shm = TargetShm({self.names['target_shm_name']!r}, is_owner=False)
shm.close()
"""
            )

        self.assert_all_missing()


if __name__ == "__main__":
    unittest.main()
