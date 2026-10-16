import subprocess
import sys
import tempfile
import unittest
from multiprocessing import shared_memory
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from camera import camera_runner
from common.ipc import ProductIpc


class CameraRunnerIpcTest(unittest.TestCase):
    def setUp(self):
        suffix = uuid4().hex
        self.camera_name = f"camera_runner_{suffix}"
        self.sem_name = f"/camera_runner_{suffix}"
        self.tempdir = tempfile.TemporaryDirectory()
        self.ipc = ProductIpc(
            camera_shm_name=self.camera_name,
            flange_shm_name=f"flange_runner_{suffix}",
            target_shm_name=f"target_runner_{suffix}",
            camera_sem_name=self.sem_name,
            lock_path=str(Path(self.tempdir.name) / "supervisor.lock"),
        )

    def tearDown(self):
        self.ipc.close()
        self.tempdir.cleanup()

    def test_open_camera_ipc_requires_supervisor_resources(self):
        with patch.object(camera_runner, "SHM_NAME", self.camera_name), patch.object(
            camera_runner, "SEM_NAME", self.sem_name
        ):
            with self.assertRaisesRegex(RuntimeError, "main.py를 먼저 실행하세요"):
                camera_runner.open_camera_ipc()

    def test_attached_worker_close_preserves_supervisor_resources(self):
        self.ipc.start()
        source = f"""
from camera import camera_runner
camera_runner.SHM_NAME = {self.camera_name!r}
camera_runner.SEM_NAME = {self.sem_name!r}
camera_shm, camera_sem = camera_runner.open_camera_ipc()
camera_shm.status = True
camera_shm.close()
camera_sem.close()
"""
        subprocess.run([sys.executable, "-c", source], check=True)

        self.assertTrue(self.ipc.camera_shm.status)
        probe = shared_memory.SharedMemory(name=self.camera_name, create=False)
        probe.close()


if __name__ == "__main__":
    unittest.main()
