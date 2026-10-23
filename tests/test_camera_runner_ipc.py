import subprocess
import sys
import tempfile
import unittest
from multiprocessing import shared_memory
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import posix_ipc

from camera import camera_runner
from common.ipc import ProductIpc
from common.shm_camera import CameraShm


class CameraRunnerIpcTest(unittest.TestCase):
    def test_direct_script_import_uses_current_checkout(self):
        root = Path(__file__).resolve().parents[1]
        for script in ('camera_runner.py', 'test_sem.py'):
            with self.subTest(script=script):
                source = f'''
import runpy
import sys
from pathlib import Path
sys.path[0] = {str(root / 'camera')!r}
runpy.run_path({str(root / 'camera' / script)!r}, run_name='import_probe')
import common.shm_camera
assert Path(common.shm_camera.__file__).resolve() == Path({str(root / 'common/shm_camera.py')!r})
'''
                subprocess.run([sys.executable, '-c', source], cwd='/tmp', check=True,
                               capture_output=True, text=True, timeout=10)

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


class CameraRunnerLifecycleTest(unittest.TestCase):
    def run_runner(self, shm, sem, camera_factory):
        with patch.object(camera_runner, 'open_camera_ipc', return_value=(shm, sem)), patch.object(
            camera_runner, 'Gemini336', camera_factory
        ), patch.object(camera_runner.signal, 'signal'), patch.object(
            camera_runner.os, 'sched_setaffinity', create=True
        ), patch.object(camera_runner, 'is_running', True):
            camera_runner.main()

    def test_restart_continues_existing_sequence(self):
        with CameraShm(f'runner_sequence_{uuid4().hex}', is_owner=True,
                       color_shape=(1, 2, 3), depth_shape=(1, 2, 1)) as shm:
            shm.sequence_id = 108000
            sem, camera = MagicMock(), MagicMock()
            sem.value = 0
            frame = camera.get_frame.return_value
            frame.get_color_frame.return_value.get_data.return_value = np.ones((1, 2, 3), np.uint8)
            frame.get_depth_frame.return_value.get_data.return_value = np.ones((1, 2, 1), np.uint16)
            frame.get_depth_frame.return_value.get_global_timestamp_us.return_value = 123
            def stop_after_frame():
                camera_runner.is_running = False
            sem.release.side_effect = stop_after_frame
            with patch.object(shm, 'close') as close:
                self.run_runner(shm, sem, MagicMock(return_value=camera))
                close.assert_called_once()
            self.assertEqual(shm.read().sequence_id, 108001)
            sem.close.assert_called_once()
            camera.stop_camera.assert_called_once()

    def test_pending_notification_does_not_accumulate(self):
        name = f'/runner_notify_{uuid4().hex}'
        sem = posix_ipc.Semaphore(name, flags=posix_ipc.O_CREX, initial_value=1)
        try:
            shm, camera = MagicMock(), MagicMock()
            def next_frame():
                camera_runner.is_running = False
                return MagicMock()
            camera.get_frame.side_effect = next_frame
            self.run_runner(shm, posix_ipc.Semaphore(name), MagicMock(return_value=camera))
            self.assertEqual(sem.value, 1)
        finally:
            sem.close()
            posix_ipc.unlink_semaphore(name)

    def test_initialization_failure_clears_status_and_closes_ipc(self):
        shm, sem = MagicMock(), MagicMock()
        shm.status = True
        self.run_runner(shm, sem, MagicMock(side_effect=RuntimeError('SDK init failed')))
        self.assertFalse(shm.status)
        shm.close.assert_called_once()
        sem.close.assert_called_once()

    def test_stop_failure_still_closes_ipc(self):
        shm, sem, camera = MagicMock(), MagicMock(), MagicMock()
        camera.get_frame.side_effect = RuntimeError('read failed')
        camera.stop_camera.side_effect = RuntimeError('stop failed')
        self.run_runner(shm, sem, MagicMock(return_value=camera))
        self.assertFalse(shm.status)
        shm.close.assert_called_once()
        sem.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
