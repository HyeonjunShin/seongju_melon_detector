import fcntl
import subprocess
import sys
import time
import unittest
from multiprocessing import shared_memory
from uuid import uuid4

import numpy as np

from common.shm_camera import CameraShm


class CameraSnapshotTest(unittest.TestCase):
    def test_incompatible_memory_size_has_restart_hint_and_preserves_owner(self):
        for size in (16, 2048):
            with self.subTest(size=size):
                owner = shared_memory.SharedMemory(create=True, size=size)
                try:
                    source = f'''
from common.shm_camera import CameraShm
try:
    CameraShm({owner.name!r}, color_shape=(1, 2, 3), depth_shape=(1, 2, 1))
except RuntimeError as exc:
    assert 'main.py' in str(exc) and '크기' in str(exc), str(exc)
else:
    raise AssertionError('incompatible layout accepted')
'''
                    subprocess.run([sys.executable, '-c', source], check=True, timeout=10)
                    probe = shared_memory.SharedMemory(name=owner.name)
                    probe.close()
                finally:
                    owner.close()
                    owner.unlink()

    def test_concurrent_producer_keeps_timestamp_rgb_depth_and_sequence_together(self):
        name = f'camera_pair_{uuid4().hex}'
        with CameraShm(name, is_owner=True, color_shape=(64, 96, 3),
                       depth_shape=(64, 96, 1)) as shm:
            source = f'''
import time
import numpy as np
from common.shm_camera import CameraShm
with CameraShm({name!r}, color_shape=(64, 96, 3), depth_shape=(64, 96, 1)) as shm:
    for value in range(1, 101):
        shm.write(value, np.full((64, 96, 3), value, np.uint8),
                  np.full((64, 96, 1), value * 10, np.uint16))
        time.sleep(0.001)
'''
            worker = subprocess.Popen([sys.executable, '-c', source])
            last_sequence = 0
            deadline = time.monotonic() + 10
            try:
                while last_sequence < 100:
                    self.assertLess(time.monotonic(), deadline, 'producer timed out')
                    frame = shm.read(last_sequence_id=last_sequence)
                    if frame is None:
                        time.sleep(0.0001)
                        continue
                    self.assertEqual(frame.timestamp, frame.sequence_id)
                    np.testing.assert_array_equal(frame.color, frame.timestamp)
                    np.testing.assert_array_equal(frame.depth, frame.timestamp * 10)
                    last_sequence = frame.sequence_id
                self.assertEqual(worker.wait(timeout=10), 0)
            finally:
                if worker.poll() is None:
                    worker.kill()
                    worker.wait()

    def test_single_buffer_returns_latest_pair_and_skips_duplicates(self):
        name = f'camera_latest_{uuid4().hex}'
        with CameraShm(name, is_owner=True, color_shape=(1, 2, 3),
                       depth_shape=(1, 2, 1)) as shm:
            self.assertEqual(shm.total_bytes, shm.header_bytes + shm.frame_dtype.itemsize)
            self.assertIsNone(shm.read(last_sequence_id=0))
            shm.sequence_id = 100
            for value in (1, 2, 3):
                shm.write(value, np.full((1, 2, 3), value, np.uint8),
                          np.full((1, 2, 1), value * 10, np.uint16))
            latest = shm.read(last_sequence_id=100)
            self.assertEqual(latest.sequence_id, 103)
            self.assertEqual(latest.timestamp, 3)
            np.testing.assert_array_equal(latest.color, 3)
            np.testing.assert_array_equal(latest.depth, 30)
            self.assertIsNone(shm.read(last_sequence_id=latest.sequence_id))
            shm.status = False
            self.assertIsNone(shm.read(last_sequence_id=0))

    def test_reads_and_writes_wait_for_another_process_lock(self):
        name = f'camera_lock_{uuid4().hex}'
        with CameraShm(name, is_owner=True, color_shape=(1, 2, 3),
                       depth_shape=(1, 2, 1)) as shm:
            for operation in ('shm.read()', 'shm.write(2)'):
                with self.subTest(operation=operation):
                    source = f'''
from common.shm_camera import CameraShm
with CameraShm({name!r}, color_shape=(1, 2, 3), depth_shape=(1, 2, 1)) as shm:
    print('ready', flush=True)
    {operation}
    print('done', flush=True)
'''
                    fcntl.flock(shm.shm._fd, fcntl.LOCK_EX)
                    worker = subprocess.Popen([sys.executable, '-c', source],
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                              text=True)
                    try:
                        self.assertEqual(worker.stdout.readline().strip(), 'ready')
                        with self.assertRaises(subprocess.TimeoutExpired):
                            worker.communicate(timeout=0.2)
                    finally:
                        fcntl.flock(shm.shm._fd, fcntl.LOCK_UN)
                        try:
                            output, errors = worker.communicate(timeout=10)
                        except subprocess.TimeoutExpired:
                            worker.kill()
                            worker.communicate()
                            raise
                    self.assertEqual(worker.returncode, 0, errors)
                    self.assertIn('done', output)

    def test_snapshot_survives_slot_reuse_in_another_process(self):
        name = f'camera_snapshot_{uuid4().hex}'
        with CameraShm(name, is_owner=True, color_shape=(1, 2, 3),
                       depth_shape=(1, 2, 1)) as shm:
            shm.write(1, np.full((1, 2, 3), 1, np.uint8),
                      np.full((1, 2, 1), 10, np.uint16))
            frame = shm.read()
            source = f'''
import numpy as np
from common.shm_camera import CameraShm
with CameraShm({name!r}, color_shape=(1, 2, 3), depth_shape=(1, 2, 1)) as shm:
    for value in (2, 3):
        shm.write(value, np.full((1, 2, 3), value, np.uint8),
                  np.full((1, 2, 1), value * 10, np.uint16))
'''
            subprocess.run([sys.executable, '-c', source], check=True, timeout=10)
            self.assertEqual(frame.timestamp, 1)
            np.testing.assert_array_equal(frame.color, 1)
            np.testing.assert_array_equal(frame.depth, 10)
            latest = shm.read()
            self.assertEqual(latest.timestamp, 3)
            np.testing.assert_array_equal(latest.color, 3)
            np.testing.assert_array_equal(latest.depth, 30)


if __name__ == '__main__':
    unittest.main()
