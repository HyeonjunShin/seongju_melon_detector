import unittest
from multiprocessing import shared_memory
import subprocess
import sys
from uuid import uuid4

import numpy as np

from common.shm_camera import CameraShm
from common.shm_flange import FlangeShm
from common.shm_target import TargetShm


class SharedMemoryOwnershipTest(unittest.TestCase):
    def setUp(self):
        suffix = uuid4().hex
        self.camera_name = f"camera_test_{suffix}"
        self.flange_name = f"flange_test_{suffix}"
        self.target_name = f"target_test_{suffix}"
        self.owners = []

    def tearDown(self):
        for owner in reversed(self.owners):
            owner.close()

    def assert_removed(self, name):
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=name, create=False)

    def run_worker(self, source):
        subprocess.run([sys.executable, "-c", source], check=True)

    def test_camera_owner_is_exclusive_and_attached_close_preserves_data(self):
        owner = CameraShm(
            self.camera_name,
            is_owner=True,
            color_shape=(1, 2, 3),
            depth_shape=(1, 2, 1),
        )
        self.owners.append(owner)
        self.assertFalse(owner.status)
        self.assertEqual(owner.sequence_id, 0)

        self.run_worker(
            f"""
import numpy as np
from common.shm_camera import CameraShm
shm = CameraShm({self.camera_name!r}, is_owner=False, color_shape=(1, 2, 3), depth_shape=(1, 2, 1))
shm.write(7, np.ones((1, 2, 3), np.uint8), np.full((1, 2, 1), 9, np.uint16))
shm.close()
"""
        )
        self.assertEqual(owner.read().timestamp, 7)

        with self.assertRaises(FileExistsError):
            CameraShm(
                self.camera_name,
                is_owner=True,
                color_shape=(1, 2, 3),
                depth_shape=(1, 2, 1),
            )

        owner.close()
        self.assert_removed(self.camera_name)

    def test_flange_owner_unlinks_and_attached_instance_exchanges_state(self):
        owner = FlangeShm(self.flange_name, is_owner=True)
        self.owners.append(owner)
        self.assertFalse(owner.status)
        self.assertEqual(owner.sequence, 0)
        self.assertEqual(owner.head, 0)

        self.run_worker(
            f"""
from common.shm_flange import FlangeShm
shm = FlangeShm({self.flange_name!r}, is_owner=False)
shm.status = True
shm.close()
"""
        )
        self.assertTrue(owner.status)

        owner.close()
        self.assert_removed(self.flange_name)

    def test_target_owner_unlinks_and_attached_instance_exchanges_data(self):
        owner = TargetShm(self.target_name, is_owner=True)
        self.owners.append(owner)
        self.assertFalse(owner.status)
        self.assertEqual(owner.index, 0)

        self.run_worker(
            f"""
from common.shm_target import TargetShm
shm = TargetShm({self.target_name!r}, is_owner=False)
shm.write(timestamp=11, detected=True, score=0.75)
shm.close()
"""
        )
        self.assertEqual(owner.read().timestamp, 11)
        self.assertTrue(owner.read().detected)

        owner.close()
        self.assert_removed(self.target_name)


if __name__ == "__main__":
    unittest.main()
