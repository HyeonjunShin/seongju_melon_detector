import unittest
from multiprocessing import shared_memory
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

        attached = CameraShm(
            self.camera_name,
            is_owner=False,
            color_shape=(1, 2, 3),
            depth_shape=(1, 2, 1),
        )
        owner.write(7, np.ones((1, 2, 3), np.uint8), np.full((1, 2, 1), 9, np.uint16))
        attached.close()

        reopened = CameraShm(
            self.camera_name,
            is_owner=False,
            color_shape=(1, 2, 3),
            depth_shape=(1, 2, 1),
        )
        self.assertEqual(reopened.read().timestamp, 7)
        reopened.close()

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

        attached = FlangeShm(self.flange_name, is_owner=False)
        attached.status = True
        self.assertTrue(owner.status)
        attached.close()

        owner.close()
        self.assert_removed(self.flange_name)

    def test_target_owner_unlinks_and_attached_instance_exchanges_data(self):
        owner = TargetShm(self.target_name, is_owner=True)
        self.owners.append(owner)
        self.assertFalse(owner.status)
        self.assertEqual(owner.index, 0)

        attached = TargetShm(self.target_name, is_owner=False)
        attached.write(timestamp=11, detected=True, score=0.75)
        attached.close()
        self.assertEqual(owner.read().timestamp, 11)
        self.assertTrue(owner.read().detected)

        owner.close()
        self.assert_removed(self.target_name)


if __name__ == "__main__":
    unittest.main()
