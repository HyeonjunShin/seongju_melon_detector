import unittest
from uuid import uuid4

import numpy as np

from common.shm_camera import CameraShm


class CameraTimestampTest(unittest.TestCase):
    def test_separate_timestamps_and_latest_producer_intervals(self):
        with CameraShm(f'timing_{uuid4().hex}', is_owner=True,
                       color_shape=(1, 2, 3), depth_shape=(1, 2, 1)) as shm:
            shm.write(1_000_000, 1_000_100)
            first = shm.read()
            self.assertTrue(np.isnan(first.interval_color_us))
            self.assertTrue(np.isnan(first.interval_depth_us))
            shm.write(1_016_667, 1_016_766)
            shm.write(1_033_334, 1_050_100)
            latest = shm.read()
            self.assertEqual(latest.timestamp_color, 1_033_334)
            self.assertEqual(latest.timestamp_depth, 1_050_100)
            self.assertEqual(latest.interval_color_us, 16_667)
            self.assertEqual(latest.interval_depth_us, 33_334)
            self.assertEqual(latest.timestamp, latest.timestamp_color)
            # Resetting the timestamp clock must not report a huge unsigned gap.
            shm.write(100, 200)
            self.assertTrue(np.isnan(shm.read().interval_color_us))
