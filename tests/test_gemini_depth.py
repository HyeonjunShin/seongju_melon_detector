import threading
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np

from camera.gemini import Gemini
from common.shm_camera import CameraShm


class GeminiDepthTest(unittest.TestCase):
    def test_callback_depth_is_float32_meters_and_can_be_published(self):
        # Bypass hardware initialization; exercise the real callback and SHM writer.
        camera = Gemini.__new__(Gemini)
        camera.stop_event = threading.Event()
        camera.frame_condition = threading.Condition()
        camera.received_count = 0
        camera.prev_callback_timestamp_color = None
        color_frame = Mock()
        color_frame.get_data.return_value = np.arange(6, dtype=np.uint8).tobytes()
        color_frame.get_height.return_value = 1
        color_frame.get_width.return_value = 2
        color_frame.get_system_timestamp_us.return_value = 123
        depth_frame = Mock()
        depth_frame.get_data.return_value = np.array([0, 12345], np.uint16).tobytes()
        depth_frame.get_height.return_value = 1
        depth_frame.get_width.return_value = 2
        depth_frame.get_depth_scale.return_value = 0.1
        depth_frame.get_system_timestamp_us.return_value = 120
        frames = Mock()
        frames.get_color_frame.return_value = color_frame
        frames.get_depth_frame.return_value = depth_frame

        camera.on_new_frame(frames)
        color, depth, timestamp_color, timestamp_depth = camera.wait_for_new_frame(timeout=0)
        self.assertEqual(depth.dtype, np.dtype(np.float32))
        np.testing.assert_allclose(depth, [[0, 1.2345]])
        depth_frame.get_data.return_value = np.array([65535, 12345], np.uint16).tobytes()
        camera.on_new_frame(frames)
        _, invalid_depth, _, _ = camera.wait_for_new_frame(timeout=0)
        np.testing.assert_allclose(invalid_depth, [[0, 1.2345]])
        with CameraShm(
            f"gemini_depth_{uuid4().hex}",
            is_owner=True,
            color_shape=(2, 1, 3),
            depth_shape=(2, 1, 1),
        ) as shm:
            shm.write(timestamp_color, timestamp_depth, color, depth)
            frame = shm.read()
            self.assertEqual(frame.timestamp_color, 123)
            self.assertEqual(frame.timestamp_depth, 120)
            self.assertEqual(frame.depth.dtype, np.dtype(np.float32))
            np.testing.assert_allclose(frame.depth.ravel(), [0, 1.2345])

        color_frame.get_system_timestamp_us.return_value = 20123
        depth_frame.get_system_timestamp_us.return_value = 40120
        with patch('builtins.print') as log:
            camera.on_new_frame(frames)
        log.assert_any_call('[DEPTH CALLBACK GAP] 40.000 ms')
        self.assertEqual(camera.prev_callback_timestamp_depth, 40120)


if __name__ == "__main__":
    unittest.main()
