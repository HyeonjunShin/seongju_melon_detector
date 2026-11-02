import unittest
from unittest.mock import Mock, patch

import numpy as np

from camera import test_sem
from common.shm_camera import CAMERA_HEIGHT, CAMERA_WIDTH, Frame


class CameraPreviewTest(unittest.TestCase):
    def test_current_frame_renders_rgb_and_meter_depth_and_closes_resources(self):
        color = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), [10, 20, 30], np.uint8)
        depth = np.ones((CAMERA_HEIGHT, CAMERA_WIDTH, 1), np.float32)
        depth[200, 100] = np.nan
        depth[200, 200] = 5.0
        frame = Frame(123, 123, color, depth, sequence_id=88)
        shm, sem = Mock(), Mock()
        shm.read.return_value = frame
        with patch.object(test_sem, 'CameraShm', return_value=shm), patch.object(
            test_sem.posix_ipc, 'Semaphore', return_value=sem
        ), patch.object(test_sem.signal, 'signal'), patch.object(
            test_sem.cv2, 'namedWindow'
        ), patch.object(test_sem.cv2, 'resizeWindow'), patch.object(
            test_sem.cv2, 'destroyAllWindows'
        ), patch.object(test_sem.cv2, 'waitKey', return_value=ord('q')), patch.object(
            test_sem.cv2, 'imshow'
        ) as show, patch('sys.argv', ['test_sem', '--scale', '1']):
            test_sem.main()
        grid = show.call_args.args[1]
        self.assertEqual(grid.shape, (CAMERA_HEIGHT * 2, CAMERA_WIDTH * 2, 3))
        np.testing.assert_array_equal(grid[200, 100], [30, 20, 10])
        np.testing.assert_array_equal(grid[200, CAMERA_WIDTH + 100], [0, 0, 0])
        np.testing.assert_array_equal(
            grid[CAMERA_HEIGHT + 200, CAMERA_WIDTH + 200], [0, 0, 0]
        )
        self.assertTrue(grid[200, CAMERA_WIDTH + 200].any())
        shm.close.assert_called_once()
        sem.close.assert_called_once()

    def test_scaled_preview_preserves_layout_and_depth_masks(self):
        color = np.full((530, 848, 3), [10, 20, 30], np.uint8)
        depth = np.ones((530, 848, 1), np.float32)
        depth[200, 100] = np.nan
        depth[200, 200] = 5.
        grid = test_sem.render_preview(Frame(1, 1, color, depth, 88), fps=60., scale=.5)
        self.assertEqual(grid.shape, (530, 848, 3))
        np.testing.assert_array_equal(grid[100, 50], [30, 20, 10])
        np.testing.assert_array_equal(grid[100, 424 + 50], [0, 0, 0])
        np.testing.assert_array_equal(grid[265 + 100, 424 + 100], [0, 0, 0])
        self.assertTrue(grid[100, 424 + 100].any())
