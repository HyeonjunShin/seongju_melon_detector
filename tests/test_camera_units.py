import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from camera import data_collection, test_picking
from camera import camera_runner
from camera.gemini import Gemini
from common.shm_camera import CameraShm
from pyorbbecsdk import OBFormat
from uuid import uuid4


class CameraUnitsTest(unittest.TestCase):
    def test_picking_layout_matches_runner_and_preserves_pixel_locations(self):
        width, height, _ = camera_runner.COLOR_SHAPE
        self.assertEqual((test_picking.IMAGE_WIDTH, test_picking.IMAGE_HEIGHT), (width, height))
        with CameraShm(f'picking_layout_{uuid4().hex}', is_owner=True) as shm:
            self.assertEqual(shm.color_shape, test_picking.COLOR_SHAPE)
            self.assertEqual(shm.depth_shape, test_picking.DEPTH_SHAPE)
            color = np.zeros((height, width, 3), np.uint8)
            depth = np.zeros((height, width), np.float32)
            color[20, 40] = [1, 2, 3]
            depth[20, 40] = 1.25
            shm.write(1, 1, color, depth)
            frame = shm.read()
            np.testing.assert_array_equal(frame.color[20, 40], [1, 2, 3])
            self.assertEqual(frame.depth[20, 40, 0], 1.25)

    def test_runner_calibration_reaches_picking_in_opencv_order(self):
        camera = Gemini.__new__(Gemini)
        camera.fx_color, camera.fy_color = 500., 510.
        camera.cx_color, camera.cy_color = 424., 265.
        for i, name in enumerate(('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')):
            setattr(camera, f'{name}_color', i * 0.01)
        with CameraShm(f'picking_calibration_{uuid4().hex}', is_owner=True,
                       color_shape=(1, 2, 3), depth_shape=(1, 2, 1)) as shm:
            shm.intrinsics = camera.get_color_calibration()
            matrix, distortion = shm.intrinsics
            np.testing.assert_array_equal(matrix, [[500, 0, 424], [0, 510, 265], [0, 0, 1]])
            np.testing.assert_allclose(distortion, [0, .01, .02, .03, .04, .05, .06, .07])
            source = f'''
import numpy as np
from common.shm_camera import CameraShm
with CameraShm({shm.shm_name!r}, color_shape=(1, 2, 3), depth_shape=(1, 2, 1)) as shm:
    matrix, distortion = shm.intrinsics
    np.testing.assert_array_equal(matrix, [[500, 0, 424], [0, 510, 265], [0, 0, 1]])
    np.testing.assert_allclose(distortion, [0, .01, .02, .03, .04, .05, .06, .07])
'''
            subprocess.run([sys.executable, '-c', source], check=True, timeout=10)

    def test_preview_renders_current_resolution_and_maps_depth_click(self):
        color = np.zeros((530, 848, 3), np.uint8)
        depth = np.full_like(color, 100)
        preview = test_picking.render_preview(color, depth, None, 60.)
        self.assertEqual(preview.shape, (530, 848, 3))
        self.assertEqual(test_picking.preview_to_pixel(672, 116), (425, 266))
        self.assertEqual(test_picking.preview_to_pixel(40, 20), (40, 20))
        self.assertIsNone(test_picking.preview_to_pixel(848, 530))

    def test_picking_preserves_meter_depth_and_plane_normal(self):
        depth = np.full((61, 61), 1.25, np.float32)
        intrinsic = np.array([[100., 0., 30.], [0., 100., 30.], [0., 0., 1.]])
        with patch.object(test_picking, 'K', intrinsic), patch.object(
            test_picking, 'D', np.zeros(5)
        ):
            np.testing.assert_allclose(
                test_picking.pixel_to_camera_xyz(30, 30, depth), [0, 0, 1.25]
            )
            np.testing.assert_allclose(
                test_picking.estimate_surface_normal(30, 30, depth), [0, 0, 1]
            )

    def test_saved_depth_preserves_float32_meters(self):
        frame = Mock()
        frame.get_width.return_value = 2
        frame.get_height.return_value = 1
        frame.get_depth_scale.return_value = 0.1
        frame.get_format.return_value = OBFormat.Y16
        frame.get_data.return_value = np.array([0, 12345], np.uint16).tobytes()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            data_collection.os, 'getcwd', return_value=directory
        ):
            data_collection.save_depth_frame(frame, 123)
            saved = np.load(Path(directory) / 'outputs/depth_images/depth_2_1_123.npy')
            self.assertEqual(saved.dtype, np.dtype(np.float32))
            np.testing.assert_allclose(saved, [[0, 1.2345]])
