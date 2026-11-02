import unittest
from uuid import uuid4
from unittest.mock import patch

import numpy as np

from common.shm_detection import DetectionShm, OBJECT_DTYPE
from detector.xpu_detection import build_base_detections
from camera.test_picking import T_FLANGE_CAMERA


class DetectionTest(unittest.TestCase):
    def test_snapshots_grow_shrink_and_clear(self):
        owner = DetectionShm('detection_test_' + uuid4().hex, is_owner=True)
        # This test shares the owner process/resource tracker. Production readers
        # are independently launched processes.
        with patch("common.shm_detection.unregister"):
            reader = DetectionShm(owner.shm.name)
        try:
            for publication, count in enumerate((0, 1, 8, 2, 0), 1):
                records = np.zeros(count, dtype=OBJECT_DTYPE)
                records['TF'] = np.eye(4)
                owner.write(123, count + 1, records, latency_ms=12.5)
                snapshot = reader.read()
                self.assertEqual(snapshot.count, count)
                self.assertEqual(snapshot.timestamp, 123)
                self.assertEqual(snapshot.publication_count, publication)
                self.assertEqual(snapshot.latency_ms, 12.5)
                np.testing.assert_array_equal(snapshot.objects, records)
            with self.assertRaises(ValueError):
                owner.write(1, 1, np.zeros(9, dtype=OBJECT_DTYPE))
            owner.write(1, 1, [], status=False)
            self.assertFalse(reader.read().status)
        finally:
            reader.close()
            owner.close()

    def test_period_uses_consecutive_writer_timestamps_and_resets(self):
        owner = DetectionShm('period_test_' + uuid4().hex, is_owner=True)
        try:
            for timestamp, expected in ((1000000, None), (1033333, 33333),
                                        (1083333, 50000), (1083333, None),
                                        (900000, None), (0, None), (1100000, None)):
                owner.write(timestamp, 1, [])
                value = owner.read().interval_us
                if expected is None:
                    self.assertTrue(np.isnan(value))
                else:
                    self.assertEqual(value, expected)
        finally:
            owner.close()

    def test_planar_body_pose_in_base_and_confidence_limit(self):
        matrix = np.array([[100., 0, 40], [0, 100., 40], [0, 0, 1.]])
        depth = np.ones((81, 81, 1), np.float32)
        base = np.eye(4)
        base[:3, 3] = [1., 2., 3.]
        predictions = [(score, np.array([10., 10., 70., 70.]),
                        np.array([[40., 40.], [60., 40.]]), np.array([1., 1.]))
                       for score in np.linspace(.5, 1., 10)]
        objects = build_base_detections(predictions, depth, matrix, np.zeros(8), base)
        self.assertEqual(len(objects), 8)
        self.assertTrue(np.all(np.diff(objects['score']) <= 0))
        expected = base @ T_FLANGE_CAMERA
        np.testing.assert_allclose(objects[0]['TF'][:3, 3],
                                   (expected @ [0, 0, 1, 1])[:3])
        np.testing.assert_allclose(objects[0]['TF'][:3, :3], expected[:3, :3], atol=1e-7)
        self.assertEqual(len(build_base_detections(
            predictions, depth * 0, matrix, np.zeros(8), base)), 0)
        self.assertEqual(len(build_base_detections(
            predictions, depth, matrix, np.zeros(8), base * np.nan)), 0)
        self.assertEqual(len(build_base_detections(
            [], depth, matrix, np.zeros(8), base)), 0)

    def test_x_axis_follows_body_to_knot_and_rejects_undefined_direction(self):
        matrix = np.array([[100., 0, 40], [0, 100., 40], [0, 0, 1.]])
        depth = np.ones((81, 81), np.float32)
        prediction = (.9, np.array([10., 10., 70., 70.]),
                      np.array([[40., 40.], [40., 60.]]), np.ones(2))
        objects = build_base_detections([prediction], depth, matrix, np.zeros(8), np.eye(4))
        rotation_camera = np.linalg.inv(T_FLANGE_CAMERA[:3, :3]) @ objects[0]['TF'][:3, :3]
        np.testing.assert_allclose(rotation_camera[:, 0], [0, 1, 0], atol=1e-7)
        np.testing.assert_allclose(rotation_camera[:, 2], [0, 0, 1], atol=1e-7)
        np.testing.assert_allclose(rotation_camera.T @ rotation_camera, np.eye(3), atol=1e-7)
        self.assertAlmostEqual(np.linalg.det(rotation_camera), 1.)
        prediction[2][1] = prediction[2][0]
        self.assertEqual(len(build_base_detections(
            [prediction], depth, matrix, np.zeros(8), np.eye(4))), 0)
        prediction[2][1] = [40., 60.]
        prediction[3][1] = 0.
        self.assertEqual(len(build_base_detections(
            [prediction], depth, matrix, np.zeros(8), np.eye(4))), 0)

    def test_body_depth_and_normal_failures_but_missing_knot_depth_is_allowed(self):
        matrix = np.array([[100., 0, 40], [0, 100., 40], [0, 0, 1.]])
        prediction = (.9, np.array([10., 10., 70., 70.]),
                      np.array([[20., 40.], [65., 40.]]), np.ones(2))
        for kind in ('body', 'knot', 'normal', 'ok'):
            depth = np.ones((81, 81), np.float32)
            if kind == 'body':
                depth[:] = 0
            elif kind == 'knot':
                depth[:, 50:] = 0
            elif kind == 'normal':
                depth += np.random.default_rng(42).normal(0, .008, depth.shape).astype(np.float32)
            diagnostics = []
            rows = build_base_detections([prediction], depth, matrix, np.zeros(8), np.eye(4),
                                         diagnostics=diagnostics)
            reason = diagnostics[0]['reason']
            if kind == 'body':
                self.assertEqual(reason, 'no_body_depth')
                self.assertEqual(diagnostics[0]['body_depth']['valid'], 0)
            elif kind == 'knot':
                self.assertEqual(reason, 'ok')
                self.assertEqual(len(rows), 1)
                self.assertNotIn('knot_depth', diagnostics[0])
            elif kind == 'normal':
                self.assertEqual(reason, 'ok')
                self.assertGreater(diagnostics[0]['normal']['rms_m'], .004)
            else:
                self.assertEqual(reason, 'ok')
                self.assertEqual(len(rows), 1)


    def test_surface_quality_filters_are_disabled_but_can_be_reenabled(self):
        from camera.test_picking import estimate_surface_normal
        matrix = np.array([[100., 0, 40], [0, 100., 40], [0, 0, 1.]])
        checker = np.indices((81, 81)).sum(axis=0) % 2 * 2 - 1
        for noise_m in (.0032, .006):
            depth = (1. + checker * noise_m).astype(np.float32)
            info = {}
            normal = estimate_surface_normal(40, 40, depth, matrix, np.zeros(8), diagnostics=info)
            if noise_m == .0032:
                self.assertIsNotNone(normal)
                self.assertGreater(info['rms_m'], .003)
                np.testing.assert_allclose(normal, [0, 0, 1], atol=1e-4)
            else:
                self.assertIsNotNone(normal)
                self.assertEqual(info['reason'], 'ok')
                with patch('camera.test_picking.NORMAL_FILTERS_ENABLED', True):
                    self.assertIsNone(estimate_surface_normal(40, 40, depth, matrix, np.zeros(8), diagnostics={}))

    def test_unfiltered_normal_keeps_depth_band_points_and_accepts_three_samples(self):
        from camera.test_picking import estimate_surface_normal
        matrix = np.array([[100., 0, 40], [0, 100., 40], [0, 0, 1.]])
        checker = np.indices((81, 81)).sum(axis=0) % 2 * 2 - 1
        info = {}
        self.assertIsNotNone(estimate_surface_normal(40, 40,
            1. + checker * .05, matrix, np.zeros(8), diagnostics=info))
        self.assertEqual(info['kept'], info['valid'])
        depth = np.zeros((81, 81))
        depth[39, 39] = depth[40, 40] = depth[39, 41] = 1.
        self.assertIsNotNone(estimate_surface_normal(40, 40, depth, matrix, np.zeros(8), diagnostics={}))
        depth[39, 41] = np.nan
        self.assertIsNone(estimate_surface_normal(40, 40, depth, matrix, np.zeros(8), diagnostics={}))

    def test_image_direction_defines_x_axis_on_tilted_body_plane(self):
        import cv2
        matrix = np.array([[100., 0, 40], [0, 120., 40], [0, 0, 1.]])
        depth = np.ones((81, 81), np.float32)
        prediction = (.9, np.array([10., 10., 70., 70.]),
                      np.array([[40., 40.], [60., 50.]]), np.ones(2))
        normal = np.array([.5, .2, np.sqrt(.71)])
        with patch('camera.test_picking.estimate_surface_normal', return_value=normal):
            rows = build_base_detections([prediction], depth, matrix, np.zeros(8), np.eye(4))
        camera_tf = np.linalg.inv(T_FLANGE_CAMERA) @ rows[0]['TF']
        np.testing.assert_allclose(camera_tf[:3, 2], normal, atol=1e-7)
        self.assertAlmostEqual(float(camera_tf[:3, 0] @ normal), 0.)
        points = np.array([camera_tf[:3, 3], camera_tf[:3, 3] + .05 * camera_tf[:3, 0]])
        pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), matrix, np.zeros(8))
        direction = np.diff(pixels.reshape(2, 2), axis=0)[0]
        np.testing.assert_allclose(direction / np.linalg.norm(direction),
                                   np.array([20., 10.]) / np.linalg.norm([20., 10.]), atol=1e-7)


class PoseOverlayTest(unittest.TestCase):
    def test_axes_project_published_base_pose_back_into_camera(self):
        from detector.xpu_detection import draw_base_poses
        from camera.test_picking import T_FLANGE_CAMERA
        image = np.zeros((200, 240, 3), np.uint8)
        matrix = np.array([[200., 0, 120.], [0, 200., 100.], [0, 0, 1.]])
        base_camera = np.eye(4)
        base_camera[:3, :3] = T_FLANGE_CAMERA[:3, :3]
        base_camera[:3, 3] = [1., 2., 3.]
        camera_object = np.eye(4)
        camera_object[:3, 3] = [0., 0., 1.]
        records = np.zeros(1, dtype=OBJECT_DTYPE)
        records['TF'][0] = base_camera @ camera_object
        with patch('detector.xpu_detection.cv2.arrowedLine') as arrow:
            draw_base_poses(image, records, matrix, np.zeros(8), base_camera)
        self.assertEqual(arrow.call_count, 3)
        x, y, z = arrow.call_args_list
        self.assertEqual(x.args[1:4], ((120, 100), (130, 100), (0, 0, 255)))
        self.assertEqual(y.args[1:4], ((120, 100), (120, 110), (0, 255, 0)))
        self.assertEqual(z.args[1:4], ((120, 100), (120, 100), (255, 0, 0)))
        self.assertTrue(image.any())

    def test_rejected_pose_overlay_and_detail_log_show_depth_availability(self):
        from detector.xpu_detection import draw_pose_diagnostics, format_pose_diagnostic
        image = np.zeros((100, 500, 3), np.uint8)
        detail = dict(reason='no_body_depth', bbox=np.array([10., 10., 60., 60.]),
                      body_depth=dict(valid=0, total=961, median_m=None))
        draw_pose_diagnostics(image, [detail])
        self.assertTrue(image.any())
        log = format_pose_diagnostic(detail)
        self.assertIn('no_body_depth', log)
        self.assertIn("'valid': 0", log)
        self.assertIn("'median_m': None", log)

    def test_empty_and_behind_camera_poses_are_skipped(self):
        from detector.xpu_detection import draw_base_poses
        image = np.zeros((100, 100, 3), np.uint8)
        records = np.zeros(1, dtype=OBJECT_DTYPE)
        records['TF'][0] = np.eye(4)
        records['TF'][0, 2, 3] = -1.
        with patch('detector.xpu_detection.cv2.arrowedLine') as arrow:
            draw_base_poses(image, records, np.eye(3), np.zeros(8), np.eye(4))
            draw_base_poses(image, [], np.eye(3), np.zeros(8), None)
        arrow.assert_not_called()


if __name__ == '__main__':
    unittest.main()
