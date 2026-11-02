import unittest
import io
import itertools
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import numpy as np

from detector import xpu_detection
from common.shm_camera import Frame


class TensorRTDetectionTest(unittest.TestCase):
    def test_headless_loop_reports_stage_times_and_skipped_frames_without_gui(self):
        color = np.zeros((530, 848, 3), np.uint8)
        shm = Mock()
        shm.intrinsics = (np.eye(3), np.zeros(8))
        flange = Mock(status=False)
        detections = Mock()
        shm.read.side_effect = [Frame(1, 1, color, None, 1), Frame(3, 3, color, None, 3)]
        sem = Mock()
        def acquire(**kwargs):
            if sem.acquire.call_count == 2:
                xpu_detection.is_running = False
        sem.acquire.side_effect = acquire
        model = Mock()
        model.detect_all.return_value = []
        output = io.StringIO()
        with patch('sys.argv', ['detection', '--no-display']), patch(
            'common.shm_camera.CameraShm', return_value=shm
        ), patch('posix_ipc.Semaphore', return_value=sem), patch.object(
            xpu_detection, 'Detector', return_value=model
        ), patch.object(xpu_detection, 'FlangeShm', return_value=flange), patch.object(
            xpu_detection, 'DetectionShm', return_value=detections
        ) as detection_factory, patch('signal.signal'), patch.object(
            xpu_detection.time, 'perf_counter', side_effect=itertools.count(0., .2)
        ), patch.object(xpu_detection.cv2, 'namedWindow') as window, patch.object(
            xpu_detection.cv2, 'imshow'
        ) as show, patch.object(xpu_detection.cv2, 'waitKey') as pump, redirect_stdout(output):
            xpu_detection.main()
        for text in ('FPS=', 'wait=', 'shm=', 'infer=', 'draw=0.00ms', 'gui=0.00ms', 'skipped=1'):
            self.assertIn(text, output.getvalue())
        window.assert_not_called()
        show.assert_not_called()
        pump.assert_not_called()
        shm.close.assert_called_once()
        sem.close.assert_called_once()
        flange.close.assert_called_once()
        detection_factory.assert_called_once_with(is_owner=False)
        detections.close.assert_called_once()
        self.assertEqual(detections.write.call_count, 3)

    def make_detector(self, objects, threshold=.3):
        engine = Mock(return_value=objects)
        engine.metadata = {'confidence': .3}
        with patch.object(xpu_detection, 'TensorRTModel', return_value=engine):
            detector = xpu_detection.Detector('test.engine', conf_threshold=threshold)
        return detector, engine

    def test_selects_top_one_and_preserves_original_pixels_and_probabilities(self):
        objects = [dict(score=.4), dict(score=.9, bbox=[100, 120, 200, 240],
                   body=[140, 160], knot=[170, 210], visibility_probability=[.95, .1])]
        detector, engine = self.make_detector(objects)
        rgb = np.zeros((530, 848, 3), np.uint8)
        rgb[0, 0] = [10, 20, 30]
        score, box, points, visibility = detector.detect(rgb)
        self.assertEqual(score, .9)
        np.testing.assert_array_equal(box, [100, 120, 200, 240])
        np.testing.assert_array_equal(points, [[140, 160], [170, 210]])
        np.testing.assert_allclose(visibility, [.95, .1])
        supplied = engine.call_args.args[0]
        self.assertEqual(supplied.shape, (530, 848, 3))
        np.testing.assert_array_equal(supplied[0, 0], [30, 20, 10])

    def test_no_detection_and_higher_confidence_filter(self):
        for objects in ([], [dict(score=.4)]):
            with self.subTest(objects=objects):
                detector, _ = self.make_detector(objects, threshold=.5)
                self.assertEqual(detector.detect(np.zeros((530, 848, 3), np.uint8)),
                                 (0., None, None, None))

    def test_rejects_threshold_below_engine_filter(self):
        with self.assertRaisesRegex(ValueError, 'confidence'):
            self.make_detector([], threshold=.2)

    def test_detect_all_filters_confidence_and_draws_every_object(self):
        objects = [
            dict(score=score, bbox=box, body=body, knot=knot,
                 visibility_probability=[.95, .9])
            for score, box, body, knot in (
                (.4, [10, 80, 60, 160], [30, 110], [40, 140]),
                (.9, [100, 120, 200, 240], [140, 160], [170, 210]),
                (.3, [250, 120, 350, 240], [290, 160], [320, 210]),
            )
        ]
        detector, _ = self.make_detector(objects, threshold=.4)
        rgb = np.zeros((530, 848, 3), np.uint8)
        predictions = detector.detect_all(rgb)
        self.assertEqual([item[0] for item in predictions], [.4, .9])
        display = xpu_detection.draw_detections(rgb, predictions)
        for y, x in ((80, 10), (120, 100)):
            np.testing.assert_array_equal(display[y, x], [0, 140, 255])
        for y, x in ((110, 30), (160, 140)):
            np.testing.assert_array_equal(display[y, x], [255, 0, 0])
        self.assertEqual(display[120, 250].sum(), 0)
        self.assertEqual(rgb.sum(), 0)
        detector.engine.return_value = []
        self.assertEqual(detector.detect_all(rgb), [])
        np.testing.assert_array_equal(xpu_detection.draw_detections(rgb, []), rgb)

    def test_rendering_uses_original_pixel_coordinates(self):
        rgb = np.zeros((530, 848, 3), np.uint8)
        display = xpu_detection.draw_detection(
            rgb, .9, [100, 120, 200, 240], [[140, 160], [170, 210]], [.95, .1]
        )
        np.testing.assert_array_equal(display[120, 100], [0, 140, 255])
        np.testing.assert_array_equal(display[160, 140], [255, 0, 0])
        self.assertEqual(rgb.sum(), 0)
