import unittest

import numpy as np
import torch

from xpu_detection import DetectionEMA, PyTorchDetector, prepare_pytorch_input


class PyTorchDetectionTest(unittest.TestCase):
    def test_ema_smooths_bbox_and_keypoints_and_resets_after_miss(self):
        ema = DetectionEMA(alpha=0.25)
        first_box = np.array([0, 0, 100, 100], dtype=np.float32)
        first_kpts = np.array([[20, 30], [80, 70]], dtype=np.float32)
        _, box, kpts, _ = ema.update(0.9, first_box, first_kpts, np.ones(2))
        np.testing.assert_allclose(box, first_box)
        np.testing.assert_allclose(kpts, first_kpts)

        _, box, kpts, _ = ema.update(
            0.8,
            np.array([40, 40, 140, 140], dtype=np.float32),
            np.array([[60, 70], [120, 110]], dtype=np.float32),
            np.ones(2),
        )
        np.testing.assert_allclose(box, [10, 10, 110, 110])
        np.testing.assert_allclose(kpts, [[30, 40], [90, 80]])

        ema.update(0.0, None, None, None)
        _, box, kpts, _ = ema.update(0.7, first_box + 200, first_kpts + 200, np.ones(2))
        np.testing.assert_allclose(box, first_box + 200)
        np.testing.assert_allclose(kpts, first_kpts + 200)

    def test_preprocess_matches_training_canvas(self):
        frame = np.full((720, 1280, 3), 255, dtype=np.uint8)

        tensor = prepare_pytorch_input(frame, torch.device("cpu"))

        self.assertEqual(tuple(tensor.shape), (1, 3, 384, 640))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor[:, :, :12].count_nonzero().item(), 0)
        self.assertEqual(tensor[:, :, 12:372].min().item(), 1.0)
        self.assertEqual(tensor[:, :, 372:].count_nonzero().item(), 0)

    def test_detector_returns_one_to_one_prediction(self):
        detector = PyTorchDetector(
            "runs/dual_v2/best.pt", device="cpu", conf_threshold=0.0
        )

        score, bbox, keypoints, visibility = detector.detect(
            np.zeros((720, 1280, 3), dtype=np.uint8)
        )

        self.assertGreaterEqual(score, 0.0)
        self.assertEqual(bbox.shape, (4,))
        self.assertEqual(keypoints.shape, (2, 2))
        self.assertEqual(visibility.shape, (2,))


if __name__ == "__main__":
    unittest.main()
