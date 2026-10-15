import math
import unittest

from claud import DualAssignmentPoseLoss, mean_keypoint_error


class TrainingSelectionTest(unittest.TestCase):
    def test_default_keypoint_weight_is_ten(self):
        self.assertEqual(DualAssignmentPoseLoss().kpt_weight, 10.0)

    def test_mean_keypoint_error_uses_visible_keypoints(self):
        metrics = {"kpt_err_px": [12.0, 8.0], "kpt_err_count": [4, 2]}

        self.assertEqual(mean_keypoint_error(metrics), 10.0)

    def test_missing_keypoint_matches_cannot_be_best(self):
        metrics = {"kpt_err_px": [0.0, 0.0], "kpt_err_count": [0, 0]}

        self.assertTrue(math.isinf(mean_keypoint_error(metrics)))


if __name__ == "__main__":
    unittest.main()
