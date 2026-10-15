import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torchvision import tv_tensors

from datasets import ChamaePackDataset


class DatasetVisibilityTest(unittest.TestCase):
    def test_keypoints_outside_transformed_image_become_invisible(self):
        with tempfile.TemporaryDirectory() as tmp:
            label_path = Path(tmp) / "sample.txt"
            label_path.write_text(
                "0 0 0 100 100 10 10 1 20 20 1\n", encoding="utf-8"
            )
            dataset = ChamaePackDataset(
                [(Path(tmp) / "sample.png", label_path)], is_train=False
            )

            def transform(sample):
                return {
                    "image": torch.zeros((3, 384, 640)),
                    "boxes": tv_tensors.BoundingBoxes(
                        [[0.0, 0.0, 100.0, 100.0]],
                        format="XYXY",
                        canvas_size=(384, 640),
                    ),
                    "keypoints": tv_tensors.KeyPoints(
                        [[[100.0, 100.0], [640.0, 100.0]]],
                        canvas_size=(384, 640),
                    ),
                }

            dataset.transform = transform
            with patch("datasets.read_image", return_value=torch.zeros((3, 720, 1280))):
                *_, visibility = dataset[0]

        torch.testing.assert_close(visibility, torch.tensor([[1.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
