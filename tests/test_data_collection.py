import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np


class DataCollectionTest(unittest.TestCase):
    def test_records_original_rgb_and_uint16_depth_without_detection(self):
        collection = importlib.import_module("training.data_collection")
        color = np.full((720, 1280, 3), [10, 100, 200], dtype=np.uint8)
        depth = np.full((720, 1280, 1), 1234, dtype=np.uint16)
        color_frame = SimpleNamespace(get_data=lambda: color.tobytes())
        depth_frame = SimpleNamespace(
            get_data=lambda: depth.tobytes(), get_global_timestamp_us=lambda: 123456
        )
        camera = Mock()
        camera.get_frame.side_effect = [
            SimpleNamespace(
                get_color_frame=lambda: SimpleNamespace(get_data=lambda: bytes(color.size)),
                get_depth_frame=lambda: depth_frame,
            ),
            SimpleNamespace(get_color_frame=lambda: color_frame, get_depth_frame=lambda: depth_frame),
        ]
        with tempfile.TemporaryDirectory() as directory:
            rgb_dir, depth_dir = collection.setup_save_directory(directory)
            with patch.object(collection, "setup_save_directory", return_value=(rgb_dir, depth_dir)), patch.object(
                collection, "Gemini336", return_value=camera
            ), patch.object(collection.cv2, "imshow"), patch.object(
                collection.cv2, "waitKey", side_effect=[32, ord("q")]
            ), patch.object(collection.cv2, "destroyAllWindows"):
                collection.main()
            rgb_files = list(Path(rgb_dir).glob("*.png"))
            depth_files = list(Path(depth_dir).glob("*.png"))
            self.assertEqual(len(rgb_files), 1)
            self.assertEqual(rgb_files[0].name, "melon_123456_blur.png")
            self.assertEqual([p.name for p in depth_files], [rgb_files[0].name])
            np.testing.assert_array_equal(cv2.imread(str(rgb_files[0])), color[:, :, ::-1])
            saved_depth = cv2.imread(str(depth_files[0]), cv2.IMREAD_UNCHANGED)
            self.assertEqual(saved_depth.dtype, np.uint16)
            np.testing.assert_array_equal(saved_depth, depth[:, :, 0])
        camera.stop_camera.assert_called_once()

    def test_camera_is_stopped_when_capture_raises(self):
        collection = importlib.import_module("training.data_collection")
        camera = Mock()
        camera.get_frame.side_effect = RuntimeError("capture failed")
        with patch.object(collection, "setup_save_directory", return_value=("rgb", "depth")), patch.object(
            collection, "Gemini336", return_value=camera
        ), patch.object(collection.cv2, "destroyAllWindows"):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                collection.main()
        camera.stop_camera.assert_called_once()


if __name__ == "__main__":
    unittest.main()
