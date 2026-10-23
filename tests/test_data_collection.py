import importlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


class DataCollectionTest(unittest.TestCase):
    def test_save_error_is_reported_even_when_no_more_frames_arrive(self):
        collection = importlib.import_module('training.data_collection')
        from training.recording import RecordingBuffer
        buffer = RecordingBuffer()
        buffer.configure(True)
        buffer.offer(1, np.zeros((8, 12, 3), np.uint8), np.ones((8, 12), np.uint16), None)

        def fail(packet):
            raise OSError('disk full')

        worker = collection.CollectionWorker(buffer, SimpleNamespace(save=fail))
        worker.thread.start()
        worker.thread.join(timeout=1)
        try:
            self.assertIsInstance(worker.error, OSError)
            self.assertFalse(buffer.stats()['enabled'])
        finally:
            worker.stop.set()
            worker.thread.join(timeout=2)

    def test_storage_overlaps_for_two_frames_and_drains_on_stop(self):
        collection = importlib.import_module('training.data_collection')
        from training.recording import RecordingBuffer
        buffer = RecordingBuffer()
        buffer.configure(True)
        for timestamp in (1, 2):
            buffer.offer(timestamp, np.zeros((8, 12, 3), np.uint8),
                         np.ones((8, 12), np.uint16), None)
        started = []
        lock = threading.Lock()
        both_started, release = threading.Event(), threading.Event()

        def save(packet):
            with lock:
                started.append(packet['timestamp_us'])
                if len(started) == 2:
                    both_started.set()
            release.wait(timeout=2)

        worker = collection.CollectionWorker(buffer, SimpleNamespace(save=save))
        worker.thread.start()
        try:
            self.assertTrue(both_started.wait(timeout=1))
        finally:
            buffer.configure(False)
            worker.stop.set()
            release.set()
            worker.thread.join(timeout=3)
        self.assertFalse(worker.thread.is_alive())
        self.assertIsNone(worker.error)
        self.assertEqual(sorted(started), [1, 2])

    def test_quit_drains_accepted_frames_and_preserves_preview_shm(self):
        collection = importlib.import_module('training.data_collection')
        from training.recording import RecordingBuffer
        buffer = RecordingBuffer()
        original_configure = buffer.configure
        color = np.zeros((8, 12, 3), np.uint8)
        depth = np.ones((8, 12), np.uint16)

        def configure(enabled):
            original_configure(enabled)
            if enabled:
                for timestamp in (100, 100, 101):
                    buffer.offer(timestamp, color, depth, None)

        with tempfile.TemporaryDirectory() as directory, patch.object(
            collection, 'SAVE_PATH', directory
        ), patch.object(collection, 'connect', return_value=(None, buffer)), patch.object(
            collection, 'CameraShm'
        ) as camera_class, patch.object(collection.cv2, 'imshow'), patch.object(
            collection.cv2, 'waitKey', side_effect=[32, ord('q')]
        ), patch.object(collection.cv2, 'destroyAllWindows'), patch.object(
            buffer, 'configure', side_effect=configure
        ):
            camera_class.return_value.status = True
            camera_class.return_value.read.return_value = SimpleNamespace(color=color)
            collection.main()
            self.assertEqual(len(list((Path(directory) / 'rgb').glob('*.png'))), 2)
            self.assertEqual(buffer.stats()['pending'], 0)
            self.assertFalse(buffer.stats()['enabled'])
            camera_class.return_value.close.assert_called_once()

    def test_saves_original_pair_and_flange_once_per_timestamp(self):
        collection = importlib.import_module('training.data_collection')
        color = np.full((8, 12, 3), [10, 100, 200], dtype=np.uint8)
        packet = dict(timestamp_us=123456, color=color,
                      depth=np.full((8, 12), 1234, dtype=np.uint16),
                      flange=dict(matched_timestamp_us=123450, time_diff_us=6, TF=np.eye(4)))
        with tempfile.TemporaryDirectory() as directory:
            writer = collection.FrameWriter(directory)
            self.assertTrue(writer.save(packet))
            # A repeated timestamp stays a duplicate even if the image changes.
            self.assertFalse(writer.save(dict(packet, color=np.zeros_like(color))))
            rgb_files = list((Path(directory) / 'rgb').glob('*.png'))
            self.assertEqual(len(rgb_files), 1)
            filename = rgb_files[0].name
            np.testing.assert_array_equal(cv2.imread(str(rgb_files[0])), color[:, :, ::-1])
            depth = cv2.imread(str(Path(directory) / 'depth' / filename), cv2.IMREAD_UNCHANGED)
            self.assertEqual(depth.dtype, np.uint16)
            self.assertTrue((depth == 1234).all())
            meta = json.loads((Path(directory) / 'flange' / (rgb_files[0].stem + '.json')).read_text())
            self.assertEqual(meta['timestamp_us'], 123456)
            self.assertEqual(meta['flange']['time_diff_us'], 6)
            self.assertEqual(meta['flange']['TF'], np.eye(4).tolist())
            self.assertTrue(writer.save(dict(packet, timestamp_us=123457, flange=None)))
            missing = next((Path(directory) / 'flange').glob('melon_123457_*.json'))
            self.assertIsNone(json.loads(missing.read_text())['flange'])

    def test_write_failure_does_not_count_or_mark_timestamp_saved(self):
        collection = importlib.import_module('training.data_collection')
        packet = dict(timestamp_us=1, color=np.zeros((8, 12, 3), np.uint8),
                      depth=np.ones((8, 12), np.uint16), flange=None)
        with tempfile.TemporaryDirectory() as directory:
            writer = collection.FrameWriter(directory)
            with patch.object(collection.cv2, 'imwrite', return_value=False):
                with self.assertRaises(OSError):
                    writer.save(packet)
            self.assertEqual(writer.saved_count, 0)
            self.assertTrue(writer.save(packet))


if __name__ == '__main__':
    unittest.main()
