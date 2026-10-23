import tempfile
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from queue import Empty
from types import SimpleNamespace

import numpy as np

from training import recording


class RecordingTest(unittest.TestCase):
    def test_pause_waits_for_inflight_copy_before_drain(self):
        buffer = recording.RecordingBuffer()
        buffer.configure(True)
        copying, release, paused = threading.Event(), threading.Event(), threading.Event()

        class DelayedColor:
            def copy(self):
                copying.set()
                release.wait(timeout=2)
                return np.zeros((2, 3, 3), np.uint8)

        producer = threading.Thread(target=buffer.offer, args=(
            1, DelayedColor(), np.ones((2, 3), np.uint16), None))
        def pause():
            buffer.configure(False)
            paused.set()

        producer.start()
        self.assertTrue(copying.wait(timeout=1))
        stopper = threading.Thread(target=pause)
        stopper.start()
        try:
            self.assertFalse(paused.wait(timeout=0.05))
        finally:
            release.set()
            producer.join(timeout=2)
            stopper.join(timeout=2)
        self.assertTrue(paused.is_set())
        self.assertEqual(buffer.take(timeout=0)['timestamp_us'], 1)
        buffer.offer(2, np.zeros((2, 3, 3), np.uint8), np.ones((2, 3), np.uint16), None)
        with self.assertRaises(Empty):
            buffer.take(timeout=0)

    def test_bounded_queue_copies_frames_and_flange_without_blocking(self):
        buffer = recording.RecordingBuffer(maxsize=1)
        color = np.zeros((2, 3, 3), dtype=np.uint8)
        depth = np.full((2, 3), 1234, dtype=np.uint16)
        tf = np.eye(4)
        flange = SimpleNamespace(status=True, read=lambda ts: SimpleNamespace(
            matched_ts=ts - 2, time_diff_us=2, TF=tf))
        buffer.offer(100, color, depth, flange)
        with self.assertRaises(Empty):
            buffer.take(timeout=0)
        buffer.configure(True)
        buffer.offer(100, color, depth, flange)
        buffer.offer(101, color, depth, flange)
        color[:] = 99
        depth[:] = 0
        tf[:] = 0
        packet = buffer.take(timeout=0)
        self.assertEqual(packet['timestamp_us'], 100)
        self.assertEqual(packet['color'].max(), 0)
        self.assertEqual(packet['depth'].min(), 1234)
        np.testing.assert_array_equal(packet['flange']['TF'], np.eye(4))
        self.assertEqual(packet['flange']['matched_timestamp_us'], 98)
        self.assertEqual(buffer.stats()['dropped'], 1)
        buffer.offer(102, color, depth, None)
        self.assertIsNone(buffer.take(timeout=0)['flange'])

    def test_manager_transfers_frame_and_preserves_queue_on_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            address = str(Path(directory) / 'recorder.sock')
            with recording.RecordingServer(address=address, maxsize=2) as server:
                manager, remote = recording.connect(address)
                remote.configure(True)
                server.buffer.offer(7, np.zeros((2, 3, 3), np.uint8),
                                    np.ones((2, 3), np.uint16), None)
                remote.configure(False)
                packet = remote.take(timeout=1)
                self.assertEqual(packet['timestamp_us'], 7)
                self.assertEqual(packet['depth'].dtype, np.uint16)
                self.assertFalse(remote.stats()['enabled'])
                with self.assertRaises(Empty):
                    remote.take(timeout=0)

    def test_separate_process_receives_numpy_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            address = str(Path(directory) / 'recorder.sock')
            with recording.RecordingServer(address=address) as server:
                server.buffer.configure(True)
                server.buffer.offer(42, np.zeros((2, 3, 3), np.uint8),
                                    np.full((2, 3), 1234, np.uint16), None)
                source = f'''
from training.recording import connect
manager, buffer = connect({address!r})
packet = buffer.take(timeout=1)
assert packet['timestamp_us'] == 42
assert (packet['depth'] == 1234).all()
assert packet['flange'] is None
buffer.configure(False)
'''
                subprocess.run([sys.executable, '-c', source], check=True, timeout=10)
                self.assertFalse(server.buffer.enabled.is_set())


if __name__ == '__main__':
    unittest.main()
