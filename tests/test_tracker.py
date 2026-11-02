import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np

from common.shm_detection import OBJECT_DTYPE
from detector.tracker import MultiObjectTracker, predict_pose, process_frame


def detections(*positions):
    rows = np.zeros(len(positions), dtype=OBJECT_DTYPE)
    rows['score'] = .9
    rows['TF'] = np.eye(4)
    for row, position in zip(rows, positions):
        row['TF'][:3, 3] = position
    return rows


class TrackerTest(unittest.TestCase):
    def test_id_survives_detection_reordering_and_velocity_converges(self):
        tracker = MultiObjectTracker()
        first = tracker.update(1000000, detections([0, 0, 1], [.5, 0, 1]))
        ids = first['track_id'].copy()
        for step in range(1, 41):
            t = step / 16
            rows = detections([.1 * t, 0, 1], [.5 + .1 * t, 0, 1])
            result = tracker.update(1000000 + step * 62500, rows[::-1])
        np.testing.assert_array_equal(result['track_id'], ids)
        np.testing.assert_allclose(result['velocity'], [[.1, 0, 0]] * 2, atol=.003)
        np.testing.assert_allclose(result['TF'][:, :3, 3], [[.25, 0, 1], [.75, 0, 1]], atol=.002)
        self.assertTrue(result['confirmed'].all())
        self.assertTrue(result['observed'].all())
        for covariance in result['covariance']:
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-12)

    def test_variable_dt_missed_frame_reacquisition_and_expiry(self):
        tracker = MultiObjectTracker(max_unseen=.30)
        for offset in [0, 62500, 140000, 200000]:
            result = tracker.update(1000000 + offset, detections([offset / 1e6 * .2, 0, 1]))
        track_id = result['track_id'][0]
        missed = tracker.update(1250000, detections())
        self.assertFalse(missed['observed'][0])
        self.assertGreater(missed['TF'][0, 0, 3], result['TF'][0, 0, 3])
        self.assertEqual(missed['last_seen'][0], 1200000)
        recovered = tracker.update(1300000, detections([.06, 0, 1]))
        self.assertEqual(recovered['track_id'][0], track_id)
        self.assertEqual(len(tracker.update(1700000, detections())), 1)
        self.assertEqual(len(tracker.update(10000000, detections())), 0)
        new = tracker.update(11000000, detections([.16, 0, 1]))
        self.assertNotEqual(new['track_id'][0], track_id)

    def test_gating_is_part_of_assignment_and_matching_is_one_to_one(self):
        tracker = MultiObjectTracker(max_distance=.1)
        first = tracker.update(1000000, detections([0, 0, 1], [.09, 0, 1]))
        second = tracker.update(1062500, detections([.01, 0, 1], [0, .09, 1]))
        self.assertEqual(len(second), 2)
        np.testing.assert_array_equal(first['track_id'], second['track_id'])
        self.assertTrue(second['observed'].all())
        self.assertEqual(len(set(second['track_id'])), 2)

    def test_bounded_capacity_validation_and_monotonic_timestamps(self):
        tracker = MultiObjectTracker()
        positions = [[i * .1, 0, .5] for i in range(8)]
        initial = tracker.update(1000000, detections(*positions))
        self.assertEqual(len(initial), 8)
        with self.assertRaises(ValueError):
            tracker.update(1000000, detections())
        with self.assertRaises(ValueError):
            tracker.update(900000, detections())
        with self.assertRaises(ValueError):
            tracker.update(1062500, detections(*positions, [2, 0, 1]))
        malformed = detections([np.nan, 0, 1])
        with self.assertRaises(ValueError):
            tracker.update(1062500, malformed)
        self.assertEqual(tracker.timestamp, 1000000)
        replacement = tracker.update(1125000, detections([1.2, 0, .5]))
        self.assertEqual(len(replacement), 8)
        self.assertEqual(replacement['observed'].sum(), 1)
        tracker.reset()
        new = tracker.update(500000, detections([0, 0, 1]))
        self.assertGreater(new['track_id'][0], initial['track_id'].max())

    def test_servo_prediction_rejects_unconfirmed_and_stale_tracks(self):
        tracker = MultiObjectTracker()
        row = tracker.update(1000000, detections([0, 0, 1]))[0]
        with self.assertRaises(ValueError):
            predict_pose(row, 1000000, 1002000)
        for step in range(1, 10):
            row = tracker.update(1000000 + step * 62500,
                                 detections([.1 * step / 16, 0, 1]))[0]
        timestamp = tracker.timestamp
        pose = predict_pose(row, timestamp, timestamp + 2000)
        np.testing.assert_allclose(pose[:3, 3], row['TF'][:3, 3] + row['velocity'] * .002)
        with self.assertRaises(ValueError):
            predict_pose(row, timestamp, timestamp + 400000, max_age=.30)
        with self.assertRaises(ValueError):
            predict_pose(row, timestamp, timestamp - 1)

    def test_runtime_deduplicates_and_invalidates_stalled_or_inactive_input(self):
        from common.shm_detection import DetectionFrame
        tracker = MultiObjectTracker()
        frame = DetectionFrame(True, 1000000, 1, detections([0, 0, 1]))
        rows, status = process_frame(tracker, frame, 1010000)
        self.assertTrue(status)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(process_frame(tracker, frame, 1020000))
        rows, status = process_frame(tracker, frame, 1400000)
        self.assertFalse(status)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows["observed"][0])
        frame.status = False
        self.assertFalse(process_frame(tracker, frame, 1400000)[1])

    def test_coasting_prediction_uses_state_epoch_but_last_observation_freshness(self):
        tracker = MultiObjectTracker(min_hits=1)
        for step in range(10):
            tracker.update(1000000 + step * 62500, detections([step * .00625, 0, 1]))
        row = tracker.update(1625000, detections())[0]
        self.assertFalse(row['observed'])
        pose = predict_pose(row, 1625000, 1627000)
        np.testing.assert_allclose(pose[:3, 3], row['TF'][:3, 3] + .002 * row['velocity'])
        # 250 ms after state, but 312.5 ms after last actual observation.
        with self.assertRaises(ValueError):
            predict_pose(row, 1625000, 1875000, max_age=.30)

    def test_capacity_replaces_a_later_coasting_track_without_disturbing_matched_ids(self):
        tracker = MultiObjectTracker()
        initial = tracker.update(1000000, detections(*[[i * .1, 0, .5] for i in range(8)]))
        result = tracker.update(1062500, detections(*[[i * .1, 0, .5] for i in range(7)], [1.2, 0, .5]))
        np.testing.assert_array_equal(result['track_id'][:7], initial['track_id'][:7])
        self.assertNotIn(initial['track_id'][-1], result['track_id'])
        self.assertEqual(result['observed'].sum(), 8)

    def test_runtime_resets_on_clock_rollback_and_rejects_future_capture(self):
        from common.shm_detection import DetectionFrame
        tracker = MultiObjectTracker()
        first, _ = process_frame(tracker, DetectionFrame(True, 1000000, 1,
                                 detections([0, 0, 1])), 1001000)
        rolled, valid = process_frame(tracker, DetectionFrame(True, 900000, 2,
                                      detections([0, 0, 1])), 901000)
        self.assertTrue(valid)
        self.assertGreater(rolled['track_id'][0], first['track_id'][0])
        empty, valid = process_frame(tracker, DetectionFrame(True, 2000000, 3,
                                    detections([0, 0, 1])), 1000000)
        self.assertFalse(valid)
        self.assertEqual(len(empty), 0)

    def test_short_invalid_and_stale_gaps_preserve_id_without_valid_output(self):
        from common.shm_detection import DetectionFrame
        tracker = MultiObjectTracker(min_hits=1)
        first, _ = process_frame(tracker, DetectionFrame(True, 1000000, 1,
                                 detections([0, 0, 1])), 1001000)
        invalid = DetectionFrame(False, 0, 1, detections())
        rows, valid = process_frame(tracker, invalid, 1100000)
        self.assertFalse(valid)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows["observed"][0])
        second, valid = process_frame(tracker, DetectionFrame(True, 1400000, 2,
                                      detections([0, 0, 1])), 1401000)
        self.assertTrue(valid)
        self.assertEqual(second['track_id'][0], first['track_id'][0])
        rows, valid = process_frame(tracker, DetectionFrame(True, 1400000, 2,
                                    detections([0, 0, 1])), 1800000)
        self.assertFalse(valid)
        resumed, valid = process_frame(tracker, DetectionFrame(True, 1900000, 3,
                                       detections([0, 0, 1])), 1901000)
        self.assertTrue(valid)
        self.assertEqual(resumed['track_id'][0], first['track_id'][0])
        process_frame(tracker, invalid, 3000000)
        self.assertEqual(len(tracker.tracks), 1)

    def test_workspace_retention_and_optional_servo_freshness_are_separate(self):
        tracker = MultiObjectTracker(min_hits=1)
        first = tracker.update(1000000, detections([0, 0, 1]))
        coasted = tracker.update(1500000, detections())
        self.assertEqual(coasted['track_id'][0], first['track_id'][0])
        self.assertFalse(coasted['observed'][0])
        with self.assertRaises(ValueError):
            predict_pose(coasted[0], 1500000, 1502000, max_age=.30)
        self.assertEqual(len(tracker.update(2100000, detections())), 1)

    def test_confirmed_coasting_tracks_are_not_evicted_for_jitter_births(self):
        tracker = MultiObjectTracker(min_hits=1)
        first = tracker.update(1000000, detections(*[[i * .1, 0, .5] for i in range(8)]))
        second = tracker.update(1062500, detections(*[[i * .1, 0, .5] for i in range(7)], [1.2, 0, .5]))
        np.testing.assert_array_equal(second['track_id'], first['track_id'])
        self.assertFalse(second['observed'][-1])

    def test_confirmed_track_survives_long_camera_absence_inside_workspace(self):
        tracker = MultiObjectTracker(min_hits=1)
        initial = tracker.update(1000000, detections([.6, .2, .3]))
        lost = tracker.update(61000000, detections())
        self.assertEqual(lost['track_id'][0], initial['track_id'][0])
        self.assertFalse(lost['observed'][0])
        self.assertEqual(lost['last_seen'][0], 1000000)
        pose = predict_pose(lost[0], 61000000, 61002000)
        np.testing.assert_allclose(pose[:3, 3], [.6, .2, .3])
        recovered = tracker.update(61062500, detections([.6, .2, .3]))
        self.assertEqual(recovered['track_id'][0], initial['track_id'][0])
        self.assertTrue(recovered['observed'][0])

    def test_workspace_uses_predicted_spherical_radius_and_tentative_timeout(self):
        tracker = MultiObjectTracker(min_hits=1)
        tracker.update(1000000, detections([.5, 0, .5]))
        tracker.tracks[0].state[3] = .5
        self.assertEqual(len(tracker.update(1100000, detections())), 1)
        self.assertEqual(len(tracker.update(4000000, detections())), 0)
        self.assertEqual(len(tracker.update(4100000, detections([1., 1., 0.]))), 0)
        tracker = MultiObjectTracker()
        tracker.update(1000000, detections([0, 0, 1]))
        self.assertEqual(len(tracker.update(2100000, detections())), 0)

    def test_stalled_input_exposes_coasting_at_now_without_corrupting_capture_epoch(self):
        from common.shm_detection import DetectionFrame
        tracker = MultiObjectTracker(min_hits=1)
        frame = DetectionFrame(True, 1000000, 1, detections([.5, 0, .5]))
        initial, _ = process_frame(tracker, frame, 1001000)
        tracker.tracks[0].state[3] = .01
        rows, valid = process_frame(tracker, frame, 5000000)
        self.assertFalse(valid)
        self.assertEqual(rows['track_id'][0], initial['track_id'][0])
        self.assertFalse(rows['observed'][0])
        self.assertAlmostEqual(rows['TF'][0, 0, 3], .54)
        self.assertGreater(rows['covariance'][0, 0, 0], initial['covariance'][0, 0, 0])
        self.assertEqual(tracker.timestamp, 1000000)
        self.assertEqual(tracker.output_timestamp, 5000000)
        resumed, valid = process_frame(tracker, DetectionFrame(True, 4990000, 2,
                                       detections([.54, 0, .5])), 5001000)
        self.assertTrue(valid)
        self.assertEqual(resumed['track_id'][0], initial['track_id'][0])

    def test_old_observation_allowed_but_stalled_tracker_state_rejected(self):
        tracker = MultiObjectTracker(min_hits=1)
        tracker.update(1000000, detections([0, 0, 1]))
        row = tracker.update(61000000, detections())[0]
        predict_pose(row, 61000000, 61002000)
        with self.assertRaises(ValueError):
            predict_pose(row, 61000000, 62000000)

    def test_servo_prediction_rejects_workspace_exit_even_without_age_limit(self):
        tracker = MultiObjectTracker(min_hits=1)
        row = tracker.update(1000000, detections([1., 0, 0]))[0]
        row['velocity'] = [5., 0, 0]
        with self.assertRaises(ValueError):
            predict_pose(row, 1000000, 1100000)

    def test_process_loop_publishes_once_and_clears_on_stall_and_shutdown(self):
        from common.shm_detection import DetectionFrame
        from detector import tracker as module
        frame = DetectionFrame(True, 1000000, 1, detections([0, 0, 1]))
        source, destination = Mock(), Mock()
        source.read.return_value = frame
        stop = Mock()
        stop.is_set.side_effect = [False, False, False, True]
        with patch.object(module, 'DetectionShm', return_value=source) as attach_source, patch.object(
            module, 'TrackingShm', return_value=destination
        ) as attach_destination, patch.object(module.time, 'time_ns', side_effect=[
            1010000000, 1020000000, 1400000000
        ]), patch.object(module.time, 'monotonic', side_effect=[0., .01, .4]):
            self.assertEqual(module.run(MultiObjectTracker(), stop), 0)
        attach_source.assert_called_once_with()
        attach_destination.assert_called_once_with()
        self.assertEqual(destination.write.call_count, 3)
        first, stalled, shutdown = destination.write.call_args_list
        self.assertTrue(first.kwargs['status'])
        self.assertEqual(len(first.args[2]), 1)
        self.assertFalse(stalled.kwargs['status'])
        self.assertFalse(shutdown.kwargs['status'])
        source.close.assert_called_once()
        destination.close.assert_called_once()

    def test_runner_handles_clock_rollback_even_with_same_source_timestamp(self):
        from common.shm_detection import DetectionFrame
        from detector import tracker as module
        source, destination, stop = Mock(), Mock(), Mock()
        source.read.return_value = DetectionFrame(True, 1000000, 1, detections([0, 0, 1]))
        stop.is_set.side_effect = [False, False, True]
        with patch.object(module, 'DetectionShm', return_value=source), patch.object(
            module, 'TrackingShm', return_value=destination
        ), patch.object(module.time, 'time_ns', side_effect=[1010000000, 900000000]), patch.object(
            module.time, 'monotonic', side_effect=[10., 10.1]
        ):
            module.run(MultiObjectTracker(min_hits=1), stop)
        self.assertEqual(destination.write.call_count, 3)
        self.assertFalse(destination.write.call_args_list[1].kwargs['status'])

    def test_confirmation_requires_consecutive_observations_and_sessions_change(self):
        tracker = MultiObjectTracker()
        tracker.update(1000000, detections([0, 0, 1]))
        tracker.update(1062500, detections())
        for timestamp in (1125000, 1187500):
            row = tracker.update(timestamp, detections([0, 0, 1]))[0]
            self.assertFalse(row['confirmed'])
        row = tracker.update(1250000, detections([0, 0, 1]))[0]
        self.assertTrue(row['confirmed'])
        second = MultiObjectTracker().update(1000000, detections([0, 0, 1]))[0]
        self.assertNotEqual(row['session_id'], second['session_id'])



class TrackingShmTest(unittest.TestCase):
    def test_tracking_records_roundtrip_and_supervisor_owns_lifetime(self):
        from common.shm_tracking import TrackingShm
        owner = TrackingShm('tracking_test_' + uuid4().hex, is_owner=True)
        with patch('common.shm_detection.unregister'):
            client = TrackingShm(owner.shm.name)
        try:
            rows = MultiObjectTracker().update(1000000, detections([0, 0, 1]))
            client.write(1000000, 1, rows)
            result = owner.read()
            np.testing.assert_array_equal(result.objects, rows)
            self.assertEqual(result.objects['last_seen'][0], 1000000)
            name = owner.shm.name
            client.close()
            with patch('common.shm_detection.unregister'):
                probe = TrackingShm(name)
            probe.close()
        finally:
            client.close()
            owner.close()
        with self.assertRaises(FileNotFoundError):
            TrackingShm(name)


if __name__ == '__main__':
    unittest.main()
