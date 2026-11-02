"""Per-object robot-base tracking. Capture-time updates; servo-time prediction.

Run after main.py and the detector: python detector/tracker.py --help
No robot commands are sent here. A servo consumer predicts the selected ID at
its desired PC timestamp rather than reusing a 16 Hz measured position.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import signal
import secrets
import sys
import threading
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.shm_detection import DetectionShm, MAX_OBJECTS, OBJECT_DTYPE
from common.shm_tracking import TrackingShm, TRACK_DTYPE


@dataclass
class Track:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    pose: np.ndarray
    score: float
    bbox: np.ndarray
    last_seen: int
    hits: int = 1
    confirmed: bool = False
    observed: bool = True


class MultiObjectTracker:
    def __init__(self, max_distance=.10, max_unseen=1.0, min_hits=3,
                 measurement_std=.005, acceleration_std=1., initial_velocity_std=1.,
                 input_timeout=.30, workspace_radius=1.4):
        values = (max_distance, max_unseen, measurement_std, acceleration_std, initial_velocity_std, input_timeout, workspace_radius)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Distances, times and noise standard deviations must be finite and positive")
        if not isinstance(min_hits, int) or min_hits < 1:
            raise ValueError("min_hits must be a positive integer")
        self.max_distance = max_distance
        self.max_unseen = max_unseen
        self.input_timeout = input_timeout
        self.workspace_radius = workspace_radius
        self.output_timestamp = 0
        self.min_hits = min_hits
        self.measurement_std = measurement_std
        self.acceleration_std = acceleration_std
        self.initial_velocity_std = initial_velocity_std
        self.tracks = []
        self.timestamp = 0
        self.next_id = 1
        self.session_id = secrets.randbits(64) or 1

    def reset(self):
        self.tracks.clear()
        self.timestamp = 0
        # Keep IDs monotonic within this process across loss/reset.

    def expire(self, timestamp):
        """Confirmed IDs live while their predicted base position is in reach."""
        dt = max(0., (timestamp - self.timestamp) / 1e6)
        self.tracks = [t for t in self.tracks
                       if np.linalg.norm(t.state[:3] + t.state[3:] * dt) <= self.workspace_radius
                       and (t.confirmed or (timestamp - t.last_seen) / 1e6 <= self.max_unseen)]

    def motion(self, dt):
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * dt
        gain = np.vstack((np.eye(3) * dt * dt / 2, np.eye(3) * dt))
        return transition, gain @ gain.T * self.acceleration_std ** 2

    def update(self, timestamp, detections):
        rows = np.asarray(detections, dtype=OBJECT_DTYPE)
        if rows.ndim != 1 or len(rows) > MAX_OBJECTS:
            raise ValueError("Expected at most eight detections")
        if (not isinstance(timestamp, (int, np.integer)) or timestamp <= self.timestamp
                or not all(np.isfinite(rows[name]).all() for name in ("TF", "bbox", "score"))):
            raise ValueError("Expected finite detections and a positive increasing capture timestamp")
        dt = (timestamp - self.timestamp) / 1e6 if self.timestamp else 0.
        # Let a new observation correct an edge-of-workspace prediction before
        # dropping it. Tentative candidates still expire to avoid permanent noise IDs.
        self.tracks = [t for t in self.tracks
                       if t.confirmed or (timestamp - t.last_seen) / 1e6 <= self.max_unseen]
        transition, process_noise = self.motion(dt)
        for track in self.tracks:
            track.state = transition @ track.state
            track.covariance = transition @ track.covariance @ transition.T + process_noise
            track.observed = False

        matched = set()
        if self.tracks and len(rows):
            distances = np.linalg.norm(
                np.array([t.state[:3] for t in self.tracks])[:, None, :]
                - rows['TF'][None, :, :3, 3], axis=2)
            n = len(self.tracks)
            # Dummy columns allow unmatched tracks. Gate before assignment so an
            # impossible pair cannot displace two feasible matches.
            cost = np.full((n, len(rows) + n), n + 1., dtype=float)
            cost[:, :len(rows)] = np.where(
                distances <= self.max_distance, distances / self.max_distance, 1e6)
            track_indices, detection_indices = linear_sum_assignment(cost)
            for ti, di in zip(track_indices, detection_indices):
                if di >= len(rows) or distances[ti, di] > self.max_distance:
                    continue
                track, row = self.tracks[ti], rows[di]
                noise = np.eye(3) * self.measurement_std ** 2
                innovation_cov = track.covariance[:3, :3] + noise
                kalman_gain = np.linalg.solve(innovation_cov, track.covariance[:3, :]).T
                track.state += kalman_gain @ (row['TF'][:3, 3] - track.state[:3])
                residual = np.eye(6)
                residual[:, :3] -= kalman_gain
                track.covariance = (residual @ track.covariance @ residual.T
                                    + kalman_gain @ noise @ kalman_gain.T)
                track.covariance = (track.covariance + track.covariance.T) / 2
                track.pose = row['TF'].copy()
                track.score = float(row['score'])
                track.bbox = row['bbox'].copy()
                track.last_seen = int(timestamp)
                track.hits += 1
                track.confirmed |= track.hits >= self.min_hits
                track.observed = True
                matched.add(di)
        for track in self.tracks:
            if not track.observed:
                track.hits = 0
        for di in np.argsort(-rows['score'], kind='stable'):
            if di in matched or np.linalg.norm(rows[di]["TF"][:3, 3]) > self.workspace_radius:
                continue
            if len(self.tracks) == MAX_OBJECTS:
                missing = [t for t in self.tracks if not t.observed and not t.confirmed]
                if not missing:
                    break
                self.tracks.remove(min(missing, key=lambda t: (t.confirmed, t.last_seen)))
            row = rows[di]
            state = np.r_[row['TF'][:3, 3], np.zeros(3)]
            covariance = np.diag([self.measurement_std ** 2] * 3
                                 + [self.initial_velocity_std ** 2] * 3)
            self.tracks.append(Track(self.next_id, state, covariance, row['TF'].copy(),
                                     float(row['score']), row['bbox'].copy(), int(timestamp),
                                     confirmed=self.min_hits == 1))
            self.next_id += 1
        self.timestamp = int(timestamp)
        self.output_timestamp = self.timestamp
        self.expire(timestamp)
        return self.snapshot()

    def snapshot(self, timestamp=None):
        dt = 0. if timestamp is None else (timestamp - self.timestamp) / 1e6
        if dt < 0:
            raise ValueError("Snapshot cannot precede the filter state")
        transition, process_noise = self.motion(dt)
        rows = np.zeros(len(self.tracks), dtype=TRACK_DTYPE)
        for row, track in zip(rows, sorted(self.tracks, key=lambda t: t.track_id)):
            row['track_id'] = track.track_id
            row['session_id'] = self.session_id
            row['confirmed'], row['observed'] = track.confirmed, track.observed
            row['last_seen'], row['score'] = track.last_seen, track.score
            row['bbox'] = track.bbox
            row['TF'] = track.pose
            state = transition @ track.state
            row['TF'][:3, 3] = state[:3]
            row['velocity'] = state[3:]
            row['covariance'] = transition @ track.covariance @ transition.T + process_noise
        return rows


def predict_pose(track, state_timestamp, target_timestamp, max_age=None, workspace_radius=1.4, max_state_age=.30):
    """Predict a confirmed track at PC time (µs) while inside the workspace.

    max_age=None permits unobserved coasting; set seconds to enforce freshness.

    This does not select a picking target or check robot reachability. Orientation
    is held at the last observation; rotational velocity is not estimated.
    """
    if (not isinstance(state_timestamp, (int, np.integer))
            or not isinstance(target_timestamp, (int, np.integer))
            or state_timestamp <= 0 or int(track['last_seen']) <= 0
            or not track['confirmed']
            or not np.isfinite(workspace_radius) or workspace_radius <= 0
            or not np.isfinite(max_state_age) or max_state_age <= 0
            or (target_timestamp - state_timestamp) / 1e6 > max_state_age
            or (max_age is not None and (not np.isfinite(max_age) or max_age <= 0))
            or state_timestamp < int(track['last_seen'])
            or target_timestamp < state_timestamp
            or (max_age is not None and (target_timestamp - int(track['last_seen'])) / 1e6 > max_age)
            or not np.isfinite(track['TF']).all() or not np.isfinite(track['velocity']).all()):
        raise ValueError("Track must be confirmed, finite and satisfy the requested time limits")
    pose = track['TF'].copy()
    pose[:3, 3] += track['velocity'] * ((target_timestamp - state_timestamp) / 1e6)
    if np.linalg.norm(pose[:3, 3]) > workspace_radius:
        raise ValueError("Predicted position is outside the robot-base workspace sphere")
    return pose


def process_frame(tracker, frame, now_us):
    """Return (tracks, valid-input status), or None for an already processed frame."""
    if now_us < tracker.timestamp:
        tracker.reset()
    age = (now_us - frame.timestamp) / 1e6
    if frame.timestamp > now_us:
        tracker.output_timestamp = 0
        return np.empty(0, dtype=TRACK_DTYPE), False
    if not frame.status or frame.timestamp <= 0 or not 0 <= age <= tracker.input_timeout:
        tracker.expire(now_us)
        for track in tracker.tracks:
            track.observed = False
            track.hits = 0
        tracker.output_timestamp = now_us
        return tracker.snapshot(now_us), False
    if frame.timestamp == tracker.timestamp:
        return None
    if frame.timestamp < tracker.timestamp:
        tracker.reset()
    try:
        return tracker.update(frame.timestamp, frame.objects), True
    except ValueError:
        tracker.expire(now_us)
        for track in tracker.tracks:
            track.observed = False
            track.hits = 0
        tracker.output_timestamp = now_us
        return tracker.snapshot(now_us), False


def run(tracker, stop_event):
    """Attach only; main.py owns both segments. Polling is not a servo loop."""
    try:
        source = DetectionShm()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Detection SHM 연결 실패: {exc}. main.py를 먼저 실행하세요.")
        return 1
    try:
        destination = TrackingShm()
    except (FileNotFoundError, RuntimeError) as exc:
        source.close()
        print(f"Tracking SHM 연결 실패: {exc}. main.py를 먼저 실행하세요.")
        return 1
    next_poll = 0.
    try:
        print("[Tracker] detection_shm → tracking_shm (robot-base m, m/s)")
        while not stop_event.is_set():
            frame = source.read()
            now_us = time.time_ns() // 1000
            new_frame = frame.status and frame.timestamp > 0 and frame.timestamp != tracker.timestamp
            poll_time = time.monotonic()
            if new_frame or poll_time >= next_poll:
                result = process_frame(tracker, frame, now_us)
                next_poll = poll_time + .05  # 20 Hz coast publication during input loss.
                if result is not None:
                    rows, valid = result
                    destination.write(tracker.output_timestamp, frame.sequence_id, rows, status=valid)
            stop_event.wait(.001)
    finally:
        destination.write(0, 0, [], status=False)
        destination.close()
        source.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description="3D Hungarian + constant-velocity Kalman tracker")
    parser.add_argument('--max-distance', type=float, default=.10, help='3D matching gate, meters')
    parser.add_argument('--max-unseen', type=float, default=1.0, help='Unconfirmed candidate timeout, seconds; confirmed IDs use workspace bounds')
    parser.add_argument('--workspace-radius', type=float, default=1.4, help='Robot-base sphere radius, meters')
    parser.add_argument('--input-timeout', type=float, default=.30, help='Reject stale input for control, seconds')
    parser.add_argument('--min-hits', type=int, default=3, help='Consecutive matches before confirmation')
    parser.add_argument('--measurement-std', type=float, default=.005, help='Position noise std, meters')
    parser.add_argument('--acceleration-std', type=float, default=1., help='Acceleration noise std, m/s²')
    parser.add_argument('--initial-velocity-std', type=float, default=1., help='Initial velocity std, m/s')
    args = parser.parse_args()
    try:
        tracker = MultiObjectTracker(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))
    stop_event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop_event.set())
    return run(tracker, stop_event)


if __name__ == '__main__':
    raise SystemExit(main())
