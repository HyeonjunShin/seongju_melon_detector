# Per-object robot-base tracker

Run these in separate terminals after restarting the old processes:

```bash
python main.py
python camera/camera_runner.py
python detector/xpu_detection.py
python detector/tracker.py
```

The existing robot flange receiver must also be running. Camera and detector
supply timestamped observations (the tracker does not assume a fixed frame rate); the tracker polls for new snapshots without
consuming the camera semaphore. Run one tracker process at a time.

`main.py`/`ProductIpc` alone creates and unlinks `tracking_shm`. Tracker and servo
consumers only attach. Restart all attached clients when main.py restarts.
On tracker termination the segment remains, with status=False and no objects.
The dashboard includes a TRACKER position/velocity table.

## Matching and estimation

Each observation is already in robot-base coordinates. A six-state Kalman filter
estimates `[x,y,z,vx,vy,vz]` independently for each object using the actual capture
timestamp difference, in seconds. Hungarian assignment uses Euclidean distance
from the predicted position. The distance gate is applied before assignment;
unmatched dummy slots prevent impossible pairs from stealing feasible matches.

Default options are starting values, not measured bounds for this installation:

| Option | Default | Meaning |
| --- | --- | --- |
| `--max-distance` | 0.10 m | Maximum predicted-to-measured 3D distance |
| `--max-unseen` | 1.0 s | Timeout for unconfirmed candidates only |
| `--workspace-radius` | 1.4 m | Base-origin sphere for confirmed ID retention |
| `--input-timeout` | 0.30 s | Reject stale detector input without immediately forgetting IDs |
| `--min-hits` | 3 | Consecutive observations needed to confirm a new ID |
| `--measurement-std` | 0.005 m | Position measurement standard deviation |
| `--acceleration-std` | 1.0 m/s² | Independent per-step acceleration noise std |
| `--initial-velocity-std` | 1.0 m/s | Initial velocity uncertainty |

For example:

```bash
python detector/tracker.py --workspace-radius 1.4 --max-distance 0.10
```

Increase the distance gate if valid motion cannot be associated, but a larger
gate raises ID-switch risk near other melons. Measure static base-frame position
noise and actual motion before tuning the noise values. Identical objects
crossing or disappearing together can still exchange IDs; position matching
alone cannot guarantee identity.

Confirmed IDs have no age-based expiry while their predicted base position is
inside the sphere `x²+y²+z² <= 1.4²`. This covers leaving the camera field of view.
Predictions use the last estimated velocity; they are not new measurements.
A newly observed pose can correct an edge prediction before workspace removal.
New detections outside the sphere are ignored. Tentative candidates still expire
after max_unseen; they must meet min_hits to become persistent.

With all eight slots occupied, new observations can replace unconfirmed coasting
tracks; confirmed identities keep their slots until leaving the sphere. If eight
persistent tracks remain inside, new object registration waits for a free slot.
A picking controller will need an explicit completed-target removal policy when
that control integration is implemented.

Healthy empty detection frames advance predictions at the input rate. Missing,
inactive or stale detector input now publishes retained predictions at 20 Hz,
with status=False and observed=False. The output timestamp is the prediction
epoch, while last_seen stays at the last actual measurement. The internal filter
still stays at its last processed capture epoch, so delayed new frames do not
cause a false clock-rollback reset. Backward capture timestamps reset identities.
Coast scheduling uses monotonic time; all data epochs remain PC wall-clock µs.
Duplicate capture timestamps are not processed twice.

Position and linear velocity are filtered. Orientation is the latest observed
image Body-to-Knot/+Z-normal orientation (depth from Body only); it is held during prediction, not filtered
and not extrapolated with angular velocity.

## Shared-memory contract

Use `from common.shm_tracking import TrackingShm`. `read()` returns the same
snapshot envelope as DetectionShm with an extended object layout:

- `status`: detector input was usable at the last tracker update. This alone does
  not mean that a track is confirmed or suitable for picking.
- `timestamp`: state/prediction epoch, PC microseconds. For normal updates this
  is capture time; for input-loss coasting snapshots it is the prediction time.
- `sequence_id`: input camera sequence; `publication_count`: tracker updates.
- `objects` / `count`: zero to eight tracks, ordered by track ID.

Each track contains:

| Field | Meaning |
| --- | --- |
| `session_id`, `track_id` | Use this pair for identity; session changes on tracker process restart |
| `confirmed` | At least min_hits consecutive observations received |
| `observed` | Matched in this frame; False means coasting |
| `last_seen` | Last actual observation's capture timestamp, PC µs |
| `TF` | Filtered position (m), last observed orientation, robot-base frame |
| `velocity` | Base-frame `[vx,vy,vz]`, m/s |
| `covariance` | 6×6 uncertainty for `[x,y,z,vx,vy,vz]` at snapshot timestamp |
| `score`, `bbox` | Latest matched detection's confidence and pixel bbox |

The displayed sigma is the largest XYZ marginal standard deviation (mm). AGE(s)
is PC time since the last actual observation, not time since a prediction update.
ID counters continue across loss/reset inside one process. A new process starts
a new random session, so a picking target must not be silently reassigned to a
new session with the same numeric track ID.

## Servo-time prediction

The tracker updates at the detection rate (20 Hz coasting if input is lost), not 500 Hz. A separate servo controller
reads a snapshot and predicts its selected identity at each 2 ms command tick:

```python
import time
from common.shm_tracking import TrackingShm
from detector.tracker import predict_pose

buffer = TrackingShm()  # create once; close when controller exits
try:
    frame = buffer.read()
    # selected_session and selected_id must come from your target selection.
    matches = [obj for obj in frame.objects
               if int(obj['session_id']) == selected_session
               and int(obj['track_id']) == selected_id]
    # Set allow_coasting according to the controller's pursuit policy. A False
    # frame.status means detector input is unavailable, not that rows are empty.
    if matches and (frame.status or allow_coasting):
        target_time_us = time.time_ns() // 1000 + command_delay_us
        pose = predict_pose(matches[0], frame.timestamp, target_time_us,
                            max_age=None, workspace_radius=1.4, max_state_age=0.30)
        # Apply your controller's uncertainty/observation policy before using pose.
finally:
    buffer.close()
```

`predict_pose` rejects unconfirmed tracks, target times before the state time,
positions outside workspace_radius, and predictions more than max_state_age
(default 0.30 s) beyond the latest published state epoch. This last check detects
a stalled/dead tracker even when old observations are allowed. The controller can
set max_state_age for its target-time horizon. max_age defaults to None, allowing
old observations to coast; optionally set e.g. 0.30 to impose an observation-age
limit. These two ages serve different purposes.

The function preserves orientation and predicts position as
`filtered_xyz + velocity * dt`. It does not send robot commands, check actual arm
reachability/collisions or predict gripper closure. The 1.4 m sphere is the user's
retention boundary and is not the robot's full kinematic workspace. During long
absence the estimate may drift and covariance grows; monitor AGE and sigma and
reacquire the object when it re-enters the image. Reacquisition still uses the
configured distance gate; identical/crossing objects can exchange identities.
This Python process does not guarantee hard-real-time 2 ms scheduling.
