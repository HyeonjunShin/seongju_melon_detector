# Tracker input: detection_shm

Start main.py, the robot flange receiver, camera_runner.py, then
`python detector/xpu_detection.py`. Only main.py (ProductIpc) creates and unlinks `detection_shm`.
The detector attaches for writing and clears status/count on shutdown. Existing `target_shm` remains the single robot target.
Only one detector producer may run at a time. Consumers should be independent
processes, like the existing camera/flange SHM consumers.

```python
from common.shm_detection import DetectionShm

buffer = DetectionShm()  # attach after main.py startup
try:
    frame = buffer.read()
    if frame.status:
        for obj in frame.objects:  # frame.count is 0..8
            base_from_object = obj['TF']  # 4x4, translation in meters
            bbox = obj['bbox']           # original image pixel xyxy
            score = obj['score']
finally:
    buffer.close()
```

Each read returns an independent, locked snapshot. Use `sequence_id` to avoid
processing a frame twice and `timestamp` for camera capture time (microseconds).
The tracker should check sequence progress to detect a crashed/stalled producer.
The SHM remains available across detector restarts. Reconnect all clients after
main.py restarts.

Objects are sorted by confidence, with at most eight valid poses. An array index
is not a track ID. A healthy frame with no valid objects has status=True and
count=0. Missing camera input (100 ms semaphore timeout), inactive flange or
invalid calibration/flange data produces status=False and count=0.

Geometry reuses camera/test_picking.py: Body depth patch median, inward local
surface normal as +Z, and the image Body-to-Knot direction lifted into the
Body tangent plane as +X (+Y = +Z cross +X), then
`T_base_flange @ T_flange_camera @ T_camera_object`. The picking Z offset is
applied as well. FlangeShm chooses the nearest available timestamp, exactly as
picking does; no extra flange time-difference cutoff is introduced here.

Both keypoints require visibility >= 0.5, but only Body requires valid depth.
Knot is used solely as a 2D image direction. +X is the intersection of the Body
tangent plane and the viewing plane through the Body/Knot rays, oriented toward
Knot in the image. No Knot depth is read. A degenerate direction rejects the pose.

## Terminal monitor

`python main.py` shows colored status sections and a compact detection table:
score, base XYZ in meters and fixed-XYZ RPY in degrees. `python main.py --details`
also shows full 4x4 transforms and bounding boxes. Redirected output contains
no ANSI cursor/color codes. P1..P8 are per-frame display indices, not track IDs.

The monitor summary is ordered Robot, Camera, Detector, with STATUS, RATE (Hz),
PERIOD (ms), and TIMESTAMP (microseconds). Detector RATE uses publication_count,
not the input camera sequence: skipped camera frames do not inflate the rate.
PERIOD is the latest consecutive timestamp difference divided by 1000:
Robot uses the last two flange samples, Camera uses its producer's Color interval,
and Detector stores interval_us between consecutive processed input timestamps.
It is not the one-second dashboard polling interval or inference latency.
The first sample, non-increasing timestamps and inactive data show --.
After this layout update, restart main.py and all clients to recreate the SHM.

## Per-object tracking

Run `python detector/tracker.py` to publish filtered IDs, base-frame positions
and linear velocities to `tracking_shm`. main.py owns that segment as well.
See [tracker.md](tracker.md) for configuration and servo-time prediction.

## Pose rejection diagnosis

The detector image labels rejected boxes with `Pose: <reason>` and Body
patch depth median (meters) plus valid/total sample counts. Terminal summaries
aggregate reasons once a second; `[Pose Detail]` shows one example per reason,
including surface-normal sample counts, depth percentiles, RMS and thickness.
`camera/test_picking.py` sets `NORMAL_FILTERS_ENABLED=False`: depth-band, RMS
and thickness filters are bypassed. The configured RMS/thickness thresholds apply
only if that switch is re-enabled. Positive finite depth and at least three
non-collinear points are still required for a mathematically defined normal.

- `no_body_depth`: no positive finite depth in the Body patch, or the Body
  pixel is outside the depth image. Knot depth is not used.
- `keypoint_visibility`: at least one keypoint probability is below 0.5.
- `normal_insufficient_depth[_after_filter]`: insufficient surface samples.
- `normal_plane_rms_too_large`: local plane RMS exceeds 0.004 m (filters enabled only).
- `normal_thick_cloud`: smallest/second SVD singular-value ratio exceeds 0.35 (filters enabled only).
- `undefined_body_knot_axis`: no usable Body-to-Knot direction in the tangent plane.
- `flange_inactive`, `invalid_flange`, `invalid_calibration`, `camera_inactive`:
  the frame was not eligible for base-pose calculation at all.
