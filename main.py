import signal
import sys
import threading
import time
from collections import Counter

import numpy as np

from common.ipc import ProductIpc


def rotation_rpy(rotation):
    """Fixed-XYZ Euler angles in degrees, or None for invalid rotations."""
    if not (
        np.isfinite(rotation).all()
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        return None
    cosine_pitch = np.hypot(rotation[0, 0], rotation[1, 0])
    pitch = np.arctan2(-rotation[2, 0], cosine_pitch)
    if cosine_pitch > 1e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0  # At gimbal lock, choose yaw=0 as the representative pose.
    roll, pitch, yaw = np.rad2deg([roll, pitch, yaw])
    # Avoid displaying -0.000 for exact axis-aligned poses.
    roll, pitch, yaw = [
        0.0 if abs(value) < 0.0005 else value for value in (roll, pitch, yaw)
    ]
    return roll, pitch, yaw


def pose_lines(transform):
    """Show the complete base-frame pose, with fixed-XYZ Euler angles."""
    if transform is None:
        return ["  TF: no data"]
    xyz = transform[:3, 3]
    rotation = transform[:3, :3]
    lines = [
        f"  위치   X {xyz[0]:.6f}  Y {xyz[1]:.6f}  Z {xyz[2]:.6f} m",
        "  XYZ (mm): " + "  ".join(f"{value * 1000:.3f}" for value in xyz),
    ]
    rpy = rotation_rpy(rotation)
    if rpy is not None:
        roll, pitch, yaw = rpy
        lines.append(f"  RPY (deg): roll={roll:.3f}  pitch={pitch:.3f}  yaw={yaw:.3f}")
    else:
        lines.append("  RPY (deg): unavailable (invalid rotation)")
    lines.append(
        "  +Z (base): " + "  ".join(f"{value:.6f}" for value in rotation[:, 2])
    )
    lines.append("  TF (4x4, translation=m):")
    lines.extend(
        "    [" + "  ".join(f"{value: .9g}" for value in row) + "]" for row in transform
    )
    return lines


def flange_pose_table(transform):
    lines = [
        "       X(m)       Y(m)       Z(m) │   ROLL(°)  PITCH(°)    YAW(°)",
        "  " + "─" * 70,
    ]
    xyz = transform[:3, 3] if transform is not None else [np.nan] * 3
    rpy = rotation_rpy(transform[:3, :3]) if transform is not None else None
    position = " ".join(
        f"{value:+10.6f}" if np.isfinite(value) else f"{'--':>10}" for value in xyz
    )
    angles = (
        " ".join(f"{value:9.3f}" for value in rpy)
        if rpy is not None
        else " ".join([f"{'--':>9}"] * 3)
    )
    lines.append(f"  {position} │ {angles}")
    return lines


def tracking_table(frame):
    lines = ["", "TRACKS (robot base)"]
    lines.append(
        f"  tracking_shm: {'ACTIVE' if frame.status else 'IDLE'}  count={frame.count}/8  timestamp={frame.timestamp}"
    )
    if frame.count:
        lines.extend(
            [
                "  ID STATE    │    X(m)    Y(m)    Z(m) │ VX(m/s) VY(m/s) VZ(m/s) │ sigma(mm)  AGE(s)",
                "  " + "─" * 86,
            ]
        )
    for row in frame.objects:
        state = (
            "COASTING"
            if not row["observed"]
            else ("TRACKED" if row["confirmed"] else "NEW")
        )
        xyz = " ".join(f"{value:+7.3f}" for value in row["TF"][:3, 3])
        velocity = " ".join(f"{value:+7.3f}" for value in row["velocity"])
        sigma = np.sqrt(max(0.0, np.max(np.diag(row["covariance"])[:3]))) * 1000
        age = max(0., (time.time_ns() // 1000 - int(row["last_seen"])) / 1e6)
        lines.append(
            f"  {int(row['track_id']):2d} {state:<8} │ {xyz} │ {velocity} │ {sigma:9.2f}  {age:6.2f}"
        )
    if not frame.count:
        lines.append("  No tracks - run detector/tracker.py after main.py and detector")
    lines.append(
        "  NEW=tentative · TRACKED=confirmed · COASTING=prediction without observation"
    )
    return lines


def build_dashboard(
    ipc, previous: dict | None, now: float, detection=None, compact=False, tracking=None
) -> tuple[str, dict]:
    camera = ipc.camera_shm
    flange = ipc.flange_shm
    target = ipc.target_shm

    frame = camera.read()
    flange_data, flange_sequence = flange.get_ordered_data()
    flange_latest = flange_data[-1] if flange_data is not None else None
    flange_timestamp_counts = (
        Counter(int(row["timestamp_us"]) for row in flange_data)
        if flange_data is not None
        else Counter()
    )
    flange_duplicates = (
        ", ".join(
            f"{timestamp} us × {count}회"
            for timestamp, count in flange_timestamp_counts.items()
            if count > 1
        )
        or "없음"
    )
    target_latest = target.read()

    counters = {
        "camera": int(camera.sequence_id),
        "flange": int(flange_sequence),
        "target": (int(target_latest.timestamp), int(target.index)),
        "detector": detection.publication_count if detection is not None else 0,
    }
    if previous is None:
        elapsed = 0.0
        last_change = {name: now for name in counters}
        rates = {name: 0.0 for name in counters}
    else:
        elapsed = max(now - previous["time"], 0.0)
        last_change = previous["last_change"].copy()
        rates = {}
        for name, current in counters.items():
            changed = current != previous["counters"][name]
            if changed:
                last_change[name] = now
            if name == "target":
                rates[name] = (1.0 / elapsed) if changed and elapsed else 0.0
            else:
                delta = current - previous["counters"][name]
                rates[name] = max(delta, 0) / elapsed if elapsed else 0.0

    flange_timing = "  주기: 측정 불가 (이전 데이터 없음 또는 새 데이터 없음)"
    if rates["flange"] > 0:
        flange_timing = (
            f"  주기 (이전/현재 sequence·경과 시간 기준): "
            f"평균={1000.0 / rates['flange']:.3f} ms  "
            f"수신 수={counters['flange'] - previous['counters']['flange']}  "
            f"경과={elapsed:.3f} s"
        )

    stale = {name: now - changed_at for name, changed_at in last_change.items()}
    camera_state = "ACTIVE" if camera.status else "IDLE"
    flange_state = "ACTIVE" if flange.status else "IDLE"
    target_state = "ACTIVE" if target.status else "IDLE"

    if flange_latest is None:
        flange_timestamp = "-"
        flange_tf = None
    else:
        flange_timestamp = int(flange_latest["timestamp_us"])
        flange_tf = flange_latest["TF"].copy()

    valid_depth = frame.depth[(frame.depth > 0) & np.isfinite(frame.depth)]
    depth_range = (
        f"min={valid_depth.min():.4f}  median={np.median(valid_depth):.4f}  "
        f"max={valid_depth.max():.4f} m"
        if valid_depth.size
        else "min=-  median=-  max=- m"
    )
    center_y, center_x = frame.depth.shape[0] // 2, frame.depth.shape[1] // 2
    camera_timing = []
    for label, stream in (("Color", "color"), ("Depth", "depth")):
        timestamp = getattr(frame, f"timestamp_{stream}")
        interval = getattr(frame, f"interval_{stream}_us")
        gap = (
            f"{interval / 1000:.3f} ms"
            if camera.status and np.isfinite(interval)
            else "--"
        )
        camera_timing.append(f"  {label} timestamp={timestamp} us  Δt={gap}")
    detection_lines = ["", "DETECTIONS (robot base)"]
    if detection is None:
        detection_lines.append("  detection_shm: WAITING - no snapshot")
    else:
        detection_lines.extend(
            [
                f"  detection_shm: {'ACTIVE' if detection.status else 'IDLE'}  "
                f"count={detection.count}/8",
                f"  timestamp={detection.timestamp}  seq={detection.sequence_id}",
                f"  processed={detection.publication_count}  period={detection.interval_us / 1000:.3f} ms",
            ]
        )
        if compact and detection.count:
            detection_lines.extend(
                [
                    "  ID  SCORE │     X(m)     Y(m)     Z(m) │  ROLL(°) PITCH(°)   YAW(°)",
                    "  " + "─" * 72,
                ]
            )
        for index, obj in enumerate(detection.objects, 1):
            if compact:
                xyz = obj["TF"][:3, 3]
                rpy = rotation_rpy(obj["TF"][:3, :3])
                angles = (
                    " ".join(f"{angle:8.2f}" for angle in rpy)
                    if rpy is not None
                    else "   invalid rotation"
                )
                detection_lines.append(
                    f"  P{index:<2} {obj['score']:5.3f} │ "
                    + " ".join(f"{value:+8.4f}" for value in xyz)
                    + f" │ {angles}"
                )
                continue
            detection_lines.append(
                f"  P{index} score={obj['score']:.3f}  BBox={obj['bbox'].tolist()}"
            )
            detection_lines.extend(pose_lines(obj["TF"]))
        if detection.count == 0:
            detection_lines.append("  No valid poses")

    def period_text(interval_us, active):
        return (
            f"{interval_us / 1000:.3f} ms"
            if active and np.isfinite(interval_us) and interval_us > 0
            else "--"
        )

    robot_interval_us = (
        int(flange_data[-1]["timestamp_us"]) - int(flange_data[-2]["timestamp_us"])
        if flange_data is not None and len(flange_data) >= 2
        else np.nan
    )
    robot_period = period_text(robot_interval_us, flange.status)
    camera_period = period_text(frame.interval_color_us, camera.status)
    detector_state = (
        "WAITING" if detection is None else ("ACTIVE" if detection.status else "IDLE")
    )
    detector_timestamp = (
        detection.timestamp
        if detection is not None and detection.timestamp > 0
        else "--"
    )
    detector_period = (
        period_text(detection.interval_us, detection.status)
        if detection is not None
        else "--"
    )
    header = f"{'NAME':<10} │ {'STATUS':<8} │ {'RATE':>10} │ {'PERIOD (ms)':>12} │ {'TIMESTAMP (µs)':>20}"
    summary_rows = [
        f"{'Robot':<10} │ {flange_state:<8} │ {rates['flange']:7.1f} Hz │ {robot_period:>12} │ {flange_timestamp:>20}",
        f"{'Camera':<10} │ {camera_state:<8} │ {rates['camera']:7.1f} Hz │ {camera_period:>12} │ {frame.timestamp:>20}",
        f"{'Detector':<10} │ {detector_state:<8} │ {rates['detector']:7.1f} Hz │ {detector_period:>12} │ {detector_timestamp:>20}",
    ]
    lines = [
        "SEONGJU MELON DETECTOR - SHARED MEMORY STATUS",
        "═" * 78,
        header,
        "─" * 78,
        *summary_rows,
        "═" * 78,
        "",
        "CAMERA",
        f"{camera.shm_name:<14} {camera_state:<8} {rates['camera']:>7.1f} Hz  {stale['camera']:>10.1f} s",
        f"  RGB    {frame.color.shape[1]} × {frame.color.shape[0]} · "
        f"{frame.color.dtype} · {frame.color.shape[2]} ch",
        f"  Depth  {frame.depth.shape[1]} × {frame.depth.shape[0]} · "
        f"{frame.depth.dtype} · {frame.depth.shape[2]} ch",
        f"  Depth valid={valid_depth.size}/{frame.depth.size}  {depth_range}",
        f"  Depth center (u={center_x}, v={center_y}): "
        f"{frame.depth[center_y, center_x].item():.4f} m",
        f"  지연   {camera.latency:.2f} ms",
        f"  timestamp={frame.timestamp}  seq={counters['camera']}  index={camera.index}",
        *camera_timing,
        "",
        "FLANGE",
        f"{flange.shm_name:<14} {flange_state:<8} {rates['flange']:>7.1f} Hz  {stale['flange']:>10.1f} s",
        f"  timestamp_us={flange_timestamp}  seq={counters['flange']}  head={flange.head}",
        f"  buffer={len(flange_data) if flange_data is not None else 0}/{flange.buffer_size}",
        f"  버퍼 내 timestamp 중복: {flange_duplicates}",
        flange_timing,
        *flange_pose_table(flange_tf),
        *(pose_lines(flange_tf) if not compact else []),
        "",
        "TARGET",
        f"{target.shm_name:<14} {target_state:<8} {rates['target']:>7.1f} Hz  {stale['target']:>10.1f} s",
        f"  검출   {'YES' if target_latest.detected else 'NO'} · "
        f"신뢰도 {target_latest.score:.3f}",
        f"  BBox   {target_latest.bbox.tolist()}",
        f"  timestamp={target_latest.timestamp}  index={target.index}",
        *pose_lines(target_latest.TF if target_latest.timestamp > 0 else None),
        *detection_lines,
        *(tracking_table(tracking) if tracking is not None else []),
        "=" * 72,
        "TF/XYZ/+Z: robot base frame | RPY: fixed XYZ, R=Rz(yaw)@Ry(pitch)@Rx(roll)",
        "PERIOD = (현재 timestamp - 직전 데이터 timestamp) / 1000 ms",
        "Camera: Color 간격 · Detector: 처리한 입력 프레임 timestamp 간격",
        "Ctrl+C: 종료 | -- = 측정값 없음 · 첫 화면의 RATE는 0.0 Hz",
        "target Hz는 polling으로 관측된 값입니다.",
    ]
    if compact:
        # Keep numeric summaries; full transforms remain available with --details.
        lines = [
            line
            for line in lines
            if not line.startswith(
                ("  TF (4x4", "    [", "  BBox", "  +Z (base)", "  XYZ (mm)")
            )
        ]
    state = {"time": now, "counters": counters, "last_change": last_change}
    return "\n".join(lines), state


def style_dashboard(text, color=False):
    """Section dividers and optional ANSI styling, with plain redirected output."""
    sections = {
        "CAMERA": "CAMERA · RGB / DEPTH",
        "FLANGE": "FLANGE · ROBOT FEEDBACK",
        "TARGET": "TARGET · SINGLE ROBOT TARGET",
        "DETECTIONS (robot base)": "DETECTIONS · TRACKER INPUT / ROBOT BASE",
        "TRACKS (robot base)": "TRACKER · POSITION / VELOCITY",
    }
    output = []
    for line in text.splitlines():
        if line in sections:
            line = "┌─ " + sections[line] + " " + "─" * max(0, 74 - len(sections[line]))
            if color:
                line = f"\033[1;36m{line}\033[0m"
        elif line.startswith("SEONGJU MELON"):
            line = "  SEONGJU MELON  /  LIVE MONITOR"
            if color:
                line = f"\033[1;32m{line}\033[0m"
        elif line.startswith(("====", "----")):
            line = "─" * 78
        if color:
            for word, code in (("ACTIVE", "32"), ("IDLE", "33"), ("WAITING", "33")):
                line = line.replace(word, f"\033[{code}m{word}\033[0m")
        output.append(line)
    return "\n".join(output)


def print_dashboard(text: str) -> None:
    is_terminal = sys.stdout.isatty()
    output = style_dashboard(text, color=is_terminal)
    # Clear before repaint: cursor-home alone leaves tails of longer old rows.
    prefix = "\033[2J\033[H" if is_terminal else ""
    suffix = "\033[J" if is_terminal else ""
    print(f"{prefix}{output}{suffix}", file=sys.stdout, flush=True)


def install_signal_handlers(stop_event: threading.Event) -> None:
    def stop(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)


def run(stop_event: threading.Event | None = None, details=False) -> int:
    stop_event = stop_event or threading.Event()
    with ProductIpc() as ipc:
        previous = None
        while not stop_event.is_set():
            dashboard, previous = build_dashboard(
                ipc,
                previous,
                time.monotonic(),
                detection=ipc.detection_shm.read(),
                compact=not details,
                tracking=ipc.tracking_shm.read(),
            )
            print_dashboard(dashboard)
            stop_event.wait(0.1)
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Live shared-memory dashboard")
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show full 4x4 transforms and bounding boxes",
    )
    args = parser.parse_args()
    stop_event = threading.Event()
    install_signal_handlers(stop_event)
    return run(stop_event, details=args.details)


if __name__ == "__main__":
    raise SystemExit(main())
