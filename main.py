import signal
import sys
import threading
import time

import numpy as np

from common.ipc import ProductIpc


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
    if (np.isfinite(rotation).all()
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
        cosine_pitch = np.hypot(rotation[0, 0], rotation[1, 0])
        pitch = np.arctan2(-rotation[2, 0], cosine_pitch)
        if cosine_pitch > 1e-8:
            roll = np.arctan2(rotation[2, 1], rotation[2, 2])
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        else:
            roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
            yaw = 0.  # At gimbal lock, choose yaw=0 as the representative pose.
        roll, pitch, yaw = np.rad2deg([roll, pitch, yaw])
        # Avoid displaying -0.000 for exact axis-aligned poses.
        roll, pitch, yaw = [0. if abs(value) < .0005 else value for value in (roll, pitch, yaw)]
        lines.append(f"  RPY (deg): roll={roll:.3f}  pitch={pitch:.3f}  yaw={yaw:.3f}")
    else:
        lines.append("  RPY (deg): unavailable (invalid rotation)")
    lines.append("  +Z (base): " + "  ".join(f"{value:.6f}" for value in rotation[:, 2]))
    lines.append("  TF (4x4, translation=m):")
    lines.extend("    [" + "  ".join(f"{value: .9g}" for value in row) + "]" for row in transform)
    return lines


def build_dashboard(ipc, previous: dict | None, now: float) -> tuple[str, dict]:
    camera = ipc.camera_shm
    flange = ipc.flange_shm
    target = ipc.target_shm

    frame = camera.read()
    flange_data, flange_sequence = flange.get_ordered_data()
    flange_latest = flange_data[-1] if flange_data is not None else None
    target_latest = target.read()

    counters = {
        "camera": int(camera.sequence_id),
        "flange": int(flange_sequence),
        "target": (int(target_latest.timestamp), int(target.index)),
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
        f"min={valid_depth.min():.0f}  median={np.median(valid_depth):.0f}  "
        f"max={valid_depth.max():.0f} mm"
        if valid_depth.size else "min=-  median=-  max=- mm"
    )
    center_y, center_x = frame.depth.shape[0] // 2, frame.depth.shape[1] // 2
    lines = [
        "SEONGJU MELON DETECTOR - SHARED MEMORY STATUS",
        "=" * 72,
        f"{'NAME':<14} {'STATUS':<8} {'RATE':>10}  {'LAST CHANGE':>12}",
        "-" * 72,
        f"{camera.shm_name:<14} {camera_state:<8} {rates['camera']:>7.1f} Hz  " f"{stale['camera']:>10.1f} s",
        f"{flange.shm_name:<14} {flange_state:<8} {rates['flange']:>7.1f} Hz  " f"{stale['flange']:>10.1f} s",
        f"{target.shm_name:<14} {target_state:<8} {rates['target']:>7.1f} Hz  " f"{stale['target']:>10.1f} s",
        "",
        "CAMERA",
        f"  RGB    {frame.color.shape[1]} × {frame.color.shape[0]} · "
        f"{frame.color.dtype} · {frame.color.shape[2]} ch",
        f"  Depth  {frame.depth.shape[1]} × {frame.depth.shape[0]} · "
        f"{frame.depth.dtype} · {frame.depth.shape[2]} ch",
        f"  Depth valid={valid_depth.size}/{frame.depth.size}  {depth_range}",
        f"  Depth center (u={center_x}, v={center_y}): "
        f"{frame.depth[center_y, center_x].item()} mm",
        f"  지연   {camera.latency:.2f} ms",
        f"  timestamp={frame.timestamp}  seq={counters['camera']}  index={camera.index}",
        "",
        "FLANGE",
        f"  timestamp_us={flange_timestamp}  seq={counters['flange']}  head={flange.head}",
        f"  buffer={len(flange_data) if flange_data is not None else 0}/{flange.buffer_size}",
        *pose_lines(flange_tf),
        "",
        "TARGET",
        f"  검출   {'YES' if target_latest.detected else 'NO'} · " f"신뢰도 {target_latest.score:.3f}",
        f"  BBox   {target_latest.bbox.tolist()}",
        f"  timestamp={target_latest.timestamp}  index={target.index}",
        *pose_lines(target_latest.TF if target_latest.timestamp > 0 else None),
        "=" * 72,
        "TF/XYZ/+Z: robot base frame | RPY: fixed XYZ, R=Rz(yaw)@Ry(pitch)@Rx(roll)",
        "Ctrl+C: 종료 | target Hz는 polling으로 관측된 값입니다.",
    ]
    state = {"time": now, "counters": counters, "last_change": last_change}
    return "\n".join(lines), state


def print_dashboard(text: str) -> None:
    print(f"\033[2J\033[H{text}", file=sys.stdout, flush=True)


def install_signal_handlers(stop_event: threading.Event) -> None:
    def stop(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)


def run(stop_event: threading.Event | None = None) -> int:
    stop_event = stop_event or threading.Event()
    with ProductIpc() as ipc:
        previous = None
        while not stop_event.is_set():
            dashboard, previous = build_dashboard(ipc, previous, time.monotonic())
            print_dashboard(dashboard)
            stop_event.wait(1.0)
    return 0


def main() -> int:
    stop_event = threading.Event()
    install_signal_handlers(stop_event)
    return run(stop_event)


if __name__ == "__main__":
    raise SystemExit(main())
