import signal
import sys
import threading
import time

from common.ipc import ProductIpc


def build_dashboard(ipc, previous: dict | None, now: float) -> tuple[str, dict]:
    camera = ipc.camera_shm
    flange = ipc.flange_shm
    target = ipc.target_shm

    frame = camera.read()
    flange_data, _ = flange.get_ordered_data()
    flange_latest = flange_data[-1] if flange_data is not None else None
    target_latest = target.read()

    counters = {
        "camera": int(camera.sequence_id),
        "flange": int(flange.sequence),
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
        flange_detail = "timestamp=- XYZ=(-, -, -)"
    else:
        xyz = flange_latest["TF"][:3, 3]
        flange_detail = (
            f"timestamp={int(flange_latest['timestamp_us'])} "
            f"XYZ=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) m"
        )

    target_xyz = target_latest.TF[:3, 3]
    lines = [
        "SEONGJU MELON DETECTOR - IPC STATUS",
        "=" * 72,
        f"camera_shm  {camera_state:<6}  {rates['camera']:.1f} Hz  "
        f"stale={stale['camera']:.1f}s  seq={counters['camera']}",
        f"  timestamp={frame.timestamp} latency={camera.latency:.2f} ms  "
        f"RGB {frame.color.dtype}{frame.color.shape}  "
        f"Depth {frame.depth.dtype}{frame.depth.shape}",
        f"shm_flange {flange_state:<6}  {rates['flange']:.1f} Hz  "
        f"stale={stale['flange']:.1f}s  seq={counters['flange']}",
        f"  {flange_detail}",
        f"shm_target  {target_state:<6}  observed {rates['target']:.1f} Hz  "
        f"stale={stale['target']:.1f}s  index={target.index}",
        f"  timestamp={target_latest.timestamp} detected={target_latest.detected} "
        f"score={target_latest.score:.3f} bbox={target_latest.bbox.tolist()}",
        f"  XYZ=({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, " f"{target_xyz[2]:.3f}) m",
        "=" * 72,
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
