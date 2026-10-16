import signal
import threading

from common.ipc import ProductIpc


def install_signal_handlers(stop_event: threading.Event) -> None:
    def stop(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)


def run(stop_event: threading.Event | None = None) -> int:
    stop_event = stop_event or threading.Event()
    with ProductIpc():
        print("✅ Shared Memory 및 Semaphore 준비 완료")
        stop_event.wait()
    return 0


def main() -> int:
    stop_event = threading.Event()
    install_signal_handlers(stop_event)
    return run(stop_event)


if __name__ == "__main__":
    raise SystemExit(main())
