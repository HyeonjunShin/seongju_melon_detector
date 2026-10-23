"""Optional, bounded recording channel; camera control never waits for disk I/O."""

import fcntl
import os
import threading
from multiprocessing.managers import BaseManager
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np

# The Unix socket lives in a private directory, accessible only by this user.
SOCKET_DIRECTORY = Path('/tmp') / f'seongju-melon-recorder-{os.getuid()}'
ADDRESS = str(SOCKET_DIRECTORY / 'queue.sock')
AUTHKEY = b'seongju-melon-recording'
QUEUE_SIZE = 8


class RecordingBuffer:
    def __init__(self, maxsize=QUEUE_SIZE):
        self.queue = Queue(maxsize=maxsize)
        self.enabled = threading.Event()
        self.dropped = 0
        self.offer_lock = threading.Lock()

    def configure(self, enabled):
        # On pause/quit, finish any in-flight copy before the consumer drains.
        with self.offer_lock:
            if enabled:
                self.enabled.set()
            else:
                self.enabled.clear()

    def offer(self, timestamp_us, color, depth, flange_shm):
        if not self.enabled.is_set():
            return
        with self.offer_lock:
            if not self.enabled.is_set():
                return
            if self.queue.full():
                self.dropped += 1
                return
            try:
                flange = None
                if flange_shm is not None and flange_shm.status:
                    pose = flange_shm.read(timestamp_us)
                    if pose.matched_ts > 0 and np.isfinite(pose.TF).all():
                        flange = {
                            'matched_timestamp_us': pose.matched_ts,
                            'time_diff_us': pose.time_diff_us,
                            'TF': pose.TF.copy(),
                        }
                self.queue.put_nowait({
                    'timestamp_us': int(timestamp_us),
                    'color': color.copy(),
                    'depth': depth.copy(),
                    'flange': flange,
                })
            except Full:
                self.dropped += 1

    def take(self, timeout=0.2):
        return self.queue.get(timeout=timeout)

    def stats(self):
        return dict(enabled=self.enabled.is_set(), pending=self.queue.qsize(), dropped=self.dropped)


class RecordingManager(BaseManager):
    pass


RecordingManager.register('get_buffer')


def connect(address=ADDRESS):
    manager = RecordingManager(address=address, authkey=AUTHKEY)
    manager.connect()
    return manager, manager.get_buffer()


class RecordingServer:
    def __init__(self, address=ADDRESS, maxsize=QUEUE_SIZE):
        self.address = address
        self.buffer = RecordingBuffer(maxsize)
        self.server = None
        self.thread = None
        self.lock_file = None

    def __enter__(self):
        directory = Path(self.address).parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if directory == SOCKET_DIRECTORY:
            info = directory.stat()
            if info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise PermissionError(f'Recording directory must be private: {directory}')
        self.lock_file = open(self.address + '.lock', 'a')
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Only the lock holder may remove a socket left by a crashed runner.
            path = Path(self.address)
            if path.is_socket():
                path.unlink()

            class ServerManager(RecordingManager):
                pass

            ServerManager.register('get_buffer', callable=lambda: self.buffer,
                                   exposed=('configure', 'take', 'stats'))
            self.server = ServerManager(address=self.address, authkey=AUTHKEY).get_server()
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
        except Exception:
            self.close()
            raise
        return self

    def close(self):
        self.buffer.configure(False)
        if self.server is not None:
            if hasattr(self.server, 'stop_event'):
                self.server.stop_event.set()
            self.server.listener.close()
        if self.thread is not None:
            self.thread.join(timeout=1)
        if self.lock_file is not None:
            self.lock_file.close()
            self.lock_file = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
