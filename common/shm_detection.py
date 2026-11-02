"""Latest robot-base detections for a tracker (meters, row-major 4x4 TF)."""
import fcntl
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister

import numpy as np

SHM_NAME = "detection_shm"
MAX_OBJECTS = 8
OBJECT_DTYPE = np.dtype([
    ("score", np.float64), ("bbox", np.float64, (4,)),
    ("TF", np.float64, (4, 4)),
])
FRAME_DTYPE = np.dtype([
    ("status", np.bool_), ("timestamp", np.uint64),
    ("sequence_id", np.uint64), ("count", np.uint32),
    ("publication_count", np.uint64), ("latency_ms", np.float64),
    ("interval_us", np.float64),
    ("objects", OBJECT_DTYPE, (MAX_OBJECTS,)),
])


@dataclass
class DetectionFrame:
    status: bool
    timestamp: int
    sequence_id: int
    objects: np.ndarray
    publication_count: int = 0
    latency_ms: float = float("nan")
    interval_us: float = float("nan")

    @property
    def count(self):
        return len(self.objects)


class DetectionShm:
    frame_dtype = FRAME_DTYPE
    object_dtype = OBJECT_DTYPE

    def __init__(self, shm_name=SHM_NAME, is_owner=False):
        self.is_owner = is_owner
        self._local_lock = threading.Lock()
        self.shm = shared_memory.SharedMemory(
            name=shm_name, create=is_owner,
            size=self.frame_dtype.itemsize if is_owner else 0,
        )
        # Independently launched consumers must not unlink the supervisor's SHM.
        if not is_owner:
            unregister(self.shm._name, "shared_memory")
        if self.shm.size != self.frame_dtype.itemsize:
            self.shm.close()
            raise RuntimeError("Detection SHM layout mismatch; restart its owner")
        self._data = np.ndarray((1,), dtype=self.frame_dtype, buffer=self.shm.buf)
        if is_owner:
            self._data[:] = np.zeros(1, dtype=self.frame_dtype)
            self._data[0]["latency_ms"] = np.nan
            self._data[0]["interval_us"] = np.nan

    @contextmanager
    def _lock(self, operation):
        with self._local_lock:
            fcntl.flock(self.shm._fd, operation)
            try:
                yield
            finally:
                fcntl.flock(self.shm._fd, fcntl.LOCK_UN)

    def write(self, timestamp, sequence_id, objects, status=True, latency_ms=float("nan")):
        objects = np.asarray(objects, dtype=self.object_dtype)
        if objects.ndim != 1 or len(objects) > MAX_OBJECTS:
            raise ValueError("Expected at most eight detection records")
        with self._lock(fcntl.LOCK_EX):
            row = self._data[0]
            previous_timestamp = int(row["timestamp"])
            row["interval_us"] = (
                timestamp - previous_timestamp
                if previous_timestamp > 0 and timestamp > previous_timestamp
                else np.nan
            )
            if timestamp > 0:
                row["publication_count"] += 1
            row["latency_ms"] = latency_ms
            row["status"] = status
            row["timestamp"] = timestamp
            row["sequence_id"] = sequence_id
            row["count"] = len(objects)
            row["objects"] = np.zeros(MAX_OBJECTS, dtype=self.object_dtype)
            row["objects"][:len(objects)] = objects

    def read(self):
        with self._lock(fcntl.LOCK_SH):
            row = self._data[0].copy()
        return DetectionFrame(
            bool(row["status"]), int(row["timestamp"]), int(row["sequence_id"]),
            row["objects"][:int(row["count"])].copy(),
            int(row["publication_count"]), float(row["latency_ms"]),
            float(row["interval_us"]),
        )

    def close(self):
        if self.shm is None:
            return
        self._data = None
        self.shm.close()
        if self.is_owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass
        self.shm = None
