import fcntl
import threading
from contextlib import contextmanager

import numpy as np
from multiprocessing import shared_memory
from dataclasses import dataclass
from multiprocessing.resource_tracker import unregister

CAMERA_WIDTH = 848
CAMERA_HEIGHT = 530
COLOR_SHAPE = (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
DEPTH_SHAPE = (CAMERA_HEIGHT, CAMERA_WIDTH, 1)


@dataclass(slots=True)
class Frame:
    timestamp_color: int
    timestamp_depth: int
    color: np.ndarray
    depth: np.ndarray  # float32 meters; SDK raw values must be scaled before write().
    sequence_id: int = 0
    interval_color_us: float = float("nan")
    interval_depth_us: float = float("nan")

    @property
    def timestamp(self):
        """Keep RGB-D consumers on the existing color timestamp timebase."""
        return self.timestamp_color


class CameraShm:
    def __init__(
        self,
        shm_name: str,
        is_owner: bool = False,
        color_shape: tuple = COLOR_SHAPE,
        depth_shape: tuple = DEPTH_SHAPE,
    ):
        self._shm_name = shm_name
        self.is_owner = is_owner
        self._local_lock = threading.Lock()

        self.color_shape = color_shape
        self.depth_shape = depth_shape

        self.header_dtype = np.dtype(
            [
                ("status", np.bool_),
                ("index", np.int64),
                ("latency", np.float64),
                ("sequence_id", np.uint64),
                ("camera_matrix", np.float64, (3, 3)),
                ("distortion", np.float64, (8,)),
            ]
        )

        self.frame_dtype = np.dtype(
            [
                ("timestamp_color", np.uint64),
                ("timestamp_depth", np.uint64),
                ("interval_color_us", np.float64),
                ("interval_depth_us", np.float64),
                ("color", np.uint8, color_shape),
                ("depth", np.float32, depth_shape),
            ]
        )

        self.header_bytes = self.header_dtype.itemsize
        self.total_bytes = self.header_bytes + self.frame_dtype.itemsize

        if self.is_owner:
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name, create=True, size=self.total_bytes
            )
            self.shm.buf[: self.total_bytes] = b"\x00" * self.total_bytes
        else:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
            unregister(self.shm._name, "shared_memory")
            if self.shm.size != self.total_bytes:
                self.shm.close()
                raise RuntimeError(
                    "카메라 Shared Memory 크기가 다릅니다. main.py와 카메라 프로세스를 재시작하세요."
                )

        self._header_arr = np.ndarray(
            (1,), dtype=self.header_dtype, buffer=self.shm.buf
        )
        self.slots = np.ndarray(
            (1,), dtype=self.frame_dtype, buffer=self.shm.buf, offset=self.header_bytes
        )

    @property
    def intrinsics(self):
        """Color intrinsics for aligned depth: OpenCV camera matrix and distortion."""
        with self._frame_lock(fcntl.LOCK_SH):
            return (
                self._header_arr[0]["camera_matrix"].copy(),
                self._header_arr[0]["distortion"].copy(),
            )

    @intrinsics.setter
    def intrinsics(self, calibration):
        matrix, distortion = calibration
        with self._frame_lock(fcntl.LOCK_EX):
            self._header_arr[0]["camera_matrix"] = matrix
            self._header_arr[0]["distortion"] = distortion

    @property
    def shm_name(self) -> str:
        return self._shm_name

    @shm_name.setter
    def shm_name(self, shm_name):
        self._shm_name = shm_name

    @property
    def status(self) -> bool:
        return bool(self._header_arr[0]["status"])

    @status.setter
    def status(self, val: bool):
        self._header_arr[0]["status"] = val

    @property
    def index(self) -> int:
        return int(self._header_arr[0]["index"])

    @index.setter
    def index(self, val: int):
        self._header_arr[0]["index"] = val

    @property
    def latency(self) -> float:
        return float(self._header_arr[0]["latency"])

    @latency.setter
    def latency(self, val: float):
        self._header_arr[0]["latency"] = val

    @property
    def sequence_id(self):
        return self._header_arr[0]["sequence_id"]

    @sequence_id.setter
    def sequence_id(self, val: int):
        self._header_arr[0]["sequence_id"] = val

    @contextmanager
    def _frame_lock(self, operation):
        # flock coordinates independently opened SHM descriptors across processes;
        # the local lock also serializes threads sharing this instance.
        with self._local_lock:
            fcntl.flock(self.shm._fd, operation)
            try:
                yield
            finally:
                fcntl.flock(self.shm._fd, fcntl.LOCK_UN)

    def write(
        self,
        timestamp_color: int | np.uint64,
        timestamp_depth: int | np.uint64,
        color_data: np.ndarray | None = None,
        depth_data: np.ndarray | None = None,
    ):
        # Validate shapes before touching the published frame.
        color = None if color_data is None else color_data.reshape(self.color_shape)
        depth = (
            None
            if depth_data is None
            else depth_data.astype(np.float32, copy=False).reshape(self.depth_shape)
        )
        with self._frame_lock(fcntl.LOCK_EX):
            # If a write fails or the writer exits midway, consumers must not
            # accept a partially updated frame as a new valid snapshot.
            self.status = False
            slot = self.slots[0]
            for stream, timestamp in (("color", timestamp_color), ("depth", timestamp_depth)):
                previous = int(slot[f"timestamp_{stream}"])
                slot[f"interval_{stream}_us"] = (
                    int(timestamp) - previous
                    if int(self.sequence_id) > 0 and previous > 0 and int(timestamp) >= previous
                    else np.nan
                )
            slot["timestamp_color"] = timestamp_color
            slot["timestamp_depth"] = timestamp_depth
            if color is not None:
                slot["color"][:] = color
            if depth is not None:
                slot["depth"][:] = depth
            self.sequence_id = int(self.sequence_id) + 1
            self.status = True

    def read(self, last_sequence_id: int | None = None) -> Frame | None:
        """Copy the latest frame; optionally skip inactive or already-read data.

        The returned arrays belong to the consumer. Process them after this
        method returns, outside the shared-memory lock. No frames are queued.
        """
        with self._frame_lock(fcntl.LOCK_SH):
            sequence_id = int(self.sequence_id)
            if last_sequence_id is not None and (
                not self.status or sequence_id <= last_sequence_id
            ):
                return None
            slot = self.slots[0]
            return Frame(
                timestamp_color=int(slot["timestamp_color"]),
                timestamp_depth=int(slot["timestamp_depth"]),
                color=slot["color"].copy(),
                depth=slot["depth"].copy(),
                sequence_id=sequence_id,
                interval_color_us=float(slot["interval_color_us"]),
                interval_depth_us=float(slot["interval_depth_us"]),
            )

    def close(self):
        """메모리 참조 해제 및 Shared Memory 닫기"""
        self._header_arr = None
        self.slots = None

        if hasattr(self, "shm") and self.shm is not None:
            self.shm.close()
            if self.is_owner:
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass
            self.shm = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
