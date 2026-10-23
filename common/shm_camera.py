import fcntl
import threading
from contextlib import contextmanager

import numpy as np
from multiprocessing import shared_memory
from dataclasses import dataclass
from multiprocessing.resource_tracker import unregister


@dataclass(slots=True)
class Frame:
    timestamp: int
    color: np.ndarray
    depth: np.ndarray
    sequence_id: int = 0


class CameraShm:
    def __init__(
        self,
        shm_name: str,
        is_owner: bool = False,
        color_shape: tuple = (720, 1280, 3),
        depth_shape: tuple = (720, 1280, 1),
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
            ]
        )

        self.frame_dtype = np.dtype(
            [
                ("timestamp", np.uint64),
                ("color", np.uint8, color_shape),
                ("depth", np.uint16, depth_shape),
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

        self._header_arr = np.ndarray(
            (1,), dtype=self.header_dtype, buffer=self.shm.buf
        )
        self.slots = np.ndarray(
            (1,), dtype=self.frame_dtype, buffer=self.shm.buf, offset=self.header_bytes
        )

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
        timestamp: int | np.uint64,
        color_data: np.ndarray | None = None,
        depth_data: np.ndarray | None = None,
    ):
        # Validate shapes before touching the published frame.
        color = None if color_data is None else color_data.reshape(self.color_shape)
        depth = (None if depth_data is None else
                 depth_data.view(np.uint16).reshape(self.depth_shape))
        with self._frame_lock(fcntl.LOCK_EX):
            # If a write fails or the writer exits midway, consumers must not
            # accept a partially updated frame as a new valid snapshot.
            self.status = False
            slot = self.slots[0]
            slot["timestamp"] = timestamp
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
                timestamp=int(slot["timestamp"]),
                color=slot["color"].copy(),
                depth=slot["depth"].copy(),
                sequence_id=sequence_id,
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
