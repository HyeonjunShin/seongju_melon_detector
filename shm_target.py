from dataclasses import dataclass
from multiprocessing import shared_memory
import numpy as np
from multiprocessing.resource_tracker import unregister


@dataclass(slots=True)
class TargetTF:
    timestamp: int
    detected: bool
    score: float
    bbox: np.ndarray  # (4,) [x1, y1, x2, y2]
    TF: np.ndarray  # (4, 4) Homogeneous Transformation Matrix (Position + Orientation)

    @classmethod
    def empty(cls, timestamp: int = 0) -> "TargetTF":
        return cls(
            timestamp=timestamp,
            detected=False,
            score=0.0,
            bbox=np.zeros((4,), dtype=np.float64),
            TF=np.eye(4, dtype=np.float64),
        )


class TargetShm:
    """
    struct TrackObjSlot {
        uint64_t timestamp;          // 8 bytes
        bool detected;               // 1 byte
        uint8_t padding[7];          // 7 bytes (8바이트 정렬)
        double score;                // 8 bytes
        double bbox[4];              // 32 bytes
        double TF[16];               // 128 bytes (4x4 Matrix)
    };                               // Total = 184 bytes

    struct DetectorBufferHeader {
        bool status;                 // 1 byte
        uint8_t padding[7];          // 7 bytes (8바이트 정렬)
        int64_t index;         // 8 bytes
    };                               // Total = 16 bytes

    Total Shared Memory Size = 16 + (184 * 2) = 384 bytes
    """

    def __init__(self, shm_name: str, is_owner: bool = False):
        self._shm_name = shm_name
        self.is_owner = is_owner

        self.header_dtype = np.dtype(
            [
                ("status", np.bool_),
                ("padding", np.uint8, (7,)),
                ("index", np.int64),
            ]
        )

        self.slot_dtype = np.dtype(
            [
                ("timestamp", np.uint64),
                ("detected", np.bool_),
                ("padding", np.uint8, (7,)),
                ("score", np.float64),
                ("bbox", np.float64, (4,)),
                ("TF", np.float64, (4, 4)),  # 4x4 Row-major Matrix
            ]
        )

        self.header_bytes = self.header_dtype.itemsize
        self.slot_bytes = self.slot_dtype.itemsize
        self.total_bytes = self.header_bytes + (2 * self.slot_bytes)

        if self.is_owner:
            try:
                self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
                print(f"[Python SHM] 기존 공유 메모리 '{self.shm_name}'에 재연결되었습니다.")

            except FileNotFoundError:
                self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=self.total_bytes)
                self.shm.buf[: self.total_bytes] = b"\x00" * self.total_bytes
                print(f"[Python SHM] 공유 메모리 '{self.shm_name}' 신규 생성 완료 ({self.total_bytes} bytes)")
        else:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
            unregister(self.shm._name, "shared_memory")

        # 3. Dynamic Zero-copy Mapping
        self._header_arr = np.ndarray((1,), dtype=self.header_dtype, buffer=self.shm.buf)
        self.slots = np.ndarray((2,), dtype=self.slot_dtype, buffer=self.shm.buf, offset=self.header_bytes)

    # ------------------------------------------------------------------
    # Properties (Header Zero-copy Getter/Setter)
    # ------------------------------------------------------------------
    @property
    def shm_name(self) -> str:
        return self._shm_name

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

    # ------------------------------------------------------------------
    # Data Read / Write Operations
    # ------------------------------------------------------------------
    def write(
        self,
        timestamp: int,
        detected: bool,
        score: float = 0.0,
        bbox: np.ndarray | None = None,
        TF: np.ndarray | None = None,
    ):
        next_idx = 1 - self.index
        target_slot = self.slots[next_idx]

        target_slot["timestamp"] = timestamp
        target_slot["detected"] = detected
        target_slot["score"] = score

        if bbox is not None:
            target_slot["bbox"][:] = bbox.reshape(4)
        else:
            target_slot["bbox"][:] = 0.0

        if TF is not None:
            target_slot["TF"][:] = TF.reshape(4, 4)
        else:
            target_slot["TF"][:] = np.eye(4, dtype=np.float64)
        self.index = next_idx

    def read(self) -> TargetTF:
        slot = self.slots[self.index]

        return TargetTF(
            timestamp=int(slot["timestamp"]),
            detected=bool(slot["detected"]),
            score=float(slot["score"]),
            bbox=slot["bbox"].copy(),
            TF=slot["TF"].copy(),
        )

    # ------------------------------------------------------------------
    # Resource Lifecycle Management
    # ------------------------------------------------------------------
    def close(self):
        self._header_arr = None
        self.slots = None

        if hasattr(self, "shm") and self.shm is not None:
            self.shm.close()
            if self.is_owner:
                try:
                    self.shm.unlink()
                    print(f"[Python SHM] 공유 메모리 '{self.shm_name}' 해제 완료.")
                except FileNotFoundError:
                    pass
            self.shm = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
