import numpy as np
from multiprocessing import shared_memory
from dataclasses import dataclass
from multiprocessing.resource_tracker import unregister


@dataclass(slots=True)
class Frame:
    timestamp: int
    color: np.ndarray
    depth: np.ndarray


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
        self.total_bytes = self.header_bytes + (
            self.frame_dtype.itemsize * 2
        )  # Because two slot

        if self.is_owner:
            try:
                old_shm = shared_memory.SharedMemory(name=self.shm_name)
                old_shm.close()
                old_shm.unlink()
            except FileNotFoundError:
                pass
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name, create=True, size=self.total_bytes
            )
        else:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
            unregister(self.shm._name, "shared_memory")

        self._header_arr = np.ndarray(
            (1,), dtype=self.header_dtype, buffer=self.shm.buf
        )
        self.slots = np.ndarray(
            (2,), dtype=self.frame_dtype, buffer=self.shm.buf, offset=self.header_bytes
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

    def write(
        self,
        timestamp: int | np.uint64,
        color_data: np.ndarray | None = None,
        depth_data: np.ndarray | None = None,
    ):
        next_idx = 1 - self.index
        target_slot = self.slots[next_idx]

        target_slot["timestamp"] = timestamp

        if color_data is not None:
            target_slot["color"][:] = color_data.reshape(self.color_shape)

        if depth_data is not None:
            target_slot["depth"][:] = depth_data.view(np.uint16).reshape(
                self.depth_shape
            )

        self.index = next_idx

    # def write(
    #     self,
    #     timestamp: np.uint64,
    #     color_data: np.ndarray,
    #     depth_data: np.ndarray,
    # ):
    #     next_idx = 1 - self.header.index
    #     target_slot = self.slots[next_idx]

    #     target_slot["timestamp"][0] = timestamp

    #     if color_data is not None:
    #         np.copyto(target_slot["color"], color_data.reshape(self.color_shape))

    #     if depth_data is not None:
    #         depth_data_u16 = depth_data.view(np.uint16)
    #         np.copyto(target_slot["depth"], depth_data_u16.reshape(self.depth_shape))

    #     self.header.index = next_idx

    def read(self) -> Frame:
        slot = self.slots[self.index]

        return Frame(
            timestamp=int(slot["timestamp"]),
            color=slot["color"],
            depth=slot["depth"],
        )

    # def get_latest_frame(self) -> Frame:
    #     latest_idx = self.header.index
    #     slot = self.slots[latest_idx]

    #     return Frame(
    #         timestamp=int(slot["timestamp"][0]),
    #         color=slot["color"],
    #         depth=slot["depth"],
    #     )

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
