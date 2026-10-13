from dataclasses import dataclass
from multiprocessing import shared_memory
import time
import numpy as np
from multiprocessing.resource_tracker import unregister


@dataclass(slots=True)
class FlangeTF:
    target_ts: int
    matched_ts: int
    time_diff_us: int
    TF: np.ndarray

    @classmethod
    def empty(cls, target_ts: int) -> "FlangeTF":
        return cls(
            target_ts=target_ts,
            matched_ts=0,
            time_diff_us=0,
            TF=np.eye(4, dtype=np.float64),
        )


class FlangeShm:
    """
    struct RobotFrameData {
        int64_t timestamp_us;        // 8 bytes
        double TF[16];               // 128 bytes (4x4 Matrix)
    };                               // Total = 136 bytes

    struct SharedDataBuffer {
        bool status;                 // 1 byte
        uint8_t padding[7];          // 7 bytes
        uint64_t sequence;           // 8 bytes
        uint64_t head;               // 8 bytes
        RobotFrameData buffer[100];  // 136 * 100 = 13,600 bytes
    };                               // Total = 13,616 bytes
    """

    def __init__(self, shm_name: str = "shm_flange", is_owner: bool = True):
        self.buffer_size = 100
        self._shm_name = shm_name
        self.is_owner = is_owner

        self.header_dtype = np.dtype(
            [
                ("status", np.bool_),
                ("padding", np.uint8, (7,)),
                ("sequence", np.uint64),
                ("head", np.uint64),
            ]
        )

        self.frame_dtype = np.dtype(
            [
                ("timestamp_us", np.int64),
                ("TF", np.float64, (4, 4)),  # 4x4 Row-major Matrix
            ]
        )

        self.header_bytes = self.header_dtype.itemsize
        self.shared_memory_size = self.header_bytes + (self.frame_dtype.itemsize * self.buffer_size)

        if self.is_owner:
            try:
                self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
                print(f"[Python SHM] 기존 공유 메모리 '{self.shm_name}'에 재연결되었습니다.")

            except FileNotFoundError:
                self.shm = shared_memory.SharedMemory(
                    name=self.shm_name, create=True, size=self.shared_memory_size
                )
                self.shm.buf[: self.shared_memory_size] = b"\x00" * self.shared_memory_size
                print(
                    f"[Python SHM] 공유 메모리 '{self.shm_name}' 신규 생성 완료 ({self.shared_memory_size} bytes)"
                )
        else:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
            unregister(self.shm._name, "shared_memory")

        self._header_arr = np.ndarray((1,), dtype=self.header_dtype, buffer=self.shm.buf)
        self.frame_buffer = np.ndarray(
            (self.buffer_size,), dtype=self.frame_dtype, buffer=self.shm.buf, offset=self.header_bytes
        )

    # ------------------------------------------------------------------
    # Header Real-time Properties
    # ------------------------------------------------------------------
    @property
    def status(self) -> bool:
        return bool(self._header_arr[0]["status"])

    @status.setter
    def status(self, val: bool):
        self._header_arr[0]["status"] = val

    @property
    def sequence(self) -> int:
        return int(self._header_arr[0]["sequence"])

    @sequence.setter
    def sequence(self, val: int):
        self._header_arr[0]["sequence"] = val

    @property
    def head(self) -> int:
        return int(self._header_arr[0]["head"])

    @head.setter
    def head(self, val: int):
        self._header_arr[0]["head"] = val

    @property
    def shm_name(self):
        return self._shm_name

    @shm_name.setter
    def shm_name(self, val: str):
        self._shm_name = val

    # ------------------------------------------------------------------
    # Data Access Methods
    # ------------------------------------------------------------------
    def get_ordered_data(self):
        seq = self.sequence
        h = self.head

        if seq == 0:
            return None, 0

        if seq < self.buffer_size:
            ordered_data = self.frame_buffer[:seq]
        else:
            ordered_data = np.roll(self.frame_buffer, -h, axis=0)

        return ordered_data, seq

    def read(self, target_ts_us: int) -> FlangeTF:
        seq = self.sequence
        h = self.head

        if seq == 0:
            return FlangeTF.empty(target_ts_us)

        valid_size = seq if seq < self.buffer_size else self.buffer_size
        start_idx = 0 if seq < self.buffer_size else h

        oldest_idx = start_idx
        newest_idx = (start_idx + valid_size - 1) % self.buffer_size

        oldest_ts = self.frame_buffer[oldest_idx]["timestamp_us"]
        newest_ts = self.frame_buffer[newest_idx]["timestamp_us"]

        if target_ts_us <= oldest_ts:
            best_idx = oldest_idx
        elif target_ts_us >= newest_ts:
            best_idx = newest_idx
        else:
            low = 0
            high = valid_size - 1

            while low < high:
                mid = (low + high) // 2
                real_mid_idx = (start_idx + mid) % self.buffer_size

                if self.frame_buffer[real_mid_idx]["timestamp_us"] < target_ts_us:
                    low = mid + 1
                else:
                    high = mid

            right_idx = (start_idx + high) % self.buffer_size
            left_idx = (start_idx + high - 1 + self.buffer_size) % self.buffer_size

            diff_right = abs(self.frame_buffer[right_idx]["timestamp_us"] - target_ts_us)
            diff_left = abs(self.frame_buffer[left_idx]["timestamp_us"] - target_ts_us)

            best_idx = right_idx if diff_right < diff_left else left_idx

        matched_frame = self.frame_buffer[best_idx]
        matched_ts = int(matched_frame["timestamp_us"])

        return FlangeTF(
            target_ts=target_ts_us,
            matched_ts=matched_ts,
            time_diff_us=abs(matched_ts - target_ts_us),
            TF=matched_frame["TF"],
        )

    # ------------------------------------------------------------------
    # Resource Lifecycle Management
    # ------------------------------------------------------------------
    def close(self):
        self._header_arr = None
        self.frame_buffer = None

        if hasattr(self, "shm") and self.shm is not None:
            self.shm.close()
            self.shm = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def print_terminal_dashboard(shm_name: str = "shm_flange", is_owner: bool = True):
    print(f"[Monitor] '{shm_name}' 공유 메모리 할당/연결 시도 중...")

    try:
        with FlangeShm(shm_name=shm_name, is_owner=is_owner) as reader:
            print("[Monitor] 공유 메모리가 정상 준비되었습니다. C++ 데이터 수신 대기 중...")
            time.sleep(1)

            while True:
                data, seq = reader.get_ordered_data()

                if data is None or len(data) == 0:
                    print("\033[H\033[J", end="")
                    print(f" [Hanwha Robot SHM Monitor] '{shm_name}' 수신 대기 중...")
                    print("공유 메모리 생성 완료. C++ 데이터 수신을 기다리는 중 (sequence == 0)...")
                    time.sleep(0.5)
                    continue

                latest = data[-1]
                oldest = data[0]
                dt_ms = (latest["timestamp_us"] - oldest["timestamp_us"]) / 1000.0

                latest_T = latest["TF"]
                pos_x, pos_y, pos_z = latest_T[0, 3], latest_T[1, 3], latest_T[2, 3]

                # 화면 제자리 덮어쓰기
                print("\033[H\033[J", end="")

                print(
                    "=========================================================================================="
                )
                print(f" [Hanwha Robot SHM Monitor]  누적 프레임: {seq:8d} | 저장된 버퍼 수: {len(data)}/100")
                print(f" 버퍼 시간 span: {dt_ms:8.2f} ms | 최신 Timestamp: {latest['timestamp_us']} us")
                print(
                    "=========================================================================================="
                )
                print(f" [최신 TCP Position]  X: {pos_x:8.2f} mm | Y: {pos_y:8.2f} mm | Z: {pos_z:8.2f} mm")
                print(
                    "------------------------------------------------------------------------------------------"
                )

                print(" [최신 4x4 Homogeneous Transformation Matrix (T_base_tcp)]")
                for r in range(4):
                    print(
                        f"  | {latest_T[r, 0]:8.4f}  {latest_T[r, 1]:8.4f}  {latest_T[r, 2]:8.4f}  {latest_T[r, 3]:10.2f} |"
                    )
                print(
                    "------------------------------------------------------------------------------------------"
                )

                print(" [최근 링버퍼 샘플 목록 (시간순)]")
                print(
                    " Index | Timestamp (us) |   X (mm)   |   Y (mm)   |   Z (mm)   |   R11 (R)  |   R22 (R)  |   R33 (R)  "
                )
                print(
                    "-------+----------------+------------+------------+------------+------------+------------+------------"
                )

                if len(data) > 10:
                    show_indices = list(range(3)) + [-1] + list(range(len(data) - 5, len(data)))
                else:
                    show_indices = list(range(len(data)))

                for idx in show_indices:
                    if idx == -1:
                        print(
                            "  ...  |       ...      |    ...     |    ...     |    ...     |    ...     |    ...     |    ...    "
                        )
                        continue

                    row = data[idx]
                    T = row["TF"]
                    print(
                        f" {idx:4d}  | {row['timestamp_us']:14d} | {T[0,3]:10.2f} | {T[1,3]:10.2f} | {T[2,3]:10.2f} | {T[0,0]:10.4f} | {T[1,1]:10.4f} | {T[2,2]:10.4f}"
                    )

                print(
                    "=========================================================================================="
                )
                print(" Ctrl+C를 누르면 종료됩니다.")

                time.sleep(0.05)  # 20Hz 출력

    except KeyboardInterrupt:
        print("\n[Monitor] 사용자에 의해 모니터링이 종료되었습니다.")


if __name__ == "__main__":
    # Python에서 메모리를 Create하고 실행
    print_terminal_dashboard(shm_name="shm_flange", is_owner=True)
