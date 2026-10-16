import fcntl
from multiprocessing import shared_memory

import posix_ipc

from common.shm_camera import CameraShm
from common.shm_flange import FlangeShm
from common.shm_target import TargetShm


class ProductIpc:
    def __init__(
        self,
        camera_shm_name: str = "camera_shm",
        flange_shm_name: str = "shm_flange",
        target_shm_name: str = "shm_target",
        camera_sem_name: str = "/camera_sem",
        lock_path: str = "/tmp/seongju_melon_detector.lock",
    ):
        self.camera_shm_name = camera_shm_name
        self.flange_shm_name = flange_shm_name
        self.target_shm_name = target_shm_name
        self.camera_sem_name = camera_sem_name
        self.lock_path = lock_path

        self.camera_shm = None
        self.flange_shm = None
        self.target_shm = None
        self.camera_sem = None
        self._lock_file = None

    def start(self) -> None:
        if self._lock_file is not None:
            return

        self._lock_file = open(self.lock_path, "a+")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_file.close()
            self._lock_file = None
            raise RuntimeError("main.py가 이미 실행 중입니다.") from exc

        try:
            self._remove_stale_resources()
            self.camera_shm = CameraShm(self.camera_shm_name, is_owner=True)
            self.flange_shm = FlangeShm(self.flange_shm_name, is_owner=True)
            self.target_shm = TargetShm(self.target_shm_name, is_owner=True)
            self.camera_sem = posix_ipc.Semaphore(
                self.camera_sem_name,
                flags=posix_ipc.O_CREX,
                initial_value=0,
            )
            self.camera_shm.status = False
            self.flange_shm.status = False
            self.target_shm.status = False
        except Exception:
            self.close()
            raise

    def _remove_stale_resources(self) -> None:
        for name in (
            self.camera_shm_name,
            self.flange_shm_name,
            self.target_shm_name,
        ):
            try:
                stale = shared_memory.SharedMemory(name=name, create=False)
            except FileNotFoundError:
                continue
            stale.unlink()
            stale.close()

        try:
            posix_ipc.unlink_semaphore(self.camera_sem_name)
        except posix_ipc.ExistentialError:
            pass

    def close(self) -> None:
        for resource in (self.camera_shm, self.flange_shm, self.target_shm):
            if resource is not None:
                resource.status = False

        if self.camera_sem is not None:
            self.camera_sem.close()
            try:
                posix_ipc.unlink_semaphore(self.camera_sem_name)
            except posix_ipc.ExistentialError:
                pass
            self.camera_sem = None

        for attribute in ("target_shm", "flange_shm", "camera_shm"):
            resource = getattr(self, attribute)
            if resource is not None:
                resource.close()
                setattr(self, attribute, None)

        if self._lock_file is not None:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
