"""Robot-base filtered track states; main.py owns creation and deletion."""
import numpy as np

from common.shm_detection import DetectionShm, FRAME_DTYPE, MAX_OBJECTS

SHM_NAME = "tracking_shm"
TRACK_DTYPE = np.dtype([
    ("track_id", np.uint64),
    ("session_id", np.uint64),  # Changes when the tracker process restarts.
    ("confirmed", np.bool_),
    ("observed", np.bool_),
    ("last_seen", np.uint64),  # Last matched camera timestamp, microseconds.
    ("score", np.float64),
    ("bbox", np.float64, (4,)),
    ("TF", np.float64, (4, 4)),  # Filtered XYZ; latest observed orientation.
    ("velocity", np.float64, (3,)),  # Base-frame m/s.
    ("covariance", np.float64, (6, 6)),  # [x,y,z,vx,vy,vz], SI units.
])
TRACK_FRAME_DTYPE = np.dtype([
    *[(name, FRAME_DTYPE.fields[name][0]) for name in FRAME_DTYPE.names if name != "objects"],
    ("objects", TRACK_DTYPE, (MAX_OBJECTS,)),
])


class TrackingShm(DetectionShm):
    frame_dtype = TRACK_FRAME_DTYPE
    object_dtype = TRACK_DTYPE

    def __init__(self, shm_name=SHM_NAME, is_owner=False):
        super().__init__(shm_name, is_owner)
