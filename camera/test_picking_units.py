import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def load_test_picking():
    gemini336 = types.ModuleType("camera.gemini336")
    gemini336.K = np.eye(3, dtype=np.float64)
    gemini336.D = np.zeros(5, dtype=np.float64)
    sys.modules["camera.gemini336"] = gemini336
    sys.modules.setdefault("posix_ipc", types.ModuleType("posix_ipc"))

    path = Path(__file__).with_name("test_picking.py")
    spec = importlib.util.spec_from_file_location("test_picking_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pixel_to_camera_xyz_converts_depth_millimeters_to_meters():
    picking = load_test_picking()
    depth_mm = np.full((15, 15), 5, dtype=np.uint16)

    xyz_m = picking.pixel_to_camera_xyz(7, 7, depth_mm)

    np.testing.assert_allclose(xyz_m, [0.035, 0.035, 0.005])
