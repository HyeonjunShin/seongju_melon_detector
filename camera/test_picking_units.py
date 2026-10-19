import importlib.util
import io
import sys
import types
from contextlib import redirect_stdout
from unittest.mock import patch
from pathlib import Path

import numpy as np


def load_test_picking():
    gemini336 = types.ModuleType("camera.gemini336")
    gemini336.K = np.eye(3, dtype=np.float64)
    gemini336.D = np.zeros(5, dtype=np.float64)
    path = Path(__file__).with_name("test_picking.py")
    spec = importlib.util.spec_from_file_location("test_picking_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"camera.gemini336": gemini336}):
        spec.loader.exec_module(module)
    return module


def test_pixel_to_camera_xyz_converts_depth_millimeters_to_meters():
    picking = load_test_picking()
    depth_mm = np.full((15, 15), 5, dtype=np.uint16)

    xyz_m = picking.pixel_to_camera_xyz(7, 7, depth_mm)

    np.testing.assert_allclose(xyz_m, [0.035, 0.035, 0.005])


def test_color_and_depth_inset_clicks_map_to_source_pixels():
    picking = load_test_picking()
    assert picking.preview_to_pixel(100, 50) == (100, 50)
    assert picking.preview_to_pixel(944, 16) == (2, 2)
    assert picking.preview_to_pixel(969, 28) == (102, 50)
    assert picking.preview_to_pixel(1263, 195) == (1278, 718)
    assert picking.preview_to_pixel(1264, 196) == (1264, 196)
    assert picking.preview_to_pixel(1279, 719) == (1279, 719)
    for x, y in ((-1, 0), (0, -1), (1280, 0), (0, 720)):
        assert picking.preview_to_pixel(x, y) is None


def test_preview_shows_depth_inset_and_coordinates_by_selection():
    picking = load_test_picking()
    color = np.full((720, 1280, 3), 40, dtype=np.uint8)
    depth = np.full_like(color, 200)
    selection = (1200, 700, np.array([.1, .2, .3]),
                 np.array([1., 2., 3.]), np.array([1., 2., 3.4]))
    real_put_text = picking.cv2.putText
    with patch.object(picking.cv2, "putText", wraps=real_put_text) as put_text:
        preview = picking.render_preview(color, depth, selection, 30.)
    np.testing.assert_array_equal(preview[100, 1100], [200, 200, 200])
    np.testing.assert_array_equal(preview[400, 400], [40, 40, 40])
    labels = {call.args[1]: call.args[2] for call in put_text.call_args_list}
    for text in ("Pixel (u, v): 1200, 700", "Camera XYZ (mm): 100.0, 200.0, 300.0",
                 "Robot Base XYZ (mm): 1000.0, 2000.0, 3000.0",
                 "Target XYZ (mm): 1000.0, 2000.0, 3400.0"):
        x, y = labels[text]
        (width, _), _ = picking.cv2.getTextSize(text, picking.cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
        assert 0 <= x < x + width < 1280
        assert 0 < y < 720
        assert x > 600 and y > 500  # Keep coordinates beside the bottom-right click.
    np.testing.assert_array_equal(color, np.full_like(color, 40))


def test_target_uses_camera_reference_rotation_and_base_position():
    picking = load_test_picking()
    # Tilt the flange so camera +Z and base +Z differ.
    flange = np.array([
        [1, 0, 0, 1], [0, 0, -1, 2], [0, 1, 0, 3], [0, 0, 0, 1],
    ], dtype=np.float64)
    original = flange.copy()
    camera_xyz = np.array([0., 0., 1.])
    base_xyz = picking.camera_to_base_xyz(camera_xyz, flange)
    np.testing.assert_allclose(base_xyz, [1.41343014, 1.00774738, 2.989])

    target = picking.build_target_tf(camera_xyz, flange, np.array([0., 0., 1.]))

    np.testing.assert_allclose(target[:3, 3], [1.41343014, 1.00774738, 2.989])
    np.testing.assert_allclose(target[:3, :3], [
        [0., -.939692621, .342020143],
        [0., -.342020143, -.939692621],
        [1., 0., 0.],
    ], atol=1e-9)
    np.testing.assert_array_equal(flange, original)


def test_tilted_depth_plane_gives_inward_camera_normal_and_rotation():
    picking = load_test_picking()
    picking.K = np.array([[200., 0., 7.], [0., 200., 7.], [0., 0., 1.]])
    # Plane .3*x + .4*y + sqrt(.75)*z = sqrt(.75), in meters.
    v, u = np.indices((15, 15))
    nz = np.sqrt(.75)
    depth = 1000. * nz / (.3 * (u - 7) / 200 + .4 * (v - 7) / 200 + nz)
    normal = picking.estimate_surface_normal(7, 7, depth)
    np.testing.assert_allclose(normal, [.3, .4, nz], atol=1e-6)
    target = picking.build_target_tf(np.array([0., 0., 1.]), np.eye(4), normal)
    # Express the transmitted base rotation back in the camera frame.
    rotation_camera = np.linalg.solve(picking.T_FLANGE_CAMERA[:3, :3], target[:3, :3])
    np.testing.assert_allclose(rotation_camera[:, 2], normal, atol=1e-8)
    np.testing.assert_allclose(rotation_camera.T @ rotation_camera, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(np.linalg.det(rotation_camera), 1., atol=1e-8)
    np.testing.assert_allclose(np.trace(rotation_camera), 1 + 2 * nz, atol=1e-8)


def test_normal_ignores_missing_depth_and_distant_background():
    picking = load_test_picking()
    picking.K = np.array([[200., 0., 7.], [0., 200., 7.], [0., 0., 1.]])
    depth = np.full((15, 15), 1000.)
    depth[:, :4] = 2000.
    depth[0, :] = 0.
    depth[1, :] = np.nan
    depth[2, :] = np.inf
    np.testing.assert_allclose(picking.estimate_surface_normal(7, 7, depth), [0., 0., 1.])
    np.testing.assert_allclose(picking.estimate_surface_normal(14, 14, depth), [0., 0., 1.])
    depth[:] = 2000.
    depth[5:10, 5:10] = 1000.
    np.testing.assert_allclose(picking.estimate_surface_normal(7, 7, depth), [0., 0., 1.])


def test_normal_rejects_insufficient_collinear_and_nonplanar_depth():
    picking = load_test_picking()
    picking.K = np.array([[200., 0., 7.], [0., 200., 7.], [0., 0., 1.]])
    for depth in (np.zeros((15, 15)), np.full((15, 15), np.inf)):
        assert picking.estimate_surface_normal(7, 7, depth) is None
    depth = np.zeros((15, 15))
    depth[7, :] = 1000.
    assert picking.estimate_surface_normal(7, 7, depth) is None
    v, u = np.indices((15, 15))
    depth = 1000. + 20. * ((u + v) % 2)
    assert picking.estimate_surface_normal(7, 7, depth) is None
    assert picking.estimate_surface_normal(-1, 7, depth) is None


def test_antiparallel_normal_still_produces_proper_rotation():
    picking = load_test_picking()
    target = picking.build_target_tf(np.zeros(3), np.eye(4), np.array([0., 0., -1.]))
    rotation = np.linalg.solve(picking.T_FLANGE_CAMERA[:3, :3], target[:3, :3])
    np.testing.assert_allclose(rotation[:, 2], [0., 0., -1.], atol=1e-8)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(np.linalg.det(rotation), 1., atol=1e-8)


def test_normal_uses_wider_patch_without_changing_position_depth():
    picking = load_test_picking()
    picking.K = np.array([[200., 0., 15.], [0., 200., 15.], [0., 0., 1.]])
    depth = np.full((31, 31), 1000.)
    depth[8:23, 8:23] = 0.
    depth[15, 15] = 1000.
    np.testing.assert_allclose(picking.estimate_surface_normal(15, 15, depth), [0., 0., 1.])
    # A wider position window would incorrectly select the 2m background.
    depth[:] = 2000.
    depth[8:23, 8:23] = 1000.
    np.testing.assert_allclose(picking.pixel_to_camera_xyz(15, 15, depth), [0., 0., 1.])


def test_normal_failure_logs_point_counts_and_plane_quality():
    picking = load_test_picking()
    picking.K = np.array([[200., 0., 7.], [0., 200., 7.], [0., 0., 1.]])
    v, u = np.indices((15, 15))
    output = io.StringIO()
    with redirect_stdout(output):
        assert picking.estimate_surface_normal(7, 7, 1000. + 20. * ((u + v) % 2)) is None
    log = output.getvalue()
    for field in ('reason=thick_cloud', 'valid=225', 'kept=225', 'rms_mm=', 'thickness_ratio='):
        assert field in log, log
    assert 'rms_mm=n/a' not in log
    assert 'thickness_ratio=n/a' not in log
    output = io.StringIO()
    with redirect_stdout(output):
        assert picking.estimate_surface_normal(7, 7, np.zeros((15, 15))) is None
    for field in ('reason=insufficient_depth', 'valid=0', 'kept=0', 'rms_mm=n/a'):
        assert field in output.getvalue(), output.getvalue()


def test_invalid_depth_or_pixel_has_no_camera_target():
    picking = load_test_picking()
    assert picking.pixel_to_camera_xyz(7, 7, np.zeros((15, 15))) is None
    assert picking.pixel_to_camera_xyz(7, 7, np.full((15, 15), np.inf)) is None
    assert picking.pixel_to_camera_xyz(-1, 7, np.ones((15, 15))) is None


if __name__ == "__main__":
    tests = [value for name, value in globals().copy().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} picking checks passed")
