"""
python visualization.py \ \
  --stl ./gripper1.stl \
  --xyz 0.07141 -0.011 0.05256 \
  --rpy 0 20 0 \
  --xyz2 0.0 0.0 0.355 \
  --rpy2 0 0 0

python visualization.py --no-gui \
  --xyz 0.07141 -0.011 0.05256 \
  --rpy 0 20 0 \
  --xyz2 0.0 0.0 0.355 \
  --rpy2 0 0 0
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stl",
        type=Path,
        help="STL 파일 경로 (GUI 모드에서 필요, 상대 경로는 실행 디렉터리 기준)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="GUI를 실행하지 않고 변환 행렬만 출력",
    )
    parser.add_argument(
        "--xyz",
        nargs=3,
        type=float,
        default=[0.06146, 0.001, 0.03085],
        metavar=("X", "Y", "Z"),
        help="STL 원점 기준 위치 (STL과 같은 단위)",
    )
    parser.add_argument(
        "--rpy",
        nargs=3,
        type=float,
        default=[0, 20, 0],
        metavar=("RX", "RY", "RZ"),
        help="고정 X → Y → Z축 회전각 (degree)",
    )
    parser.add_argument(
        "--xyz2",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="추가 좌표계 위치 (STL 원점 기준)",
    )
    parser.add_argument(
        "--rpy2",
        nargs=3,
        type=float,
        metavar=("RX", "RY", "RZ"),
        help="추가 좌표계 회전각 (degree, 기본: 0 0 0)",
    )
    parser.add_argument(
        "--size", type=float, default=0.05, help="좌표축 길이 (STL과 같은 단위)"
    )
    args = parser.parse_args()
    if not np.all(np.isfinite([*args.xyz, *args.rpy, args.size])) or args.size <= 0:
        parser.error("위치와 각도는 유한한 수, size는 양수여야 합니다.")
    if args.rpy2 is not None and args.xyz2 is None:
        parser.error("--rpy2는 --xyz2와 함께 사용해야 합니다.")
    if not np.all(np.isfinite((args.xyz2 or []) + (args.rpy2 or []))):
        parser.error("추가 좌표계 위치와 각도는 유한한 수여야 합니다.")

    rx, ry, rz = np.deg2rad(args.rpy)
    # 열벡터 기준: 고정 X축, Y축, Z축 순서로 회전한다.
    R = o3d.geometry.get_rotation_matrix_from_zyx([rz, ry, rx])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = args.xyz

    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.size)
    camera_frame.transform(T)
    model_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=args.size * 0.6
    )
    if args.xyz2 is not None:
        rx2, ry2, rz2 = np.deg2rad(args.rpy2 or [0, 0, 0])
        T2 = np.eye(4)
        T2[:3, :3] = o3d.geometry.get_rotation_matrix_from_zyx([rz2, ry2, rx2])
        T2[:3, 3] = args.xyz2
        extra_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.size)
        extra_frame.transform(T2)

    print("위치 단위: 입력 좌표와 동일 / 회전각: degree / 축 색상: X=빨강, Y=초록, Z=파랑")
    print("T_model_from_camera (p_model = T @ p_camera):")
    print(T)
    print("T_camera_from_model:")
    print(np.linalg.inv(T))
    if args.xyz2 is not None:
        print("T_model_from_frame2:")
        print(T2)

    if args.no_gui:
        return

    if args.stl is None:
        parser.error("--stl은 GUI 모드에서 필수입니다.")
    mesh_path = args.stl.expanduser()
    if not mesh_path.is_file():
        parser.error(f"STL 파일이 없습니다: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"STL을 읽을 수 없습니다: {mesh_path}")
    mesh.compute_vertex_normals()

    from open3d.visualization import gui

    gui.Application.instance.initialize()
    vis = o3d.visualization.O3DVisualizer("handEyeG - Camera Frame", 1920, 1080)
    vis.show_settings = True
    vis.add_geometry("handEyeG_Mesh", mesh)
    vis.add_geometry("STL_Origin", model_frame)
    vis.add_geometry("RGB_Camera", camera_frame)
    vis.add_3d_label([0, 0, 0], "STL origin")
    vis.add_3d_label(args.xyz, "RGB camera")
    if args.xyz2 is not None:
        vis.add_geometry("Frame2", extra_frame)
        vis.add_3d_label(args.xyz2, "Frame 2")
    vis.reset_camera_to_default()
    gui.Application.instance.add_window(vis)
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
