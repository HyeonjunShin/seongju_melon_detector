from model import D

detector = Detector(model_path="./model_int8.xml", conf_threshold=0.25)

    camera_shm = CameraShm(shm_name=camera_shm_name, is_owner=False)
    flange_shm = FlangeShm(shm_name=flange_shm_name, is_owner=False)
    target_shm = TargetShm(shm_name=target_shm_name, is_owner=False)

    try:
        while not stop_signal.is_set():
            if not camera_shm.status:
                target_shm.status = False
                continue
            else:
                target_shm.status = True

            if not frame_ready_signal.wait(timeout=0.035):
                print("Camera timeout!!")
                continue
            frame_ready_signal.clear()

            current_frame = camera_shm.read()

            timestamp = current_frame.timestamp
            color = current_frame.color.copy()
            depth = current_frame.depth.copy()

            # 💡 1. OpenVINO INT8 포즈 추론 (Score, BBox, Keypoints)
            score, bbox, kpts = detector.detect(color)

            detected = False
            best_score = float(score)
            best_bbox = (
                bbox.copy() if bbox is not None else np.zeros((4,), dtype=np.float64)
            )
            best_TF = np.eye(4, dtype=np.float64)

            # 💡 2. 검출 성공 시 Body Centroid & Orientation 회전 행렬 계산
            if score > 0.25 and kpts is not None:
                centroid_cam, quat_cam = compute_pose_with_kpts(
                    depth_img=depth, kpts=kpts, K=K, DIST_COEFFS=D, vis_thresh=0.5
                )

                if centroid_cam is not None:
                    # 3. 카메라 좌표계 상의 $4 \times 4$ Transform Matrix 구성
                    rot_cam_mat = R.from_quat(
                        quat_cam
                    ).as_matrix()  # Quaternion -> 3x3 Rotation Matrix

                    T_cam = np.eye(4, dtype=np.float64)
                    T_cam[:3, :3] = rot_cam_mat
                    T_cam[:3, 3] = centroid_cam

                    # 4. Camera -> Tool (Flange) 변환
                    T_flange_target = TOOL_CAM @ T_cam

                    # 5. Tool (Flange) -> Base Robot Coordinate 변환
                    flange_pose = flange_shm.read(timestamp)
                    TF_FLANGE = flange_pose.TF.copy()

                    best_TF = TF_FLANGE @ T_flange_target
                    detected = True

            # 6. 로봇 컨트롤러 공유 메모리에 TF 전송
            target_shm.write(
                timestamp=timestamp,
                detected=detected,
                score=best_score,
                bbox=best_bbox.astype(np.float64),
                TF=best_TF,
            )

    finally:
        target_shm.status = False
        target_shm.close()
