from pyorbbecsdk import (
    Pipeline,
    Config,
    Context,
    OBSensorType,
    OBFormat,
    OBPropertyID,
    OBAlignMode,
    DepthFrame,
    ColorFrame,
    VideoFrame,
)
import cv2
import numpy as np
import os

selected_points = []


def frame_to_bgr_image(frame: VideoFrame):
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image


def save_depth_frame(frame: DepthFrame, timestamp):
    if frame is None:
        return
    width = frame.get_width()
    height = frame.get_height()
    # timestamp = frame.get_system_timestamp()
    scale = frame.get_depth_scale()
    depth_format = frame.get_format()
    if depth_format != OBFormat.Y16:
        print("depth format is not Y16")
        return
    data = np.frombuffer(frame.get_data(), dtype=np.uint16)
    data = data.reshape((height, width))
    data = data.astype(np.float32) * (scale * 0.001)  # meters
    save_image_dir = os.path.join(os.getcwd(), "outputs/depth_images")
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir, exist_ok=True)
    filename = save_image_dir + "/depth_{}_{}_{}.npy".format(width, height, timestamp)
    np.save(filename, data)
    print(f"Depth saved: {filename}")


def save_color_frame(frame: ColorFrame, timestamp):
    if frame is None:
        return
    width = frame.get_width()
    height = frame.get_height()
    # timestamp = frame.get_system_timestamp()
    save_image_dir = os.path.join(os.getcwd(), "outputs/color_images")
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir, exist_ok=True)
    filename = save_image_dir + "/color_{}_{}_{}.png".format(width, height, timestamp)
    image = frame_to_bgr_image(frame)
    if image is None:
        print("failed to convert frame to image")
        return
    cv2.imwrite(filename, image)
    print(f"Color saved: {filename}")


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_points) == 2:
            selected_points.clear()
        selected_points.append((x, y))


def main():
    color_shape = (848, 530, 3, 60)
    depth_shape = (848, 530, 1, 60)
    config_file_path = "./config/gemini355_settings.json"

    # color_shape = (424, 240, 3, 60)
    # depth_shape = (480, 270, 1, 60)
    # config_file_path = "./config/gemini336_settings.json"

    ctx = Context()
    device_list = ctx.query_devices()
    pipeline = Pipeline()
    config = Config()
    device = pipeline.get_device()

    pipeline.enable_frame_sync()

    # profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    # for i in range(len(profile_list)):
    #     color_profile = profile_list[i]
    #     if color_profile.get_format() != OBFormat.RGB:
    #         continue
    #     if color_profile.get_fps() != 60:
    #         continue
    #     hw_depth_list = pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)
    #     if len(hw_depth_list) == 0:
    #         continue
    #     else:
    #         print(color_profile)
    #     for i in range(len(hw_depth_list)):
    #         depth_profile = hw_depth_list[i]
    #         if depth_profile.get_fps() != 60:
    #             continue
    #         print("   " + str(hw_depth_list[i]))

    # config.enable_stream(profile_list[0])
    # config.enable_stream(profile_list[0])

    # print(ctx)
    # print(device_list.get_count())
    # print(pipeline)
    # exit()

    color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profile_list.get_video_stream_profile(
        color_shape[0], color_shape[1], OBFormat.MJPG, color_shape[3]
    )

    depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profile_list.get_video_stream_profile(
        depth_shape[0], depth_shape[1], OBFormat.Y16, depth_shape[3]
    )

    config.enable_stream(depth_profile)
    config.enable_stream(color_profile)
    config.set_align_mode(OBAlignMode.HW_MODE)

    device.load_preset_from_json_file(config_file_path)
    pipeline.start(config)

    current_depth_work_mode = device.get_depth_work_mode()
    assert current_depth_work_mode is not None
    print("Current depth work mode: ", current_depth_work_mode)

    param = pipeline.get_camera_param()
    intr = param.rgb_intrinsic

    fx, fy = intr.fx, intr.fy
    cx, cy = intr.cx, intr.cy

    cv2.namedWindow("Alignment Check (Blended)")
    cv2.setMouseCallback("Alignment Check (Blended)", mouse_callback)

    cv2.namedWindow("Color Stream")
    cv2.setMouseCallback("Color Stream", mouse_callback)

    print("Click two points | third click: new pair | r: reset | q: quit")

    while True:
        try:
            frames = pipeline.wait_for_frames(1000)
            if not frames:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            c_width, c_height = color_frame.get_width(), color_frame.get_height()
            d_width, d_height = depth_frame.get_width(), depth_frame.get_height()

            color_img = cv2.imdecode(color_frame.get_data(), cv2.IMREAD_COLOR_RGB)
            # color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            # color_img = color_data.reshape((c_height, c_width, 3))

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            raw_depth_img = depth_data.reshape((d_height, d_width))
            # print(raw_depth_img.max())
            # invalid_mask = (raw_depth_img == 65535) | (raw_depth_img == 0)

            depth_img = (
                raw_depth_img.astype(np.float32) * depth_frame.get_depth_scale()
            )  # mm
            depth_img = depth_img * 0.001  # m
            # depth_img[invalid_mask] = 0

            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

            depth_normalized = cv2.normalize(
                depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            blended_img = cv2.addWeighted(color_img, 0.5, depth_colormap, 0.5, 0)

            for index, (u, v) in enumerate(selected_points, start=1):
                if not (0 <= u < d_width and 0 <= v < d_height):
                    continue

                z = float(depth_img[v, u])
                if z == 0:
                    text = f"P{index} pixel=({u}, {v}) Invalid depth"
                elif (intr.width, intr.height) != (d_width, d_height):
                    text = f"P{index}: RGB intrinsics resolution mismatch"
                else:
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    text = f"P{index} ({u}, {v}) " f"XYZ=({x:.4f}, {y:.4f}, {z:.4f}) m"

                point_color = (0, 255, 0) if index == 1 else (0, 255, 255)
                for image in (color_img, blended_img):
                    cv2.circle(image, (u, v), 5, point_color, -1)
                    cv2.putText(
                        image,
                        f"P{index}",
                        (u + 8, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        point_color,
                        2,
                    )
                    cv2.putText(
                        image,
                        text,
                        (10, 28 * index),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        point_color,
                        1,
                    )

            cv2.imshow("Color Stream", color_img)
            cv2.imshow("Depth (Colormap)", depth_colormap)
            cv2.imshow("Alignment Check (Blended)", blended_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                selected_points.clear()
            if key == 32:
                timestamp = color_frame.get_system_timestamp_us()
                save_color_frame(color_frame, timestamp)
                save_depth_frame(depth_frame, timestamp)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Runtime error: {e}")
            continue

    # 종료 처리
    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
