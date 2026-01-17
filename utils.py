import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

model = YOLO(model_path)
print(f"成功加载模型{model_path}")
# 创建管道
pipeline = rs.pipeline()

# 创建配置对象
config = rs.config()

# 启用彩色和深度流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 开始流传输
profile = pipeline.start(config)

# 获取深度传感器的深度标尺（单位：米）
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth Scale: {depth_scale}")

align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # 等待一组帧（深度和彩色）
        frames = pipeline.wait_for_frames()

        # 对齐帧
        aligned_frames = align.process(frames)

        # 获取对齐后的帧
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # 转换为numpy数组 Cir:低运行效率段，考虑优化
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        with
#预留接口，从yolo模型中获取目标框中心点，提取中心点坐标，再进行映射

        # 应用颜色映射到深度图像（用于可视化）
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET)

        # 显示图像
        cv2.imshow('Color', color_image)
        cv2.imshow('Depth', depth_colormap)

        if cv2.waitKey(1) == ord('q'):
            break
finally:
    # 停止管道
    pipeline.stop()
    cv2.destroyAllWindows()
