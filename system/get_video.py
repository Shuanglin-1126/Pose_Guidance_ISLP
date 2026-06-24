import cv2
import os
import glob
from typing import List, Optional
import numpy as np


class Image2Video:
    """图片合成视频工具类"""

    def __init__(self):
        # 默认参数（可根据需求调整）
        self.fps = 1  # 视频帧率（每秒播放的图片数）
        self.video_size = (1920, 1080)  # 视频分辨率（宽, 高）
        self.video_format = "mp4"  # 输出视频格式
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 编码格式（适配mp4）

    def get_sorted_images(self, img_dir: str, img_formats: List[str] = None) -> List[str]:
        """
        读取指定目录下的图片并按文件名排序
        :param img_dir: 图片目录路径
        :param img_formats: 支持的图片格式，默认['jpg', 'jpeg', 'png', 'bmp']
        :return: 排序后的图片路径列表
        """
        if img_formats is None:
            img_formats = ['jpg', 'jpeg', 'png', 'bmp']

        # 拼接所有支持的图片路径
        img_paths = []
        for fmt in img_formats:
            img_paths.extend(glob.glob(os.path.join(img_dir, f"*.{fmt}")))
            img_paths.extend(glob.glob(os.path.join(img_dir, f"*.{fmt.upper()}")))

        # 按文件名排序（确保图片顺序正确）
        img_paths.sort(key=lambda x: os.path.basename(x))

        if not img_paths:
            raise FileNotFoundError(f"目录 {img_dir} 下未找到支持的图片文件")
        return img_paths

    def resize_image(self, img: cv2.Mat) -> cv2.Mat:
        """将图片缩放/裁剪到指定视频尺寸"""
        h, w = img.shape[:2]
        target_w, target_h = self.video_size

        # 计算缩放比例，保持宽高比
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 创建空白画布，将缩放后的图片居中放置
        img_padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        img_padded[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = img_resized

        return img_padded

    def convert(self, img_dir: str, output_path: str, fps: Optional[int] = None, video_size: Optional[tuple] = None):
        """
        核心方法：将图片合成视频
        :param img_dir: 存放图片的目录
        :param output_path: 输出视频路径（如./output.mp4）
        :param fps: 自定义帧率，覆盖默认值
        :param video_size: 自定义视频尺寸，覆盖默认值（宽, 高）
        """
        # 更新参数（如果传入自定义值）
        if fps is not None:
            self.fps = fps
        if video_size is not None:
            self.video_size = video_size

        # 1. 读取并排序图片
        print(f"正在读取图片目录：{img_dir}")
        img_paths = self.get_sorted_images(img_dir)
        print(f"共找到 {len(img_paths)} 张图片")

        # 2. 初始化视频写入器
        # 先读取第一张图片，确认通道数（兼容灰度图）
        first_img = cv2.imread(img_paths[0])
        if len(first_img.shape) == 2:  # 灰度图转彩色
            first_img = cv2.cvtColor(first_img, cv2.COLOR_GRAY2BGR)
        first_img_resized = self.resize_image(first_img)

        video_writer = cv2.VideoWriter(
            output_path,
            self.fourcc,
            self.fps,
            self.video_size
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"无法创建视频文件：{output_path}，请检查路径和格式")

        # 3. 逐张写入图片
        print("开始合成视频...")
        for idx, img_path in enumerate(img_paths):
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告：跳过损坏的图片 {img_path}")
                continue

            # 灰度图转彩色
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # 调整图片尺寸
            img_resized = self.resize_image(img)

            # 写入视频帧
            video_writer.write(img_resized)

            # 打印进度
            if (idx + 1) % 10 == 0:
                print(f"已处理 {idx + 1}/{len(img_paths)} 张图片")

        # 4. 释放资源
        video_writer.release()
        print(f"视频合成完成！输出路径：{output_path}")


# ---------------------- 示例使用 ----------------------
if __name__ == "__main__":
    # 初始化工具类
    converter = Image2Video()

    # 自定义参数（按需修改）
    IMG_DIR = r"D:\SLR_dataset\CSL_Daily\video_100\S000017_P0000_T00"  # 存放图片的文件夹路径
    OUTPUT_VIDEO = "./video/csl_daily_049.mp4"  # 输出视频路径
    FPS = 2  # 每秒播放2张图片
    VIDEO_SIZE = (1280, 720)  # 视频分辨率（720P）

    # 执行合成
    try:
        converter.convert(
            img_dir=IMG_DIR,
            output_path=OUTPUT_VIDEO,
            fps=FPS,
            video_size=VIDEO_SIZE
        )
    except Exception as e:
        print(f"合成失败：{e}")