import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
import os
import warnings

warnings.filterwarnings('ignore')  # 屏蔽matplotlib的无关警告


def gen_gaussian_hmap(coords, crood_scores, shape=256):
    crood_scores[crood_scores <= 0.3] = 0.

    # 1-75-86-81-32, 72-75, 78-75, 32-30-28, 36-34-32, 60-63, 66-69
    skeletons = torch.tensor([
        # 嘴巴
        [1, 75], [75, 86], [86, 81], [72, 75], [78, 75], [86, 72], [86, 78], [81, 72], [81, 78],
        # 下颌
        [32, 30], [30, 28], [36, 34], [34, 32], [28, 5], [36, 4],
        # 眼睛
        [60, 3], [3, 63], [2, 66], [2, 69],
        # 手臂
        [6, 8], [7, 9], [8, 10], [9, 11], [10, 92], [11, 113],
        # 双手
        [92, 93], [93, 94], [94, 95], [95, 96],
        [92, 97], [97, 98], [98, 99], [99, 100],
        [92, 101], [101, 102], [102, 103], [103, 104],
        [92, 105], [105, 106], [106, 107], [107, 108],
        [92, 109], [109, 110], [110, 111], [111, 112],
        [113, 114], [114, 115], [115, 116], [116, 117],
        [113, 118], [118, 119], [119, 120], [120, 121],
        [113, 122], [122, 123], [123, 124], [124, 125],
        [113, 126], [126, 127], [127, 128], [128, 129],
        [113, 130], [130, 131], [131, 132], [132, 133]
    ])
    skeletons_num = len(skeletons)

    x1, y1 = torch.meshgrid(torch.arange(shape), torch.arange(shape))
    grid1 = torch.stack([x1, y1], dim=2)  # [H,H,2]
    grid1 = grid1.repeat((skeletons_num, 1, 1, 1))  # [n,H,H,2]
    x_idx = skeletons[:, 0] - 1
    y_idx = skeletons[:, 1] - 1
    start_coord = coords[x_idx]
    end_coord = coords[y_idx]
    start_score = crood_scores[x_idx]
    end_score = crood_scores[y_idx]
    max_score = torch.min(torch.stack([start_score, end_score], dim=0), dim=0)[0]

    sigma1 = 0.37
    dis = ((grid1 - start_coord[:, None, None, :]) ** 2).sum(dim=-1) ** 0.5 + \
          ((grid1 - end_coord[:, None, None, :]) ** 2).sum(dim=-1) ** 0.5 - \
          (((start_coord - end_coord) ** 2).sum(dim=-1) ** 0.5)[:, None, None]
    hmap_sk = torch.exp(-1 * dis / (2 * sigma1 ** 2)) / (
            sigma1 * (2 * torch.pi) ** 0.5)
    hmap_sk = hmap_sk * max_score[:, None, None] * 50
    hmap = torch.max(hmap_sk, dim=0)[0]
    return hmap


def save_frames_as_images(frames, output_dir, prefix="heatmap", format="png"):
    """
    将一系列 NumPy 帧保存为多张图像文件
    Args:
        frames (list): 包含所有图像帧的列表，每个帧都是一个 NumPy 数组（RGB格式）
        output_dir (str): 输出图像文件夹路径
        prefix (str): 图像文件名前缀
        format (str): 保存格式，支持 png/jpg/jpeg
    """
    # 创建输出文件夹（不存在则创建）
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有帧并保存
    for idx, frame in enumerate(frames):
        # 生成文件名：前缀_帧序号.格式（序号补零，方便排序）
        filename = f"{prefix}_{idx:04d}.{format}"
        save_path = os.path.join(output_dir, filename)

        # 转换为PIL图像并保存（也可以用cv2，这里选择PIL保证色彩正确）
        img = Image.fromarray(frame)
        img.save(save_path)

        # 每保存10帧打印一次进度（可选）
        if (idx + 1) % 10 == 0:
            print(f"已保存 {idx + 1} 帧图像")

    print(f"\n所有图像保存完成！")
    print(f"保存路径：{output_dir}")
    print(f"总帧数：{len(frames)}")
    print(f"文件格式：{format}")


if __name__ == '__main__':
    # ========== 配置参数 ==========
    file_pth = r'F:\SLRdataset\WLASL\body_keypoint_133\00665.npz'  # 关键点文件路径
    output_dir = r'F:\chexiao\project\RAE\result\WLASL_pose'  # 图像保存文件夹
    heatmap_shape = 256  # 热图尺寸
    save_format = "png"  # 保存格式（png/jpg）
    video_name = "00665"  # 文件名前缀（对应原npz文件）

    # ========== 加载关键点数据 ==========
    with np.load(file_pth) as data:
        keypoint = data['keypoint_vedio']
        keypoint_score = data['keypoint_score_vedio']

    # 调整关键点维度（翻转最后一维）
    keypoint = torch.flip(torch.from_numpy(keypoint), dims=[-1])
    keypoint_score = torch.from_numpy(keypoint_score)
    num_frames = keypoint.shape[0]  # 获取总帧数
    print(f"加载到 {num_frames} 帧关键点数据")

    # ========== 逐帧生成热图 ==========
    all_frames = []
    for i in range(num_frames):
        # 生成高斯热图
        heatmap = gen_gaussian_hmap(keypoint[i, ...], keypoint_score[i, :], shape=heatmap_shape)

        # 将热图张量转换为 NumPy 数组并归一化到 0-255 范围
        heatmap_np = heatmap.cpu().numpy()
        # 防止除以零（如果热图全零）
        if heatmap_np.max() == heatmap_np.min():
            normalized_heatmap = np.zeros_like(heatmap_np, dtype=np.uint8)
        else:
            normalized_heatmap = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min())
            normalized_heatmap = (normalized_heatmap * 255).astype(np.uint8)

        # 转换为彩色图像（RGB）：使用viridis色彩映射
        cmap = plt.get_cmap('viridis')
        colored_heatmap = (cmap(normalized_heatmap) * 255).astype(np.uint8)
        colored_heatmap = colored_heatmap[:, :, :3]  # 移除Alpha通道

        # 将彩色帧添加到列表中
        all_frames.append(colored_heatmap)

    # ========== 保存为多张图像 ==========
    save_frames_as_images(
        frames=all_frames,
        output_dir=output_dir,
        prefix=video_name,
        format=save_format
    )