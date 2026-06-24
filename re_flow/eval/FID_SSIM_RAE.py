import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
import torchvision.transforms as transforms
from scipy import linalg
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import os
import time
import scipy
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from scipy.ndimage import convolve

from src.utils.train_utils import parse_configs
from src.utils.model_utils import instantiate_from_config


# ====================== 工具函数：纯numpy实现SSIM ======================
# def calculate_ssim(img1, img2, data_range=255):
#     """纯numpy实现SSIM计算（灰度图），兼容scipy.ndimage.convolve的mode参数"""
#     win_size = 11
#     win_sigma = 1.5
#     # 生成2D高斯核
#     x = np.arange(win_size) - win_size // 2
#     win_1d = np.exp(-x ** 2 / (2 * win_sigma ** 2))
#     win_1d = win_1d / win_1d.sum()
#     win_2d = np.outer(win_1d, win_1d)  # 11×11高斯核
#
#     # 关键修复：使用constant模式（填充0），手动裁剪有效区域（等效于valid模式）
#     # 步骤1：用constant模式卷积（填充0）
#     mu1_full = convolve(img1, win_2d, mode='constant', cval=0.0)
#     mu2_full = convolve(img2, win_2d, mode='constant', cval=0.0)
#
#     # 步骤2：手动裁剪出valid区域（去掉填充部分，等效于valid模式）
#     pad = win_size // 2  # 5像素填充
#     mu1 = mu1_full[pad:-pad, pad:-pad]  # 256×256 → 246×246
#     mu2 = mu2_full[pad:-pad, pad:-pad]
#
#     # 计算方差和协方差（同样裁剪有效区域）
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
#
#     sigma1_sq_full = convolve(img1 ** 2, win_2d, mode='constant', cval=0.0)
#     sigma1_sq = sigma1_sq_full[pad:-pad, pad:-pad] - mu1_sq
#
#     sigma2_sq_full = convolve(img2 ** 2, win_2d, mode='constant', cval=0.0)
#     sigma2_sq = sigma2_sq_full[pad:-pad, pad:-pad] - mu2_sq
#
#     sigma12_full = convolve(img1 * img2, win_2d, mode='constant', cval=0.0)
#     sigma12 = sigma12_full[pad:-pad, pad:-pad] - mu1_mu2
#
#     # SSIM核心公式
#     C1 = (0.01 * data_range) ** 2
#     C2 = (0.03 * data_range) ** 2
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()


def calculate_ssim(img1, img2, data_range=255):
    """修复版 SSIM，保证输出在 [0,1] 之间"""
    win_size = 11
    win_sigma = 1.5

    # 强制转 float，避免 uint8 溢出
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 生成高斯核
    x = np.arange(win_size) - win_size // 2
    win_1d = np.exp(-x ** 2 / (2 * win_sigma ** 2))
    win_1d /= win_1d.sum()
    win_2d = np.outer(win_1d, win_1d)

    pad = win_size // 2

    # 卷积（same 模式，最后再 crop，更稳定）
    mu1 = convolve(img1, win_2d, mode='constant', cval=0)
    mu2 = convolve(img2, win_2d, mode='constant', cval=0)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1 = convolve(img1 ** 2, win_2d, mode='constant', cval=0) - mu1_sq
    sigma2 = convolve(img2 ** 2, win_2d, mode='constant', cval=0) - mu2_sq
    sigma12 = convolve(img1 * img2, win_2d, mode='constant', cval=0) - mu1_mu2

    # 裁剪有效区域
    mu1 = mu1[pad:-pad, pad:-pad]
    mu2 = mu2[pad:-pad, pad:-pad]
    sigma1 = sigma1[pad:-pad, pad:-pad]
    sigma2 = sigma2[pad:-pad, pad:-pad]
    sigma12 = sigma12[pad:-pad, pad:-pad]

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # SSIM 公式
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1 + sigma2 + C2) + 1e-8)

    # 🔥 关键：强制限制在 [0,1]，防止数值误差
    ssim_map = np.clip(ssim_map, 0.0, 1.0)

    return ssim_map.mean()

# ====================== 适配新版本的InceptionV3特征提取器 ======================
class InceptionV3FeatureExtractor(nn.Module):
    """适配torchvision新版本的InceptionV3特征提取器（输出2048维特征）"""

    def __init__(self, device):
        super().__init__()
        # 加载预训练模型（接受默认的aux_logits=True）
        self.inception = inception_v3(pretrained=True)
        self.device = device

        # 裁剪模型：保留到avgpool层，丢弃分类层和辅助分类头
        # 逐层遍历，直到找到AvgPool2d层
        self.features = nn.Sequential()
        for name, module in self.inception.named_children():
            if name == 'avgpool':
                self.features.add_module(name, module)
                break
            # 跳过辅助分类头
            if name != 'AuxLogits':
                self.features.add_module(name, module)

        self.features.eval().to(device)

    @torch.no_grad()
    def forward(self, x):
        """
        输入：[batch, 3, 299, 299] 的归一化图像
        输出：[batch, 2048] 的特征向量
        """
        # 前向传播到avgpool层
        feat = self.features(x)
        # 展平特征：[batch, 2048, 1, 1] → [batch, 2048]
        feat = feat.view(feat.size(0), -1)
        return feat


# ====================== 核心类：FID+SSIM 计算器（仅最终算FID） ======================
class OnlineFIDSSIMCalculator:
    def __init__(
            self,
            real_dataset,  # 真实图像DataLoader
            gen_model,  # 生成模型
            log_path="./gen_metrics.log",  # 日志文件路径
            device='cuda' if torch.cuda.is_available() else 'cpu',
            batch_size=32  # 批次大小
    ):
        # 基础配置
        self.device = device
        self.batch_size = batch_size
        self.gen_model = gen_model.eval()  # 生成模型设为评估模式
        self.real_dataset = real_dataset
        self.log_path = log_path

        # 计算总图像数（批次数 × 批次大小）
        self.total_real_imgs = len(real_dataset) * batch_size
        self.total_gen_imgs = 0  # 实时更新总生成图像数

        # 初始化日志文件
        self._init_log_file()

        # ---------------------- FID相关初始化（适配新版本） ----------------------
        # 初始化InceptionV3特征提取器
        # self.inception_extractor = InceptionV3FeatureExtractor(device)

        # InceptionV3图像预处理（严格匹配官方要求）
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),  # 先转Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 提前计算真实数据的FID特征（仅算一次，避免重复）
        self._write_log("开始提取真实数据FID特征...")
        self.real_mu, self.real_sigma = self._compute_real_stats()
        self._write_log("✅ 真实数据FID特征提取完成！")

        # 累积变量初始化
        self.gen_feats = []  # 累积生成图像的FID特征
        self.ssim_scores = []  # 累积每批SSIM结果
        self.batch_count = 0  # 记录当前批次号

    def _init_log_file(self):
        """初始化日志文件：创建目录+写入日志头（避免覆盖历史记录）"""
        # 创建日志目录（如果不存在）
        log_dir = os.path.dirname(self.log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 写入日志头（追加模式）
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"生成指标记录 | 开始时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 60}\n\n")

    def _write_log(self, content):
        """通用日志写入函数：同时写入文件+打印到控制台"""
        log_content = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {content}"
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(log_content + "\n")
        print(log_content)

    @torch.no_grad()
    def _compute_real_stats(self):
        """提前计算真实数据集的FID特征均值和协方差（仅算一次）"""
        real_feats = []
        for batch in tqdm(self.real_dataset, desc="提取真实数据特征"):
            # 适配DataLoader输出
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(self.device)

            # 对每张图像单独做预处理（避免维度问题）
            fid_imgs = []
            for img in imgs:
                # 转回PIL图像做预处理（匹配preprocess的输入要求）
                pil_img = transforms.ToPILImage()(img.cpu())
                fid_img = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                fid_imgs.append(fid_img)
            fid_imgs = torch.cat(fid_imgs, dim=0)

            # 提取特征（输出：[batch, 2048]）
            feats = self.inception_extractor(fid_imgs).cpu().numpy()
            # 处理单张图像的情况（确保维度是[batch, 2048]）
            if len(feats.shape) == 1:
                feats = feats.reshape(1, -1)
            real_feats.append(feats)


        # 合并特征并计算统计量
        real_feats = np.concatenate(real_feats, axis=0)
        mu = real_feats.mean(axis=0)
        sigma = np.cov(real_feats, rowvar=False)
        return mu, sigma

    @torch.no_grad()
    def _extract_gen_feat(self, gen_imgs):
        """提取单批生成图像的FID特征（适配修复后的InceptionV3）"""
        # gen_imgs是[batch, 3, H, W]的Tensor（值域[0,1]）
        fid_imgs = []
        for img in gen_imgs:
            pil_img = transforms.ToPILImage()(img.cpu())
            fid_img = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            fid_imgs.append(fid_img)
        fid_imgs = torch.cat(fid_imgs, dim=0)

        feats = self.inception_extractor(fid_imgs).cpu().numpy()
        if len(feats.shape) == 1:
            feats = feats.reshape(1, -1)
        return feats

    def _calculate_fid(self, gen_feats):
        """计算最终FID（基于提前算好的真实特征）"""
        if len(gen_feats) == 0:
            return 0.0
        # 合并生成特征
        gen_feats = np.concatenate(gen_feats, axis=0)
        gen_mu = gen_feats.mean(axis=0)
        gen_sigma = np.cov(gen_feats, rowvar=False)

        # FID核心公式（添加正则项避免数值错误）
        cov_sqrt, _ = linalg.sqrtm(self.real_sigma @ gen_sigma, disp=False)
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real
        cov_sqrt += np.eye(cov_sqrt.shape[0]) * 1e-6  # 防止奇异矩阵
        fid_score = np.sum((self.real_mu - gen_mu) ** 2) + np.trace(self.real_sigma + gen_sigma - 2 * cov_sqrt)
        return fid_score

    def _tensor_to_pil_gray(self, tensor_img):
        """Tensor图像转PIL灰度图（numpy uint8格式）"""
        pil_img = transforms.ToPILImage()(tensor_img.clamp(0, 1).cpu())
        pil_gray = pil_img.convert('L')
        return np.array(pil_gray, dtype=np.uint8)

    def _calculate_batch_ssim(self, real_imgs, gen_imgs):
        """计算单批图像的SSIM（纯PIL+numpy实现）"""
        batch_ssim = []
        for r_tensor, g_tensor in zip(real_imgs, gen_imgs):
            r_gray = self._tensor_to_pil_gray(r_tensor)
            g_gray = self._tensor_to_pil_gray(g_tensor)
            ssim_score = calculate_ssim(r_gray, g_gray, data_range=255)
            batch_ssim.append(ssim_score)
        return np.mean(batch_ssim)

    @torch.no_grad()
    def _recon_image(self, real_image):
        """生成重建图像"""
        gen_image = self.gen_model.decode(self.gen_model.encode(real_image))
        return gen_image.clamp(0.0, 1.0)

    @torch.no_grad()
    def step(self):
        """主流程：逐批生成+算SSIM+累积FID特征"""
        for batch_idx, batch in tqdm(enumerate(self.real_dataset), total=len(self.real_dataset), desc="生成并计算SSIM"):
            self.batch_count += 1
            # 适配DataLoader输出
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(self.device).clamp(0, 1)  # 原始图像（[0,1]值域）
            self.total_gen_imgs += imgs.shape[0]

            # 生成重建图像
            gen_imgs = self._recon_image(imgs)

            # 提取生成图像的FID特征并累积
            gen_feat = self._extract_gen_feat(gen_imgs)
            self.gen_feats.append(gen_feat)

            # 计算并记录SSIM
            current_ssim = self._calculate_batch_ssim(imgs, gen_imgs)
            self.ssim_scores.append(current_ssim)
            # 记录单批SSIM到日志（修正生成图像数计算）
            self._write_log(f"📌 批次 {batch_idx} | 累计生成图像数 {self.total_gen_imgs} | SSIM: {current_ssim:.4f}")

    def get_final_metrics(self):
        """生成完成：计算最终FID+汇总日志"""
        # 1. 计算最终指标
        final_fid = self._calculate_fid(self.gen_feats)
        avg_ssim = np.mean(self.ssim_scores) if self.ssim_scores else 0.0
        std_ssim = np.std(self.ssim_scores) if self.ssim_scores else 0.0

        # 2. 写入最终汇总日志
        self._write_log("\n" + "=" * 60)
        self._write_log("📋 生成指标最终汇总")
        self._write_log(f"总生成图像数：{self.total_gen_imgs}")
        self._write_log(f"最终FID（越小越好）：{final_fid:.4f}")
        self._write_log(f"平均SSIM（越大越好）：{avg_ssim:.4f} (标准差：{std_ssim:.4f})")
        self._write_log(f"所有批次SSIM列表：{[round(s, 4) for s in self.ssim_scores]}")
        self._write_log("=" * 60 + "\n")

        # 3. 返回最终指标字典
        final_metrics = {
            "final_fid": final_fid,
            "avg_ssim": avg_ssim,
            "std_ssim": std_ssim,
            "total_gen_imgs": self.total_gen_imgs,
            "batch_count": self.batch_count,
            "all_ssim": self.ssim_scores
        }
        return final_metrics


def prepare_dataloader(
        data_path: Path,
        batch_size: int,
        workers: int,
):
    """准备数据加载器（评估模式）"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 统一图像尺寸（根据你的模型调整）
        transforms.ToTensor()  # 转Tensor（[0,1]值域）
    ])
    dataset = ImageFolder(str(data_path), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 评估阶段关闭shuffle（保证复现）
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


# ====================== 主函数 ======================
if __name__ == "__main__":
    # 1. 准备真实数据集
    real_dataset = prepare_dataloader(
        data_path=Path(r'\WLASL\origin_frame'),
        batch_size=8,
        workers=4
    )

    # 2. 加载生成模型
    rae_config, *_ = parse_configs(r'\RAE\configs\stage1\pretrained\DINOv2-B.yaml')
    RAE = instantiate_from_config(rae_config).to("cuda" if torch.cuda.is_available() else "cpu")

    # 3. 初始化计算器
    calculator = OnlineFIDSSIMCalculator(
        real_dataset=real_dataset,
        gen_model=RAE,
        log_path="./RAE_WLASL_FID_SSIM.txt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=8
    )

    # 4. 运行主流程
    calculator.step()

    # 5. 获取最终指标
    final_metrics = calculator.get_final_metrics()
    print("\n✅ 所有生成和指标计算完成！日志文件路径：", calculator.log_path)