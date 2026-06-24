import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import os
import time
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# ✅ 使用官方库，100% 正确，输出一定在 [0,1]
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from src.utils.train_utils import parse_configs
from src.utils.model_utils import instantiate_from_config


# ====================== 纯 SSIM 计算器（无FID，极简高速） ======================
class SSIMCalculator:
    def __init__(
        self,
        real_dataset,
        gen_model,
        log_path="./ssim_rae.log",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.gen_model = gen_model.eval()
        self.real_dataset = real_dataset
        self.log_path = log_path

        # ✅ 官方 SSIM，数据范围是 [0,1]
        self.ssim = SSIM(data_range=1.0).to(device)

        self.ssim_scores = []
        self.total_imgs = 0
        self._init_log()

    def _init_log(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n===== SSIM 计算开始 {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")

    def _log(self, msg):
        print(msg)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")

    @torch.no_grad()
    def _reconstruct(self, img):
        # 重建图像
        return self.gen_model.decode(self.gen_model.encode(img)).clamp(0.0, 1.0)

    def compute(self):
        self._log("开始计算 SSIM...")

        for batch_idx, batch in tqdm(enumerate(self.real_dataset), total=len(self.real_dataset)):
            imgs = batch[0].to(self.device)
            rec_imgs = self._reconstruct(imgs)

            # ✅ 计算 SSIM
            ssim_val = self.ssim(rec_imgs, imgs).item()
            self.ssim_scores.append(ssim_val)
            self.total_imgs += imgs.shape[0]

            self._log(f"批次 {batch_idx} | SSIM: {ssim_val:.4f} | 累计图像: {self.total_imgs}")

        # 最终结果
        avg_ssim = np.mean(self.ssim_scores)
        self._log("\n===== 最终结果 =====")
        self._log(f"平均 SSIM = {avg_ssim:.4f}")
        self._log(f"总图像数 = {self.total_imgs}")
        return avg_ssim


# ====================== 数据加载 ======================
def prepare_dataloader(data_path, batch_size=8, workers=4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(str(data_path), transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )


# ====================== 主函数 ======================
if __name__ == "__main__":
    # 1. 加载数据
    dataloader = prepare_dataloader(
        data_path=Path(r'F:\SLRdataset\WLASL\origin_frame'),
        batch_size=8,
        workers=4
    )

    # 2. 加载模型
    rae_config, *_ = parse_configs(r'\configs\stage1\pretrained\DINOv2-B.yaml')
    model = instantiate_from_config(rae_config).to("cuda")

    # 3. 计算 SSIM
    calculator = SSIMCalculator(dataloader, model)
    final_ssim = calculator.compute()

    print(f"\n✅ 计算完成！最终平均 SSIM = {final_ssim:.4f}")