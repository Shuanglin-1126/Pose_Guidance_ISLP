import os
import torch
import numpy as np
import lpips
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchvision import transforms
from tqdm import tqdm

# ====================== 【只需改这里】 ======================
GT_ROOT = r"F:\SLRdataset\WLASL\origin_frame"
GEN_ROOT = r"F:\SLRdataset\WLASL\gen_cx"
SAVE_FILE = "evaluation_metrics.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================================================

# 初始化指标模型
lpips_fn = lpips.LPIPS(net="alex").to(DEVICE).eval()
psnr_fn = PSNR(data_range=1.0).to(DEVICE)
ssim_fn = SSIM(data_range=1.0).to(DEVICE)


def load_image_ori(path):
    img = Image.open(path).convert("RGB")
    img = transforms.ToTensor()(img)
    return img.unsqueeze(0).to(DEVICE)


def load_image_gen(path):
    img = Image.open(path).convert("RGB")
    img = transforms.ToTensor()(img)
    return img.unsqueeze(0).to(DEVICE)

def compute_frame_metrics(gt_dir, gen_dir):
    frames = sorted([f for f in os.listdir(gt_dir) if f.endswith(('jpg', 'png'))])
    ssims, psnrs, lpips_list = [], [], []

    for f in frames:
        gt_p = os.path.join(gt_dir, f)
        gen_p = os.path.join(gen_dir, f)
        if not os.path.exists(gen_p):
            continue

        gt = load_image_ori(gt_p)
        gen = load_image_gen(gen_p)

        ssim = ssim_fn(gen, gt).item()
        psnr = psnr_fn(gen, gt).item()
        lp = lpips_fn(gen, gt).item()

        ssims.append(ssim)
        psnrs.append(psnr)
        lpips_list.append(lp)

    if len(ssims) == 0:
        return None

    return {
        "ssim": np.mean(ssims),
        "psnr": np.mean(psnrs),
        "lpips": np.mean(lpips_list)
    }

def compute_fvd(gt_dir, gen_dir):
    try:
        from torch_fidelity import calculate_metrics
        metrics = calculate_metrics(
            input1=gt_dir,
            input2=gen_dir,
            cuda=DEVICE.type == "cuda",
            isc=False,
            fid=True,
            kid=False,
            verbose=False,
        )
        return metrics["frechet_inception_distance"]
    except:
        return -1

def main():
    video_names = sorted(os.listdir(GT_ROOT))

    with open(SAVE_FILE, 'w', encoding='utf-8') as f:
        f.write("video\tSSIM\tPSNR\tLPIPS\tFVD\n")

        for name in tqdm(video_names, desc="计算进度"):
            gt_d = os.path.join(GT_ROOT, name)
            gen_d = os.path.join(GEN_ROOT, name)

            if not os.path.isdir(gt_d) or not os.path.exists(gen_d):
                continue

            m = compute_frame_metrics(gt_d, gen_d)
            if m is None:
                continue

            fvd = compute_fvd(gt_d, gen_d)

            line = f"{name}\t{m['ssim']:.4f}\t{m['psnr']:.2f}\t{m['lpips']:.4f}\t{fvd:.2f}"
            f.write(line + "\n")
            f.flush()
            print(line)

    print(f"\n✅ 全部计算完成！结果已保存到：{SAVE_FILE}")

if __name__ == "__main__":
    main()