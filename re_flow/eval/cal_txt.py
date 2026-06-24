txt_path = r'\RAE\re_flow\eval\evaluation_metrics.txt'

ssim_list, psnr_list, lpips_list, FVD_list = [], [], [], []

with open(txt_path, 'r') as f:
    lines = f.readlines()

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        ssim_list.append(float(parts[1]))
        psnr_list.append(float(parts[2]))
        lpips_list.append(float(parts[3]))
        FVD_list.append(float(parts[4]))

avg_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else 0
avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0
avg_lpips = sum(lpips_list) / len(lpips_list) if lpips_list else 0
avg_FVD = sum(FVD_list) / len(FVD_list) if FVD_list else 0

print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average LPIPS: {avg_lpips:.4f}")
print(f"Average FVD: {avg_FVD:.4f}")