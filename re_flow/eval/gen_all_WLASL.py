import torch.nn as nn
import math
from time import time
import argparse
from torchvision.utils import save_image
import torch
import sys
import os
import torchvision
from PIL import Image
from torchvision import transforms
import numpy as np
from glob import glob
from tqdm import tqdm

from src.stage1 import RAE
from src.stage2.models import Stage2ModelProtocol
from src.stage2.transport import create_transport, Sampler
from src.utils.train_utils import parse_configs
from src.utils.model_utils import instantiate_from_config

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ========== 新增：GPU加速配置 ==========
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # 固定输入尺寸，开启推理加速
torch.backends.cudnn.deterministic = False  # 牺牲确定性换速度


def load_image(image_path):
    image_path = os.path.join(image_path, "0001.jpg")
    image = Image.open(image_path).convert("RGB")

    image = torchvision.transforms.Resize(512, interpolation=Image.BICUBIC)(image)
    image = torchvision.transforms.CenterCrop(384)(image)
    image = torchvision.transforms.Resize(224, interpolation=Image.BICUBIC)(image)
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    return tensor


def _load_pose(file_path, img_size=(256, 256), device="cuda"):
    """优化：加载后直接转到GPU，避免多次传输"""
    joint = 'keypoint_vedio'
    joint_score = 'keypoint_score_vedio'
    face_idx = [74, 85, 80, 31, 71, 77, 29, 27, 35, 33, 59, 62, 65, 68]
    with np.load(file_path) as data:
        keypoints = np.concatenate((data[joint][:, :11, :],
                                    data[joint][:, -42:, :],
                                    data[joint][:][:, face_idx, :]),
                                   axis=1)
        keypoint_scores = np.concatenate((data[joint_score][:, :11],
                                          data[joint_score][:, -42:],
                                          data[joint_score][:][:, face_idx]),
                                         axis=1)
    # 优化：直接生成GPU tensor，避免后续多次传输
    keypoints = torch.tensor(keypoints, device=device)
    keypoint_scores = torch.tensor(keypoint_scores, device=device)
    keypoints = torch.flip(keypoints, dims=[-1])

    keypoints[..., 0] = (keypoints[..., 0] - (img_size[0] / 2)) / (img_size[0] / 2)
    keypoints[..., 1] = (keypoints[..., 1] - (img_size[1] / 2)) / (img_size[1] / 2)

    result = torch.cat((keypoints, keypoint_scores[..., None]), dim=-1)
    del keypoints, keypoint_scores
    return result


def process_single_video(
        npz_file,
        ref_image,
        rae,
        model,
        transport,
        sampler,
        sampler_config,
        guidance_config,
        misc_config,
        dataset_config,
        device
):
    """优化：全程GPU计算，最后统一转CPU"""
    result_ = []
    # 优化1：姿态数据直接加载到GPU
    pose = _load_pose(npz_file, device=device)
    num_frames = pose.shape[0]
    # 优化2：参考特征提前编码到GPU，避免重复计算
    reference_latent_ori = rae.encode(ref_image)

    latent_size = misc_config.get("latent_size", (768, 16, 16))
    guidance_scale = float(guidance_config.get("scale", 1.0))
    using_cfg = guidance_scale > 1.0

    num_batches = math.ceil(num_frames / 4)
    for idx_ in range(num_batches):
        batch_start = idx_ * 4
        batch_end = min((idx_ + 1) * 4, num_frames)
        batch_ = batch_end - batch_start
        if batch_ == 0:
            continue

        # 优化3：直接在GPU生成随机潜变量
        zs = torch.randn(batch_, *latent_size, device=device, dtype=torch.float32)
        # 优化4：姿态数据已在GPU，无需再to(device)
        joint_ = pose[batch_start:batch_end]

        if using_cfg:
            zs = torch.cat([zs, zs], dim=0)
            reference_latent_ = reference_latent_ori.repeat(batch_ * 2, 1, 1, 1)
            t_min, t_max = guidance_config.get("t_min", 0.0), guidance_config.get("t_max", 1.0)
            sample_model_kwargs = dict(
                joint=joint_,
                reference_image=reference_latent_,
                cfg_scale=guidance_scale,
                cfg_interval=(t_min, t_max),
            )
            model_fn = model.forward_with_cfg
        else:
            reference_latent_ = reference_latent_ori.repeat(batch_, 1, 1, 1)
            sample_model_kwargs = dict(
                joint=joint_,
                reference_image=reference_latent_,
            )
            model_fn = model.forward

        mode, sampler_params = sampler_config['mode'], sampler_config['params']
        if mode == "ODE":
            sample_fn = sampler.sample_ode(**sampler_params)
        elif mode == "SDE":
            sample_fn = sampler.sample_sde(**sampler_params)
        else:
            raise NotImplementedError(f"Invalid sampling mode {mode}.")

        # 全程GPU计算
        samples: torch.Tensor = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        samples = rae.decode(samples)
        samples = torch.clamp(samples, 0., 1.)
        # 优化5：先存在GPU，不立即转CPU
        result_.append(samples)

    # 优化6：最后统一拼接+转CPU，减少数据传输次数
    result_ = torch.cat(result_, dim=0).cpu()
    return result_, num_frames


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 验证GPU是否真的在使用
    if device == "cuda":
        print(f"✅ 使用GPU：{torch.cuda.get_device_name(0)}")
        # 可选：开启半精度推理（需模型支持）
        # model = model.half()
        # rae = rae.half()
        # ref_image = ref_image.half()
    else:
        print("❌ 未检测到GPU，使用CPU运行（速度极慢）")

    (
        rae_config,
        model_config,
        dataset_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
    ) = parse_configs(args.config)

    rae: RAE = instantiate_from_config(rae_config).to(device)
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    if model_config['ckpt_path'] is not None:
        checkpoint = torch.load(model_config['ckpt_path'], map_location="cpu")
        if "ema" in checkpoint:
            model.load_state_dict(checkpoint["ema"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    rae.eval()

    shift_dim = misc_config.get("time_dist_shift_dim", 768 * 16 * 16)
    shift_base = misc_config.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    print(f"Using time_dist_shift={time_dist_shift:.4f} = sqrt({shift_dim}/{shift_base}).")
    transport = create_transport(**transport_config['params'], time_dist_shift=time_dist_shift)
    sampler = Sampler(transport)

    npz_root = r'\WLASL\body_keypoint_133'
    output_root = r"\WLASL\gen_cx"
    os.makedirs(output_root, exist_ok=True)

    npz_files = glob(os.path.join(npz_root, "*.npz"))
    print(f"\n找到 {len(npz_files)} 个npz文件，开始批量生成...\n")

    for idx, npz_file in tqdm(enumerate(npz_files), total=len(npz_files), desc="Processing videos"):
        if idx >= 100:
            break
        try:
            file_name = os.path.splitext(os.path.basename(npz_file))[0]
            ref_image_path = os.path.join(r'\WLASL\origin_frame', file_name)
            output_dir = os.path.join(output_root, file_name)
            os.makedirs(output_dir, exist_ok=True)

            # 加载参考图后直接转GPU
            ref_image = load_image(ref_image_path).to(device)

            video_frames, num_frames = process_single_video(
                npz_file, ref_image, rae, model, transport, sampler,
                sampler_config, guidance_config, misc_config, dataset_config, device
            )

            for frame_idx in range(num_frames):
                save_path = os.path.join(output_dir, f"{frame_idx:04d}.jpg")
                save_image(video_frames[frame_idx], save_path)

            torch.cuda.empty_cache()
            tqdm.write(f"✅ [{idx + 1}/{len(npz_files)}] 完成：{file_name}（{num_frames}帧）")

        except Exception as e:
            tqdm.write(f"❌ [{idx + 1}/{len(npz_files)}] 失败：{os.path.basename(npz_file)}，错误：{str(e)}")
            torch.cuda.empty_cache()
            continue

    print(f"\n批量生成完成！所有结果保存至：{output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=r'\RAE\re_flow\config\sample\DiDH_XL_DINOv2_B.yaml',
                        help="Path to the config file.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_known_args()[0]
    main(args)