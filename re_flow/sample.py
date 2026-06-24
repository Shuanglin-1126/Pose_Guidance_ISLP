# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
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

from src.stage1 import RAE
from src.stage2.models import Stage2ModelProtocol
from src.stage2.transport import create_transport, Sampler
from src.utils.train_utils import parse_configs
from src.utils.model_utils import instantiate_from_config
from dataset import Image_Pose_Dataset, ImageScale, PoseNormalize, TransformersToTensor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_transform(pose, pose_score, image_size=(256, 256)):
    h, w = image_size
    pose[..., 0] = \
        (pose[..., 0] - (h / 2)) / (h / 2)
    pose[..., 1] = \
        (pose[..., 1] - (w / 2)) / (w / 2)
    results = torch.cat((pose, pose_score[..., None]), dim=-1)

    return results


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = torchvision.transforms.Resize(224, interpolation=Image.BICUBIC)(image)
    image = torchvision.transforms.CenterCrop(224)(image)
    tensor = transforms.ToTensor()(image).unsqueeze(0)  # (1, C, H, W)
    return tensor


def _load_pose(file_path):
    joint = 'keypoint_vedio'
    joint_score = 'keypoint_score_vedio'
    face_idx = [74, 85, 80, 31, 71, 77, 29, 27, 35, 33, 59, 62, 65, 68] # 14 nodes
    with np.load(file_path) as data:
        keypoints = np.concatenate((data[joint][:, :11, :],
                                    data[joint][:, -42:, :],
                                    data[joint][:][:, face_idx, :]),
                                   axis=1)
        keypoint_scores = np.concatenate((data[joint_score][:, :11],
                                          data[joint_score][:, -42:],
                                          data[joint_score][:][:, face_idx]),
                                   axis=1)

    # with np.load(file_path) as data:
    #     keypoints = np.concatenate((data[joint][:, :11, :],
    #                                 data[joint][:, -42:, :]),
    #                                axis=1)
    #     keypoint_scores = np.concatenate((data[joint_score][:, :11],
    #                                       data[joint_score][:, -42:]),
    #                                      axis=1)

    return torch.tensor(keypoints), torch.tensor(keypoint_scores)

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model.eval()  # important!
    rae.eval()
    shift_dim = misc_config.get("time_dist_shift_dim", 768 * 16 * 16)
    shift_base = misc_config.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(
        shift_dim / shift_base)
    print(
        f"Using time_dist_shift={time_dist_shift:.4f} = sqrt({shift_dim}/{shift_base}).")
    transport = create_transport(
        **transport_config['params'],
        time_dist_shift=time_dist_shift
    )
    sampler = Sampler(transport)
    mode, sampler_params = sampler_config['mode'], sampler_config['params']
    if mode == "ODE":
        sample_fn = sampler.sample_ode(
            **sampler_params
        )
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            **sampler_params,
            # sampling_method=args.sampling_method,
            # diffusion_form=args.diffusion_form,
            # diffusion_norm=args.diffusion_norm,
            # last_step=args.last_step,
            # last_step_size=args.last_step_size,
            # num_steps=args.num_sampling_steps,
        )
    else:
        raise NotImplementedError(f"Invalid sampling mode {mode}.")

    guidance_scale = float(guidance_config.get("scale", 1.0))
    latent_size = misc_config.get("latent_size", (768, 16, 16))

    # Labels to condition the model with (feel free to change):
    dataset_pose, dataset_pose_score = _load_pose(r'F:\SLRdataset\WLASL\body_keypoint_133\00665.npz')
    dataset_pose = torch.flip(dataset_pose, dims=[-1])
    joint_ = get_transform(dataset_pose, dataset_pose_score, (256, 256))[0:30, ...]

    using_cfg = guidance_scale > 1.0
    zs = torch.randn(dataset_config['num_frames'], *latent_size, device=device, dtype=torch.float32)
    img_ref = load_image(r'\assets\011.jpg')
    sample_path = r"\re_flow\sample_result"
    reference_latent_ = rae.encode(img_ref.to(device))

    if using_cfg:
        zs = torch.cat([zs, zs], dim=0)
        reference_latent_ = reference_latent_.repeat(dataset_config['num_frames']*2, 1, 1, 1)

        t_min, t_max = guidance_config.get("t_min", 0.0), guidance_config.get("t_max", 1.0)
        sample_model_kwargs = dict(
            joint=joint_.to(device),
            reference_image=reference_latent_,
            cfg_scale=guidance_scale,
            cfg_interval=(t_min, t_max),
        )

        model_fn = model.forward_with_cfg

    else:
        reference_latent_ = reference_latent_.repeat(dataset_config['num_frames'], 1, 1, 1)
        sample_model_kwargs = dict(
            joint=joint_.to(device),
            reference_image=reference_latent_,
        )
        model_fn = model.forward

    # Sample images:
    start_time = time()
    samples: torch.Tensor = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
    if using_cfg:
        samples, _ = samples.chunk(2, dim=0)
    # samples = vae.decode(samples / 0.18215).sample
    samples = rae.decode(samples)
    samples = torch.clamp(samples, 0., 1.)
    print(f"Sampling took {time() - start_time:.2f} seconds.")

    # Save and display images:
    if not os.path.exists(sample_path):
        os.makedirs(sample_path, exist_ok=True)

    # 保存每张图像
    for i in range(samples.shape[0]):
        save_path = os.path.join(sample_path, f"idx{i+1:03d}.png")
        save_image(samples[i], save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=r'\re_flow\config\sample\DiDH_XL_DINOv2_B.yaml',
                        help="Path to the config file.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_known_args()[0]
    main(args)
