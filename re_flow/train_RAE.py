# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
Stage-1 RAE training script with reconstruction, LPIPS, and GAN losses.

This script adapts the training logic from the Kakao Brain VQGAN trainer while
targeting the RAE autoencoder architecture used in this repository.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from glob import glob
from PIL import Image
from torchvision.utils import save_image
from omegaconf import OmegaConf
from tqdm import tqdm

from src.disc import (
    DiffAug,
    LPIPS,
    build_discriminator,
    hinge_d_loss,
    vanilla_d_loss,
    vanilla_g_loss,
)
from src.stage1 import RAE
from src.utils.model_utils import instantiate_from_config
from src.utils.train_utils import parse_configs
from src.utils.optim_utils import build_optimizer, build_scheduler



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 RAE with GAN and LPIPS losses.")
    parser.add_argument("--config", type=str, default=r'\re_flow\config\train\WLASL_RAE.yaml', help="YAML config containing a stage_1 section.")
    parser.add_argument("--data-path", type=Path, default=r'F:\SLRdataset\WLASL\origin_frame', help="Directory with ImageFolder structure.")
    parser.add_argument("--results-dir", type=str, default=r"\result", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, default=256, help="Image resolution (assumes square images).")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--global-seed", type=int, default=42, help="Override training.global_seed from the config.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    return parser.parse_args()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)

    return logger


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, current_model: torch.nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(current_model.named_parameters())
    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def calculate_adaptive_weight(
        recon_loss: torch.Tensor,
        gan_loss: torch.Tensor,
        layer: torch.nn.Parameter,
        max_d_weight: float = 1e4,
) -> torch.Tensor:
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()


def select_gan_losses(disc_kind: str, gen_kind: str):
    if disc_kind == "hinge":
        disc_loss_fn = hinge_d_loss
    elif disc_kind == "vanilla":
        disc_loss_fn = vanilla_d_loss
    else:
        raise ValueError(f"Unsupported discriminator loss '{disc_kind}'")

    if gen_kind == "vanilla":
        gen_loss_fn = vanilla_g_loss
    else:
        raise ValueError(f"Unsupported generator loss '{gen_kind}'")
    return disc_loss_fn, gen_loss_fn


def prepare_dataloader(
    data_path: Path,
    image_size: int,
    batch_size: int,
    workers: int,
):
    first_crop_size = 384
    transform = transforms.Compose(
        [
            transforms.Resize(first_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolder(str(data_path), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def save_checkpoint(
        path: str,
        step: int,
        epoch: int,
        model: torch.nn.Module,
        ema_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[LambdaLR],
        disc: torch.nn.Module,
        disc_optimizer: torch.optim.Optimizer,
        disc_scheduler: Optional[LambdaLR],
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "disc": disc.state_dict(),
        "disc_optimizer": disc_optimizer.state_dict(),
        "disc_scheduler": disc_scheduler.state_dict() if disc_scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    disc.load_state_dict(checkpoint["disc"])
    disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
    if disc_scheduler is not None and checkpoint.get("disc_scheduler") is not None:
        disc_scheduler.load_state_dict(checkpoint["disc_scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def load_sample_image(image_paths: list, target_size: int) -> torch.Tensor:
    img = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    img = [transforms.Resize((target_size, target_size))(i) for i in img]
    result = torch.stack([transforms.ToTensor()(i) for i in img], dim=0)
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    (rae_config, *_) = parse_configs(args.config)
    full_cfg = OmegaConf.load(args.config)
    training_section = full_cfg.get("training", None)
    training_cfg = OmegaConf.to_container(training_section, resolve=True) if training_section is not None else {}
    training_cfg = dict(training_cfg) if isinstance(training_cfg, dict) else {}

    gan_section = full_cfg.get("gan", None)
    gan_cfg = OmegaConf.to_container(gan_section, resolve=True) if gan_section is not None else {}
    if not gan_cfg:
        raise ValueError("Config must define a top-level 'gan' section for stage-1 training.")
    disc_cfg = gan_cfg.get("disc", {})
    if not disc_cfg:
        raise ValueError("gan.disc configuration is required for stage-1 training.")
    loss_cfg = gan_cfg.get("loss", {})
    perceptual_weight = float(loss_cfg.get("perceptual_weight", 0.0))
    disc_weight = float(loss_cfg.get("disc_weight", 0.0))
    gan_start_epoch = int(loss_cfg.get("disc_start", 0))
    disc_update_epoch = int(loss_cfg.get("disc_upd_start", gan_start_epoch))
    lpips_start_epoch = int(loss_cfg.get("lpips_start", 0))

    disc_updates = int(loss_cfg.get("disc_updates", 1))
    max_d_weight = float(loss_cfg.get("max_d_weight", 1e4))
    disc_loss_type = loss_cfg.get("disc_loss", "hinge")
    gen_loss_type = loss_cfg.get("gen_loss", "vanilla")

    batch_size = int(training_cfg.get("batch_size", 16))
    num_workers = int(training_cfg.get("num_workers", 4))
    sample_epoch = int(training_cfg.get("sample_epoch", 1))
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    log_interval = int(training_cfg.get("log_interval", 100))
    checkpoint_epoch = int(training_cfg.get("checkpoint_epoch", 1000))
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    num_epochs = int(training_cfg.get("epochs", 200))
    default_seed = int(training_cfg.get("global_seed", 0))
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)

    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*")) - 1
    model_target = str(rae_config.get("target", "stage1"))
    model_string_name = model_target.split(".")[-1]
    experiment_name = (
        f"RAE_{experiment_index:03d}-{model_string_name}"
    )
    experiment_dir = os.path.join(args.results_dir, experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.encoder.eval()
    rae.decoder.train()
    ema_model = deepcopy(rae).to(device).eval()
    ema_model.requires_grad_(False)
    # only train decoder
    rae.encoder.requires_grad_(False)
    rae.decoder.requires_grad_(True)

    optimizer, optim_msg = build_optimizer(rae.decoder.parameters(), training_cfg)
    discriminator, disc_aug = build_discriminator(disc_cfg, device)
    disc_params = [p for p in discriminator.parameters() if p.requires_grad]
    disc_optimizer, disc_optim_msg = build_optimizer(disc_params, disc_cfg)
    disc_scheduler: LambdaLR | None = None
    disc_sched_msg: Optional[str] = None

    discriminator.train()
    disc_loss_fn, gen_loss_fn = select_gan_losses(disc_loss_type, gen_loss_type)

    lpips = LPIPS().to(device)
    lpips.eval()

    scaler: GradScaler | None
    if args.precision == "fp16":
        scaler = GradScaler()
        autocast_kwargs = dict(enabled=True, dtype=torch.float16)
    elif args.precision == "bf16":
        scaler = None
        autocast_kwargs = dict(enabled=True, dtype=torch.bfloat16)
    else:
        scaler = None
        autocast_kwargs = dict(enabled=False)

    loader = prepare_dataloader(
        args.data_path, args.image_size, batch_size, num_workers
    )

    sample_image_path = [
        r'F:\SLRdataset\WLASL\origin_frame\17825\0007.jpg',
        r'F:\SLRdataset\WLASL\origin_frame\17435\0007.jpg',
        r'F:\SLRdataset\WLASL\origin_frame\23544\0007.jpg',
        r'F:\SLRdataset\WLASL\origin_frame\38946\0007.jpg',
        r'F:\SLRdataset\WLASL\origin_frame\51814\0007.jpg',
    ]
    sample_image = load_sample_image(sample_image_path, args.image_size).to(device)

    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError("Dataloader returned zero batches. Check dataset and batch size settings.")

    scheduler: LambdaLR | None = None
    sched_msg: Optional[str] = None
    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)

    if disc_cfg.get("scheduler"):
        disc_scheduler, disc_sched_msg = build_scheduler(disc_optimizer, steps_per_epoch, disc_cfg)
    start_epoch = 0
    global_step = 0
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if ckpt_path.is_file():
            start_epoch, global_step = load_checkpoint(
                ckpt_path,
                rae,
                ema_model,
                optimizer,
                scheduler,
                discriminator,
                disc_optimizer,
                disc_scheduler,
            )
            logger.info(f"[Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    num_params = sum(p.numel() for p in rae.decoder.parameters() if p.requires_grad)
    logger.info(f"Stage-1 RAE trainable parameters: {num_params / 1e6:.2f}M")
    logger.info(f"Discriminator architecture:\n{discriminator}")
    num_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    logger.info(f"Discriminator trainable parameters: {num_params / 1e6:.2f}M")
    logger.info(f"Using {disc_loss_type} discriminator loss and {gen_loss_type} generator loss.")
    logger.info(f"Perceptual (LPIPS) weight: {perceptual_weight:.6f}, GAN weight: {disc_weight:.6f}")
    logger.info(
        f"GAN training starts at epoch {gan_start_epoch}, discriminator updates start at epoch {disc_update_epoch}, LPIPS loss starts at epoch {lpips_start_epoch}.")
    if disc_aug is not None:
        logger.info(f"Using DiffAug with policies: {disc_aug}")
    else:
        logger.info("Not using DiffAug.")
    if clip_grad is not None:
        logger.info(f"Clipping gradients to max norm {clip_grad}.")
    else:
        logger.info("Not clipping gradients.")
    # print optim and schel
    logger.info(optim_msg)
    print(sched_msg if sched_msg else "No LR scheduler for generator.")
    logger.info(disc_optim_msg)
    print(disc_sched_msg if disc_sched_msg else "No LR scheduler for discriminator.")
    logger.info(f"Training for {num_epochs} epochs, batch size {batch_size} per GPU.")
    logger.info(f"Dataset contains {len(loader.dataset)} samples, {steps_per_epoch} steps per epoch.")
    logger.info(f"Starting from epoch {start_epoch} to {num_epochs}.")

    last_layer = rae.decoder.decoder_pred.weight
    gan_start_step = gan_start_epoch * steps_per_epoch
    disc_update_step = disc_update_epoch * steps_per_epoch
    lpips_start_step = lpips_start_epoch * steps_per_epoch
    for epoch in range(start_epoch, num_epochs):
        rae.decoder.train()
        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        for step, (images, _) in tqdm(enumerate(loader), total=len(loader)):
            use_gan = global_step >= gan_start_step and disc_weight > 0.0
            train_disc = global_step >= disc_update_step and disc_weight > 0.0
            use_lpips = global_step >= lpips_start_step and perceptual_weight > 0.0
            images = images.to(device, non_blocking=True)
            real_normed = images * 2.0 - 1.0
            optimizer.zero_grad(set_to_none=True)
            discriminator.eval()

            with torch.amp.autocast('cuda', **autocast_kwargs):
                with torch.no_grad():
                    z = rae.encode(images)
                recon = rae.decode(z)
                recon_normed = recon * 2.0 - 1.0
                rec_loss = F.l1_loss(recon, images)
                if use_lpips:
                    lpips_loss = lpips(real_normed, recon_normed)
                else:
                    lpips_loss = rec_loss.new_zeros(())
                recon_total = rec_loss + perceptual_weight * lpips_loss

                if use_gan:
                    fake_aug = disc_aug.aug(recon_normed)
                    logits_fake, _ = discriminator(fake_aug, None)
                    gan_loss = gen_loss_fn(logits_fake)
                else:
                    gan_loss = torch.zeros_like(recon_total)

            # Calculate adaptive weight outside autocast (autograd operation, not forward pass)
            if use_gan:
                adaptive_weight = calculate_adaptive_weight(
                    recon_total, gan_loss, last_layer, max_d_weight
                )
                total_loss = recon_total + disc_weight * adaptive_weight * gan_loss
            else:
                adaptive_weight = torch.zeros_like(recon_total)
                total_loss = recon_total

            if scaler:
                scaler.scale(total_loss).backward()
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(rae.decoder.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(rae.decoder.parameters(), clip_grad)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            update_ema(ema_model, rae, ema_decay)

            disc_metrics: Dict[str, torch.Tensor] = {}
            if train_disc:
                # Set model to eval mode and get fresh reconstruction with updated weights
                rae.decoder.eval()
                discriminator.train()
                for _ in range(disc_updates):
                    disc_optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda', **autocast_kwargs):
                        # Fresh forward pass with updated model weights (no gradient)
                        with torch.no_grad():
                            z_disc = rae.encode(images)
                            recon_disc = rae.decode(z_disc)
                            recon_disc_normed = recon_disc * 2.0 - 1.0
                        # discretize
                        fake_detached = recon_disc_normed.clamp(-1.0, 1.0)
                        fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
                        fake_input = disc_aug.aug(fake_detached)
                        real_input = disc_aug.aug(real_normed)
                        logits_fake, logits_real = discriminator(fake_input, real_input)
                        d_loss = disc_loss_fn(logits_real, logits_fake)
                    if scaler:
                        scaler.scale(d_loss).backward()
                        scaler.step(disc_optimizer)
                        scaler.update()
                    else:
                        d_loss.backward()
                        disc_optimizer.step()
                    disc_metrics = {
                        "disc_loss": d_loss.detach(),
                        "logits_real": logits_real.detach().mean(),
                        "logits_fake": logits_fake.detach().mean(),
                    }
                    if disc_scheduler is not None:
                        disc_scheduler.step()
                discriminator.eval()
                # Set model back to train mode


            epoch_metrics["recon"] += rec_loss.detach()
            epoch_metrics["lpips"] += lpips_loss.detach()
            epoch_metrics["gan"] += gan_loss.detach()
            epoch_metrics["total"] += total_loss.detach()
            num_batches += 1

            if log_interval > 0 and global_step % log_interval == 0:
                stats = {
                    "loss/total": total_loss.detach().item(),
                    "loss/recon": rec_loss.detach().item(),
                    "loss/lpips": lpips_loss.detach().item(),
                    "loss/gan": gan_loss.detach().item(),
                    "gan/weight": adaptive_weight.detach().item(),
                    "lr/generator": optimizer.param_groups[0]["lr"],
                }
                if disc_metrics:
                    stats.update(
                        {
                            "loss/disc": disc_metrics["disc_loss"].item(),
                            "disc/logits_real": disc_metrics["logits_real"].item(),
                            "disc/logits_fake": disc_metrics["logits_fake"].item(),
                            "lr/discriminator": disc_optimizer.param_groups[0]["lr"],
                        }
                    )
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
            global_step += 1

        if checkpoint_epoch > 0 and (epoch+1) % checkpoint_epoch == 0:
            ckpt_path = f"{checkpoint_dir}/epoch_{epoch:02d}.pt"
            save_checkpoint(
                ckpt_path,
                global_step,
                epoch,
                rae,
                ema_model,
                optimizer,
                scheduler,
                discriminator,
                disc_optimizer,
                disc_scheduler,
            )


        if sample_epoch > 0 and (epoch+1) % sample_epoch == 0:
            rae.decoder.eval()
            logger.info("Generating EMA samples...")

            with torch.no_grad():
                latent = rae.encode(sample_image)
                recon = rae.decode(latent)

            recon = recon.clamp(0.0, 1.0)

            # 创建保存目录
            sample_path = f"{experiment_dir}/sample/{epoch:02d}"
            if not os.path.exists(sample_path):
                os.makedirs(sample_path, exist_ok=True)

            # 保存每张图像
            for i in range(recon.shape[0]):
                save_path = os.path.join(sample_path, f"idx{i:03d}.png")
                save_image(recon[i], save_path)
            logger.info("Generating EMA samples done.")

        if num_batches > 0:
            avg_recon = (epoch_metrics["recon"] / num_batches).item()
            avg_lpips = (epoch_metrics["lpips"] / num_batches).item()
            avg_gan = (epoch_metrics["gan"] / num_batches).item()
            avg_total = (epoch_metrics["total"] / num_batches).item()
            epoch_stats = {
                "epoch/loss_total": avg_total,
                "epoch/loss_recon": avg_recon,
                "epoch/loss_lpips": avg_lpips,
                "epoch/loss_gan": avg_gan,
            }
            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
            )



if __name__ == "__main__":
    main()
