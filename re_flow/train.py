# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import os
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import math
# from torch.cuda.amp import autocast
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.utils import save_image

from src.stage1 import RAE
from src.stage2.models import Stage2ModelProtocol
from src.stage2.transport import create_transport, Sampler
from src.utils.train_utils import parse_configs
from src.utils.model_utils import instantiate_from_config
from src.utils import wandb_utils
from src.utils.optim_utils import build_optimizer, build_scheduler
from dataset import (Image_Pose_Dataset, ImageScale, PoseNormalize,
                     RandomHorizontalFlip, Stack, ToImageTensor, TransformersToTensor)


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # 跳过不需要梯度的参数（如 pos_embed），可选
        # if not param.requires_grad:
        #     continue

        # 将当前模型参数 .data 移到 CPU（不破坏计算图）
        # param_data_cpu = param.data.to(device='cpu', non_blocking=True)
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


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


def get_transform(is_train: bool = True,
                  image_size: int = 256,
                  target_size: int = 224):

    if not isinstance(image_size, tuple):
        image_size = (image_size, image_size)

    augments = [ImageScale(image_size, target_size),
                PoseNormalize(image_size)]

    if is_train:
        augments += [RandomHorizontalFlip(0.5)]

    # augments += [
    #     Stack(),
    #     ToImageTensor()]
    augments += [TransformersToTensor()]
    augmentor = transforms.Compose(augments)

    return augmentor


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    """Trains a new SiT model using config-driven hyperparameters."""
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")
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

    if rae_config is None or model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 sections.")

    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)

    misc = to_dict(misc_config)
    transport_cfg = to_dict(transport_config)
    sampler_cfg = to_dict(sampler_config)
    guidance_cfg = to_dict(guidance_config)
    training_cfg = to_dict(training_config)

    use_LORA = bool(training_cfg.get('use_LORA', False))
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)   # 4*sqrt(3)

    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    clip_grad = float(training_cfg.get("clip_grad", 1.0))
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    epochs = int(training_cfg.get("epochs", 1400))
    global_batch_size = int(training_cfg.get("global_batch_size", 2))
    num_workers = int(training_cfg.get("num_workers", 4))
    log_every = int(training_cfg.get("log_every", 100))
    ckpt_every = int(training_cfg.get("ckpt_every", 5_000))
    sample_every = int(training_cfg.get("sample_every", 10_000))
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    default_seed = int(training_cfg.get("global_seed", 0))
    global_seed = args.global_seed if args.global_seed is not None else default_seed

    if grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be >= 1.")
    if args.image_size % 16 != 0:
        raise ValueError("Image size must be divisible by 16 for the RAE encoder.")

    device = torch.device("cuda:0")

    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)

    micro_batch_size = global_batch_size // grad_accum_steps
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)
    latent_dtype = autocast_kwargs["dtype"] if use_bf16 else torch.float32

    transport_params = dict(transport_cfg.get("params", {}))
    path_type = transport_params.get("path_type", "Linear")
    prediction = transport_params.get("prediction", "velocity")
    loss_weight = transport_params.get("loss_weight")
    transport_params.pop("time_dist_shift", None)

    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))

    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    if cfg_scale_override is not None:
        guidance_scale = float(cfg_scale_override)
    guidance_method = guidance_cfg.get("method", "cfg")

    def guidance_value(key: str, default: float) -> float:
        if key in guidance_cfg:
            return guidance_cfg[key]
        dashed_key = key.replace("_", "-")
        return guidance_cfg.get(dashed_key, default)

    t_min = float(guidance_value("t_min", 0.0))
    t_max = float(guidance_value("t_max", 1.0))

    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*")) - 1
    model_target = str(model_config.get("target", "stage2"))
    model_string_name = model_target.split(".")[-1]
    precision_suffix = f"-{args.precision}" if args.precision == "bf16" else ""
    loss_weight_str = loss_weight if loss_weight is not None else "none"
    experiment_name = (
        f"{experiment_index:03d}-{model_string_name}-"
        f"{path_type}-{prediction}-{loss_weight_str}{precision_suffix}-LORA{use_LORA}"
    )
    experiment_dir = os.path.join(args.results_dir, experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    if args.wandb:
        entity = os.environ["ENTITY"]
        project = os.environ["PROJECT"]
        wandb_utils.initialize(args, entity, experiment_name, project)

    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    if args.ema:
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)

    opt_state = None
    sched_state = None
    train_steps = 0

    if model_config['ckpt_path'] is not None:
        checkpoint = torch.load(model_config['ckpt_path'], map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])

        if 'opt' in checkpoint:
            opt_state = checkpoint["opt"]
        if "scheduler" in checkpoint:
            sched_state = checkpoint["scheduler"]
        if "train_steps" in checkpoint:
            train_steps = int(checkpoint.get("train_steps", 0))

    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count/1e6:.2f}M")

    opt, opt_msg = build_optimizer(model.parameters(), training_cfg)
    if opt_state is not None:
        opt.load_state_dict(opt_state)

    transform = get_transform(is_train=True,
                              image_size=args.image_size,
                              target_size=rae_config['params']['encoder_input_size'])
    dataset = Image_Pose_Dataset(transform=transform, is_train=True, **dataset_config)
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} videos ({dataset_config['data']})")
    logger.info(
        f"Gradient accumulation: steps={grad_accum_steps}, micro batch={micro_batch_size}, "
        f"per-GPU batch={micro_batch_size * grad_accum_steps}, global batch={global_batch_size}"
    )
    logger.info(f"Precision mode: {args.precision}")

    loader_batches = len(loader)
    if loader_batches % grad_accum_steps != 0:
        raise ValueError("Number of loader batches must be divisible by grad_accum_steps when drop_last=True.")
    steps_per_epoch = loader_batches // grad_accum_steps
    if steps_per_epoch <= 0:
        raise ValueError("Gradient accumulation configuration results in zero optimizer steps per epoch.")
    schedl, sched_msg = build_scheduler(opt, steps_per_epoch, training_cfg, sched_state)
    logger.info(f"Training configured for {epochs} epochs, {steps_per_epoch} steps per epoch.")
    logger.info(opt_msg + "\n" + sched_msg)
    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = Sampler(transport)

    if sampler_mode == "ODE":
        eval_sampler = transport_sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        eval_sampler = transport_sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")

    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guidance_model_cfg = guidance_cfg.get("guidance_model")
        if guidance_model_cfg is None:
            raise ValueError("Please provide a guidance model config when using autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guidance_model_cfg).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward

    if args.ema:
        update_ema(ema, model, decay=0)
        ema.eval()
    model.eval()

    log_steps = 0
    running_loss = 0.0
    start_time = time()

    sample_dict = dataset[0]
    using_cfg = guidance_scale > 1.0
    zs = torch.randn(dataset_config['num_frames'], *latent_size, device=device, dtype=latent_dtype)
    reference_latent_ = rae.encode(sample_dict['reference_image'].unsqueeze(0).to(device))

    if using_cfg:
        zs = torch.cat([zs, zs], dim=0)
        reference_latent_ = reference_latent_.repeat(dataset_config['num_frames']*2, 1, 1, 1)

        sample_model_kwargs = dict(
            joint=torch.cat((sample_dict['joint'], sample_dict['joint']), dim=0).to(device),
            reference_image=reference_latent_,
            cfg_scale=guidance_scale,
            cfg_interval=(t_min, t_max),
        )
        if guidance_method == "autoguidance":
            if guid_model_forward is None:
                raise RuntimeError("Guidance model forward is not initialized.")
            sample_model_kwargs["additional_model_forward"] = guid_model_forward
            model_fn = ema.forward_with_autoguidance
        else:
            model_fn = ema.forward_with_cfg
    else:
        reference_latent_ = reference_latent_.repeat(dataset_config['num_frames'], 1, 1, 1)
        sample_model_kwargs = dict(
            joint=sample_dict['joint'].to(device),
            reference_image=reference_latent_,
        )
        if args.ema:
            model_fn = ema.forward
        else:
            model_fn = model.forward

    logger.info(f"Training for {epochs} epochs...")
    min_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        logger.info(f"Beginning epoch {epoch}...")
        opt.zero_grad()
        accum_counter = 0
        step_loss_accum = 0.0
        for _, data_dict in tqdm(
                enumerate(loader), total=len(loader)
        ):
            """
            image: B T 3 H W
            joint: B T V 3
            reference_image: B 3 H W
            label: B
            """

            with torch.no_grad():
                x = rae.encode(data_dict['image'].flatten(0, 1).to(device))
                reference_latent = rae.encode(data_dict['reference_image'].to(device))
                reference_latent = reference_latent.unsqueeze(1).repeat(1, dataset_config['num_frames'], 1, 1, 1).flatten(0, 1)
            model_kwargs = dict(joint=data_dict['joint'].flatten(0, 1).to(device),
                                reference_image=reference_latent)
            # x_mean, x_std, x_min, x_max = x.mean(), x.std(), x.min(), x.max()
            with torch.amp.autocast('cuda', **autocast_kwargs):
                loss_tensor = transport.training_losses(model, x, model_kwargs)["loss"].mean()
            step_loss_accum += loss_tensor.item()
            (loss_tensor / grad_accum_steps).backward()
            accum_counter += 1

            if accum_counter < grad_accum_steps:
                continue

            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            schedl.step()
            if args.ema:
                update_ema(ema, model, decay=ema_decay)
            opt.zero_grad()

            running_loss += step_loss_accum / grad_accum_steps
            log_steps += 1
            train_steps += 1
            accum_counter = 0
            step_loss_accum = 0.0

            if train_steps % log_every == 0:
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                logger.info(f"(epoch={epoch :03d} step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        {"train loss": avg_loss, "train steps/sec": steps_per_sec},
                        step=train_steps,
                    )
                running_loss = 0.0
                log_steps = 0
                start_time = time()

        if (epoch+1) % ckpt_every == 0:
            if args.ema:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "scheduler": schedl.state_dict(),
                    "train_steps": train_steps,
                    "config_path": args.config,
                    "training_cfg": training_cfg,
                    'epoch': epoch,
                    "cli_overrides": {
                        "dataset": dataset_config['data'],
                        "results_dir": args.results_dir,
                        "image_size": args.image_size,
                        "precision": args.precision,
                        "global_seed": global_seed,
                    },
                }
            else:
                checkpoint = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scheduler": schedl.state_dict(),
                    "train_steps": train_steps,
                    "config_path": args.config,
                    "training_cfg": training_cfg,
                    'epoch': epoch,
                    "cli_overrides": {
                        "dataset": dataset_config['data'],
                        "results_dir": args.results_dir,
                        "image_size": args.image_size,
                        "precision": args.precision,
                        "global_seed": global_seed,
                    },
                }
            checkpoint_path = f"{checkpoint_dir}/{epoch:03d}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        if min_loss > step_loss_accum:
            if args.ema:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "scheduler": schedl.state_dict(),
                    "train_steps": train_steps,
                    "config_path": args.config,
                    "training_cfg": training_cfg,
                    'epoch': epoch,
                    "cli_overrides": {
                        "dataset": dataset_config['data'],
                        "results_dir": args.results_dir,
                        "image_size": args.image_size,
                        "precision": args.precision,
                        "global_seed": global_seed,
                    },
                }
            else:
                checkpoint = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scheduler": schedl.state_dict(),
                    "train_steps": train_steps,
                    "config_path": args.config,
                    "training_cfg": training_cfg,
                    'epoch': epoch,
                    "cli_overrides": {
                        "dataset": dataset_config['data'],
                        "results_dir": args.results_dir,
                        "image_size": args.image_size,
                        "precision": args.precision,
                        "global_seed": global_seed,
                    },
                }
            checkpoint_path = f"{checkpoint_dir}/best.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")


        if (epoch+1) % sample_every == 0:
            model.eval()
            logger.info("Generating EMA samples...")
            if args.ema:
                model_fn = ema.forward
            else:
                model_fn = model.forward
            with torch.no_grad():
                with torch.amp.autocast('cuda', **autocast_kwargs):
                    samples = eval_sampler(zs, model_fn, **sample_model_kwargs)[-1]

                if using_cfg:
                    samples, _ = samples.chunk(2, dim=0)
                samples = rae.decode(samples.to(torch.float32))
                samples = torch.clamp(samples, 0., 1.)

                # 创建保存目录
                sample_path = f"{experiment_dir}/sample/{epoch:04d}"
                os.makedirs(sample_path, exist_ok=True)

                # 保存每张图像
                for i in range(samples.shape[0]):
                    save_path = os.path.join(sample_path, f"sample_epoch{epoch + 1}_idx{i:03d}.png")
                    save_image(samples[i], save_path)
            logger.info("Generating EMA samples done.")

        if accum_counter != 0:
            raise RuntimeError("Gradient accumulation counter not zero at epoch end.")

    model.eval()
    logger.info("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=r'\re_flow\config\train\Diff_WLASL.yaml', help="Path to the config file.")
    parser.add_argument("--results-dir", type=str, default=r"\result", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256, help="Input image resolution.")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32", help="Compute precision for training.")
    parser.add_argument("--wandb", default=False, help="Enable Weights & Biases logging.")
    parser.add_argument("--ema", default=True, help="Enable Weights & Biases logging.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument("--global-seed", type=int, default=42, help="Override training.global_seed from the config.")
    args = parser.parse_args()
    main(args)
