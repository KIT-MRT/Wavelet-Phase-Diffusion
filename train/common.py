import math
import os
from datetime import datetime

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from tqdm import tqdm

from wpd.noise import auto_J, radius_to_cutoff, wpd_noise


def sample_training_noise(latents: torch.Tensor, lfr_prob: float = 0.8):
    """Sample ψ-PD structured noise for one training step.

    The cutoff radius (latent pixels) follows a shifted exponential r = 4 + Exp(16). With
    probability `lfr_prob` Low-Frequency Randomization is applied, with the decomposition depth
    J ~ U[max(J_auto, 3), J_auto + 4] (bounded so that the low-frequency packet keeps >= 2 px).

    latents: (N, C, H, W); for video pass the frames as the batch dimension.
    Returns (noise, stats dict).
    """
    h, w = latents.shape[-2:]
    radius = 4 + np.random.exponential(scale=16)
    radius = min(radius, min(h, w) // 2)

    nyquist = min(h, w) / 2.0
    j_auto = auto_J(min(float(radius), nyquist) / nyquist)
    j_min = max(j_auto, 3)
    j_max = min(j_auto + 4, int(math.log2(min(h, w))))

    lfr = bool(np.random.random() < lfr_prob)
    if lfr and j_min <= j_max:
        J = int(np.random.randint(j_min, j_max + 1))
    else:
        lfr, J = False, None

    noise = wpd_noise(
        latents.float(),
        cutoff=radius_to_cutoff(radius, h, w),
        J=J,
        lfr=lfr,
        noise=torch.randn_like(latents.float()),
    )
    stats = {"noise/radius": float(radius), "noise/lfr": float(lfr), "noise/J": float(J if J is not None else j_auto)}
    return noise, stats


def log_step(accelerator: Accelerator, noise: torch.Tensor, loss: torch.Tensor, stats: dict, step: int):
    with torch.no_grad():
        n = noise.detach().float()
        values = torch.stack([n.mean(), n.std(), n.abs().max(), loss.detach().float()])
        values = accelerator.gather(values[None]).mean(dim=0)
    if accelerator.is_main_process:
        accelerator.log({
            "noise/mean": values[0].item(),
            "noise/std": values[1].item(),
            "noise/max_abs": values[2].item(),
            "loss": values[3].item(),
            **stats,
        }, step=step)


def launch_training(dataset, model, model_logger, args, shuffle: bool = True, num_workers: int = None):
    """LoRA training loop. Use shuffle=False for iterable (streaming) datasets."""
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=shuffle, collate_fn=lambda x: x[0],
        num_workers=args.dataset_num_workers if num_workers is None else num_workers,
    )
    accelerator = Accelerator(
        log_with="tensorboard",
        project_config=ProjectConfiguration(project_dir=args.output_path,
                                            logging_dir=os.path.join(args.output_path, "logs")),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
        dataloader_config=None if shuffle else DataLoaderConfiguration(dispatch_batches=False),
    )
    accelerator.init_trackers(datetime.now().strftime("%Y%m%d_%H%M%S"))
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    accelerator.unwrap_model(model).set_accelerator(accelerator)

    global_step = 0
    for epoch_id in range(args.num_epochs):
        for data in tqdm(dataloader, disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data, step=global_step)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, args.save_steps)
                scheduler.step()
                global_step += 1
        if args.save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, args.save_steps)
