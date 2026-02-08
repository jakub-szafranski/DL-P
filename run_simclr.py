import gc
import os
import random
import numpy as np
import tqdm
import wandb
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Literal

from utils import conf_simclr as conf, prepare_simclr_train_dataset, fine_tune
from utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    barrier,
)
from sim_clr import SimCLR, NTXentLoss, LARS, get_encoder


NUM_WORKERS = 20
os.environ["MKL_NUM_THREADS"] = f"{NUM_WORKERS}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{NUM_WORKERS}"
os.environ["OMP_NUM_THREADS"] = f"{NUM_WORKERS}"


def set_seed(seed: int, rank: int = 0) -> None:
    """
    Set random seeds for reproducibility with rank offset.

    Args:
        seed (int): Base random seed.
        rank (int): Process rank offset for different data shuffling per GPU.
    """
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)


def train_simclr_model(
    model_name: Literal["resnet50", "vit_b_16", "efficientnet_b5"],
    local_rank: int,
    world_size: int,
) -> None:
    """
    Pre-train SimCLR with DDP and periodically fine-tune for evaluation.

    Args:
        model_name (Literal): Encoder backbone name.
        local_rank (int): Local GPU rank within this node.
        world_size (int): Total number of processes.
    """
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    distributed = world_size > 1

    encoder, num_features = get_encoder(model_name)
    simclr_model = SimCLR(encoder, num_features=num_features)

    if distributed:
        simclr_model = nn.SyncBatchNorm.convert_sync_batchnorm(simclr_model)

    simclr_model = simclr_model.to(device)
    if distributed:
        simclr_model = DDP(simclr_model, device_ids=[local_rank])

    if is_main_process():
        wandb.watch(simclr_model, log="gradients", log_freq=100)
        wandb.define_metric("pretrain_step")
        wandb.define_metric("pretrain/*", step_metric="pretrain_step")

    contrastive_loss = NTXentLoss(temperature=conf.pretrain_temperature).to(device)

    optimizer = LARS(
        simclr_model.parameters(),
        lr=conf.pretrain_learning_rate,
        weight_decay=conf.pretrain_weight_decay,
        trust_coefficient=0.001,
    )

    # Warmup for 10 epochs, then Cosine Decay
    warmup_epochs = 10
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=conf.pretrain_epochs - warmup_epochs, eta_min=0)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
    )

    # Prepare dataloader with DistributedSampler
    per_gpu_batch_size = conf.pretrain_batch_size // world_size
    contrastive_dataloader = prepare_simclr_train_dataset(
        per_gpu_batch_size, NUM_WORKERS, conf.image_size, distributed=distributed
    )

    pretrain_step = 0
    epoch_iter = tqdm.tqdm(range(conf.pretrain_epochs)) if is_main_process() else range(conf.pretrain_epochs)

    for epoch in epoch_iter:
        simclr_model.train()

        if distributed and hasattr(contrastive_dataloader.sampler, "set_epoch"):
            contrastive_dataloader.sampler.set_epoch(epoch)

        for index, batch in enumerate(contrastive_dataloader):
            images, _ = batch
            images_i, images_j = images[0].to(device), images[1].to(device)

            optimizer.zero_grad()

            proj_i = simclr_model(images_i)
            proj_j = simclr_model(images_j)

            loss = contrastive_loss(proj_i, proj_j)
            loss.backward()
            optimizer.step()

            if is_main_process():
                wandb.log({"pretrain/loss": loss.item(), "pretrain_step": pretrain_step})
                pretrain_step += 1

        scheduler.step()

        # Evaluation and checkpointing
        if (epoch + 1) % conf.eval_every == 0 or epoch == conf.pretrain_epochs - 1:
            save_model = (epoch + 1) % conf.save_model_every == 0 or epoch == conf.pretrain_epochs - 1
            save_path = f"{conf.model_saved_path}/simclr_{model_name}_epoch_{epoch + 1}.pth" if save_model else None

            # Synchronize before evaluation
            barrier()

            base_model = simclr_model.module if distributed else simclr_model

            frozen_top1, frozen_top5, full_top1, full_top5, _, _ = fine_tune(
                model=base_model.encoder,
                num_features=num_features,
                device=device,
                num_workers=NUM_WORKERS,
                config=conf,
                pretrain_epoch=epoch + 1,
                save_path=save_path if is_main_process() else None,
                distributed=distributed,
                local_rank=local_rank,
            )

            if is_main_process():
                wandb.log(
                    {
                        "ft/frozen_top1": frozen_top1,
                        "ft/frozen_top5": frozen_top5,
                        "ft/full_top1": full_top1,
                        "ft/full_top5": full_top5,
                        "epoch": epoch + 1,
                    }
                )


def main() -> None:
    # Initialize distributed training
    local_rank, global_rank, world_size = setup_distributed()

    set_seed(conf.seed, global_rank)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

    if is_main_process():
        os.makedirs(conf.model_saved_path, exist_ok=True)

    barrier()

    for model_name in conf.models:
        if is_main_process():
            wandb.init(
                project="simclr",
                config={**conf.model_dump(), "encoder": model_name, "world_size": world_size},
                name=f"pretrain-{model_name}-{world_size}gpu",
                reinit=True,
            )

        barrier()

        try:
            train_simclr_model(model_name=model_name, local_rank=local_rank, world_size=world_size)
        finally:
            if is_main_process():
                wandb.finish()

        barrier()

        gc.collect()
        torch.cuda.empty_cache()

    cleanup_distributed()


if __name__ == "__main__":
    main()
