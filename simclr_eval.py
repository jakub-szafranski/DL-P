import os
import random
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.config import parse_simclr_cli, SimCLRConfig
from utils.fine_tuning import FineTuneModel, fine_tune
from utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    barrier,
)
from sim_clr import get_encoder


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


def evaluate_simclr_models(
    model_paths: list[str],
    subset_ratios: list[float],
    local_rank: int,
    world_size: int,
    conf: SimCLRConfig,
) -> None:
    """
    Load SimCLR encoders and perform fine-tuning evaluation with different subset ratios.

    Args:
        model_paths (list[str]): List of paths to saved encoder state dicts.
        subset_ratios (list[float]): List of subset ratios for fine-tuning.
        local_rank (int): Local GPU rank within this node.
        world_size (int): Total number of processes.
        conf (SimCLRConfig): Configuration object.
    """
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    distributed = world_size > 1

    if is_main_process():
        wandb.define_metric("eval_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    for i, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            if is_main_process():
                wandb.termlog(f"Model path {model_path} does not exist, skipping.")
            continue

        model_name = conf.models[0]

        encoder, num_features = get_encoder(model_name)
        ft_model = FineTuneModel(encoder, num_features)
        ft_model.load_state_dict(torch.load(model_path, map_location=device))
        encoder = ft_model.encoder
        encoder = encoder.to(device)

        if distributed:
            encoder = nn.SyncBatchNorm.convert_sync_batchnorm(encoder)

        if distributed:
            encoder = DDP(encoder, device_ids=[local_rank])

        for subset_ratio in subset_ratios:
            if is_main_process():
                wandb.termlog(f"Evaluating {model_name} with subset_ratio {subset_ratio}")

            # Synchronize before evaluation
            barrier()

            base_encoder = encoder.module if distributed else encoder

            frozen_top1, frozen_top5, full_top1, full_top5, _, _ = fine_tune(
                model=base_encoder,
                num_features=num_features,
                device=device,
                num_workers=NUM_WORKERS,
                config=conf,
                pretrain_epoch=(i+1)*conf.save_model_every,
                save_path=None,
                distributed=distributed,
                local_rank=local_rank,
                subset_ratio=subset_ratio,
            )

            if is_main_process():
                wandb.log({
                    f"eval/{model_name}/subset_{subset_ratio}/frozen_top1": frozen_top1,
                    f"eval/{model_name}/subset_{subset_ratio}/frozen_top5": frozen_top5,
                    f"eval/{model_name}/subset_{subset_ratio}/full_top1": full_top1,
                    f"eval/{model_name}/subset_{subset_ratio}/full_top5": full_top5,
                    "eval_step": i
                })


def main() -> None:
    conf = parse_simclr_cli()

    # Initialize distributed training
    local_rank, global_rank, world_size = setup_distributed()

    set_seed(conf.seed, global_rank)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

    if is_main_process():
        wandb.init(
            project="simclr_eval",
            config={**conf.model_dump(), "world_size": world_size},
            name=f"eval-{world_size}gpu",
            reinit=True,
        )

    barrier()

    subset_ratios = [0.1, 0.01]
    evaluate_simclr_models(
        model_paths=conf.eval_model_paths,
        subset_ratios=subset_ratios,
        local_rank=local_rank,
        world_size=world_size,
        conf=conf,
    )

    if is_main_process():
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
