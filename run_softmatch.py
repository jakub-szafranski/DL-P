import gc
import json
import os
import random
import numpy as np
import tqdm
import wandb
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Literal

from utils import prepare_softclr_train_dataset, prepare_stl10_test, evaluate_model, get_val_transforms
from utils import parse_softmatch_cli
from utils.config import SoftMatchConfig
from utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    barrier,
)
from sim_clr import get_encoder
from soft_match import SoftCLR, SoftMatchTrainer, ModelEMA
from datetime import datetime


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


def train_softmatch_model(
    model_name: Literal["resnet50", "vit_b_16", "efficientnet_b5"],
    local_rank: int,
    world_size: int,
    conf: SoftMatchConfig,
) -> None:
    """
    Train only the SoftMatch classification/consistency losses (no SimCLR contrastive loss).

    Args:
        model_name (Literal): Encoder backbone name.
        local_rank (int): Local GPU rank within this node.
        world_size (int): Total number of processes.
        conf (SoftMatchConfig): Configuration object.
    """
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    distributed = world_size > 1

    encoder, num_features = get_encoder(model_name)
    softclr_model = SoftCLR(encoder=encoder, num_features=num_features, num_classes=10)

    if distributed:
        softclr_model = nn.SyncBatchNorm.convert_sync_batchnorm(softclr_model)

    softclr_model = softclr_model.to(device)
    if distributed:
        softclr_model = DDP(softclr_model, device_ids=[local_rank])

    if is_main_process():
        wandb.watch(softclr_model, log="gradients", log_freq=100)
        wandb.define_metric("softmatch_step")
        wandb.define_metric("softmatch/*", step_metric="softmatch_step")

    softmatch_trainer = SoftMatchTrainer(
        num_classes=10,
        dist_align=conf.softmatch_dist_align,
        hard_label=conf.softmatch_hard_label,
        ema_p=conf.softmatch_ema_p,
        device=device,
    )

    model_ema = ModelEMA(softclr_model, decay=conf.softmatch_model_ema)
    model_ema.register()

    optimizer = torch.optim.SGD(
        softclr_model.parameters(),
        lr=conf.pretrain_learning_rate,
        weight_decay=conf.pretrain_weight_decay,
        momentum=conf.pretrain_momentum,
        nesterov=True,
    )

    # Warmup + Cosine LR
    import math

    warmup_epochs = 10

    def lr_lambda(epoch: int) -> float:
        """Combined warmup and custom partial-cosine schedule.

        Returns multiplier for base LR: lr = lr0 * multiplier.
        - linear warmup from factor 1e-4 -> 1.0 over `warmup_epochs`
        - after warmup: multiplier = cos(7*pi*k/(16*K)) where k is global epoch and K = conf.pretrain_epochs
        """
        if epoch < warmup_epochs:
            return 1e-4 + (epoch / float(max(1, warmup_epochs))) * (1.0 - 1e-4)
        K = float(max(1, conf.pretrain_epochs))
        k = float(epoch)
        return float(math.cos(7.0 * math.pi * k / (16.0 * K)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Determine unlabeled / labeled batch sizes (total across all GPUs)
    total_unlabeled_bs = conf.unlabeled_batch_size if conf.unlabeled_batch_size is not None else conf.pretrain_batch_size
    total_labeled_bs = conf.labeled_batch_size if conf.labeled_batch_size is not None else (conf.pretrain_batch_size // 8)

    per_gpu_unlabeled = max(1, total_unlabeled_bs // max(1, world_size))
    per_gpu_labeled = max(1, total_labeled_bs // max(1, world_size))

    unlabeled_loader, labeled_loader = prepare_softclr_train_dataset(
        batch_size=per_gpu_unlabeled,
        num_workers=NUM_WORKERS,
        img_size=conf.image_size,
        distributed=distributed,
        fold=conf.fold,
        labeled_batch_size=per_gpu_labeled,
    )

    step = 0
    epoch_iter = tqdm.tqdm(range(conf.pretrain_epochs)) if is_main_process() else range(conf.pretrain_epochs)

    for epoch in epoch_iter:
        softclr_model.train()

        if distributed:
            if hasattr(unlabeled_loader.sampler, "set_epoch"):
                unlabeled_loader.sampler.set_epoch(epoch)
            if hasattr(labeled_loader.sampler, "set_epoch"):
                labeled_loader.sampler.set_epoch(epoch)

        labeled_iter = iter(labeled_loader)

        for index, ulb_batch in enumerate(unlabeled_loader):
            ulb_images, _ = ulb_batch
            # dataset provides (weak, strong, simclr) views; use weak & strong for SoftMatch
            ulb_weak = ulb_images[0].to(device)
            ulb_strong = ulb_images[1].to(device)

            try:
                lb_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                lb_batch = next(labeled_iter)

            lb_images, lb_labels = lb_batch
            lb_images = lb_images.to(device)
            lb_labels = lb_labels.to(device)

            optimizer.zero_grad()

            # forward passes
            _, class_lb = softclr_model(lb_images)
            _, class_ulb_weak = softclr_model(ulb_weak)
            _, class_ulb_strong = softclr_model(ulb_strong)

            probs_lb = torch.softmax(class_lb.detach(), dim=-1)
            probs_ulb_weak = torch.softmax(class_ulb_weak.detach(), dim=-1)

            softmatch_trainer.update_prob_t(probs_lb, probs_ulb_weak)

            loss_sup = softmatch_trainer.compute_supervised_loss(class_lb, lb_labels)
            loss_unsup = softmatch_trainer.compute_unsupervised_loss(
                logits_weak=class_ulb_weak,
                logits_strong=class_ulb_strong,
                probs_weak=probs_ulb_weak,
            )

            loss = conf.softmatch_sup_weight * loss_sup + conf.softmatch_unsup_weight * loss_unsup

            loss.backward()
            optimizer.step()
            model_ema.update()

            if is_main_process():
                wandb.log({
                    "softmatch/loss": loss.item(),
                    "softmatch/loss_supervised": loss_sup.item(),
                    "softmatch/loss_unsupervised": loss_unsup.item(),
                    "softmatch_step": step,
                })
                step += 1

        scheduler.step()

        # Evaluation and checkpointing
        if (epoch + 1) % conf.eval_every == 0 or epoch == conf.pretrain_epochs - 1:
            save_model = (epoch + 1) % conf.save_model_every == 0 or epoch == conf.pretrain_epochs - 1
            save_path = f"{conf.model_saved_path}/softmatch_{model_name}_epoch_{epoch + 1}_ts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth" if save_model else None

            barrier()

            model_ema.apply_shadow()
            eval_model = softclr_model.module if distributed else softclr_model

            val_transform = get_val_transforms(96)
            test_loader = prepare_stl10_test(
                preprocess=val_transform,
                batch_size=per_gpu_unlabeled,
                num_workers=NUM_WORKERS,
                distributed=distributed,
            )
            top1, top5, per_class = evaluate_model(model=eval_model, dataloader=test_loader, device=device, distributed=distributed, num_classes=1000)

            model_ema.restore()

            if is_main_process():
                wandb.log({"eval/top1": top1, "eval/top5": top5, "epoch": epoch + 1})
                wandb.termlog(f"Epoch {epoch + 1} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")

                if save_model:
                    model_to_save = softclr_model.module if distributed else softclr_model
                    torch.save(model_to_save.state_dict(), save_path)
                    with open(save_path.replace(".pth", "_per_class_acc.json"), "w") as f:
                        json.dump({"top1": top1, "top5": top5, "per_class": per_class}, f)
                    wandb.termlog(f"Model saved to {save_path}")

            if distributed:
                barrier()


def main() -> None:
    conf = parse_softmatch_cli()

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
                project="softmatch_only",
                config={**conf.model_dump(), "encoder": model_name, "world_size": world_size},
                name=f"softmatch-{model_name}-{world_size}gpu",
                reinit=True,
            )

        barrier()

        try:
            train_softmatch_model(model_name=model_name, local_rank=local_rank, world_size=world_size, conf=conf)
        finally:
            if is_main_process():
                wandb.finish()

        barrier()

        gc.collect()
        torch.cuda.empty_cache()

    cleanup_distributed()


if __name__ == "__main__":
    main()
