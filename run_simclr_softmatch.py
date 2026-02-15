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
from utils import parse_softclr_cli, prepare_softclr_train_dataset, evaluate_model
from utils.config import SoftCLRConfig
from utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    barrier,
)
from sim_clr import get_encoder
from soft_match import SoftCLR, SoftMatchTrainer, ModelEMA, SoftNTXentLoss
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


def train_simclr_softmatch_model(
    model_name: Literal["resnet50", "vit_b_16", "efficientnet_b5"],
    local_rank: int,
    world_size: int,
    conf: SoftCLRConfig,
) -> None:
    """
    Train SimCLR along with SoftMatch using DDP and periodically perform evaluation steps.

    Args:
        model_name (Literal): Encoder backbone name.
        local_rank (int): Local GPU rank within this node.
        world_size (int): Total number of processes.
        conf (SoftCLRConfig): Configuration object.
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
        wandb.define_metric("pretrain_step")
        wandb.define_metric("pretrain/*", step_metric="pretrain_step")

    contrastive_loss = SoftNTXentLoss(temperature=conf.pretrain_temperature).to(device)

    softmatch_trainer = SoftMatchTrainer(
        num_classes=10,  # STL-10 classes
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

    # Warmup for 10 epochs, then Cosine Decay
    warmup_epochs = 10
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=conf.pretrain_epochs - warmup_epochs, eta_min=0)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
    )

    # Prepare dataloaders with DistributedSampler (unlabeled_loader: full dataset for SimCLR + unsupervised SoftMatch, labeled_loader: subset with labels for supervised SoftMatch loss)
    per_gpu_batch_size = conf.pretrain_batch_size // world_size
    unlabeled_loader, labeled_loader = prepare_softclr_train_dataset(
        batch_size=per_gpu_batch_size, 
        num_workers=NUM_WORKERS, 
        img_size=conf.image_size, 
        distributed=distributed,
    )

    pretrain_step = 0
    epoch_iter = tqdm.tqdm(range(conf.pretrain_epochs)) if is_main_process() else range(conf.pretrain_epochs)

    for epoch in epoch_iter:
        softclr_model.train()

        if distributed:
            if hasattr(unlabeled_loader.sampler, "set_epoch"):
                unlabeled_loader.sampler.set_epoch(epoch)
            if hasattr(labeled_loader.sampler, "set_epoch"):
                labeled_loader.sampler.set_epoch(epoch)

        labeled_iter = iter(labeled_loader)
        labeled_epoch = epoch
        
        for index, ulb_batch in enumerate(unlabeled_loader):
            ulb_images, _ = ulb_batch
            ulb_weak = ulb_images[0].to(device)
            ulb_softmatch = ulb_images[1].to(device)
            ulb_simclr = ulb_images[2].to(device)

            try:
                lb_batch = next(labeled_iter)
            except StopIteration:
                labeled_epoch += 1
                if distributed and hasattr(labeled_loader.sampler, "set_epoch"):
                    labeled_loader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_loader)
                lb_batch = next(labeled_iter)

            lb_images, lb_labels = lb_batch
            lb_images = lb_images.to(device)
            lb_labels = lb_labels.to(device)

            optimizer.zero_grad()

            _, class_lb = softclr_model(lb_images)

            _, class_ulb_weak = softclr_model(ulb_weak)
            proj_ulb_softmatch, class_ulb_softmatch = softclr_model(ulb_softmatch)
            proj_ulb_simclr, class_ulb_simclr = softclr_model(ulb_simclr)

            # Compute probabilities for soft contrastive loss
            probs_ulb_softmatch = torch.softmax(class_ulb_softmatch.detach(), dim=-1)
            probs_ulb_simclr = torch.softmax(class_ulb_simclr.detach(), dim=-1)

            loss_simclr = contrastive_loss(
                z_i=proj_ulb_softmatch,
                z_j=proj_ulb_simclr,
                probs_i=probs_ulb_softmatch,
                probs_j=probs_ulb_simclr,
                prob_max_mu=softmatch_trainer.prob_max_mu_t,
                prob_max_var=softmatch_trainer.prob_max_var_t,
                use_weights=epoch >= conf.soft_weights_epoch,
            )

            loss_softmatch_supervised = softmatch_trainer.compute_supervised_loss(class_lb, lb_labels)

            probs_lb = torch.softmax(class_lb.detach(), dim=-1)
            probs_ulb_weak = torch.softmax(class_ulb_weak.detach(), dim=-1)
            softmatch_trainer.update_prob_t(probs_lb, probs_ulb_weak)
            loss_softmatch_unsupervised = softmatch_trainer.compute_unsupervised_loss(
                logits_weak=class_ulb_weak,
                logits_strong=class_ulb_softmatch,
                probs_weak=probs_ulb_weak,
            )

            if conf.use_decaying_loss_weight:
                def get_decay_weight(epoch, conf):
                    progress = (conf.pretrain_epochs - epoch) / conf.pretrain_epochs
                    return conf.decaying_weight_min + 0.5 * (conf.decaying_weight_max - conf.decaying_weight_min) * (1 - np.cos(np.pi * progress))

                softmatch_loss = conf.softmatch_sup_weight * loss_softmatch_supervised + conf.softmatch_unsup_weight * loss_softmatch_unsupervised
                alpha = get_decay_weight(epoch, conf)
                loss = alpha * conf.simclr_weight * loss_simclr + (1 - alpha) * softmatch_loss * conf.softmatch_loss_weight
            else:
                loss = conf.simclr_weight * loss_simclr + conf.softmatch_sup_weight * loss_softmatch_supervised + conf.softmatch_unsup_weight * loss_softmatch_unsupervised

            loss.backward()
            optimizer.step()
            model_ema.update()

            if is_main_process():
                wandb.log({
                    "pretrain/loss": loss.item(),
                    "pretrain/loss_simclr": loss_simclr.item(),
                    "pretrain/loss_supervised": loss_softmatch_supervised.item(),
                    "pretrain/loss_unsupervised": loss_softmatch_unsupervised.item(),
                    "pretrain_step": pretrain_step,
                })
                pretrain_step += 1

        scheduler.step()

        # Evaluation and checkpointing
        if (epoch + 1) % conf.eval_every == 0 or epoch == conf.pretrain_epochs - 1:
            save_model = (epoch + 1) % conf.save_model_every == 0 or epoch == conf.pretrain_epochs - 1
            save_path = f"{conf.model_saved_path}/softclr_{model_name}_epoch_{epoch + 1}_ts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth" if save_model else None

            # Synchronize before evaluation
            barrier()

            model_ema.apply_shadow()
            eval_model = softclr_model.module if distributed else softclr_model

            val_transform = get_val_transforms(96)
            test_loader = prepare_stl10_test(
                preprocess=val_transform,
                batch_size=per_gpu_batch_size,
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

            # Ensure all ranks wait for model saving to complete
            if distributed:
                barrier()


def main() -> None:
    conf = parse_softclr_cli()

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
                project="simclr_softmatch",
                config={**conf.model_dump(), "encoder": model_name, "world_size": world_size},
                name=f"pretrain-{model_name}-{world_size}gpu",
                reinit=True,
            )

        barrier()

        try:
            train_simclr_softmatch_model(model_name=model_name, local_rank=local_rank, world_size=world_size, conf=conf)
        finally:
            if is_main_process():
                wandb.finish()

        barrier()

        gc.collect()
        torch.cuda.empty_cache()

    cleanup_distributed()


if __name__ == "__main__":
    main()
