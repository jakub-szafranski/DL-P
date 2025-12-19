import os
import random
import numpy as np
import tqdm
import wandb
import torch
from typing import Literal

from utils import (
    conf,
    prepare_simclr_train_dataset,
    fine_tune,
)
from sim_clr import (
    SimCLR,
    NTXentLoss,
    LARS,
    get_encoder,
)


NUM_WORKERS = 20
os.environ["MKL_NUM_THREADS"] = f"{NUM_WORKERS}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{NUM_WORKERS}"
os.environ["OMP_NUM_THREADS"] = f"{NUM_WORKERS}"

np.random.seed(conf.seed)
torch.manual_seed(conf.seed)
random.seed(conf.seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.set_float32_matmul_precision("high")


os.makedirs(conf.model_saved_path, exist_ok=True)


def train_simclr_model(model: Literal["resnet50", "vit_b_16", "efficientnet_b5"]) -> None:
    """Pre-train SimCLR and periodically fine-tune for evaluation.

    Args:
        model (Literal["resnet50", "vit_b_16", "efficientnet_b5"]): Encoder backbone name.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder, num_features = get_encoder(model)
    simclr_model = SimCLR(encoder, num_features=num_features).to(device)
    wandb.watch(simclr_model, log="gradients", log_freq=100)

    # Define custom step metric for pretraining to avoid jumps in plots due to fine-tuning steps
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

    contrastive_dataloader = prepare_simclr_train_dataset(conf.pretrain_batch_size, NUM_WORKERS, conf.image_size)

    pretrain_step = 0
    for epoch in tqdm.tqdm(range(conf.pretrain_epochs)):
        simclr_model.train()

        for index, batch in enumerate(contrastive_dataloader):
            images, _ = batch
            images_i, images_j = images[0].to(device), images[1].to(device)

            optimizer.zero_grad()

            proj_i = simclr_model(images_i)
            proj_j = simclr_model(images_j)

            loss = contrastive_loss(proj_i, proj_j)
            loss.backward()
            optimizer.step()

            wandb.log({"pretrain/loss": loss.item(), "pretrain_step": pretrain_step})
            pretrain_step += 1

            # For testing only - remove during actual training!
            if index == 100:
                break

        scheduler.step()

        if (epoch + 1) % conf.eval_every == 0 or epoch == conf.pretrain_epochs - 1:
            save_model = (epoch + 1) % conf.save_model_every == 0 or epoch == conf.pretrain_epochs - 1
            save_path = f"{conf.model_saved_path}/simclr_epoch_{epoch + 1}.pth" if save_model else None

            frozen_acc, full_acc = fine_tune(
                model=simclr_model.encoder,
                num_features=num_features,
                device=device,
                num_workers=NUM_WORKERS,
                config=conf,
                pretrain_epoch=epoch + 1,
                save_path=save_path,
            )
            wandb.log({
                "ft/frozen_accuracy": frozen_acc,
                "ft/full_accuracy": full_acc,
                "epoch": epoch + 1,
            })


if __name__ == "__main__":
    models = ["resnet50", "vit_b_16", "efficientnet_b5"]
    for model_name in models:
        with wandb.init(
            project="simclr",
            config={**conf.model_dump(), "encoder": model_name},
            name=f"pretrain-{model_name}",
            reinit=True,
        ):
            train_simclr_model(model_name)
