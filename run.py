import os
import random
import numpy as np
import tqdm
import wandb

import torch

from .utils import (
    conf,
    prepare_simclr_train_dataset,
    linear_evaluation,
)
from .sim_clr import (
    SimCLR,
    NTXentLoss,
    LARS,
    get_encoder,
)


NUM_WORKERS = 20
os.environ["MKL_NUM_THREADS"] = f"{NUM_WORKERS}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{NUM_WORKERS}"
os.environ["OMP_NUM_THREADS"] = f"{NUM_WORKERS}"

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.set_float32_matmul_precision("high")


os.makedirs(conf.model_saved_path, exist_ok=True)


def main():
    wandb.init(project="simclr", config=conf)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = get_encoder(conf.model).to(device)
    simclr_model = SimCLR(encoder).to(device)
    wandb.watch(simclr_model)

    contrastive_loss = NTXentLoss(temperature=conf.pretrain_temperature, device=device).to(device)

    # [...] optimized using LARS with learning rate of 4.8 (= 0.3 ×BatchSize/256)
    learning_rate = 0.3 * conf.pretrain_batch_size / 256

    # Separate parameters for weight decay (exclude bias and batch norm)
    param_weights = []
    param_biases = []
    for name, param in simclr_model.named_parameters():
        if "bias" in name or "bn" in name or "batch_normalization" in name:
            param_biases.append(param)
        else:
            param_weights.append(param)

    parameters = [
        {"params": param_weights, "weight_decay": conf.pretrain_weight_decay},
        {"params": param_biases, "weight_decay": 0.0},
    ]

    optimizer = LARS(
        parameters,
        lr=learning_rate,
        weight_decay=conf.pretrain_weight_decay,
        eta=0.001,
    )

    # Warmup for 10 epochs, then Cosine Decay
    warmup_epochs = 10
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=conf.pretrain_epochs - warmup_epochs, eta_min=0)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
    )

    contrastive_dataloader = prepare_simclr_train_dataset(conf.pretrain_batch_size, NUM_WORKERS, conf.image_size)

    for epoch in tqdm.tqdm(range(conf.pretrain_epochs)):
        simclr_model.train()
        total_loss = 0

        for index, batch in enumerate(contrastive_dataloader):
            images, _ = batch
            images_i, images_j = images[0].to(device), images[1].to(device)

            optimizer.zero_grad()

            proj_i = simclr_model(images_i)
            proj_j = simclr_model(images_j)

            loss = contrastive_loss(proj_i, proj_j)
            loss.backward()
            optimizer.step()

            wandb.log({"pretrain/loss": loss.item()})

            total_loss += loss.item()

            # For testing only - remove during actual training!
            if index == 100:
                break

        scheduler.step()

        if (epoch + 1) % conf.lin_eval_every == 0 or epoch == conf.pretrain_epochs - 1:
            save_model = True if (epoch + 1) % conf.save_model_every == 0 or epoch == conf.pretrain_epochs - 1 else False
            save_path = f"{conf.model_saved_path}/simclr_epoch_{epoch + 1}.pth" if save_model else None

            acc = linear_evaluation(
                model=simclr_model.encoder,
                config=conf,
                device=device,
                num_workers=NUM_WORKERS,
                save_model=save_model,
                save_path=save_path,
            )
            wandb.log({"lin_eval/accuracy": acc, "epoch": epoch + 1})


if __name__ == "__main__":
    main()
