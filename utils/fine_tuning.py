import copy
import gc
import json
from typing import Any

import torch
import torch.distributed as dist
import wandb
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from utils.data import prepare_stl10_train, prepare_stl10_test, get_val_transforms, STL10_NUM_CLASSES
from utils.distributed import is_main_process, get_world_size


def _ft_key(stage: str, pretrain_epoch: int, subset_ratio: float, name: str) -> str:
    """Create a W&B metric key namespaced by pretrain epoch and subset ratio.

    Args:
        stage (str): Metric stage prefix (e.g. "ft_frozen").
        pretrain_epoch (int): Pretraining epoch number.
        subset_ratio (float): Data subset ratio used for fine-tuning.
        name (str): Base metric name.

    Returns:
        str: Fully-qualified W&B metric key.
    """
    return f"{stage}/pretrain_epoch={pretrain_epoch}/subset={subset_ratio}/{name}"


class FineTuneModel(nn.Module):
    """Wrapper model for fine-tuning with a classifier head."""

    def __init__(self, encoder: nn.Module, num_features: int, num_classes: int = STL10_NUM_CLASSES):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)


def fine_tune(
    model: nn.Module,
    num_features: int,
    device: torch.device,
    num_workers: int,
    config: Any,
    pretrain_epoch: int,
    subset_ratio: float,
    save_path: str | None = None,
    distributed: bool = False,
    local_rank: int = 0,
) -> tuple[float, float, float, float, list[float], list[float]]:
    """
    Two-stage fine-tuning evaluation (does not modify original model).

    Args:
        model (nn.Module): The encoder model (will be deep-copied).
        num_features (int): Output dimension of the encoder.
        device (torch.device): Device to perform training on.
        num_workers (int): Number of subprocesses for data loading.
        config (Any): Configuration object.
        pretrain_epoch (int): Pretraining epoch at which evaluation is performed.
        subset_ratio (float): Ratio of data used for fine-tuning.
        save_path (str | None): Path to save the best model.
        distributed (bool): Whether to use distributed training.
        local_rank (int): Local GPU rank for distributed training.

    Returns:
        tuple: (frozen_top1, frozen_top5, full_top1, full_top5, frozen_per_class, full_per_class).
    """
    encoder_copy = copy.deepcopy(model)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    ])
    val_transform = get_val_transforms(96)

    world_size = get_world_size()
    per_gpu_batch_size = config.ft_frozen_batch_size // world_size

    train_loader = prepare_stl10_train(
        preprocess=train_transform, batch_size=per_gpu_batch_size, num_workers=num_workers, distributed=distributed
    )
    train_dataset = train_loader.dataset
    test_loader = prepare_stl10_test(
        preprocess=val_transform,
        batch_size=per_gpu_batch_size,
        num_workers=num_workers,
        distributed=distributed,
    )

    ft_model = FineTuneModel(encoder_copy, num_features, num_classes=STL10_NUM_CLASSES)
    ft_model = ft_model.to(device)

    # Freeze encoder BEFORE DDP wrapping (Stage 1 starts with frozen encoder)
    for param in ft_model.encoder.parameters():
        param.requires_grad = False

    if distributed:
        ft_model = DDP(ft_model, device_ids=[local_rank])

    if is_main_process():
        wandb.termlog("Stage 1: Training classifier with frozen encoder")

    frozen_top1, frozen_top5, frozen_per_class = _train_frozen(
        model=ft_model,
        train_dataset=train_dataset,
        test_loader=test_loader,
        device=device,
        config=config,
        num_workers=num_workers,
        pretrain_epoch=pretrain_epoch,
        subset_ratio=subset_ratio,
        distributed=distributed,
    )

    if is_main_process():
        wandb.termlog(f"Stage 1 complete. Top-1: {frozen_top1:.2f}% | Top-5: {frozen_top5:.2f}%")
        wandb.termlog("Stage 2: Full fine-tuning")

    # Unfreeze encoder and re-wrap with DDP for Stage 2
    # DDP must be re-constructed after changing requires_grad flags
    base_model = ft_model.module if distributed else ft_model
    for param in base_model.encoder.parameters():
        param.requires_grad = True

    if distributed:
        ft_model = DDP(base_model, device_ids=[local_rank])
    else:
        ft_model = base_model

    full_top1, full_top5, full_per_class = _train_full(
        model=ft_model,
        train_dataset=train_dataset,
        test_loader=test_loader,
        device=device,
        config=config,
        num_workers=num_workers,
        pretrain_epoch=pretrain_epoch,
        subset_ratio=subset_ratio,
        distributed=distributed,
    )

    if is_main_process():
        wandb.termlog(f"Stage 2 complete. Top-1: {full_top1:.2f}% | Top-5: {full_top5:.2f}%")

        if save_path:
            model_to_save = ft_model.module if distributed else ft_model
            torch.save(model_to_save.state_dict(), save_path)
            wandb.termlog(f"Model saved to {save_path}")

            json_path = save_path.replace(".pth", "_per_class_acc.json")
            with open(json_path, "w") as f:
                json.dump({"frozen": {"top1": frozen_top1, "top5": frozen_top5, "per_class": frozen_per_class},
                           "full": {"top1": full_top1, "top5": full_top5, "per_class": full_per_class}}, f)
            wandb.termlog(f"Per-class accuracy saved to {json_path}")

    # Ensure all ranks wait for model saving to complete
    if distributed:
        dist.barrier()

    del ft_model, encoder_copy
    gc.collect()
    torch.cuda.empty_cache()

    return frozen_top1, frozen_top5, full_top1, full_top5, frozen_per_class, full_per_class


def _train_frozen(
    model: FineTuneModel,
    train_dataset: torch.utils.data.Dataset,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Any,
    num_workers: int,
    pretrain_epoch: int,
    subset_ratio: float,
    distributed: bool = False,
) -> tuple[float, float, list[float]]:
    """
    Train only the classifier head with frozen encoder.

    Args:
        model (FineTuneModel): Fine-tuning model (or DDP-wrapped model).
        train_dataset (torch.utils.data.Dataset): Training dataset.
        test_loader (torch.utils.data.DataLoader): Test dataloader.
        device (torch.device): Device to perform training on.
        config (Any): Configuration object.
        num_workers (int): Number of subprocesses for data loading.
        pretrain_epoch (int): Pretraining epoch at which evaluation is performed.
        subset_ratio (float): Data subset ratio used for fine-tuning.
        distributed (bool): Whether to use distributed training.

    Returns:
        tuple[float, float, list[float]]: Best (top1, top5, per_class) achieved.
    """
    base_model = model.module if distributed else model

    # Create sampler for distributed training
    sampler = None
    shuffle = True
    if distributed and dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        shuffle = False

    world_size = get_world_size()
    per_gpu_batch_size = config.ft_frozen_batch_size // world_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=per_gpu_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
        sampler=sampler,
    )

    optimizer = torch.optim.SGD(
        base_model.classifier.parameters(),
        lr=config.ft_frozen_learning_rate,
        momentum=config.ft_frozen_momentum,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.ft_frozen_epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_result: tuple[float, float, list[float]] = (0.0, 0.0, [])
    for epoch in range(config.ft_frozen_epochs):
        model.train()
        base_model.encoder.eval()

        if distributed and sampler is not None:
            sampler.set_epoch(epoch)

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if is_main_process():
                wandb.log({_ft_key("ft_frozen", pretrain_epoch, subset_ratio, "train_loss"): loss.item()})

        scheduler.step()

        top1, top5, per_class = evaluate_model(model=model, dataloader=test_loader, device=device, distributed=distributed, num_classes=1000)
        if is_main_process():
            wandb.log(
                {
                    _ft_key("ft_frozen", pretrain_epoch, subset_ratio, "val_top1"): top1,
                    _ft_key("ft_frozen", pretrain_epoch, subset_ratio, "val_top5"): top5,
                    _ft_key("ft_frozen", pretrain_epoch, subset_ratio, "epoch"): epoch + 1,
                }
            )
        if top1 > best_acc:
            best_acc = top1
            best_result = (top1, top5, per_class)

    return best_result


def _train_full(
    model: FineTuneModel,
    train_dataset: torch.utils.data.Dataset,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Any,
    num_workers: int,
    pretrain_epoch: int,
    subset_ratio: float,
    distributed: bool = False,
) -> tuple[float, float, list[float]]:
    """
    Fine-tune the entire model (encoder + classifier).

    Args:
        model (FineTuneModel): Fine-tuning model (or DDP-wrapped model).
        train_dataset (torch.utils.data.Dataset): Training dataset.
        test_loader (torch.utils.data.DataLoader): Test dataloader.
        device (torch.device): Device to perform training on.
        config (Any): Configuration object.
        num_workers (int): Number of subprocesses for data loading.
        pretrain_epoch (int): Pretraining epoch at which evaluation is performed.
        subset_ratio (float): Data subset ratio used for fine-tuning.
        distributed (bool): Whether to use distributed training.

    Returns:
        tuple[float, float, list[float]]: Best (top1, top5, per_class) achieved.
    """
    # Create sampler for distributed training
    sampler = None
    shuffle = True
    if distributed and dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        shuffle = False

    world_size = get_world_size()
    per_gpu_batch_size = config.ft_full_batch_size // world_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=per_gpu_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
        sampler=sampler,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.ft_full_learning_rate,
        momentum=config.ft_full_momentum,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.ft_full_epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_result: tuple[float, float, list[float]] = (0.0, 0.0, [])
    for epoch in range(config.ft_full_epochs):
        model.train()

        if distributed and sampler is not None:
            sampler.set_epoch(epoch)

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if is_main_process():
                wandb.log({_ft_key("ft_full", pretrain_epoch, subset_ratio, "train_loss"): loss.item()})

        scheduler.step()

        top1, top5, per_class = evaluate_model(model=model, dataloader=test_loader, device=device, distributed=distributed, num_classes=1000)
        if is_main_process():
            wandb.log(
                {
                    _ft_key("ft_full", pretrain_epoch, subset_ratio, "val_top1"): top1,
                    _ft_key("ft_full", pretrain_epoch, subset_ratio, "val_top5"): top5,
                    _ft_key("ft_full", pretrain_epoch, subset_ratio, "epoch"): epoch + 1,
                }
            )
        if top1 > best_acc:
            best_acc = top1
            best_result = (top1, top5, per_class)

    return best_result


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    distributed: bool = False,
    num_classes: int = 1000,
) -> tuple[float, float, list[float]]:
    """
    Evaluate the model on the given dataloader.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation.
        device (torch.device): Device to perform evaluation on.
        distributed (bool): Whether to use distributed evaluation.
        num_classes (int): Number of classes for per-class accuracy.

    Returns:
        tuple[float, float, list[float]]: Top-1 accuracy, Top-5 accuracy, per-class accuracy.
    """
    model.eval()

    correct1 = torch.zeros(1, device=device, dtype=torch.float32)
    correct5 = torch.zeros(1, device=device, dtype=torch.float32)
    total = torch.zeros(1, device=device, dtype=torch.float32)

    class_correct = torch.zeros(num_classes, device=device, dtype=torch.float32)
    class_total = torch.zeros(num_classes, device=device, dtype=torch.float32)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[1] # SoftMatch returns (proj, logits)

            _, top5_pred = outputs.topk(5, dim=1)
            
            is_correct1 = (top5_pred[:, 0] == labels)
            is_correct5 = (top5_pred == labels.unsqueeze(1)).any(dim=1)

            correct1 += is_correct1.sum()
            correct5 += is_correct5.sum()
            total += labels.size(0)

            ones = torch.ones_like(labels, dtype=torch.float32)
            class_total.index_add_(0, labels, ones)
            class_correct.index_add_(0, labels, is_correct1.to(torch.float32))

    if distributed and dist.is_initialized():
        # Sync all metrics across GPUs
        metrics = torch.stack([correct1, correct5, total])
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        correct1, correct5, total = metrics[0], metrics[1], metrics[2]
        
        dist.all_reduce(class_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(class_total, op=dist.ReduceOp.SUM)

    t_val = total.item()
    top1_acc = (correct1.item() / t_val * 100.0) if t_val > 0 else 0.0
    top5_acc = (correct5.item() / t_val * 100.0) if t_val > 0 else 0.0
    
    per_class = (class_correct / class_total.clamp(min=1.0) * 100.0).cpu().tolist()

    return top1_acc, top5_acc, per_class