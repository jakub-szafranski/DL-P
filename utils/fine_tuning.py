import copy
from typing import Any

import torch
import torch.distributed as dist
import wandb
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils.data import prepare__ImageNetTrain, prepare__ImageNetTest
from utils.distributed import is_main_process, get_world_size


def _ft_key(stage: str, pretrain_epoch: int, name: str) -> str:
    """Create a W&B metric key namespaced by pretrain epoch.

    Args:
        stage (str): Metric stage prefix (e.g. "ft_frozen").
        pretrain_epoch (int): Pretraining epoch number.
        name (str): Base metric name.

    Returns:
        str: Fully-qualified W&B metric key.
    """

    return f"{stage}/pretrain_epoch={pretrain_epoch}/{name}"


class FineTuneModel(nn.Module):
    """Wrapper model for fine-tuning with a classifier head."""

    def __init__(self, encoder: nn.Module, num_features: int, num_classes: int = 1000):
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
    save_path: str | None = None,
    distributed: bool = False,
    local_rank: int = 0,
) -> tuple[float, float]:
    """
    Two-stage fine-tuning evaluation (does not modify original model).

    Args:
        model (nn.Module): The encoder model (will be deep-copied).
        num_features (int): Output dimension of the encoder.
        device (torch.device): Device to perform training on.
        num_workers (int): Number of subprocesses for data loading.
        config (Any): Configuration object.
        pretrain_epoch (int): Pretraining epoch at which evaluation is performed.
        save_path (str | None): Path to save the best model.
        distributed (bool): Whether to use distributed training.
        local_rank (int): Local GPU rank for distributed training.

    Returns:
        tuple[float, float]: Accuracy after stage 1 (frozen) and stage 2 (full).
    """
    # Deep copy encoder to avoid modifying original SimCLR model
    encoder_copy = copy.deepcopy(model)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    world_size = get_world_size()
    per_gpu_batch_size = config.ft_frozen_batch_size // world_size

    full_train_dataset = prepare__ImageNetTrain(
        preprocess=train_transform, batch_size=per_gpu_batch_size, num_workers=num_workers, distributed=distributed
    ).dataset
    train_dataset_subset = _get_data_subset(full_train_dataset, config.ft_subset_ratio)
    test_loader = prepare__ImageNetTest(
        preprocess=val_transform,
        batch_size=per_gpu_batch_size,
        num_workers=num_workers,
        distributed=distributed,
    )

    ft_model = FineTuneModel(encoder_copy, num_features)
    ft_model = ft_model.to(device)

    # Freeze encoder BEFORE DDP wrapping (Stage 1 starts with frozen encoder)
    for param in ft_model.encoder.parameters():
        param.requires_grad = False

    if distributed:
        ft_model = DDP(ft_model, device_ids=[local_rank])

    if is_main_process():
        wandb.termlog("Stage 1: Training classifier with frozen encoder")

    frozen_acc = _train_frozen(
        model=ft_model,
        train_dataset=train_dataset_subset,
        test_loader=test_loader,
        device=device,
        config=config,
        num_workers=num_workers,
        pretrain_epoch=pretrain_epoch,
        distributed=distributed,
    )

    if is_main_process():
        wandb.termlog(f"Stage 1 complete. Best accuracy: {frozen_acc:.2f}%")
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

    full_acc = _train_full(
        model=ft_model,
        train_dataset=train_dataset_subset,
        test_loader=test_loader,
        device=device,
        config=config,
        num_workers=num_workers,
        pretrain_epoch=pretrain_epoch,
        distributed=distributed,
    )

    if is_main_process():
        wandb.termlog(f"Stage 2 complete. Best accuracy: {full_acc:.2f}%")

        if save_path:
            model_to_save = ft_model.module if distributed else ft_model
            torch.save(model_to_save.state_dict(), save_path)
            wandb.termlog(f"Model saved to {save_path}")

    # Ensure all ranks wait for model saving to complete
    if distributed:
        dist.barrier()

    return frozen_acc, full_acc


def _train_frozen(
    model: FineTuneModel,
    train_dataset: torch.utils.data.Dataset,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Any,
    num_workers: int,
    pretrain_epoch: int,
    distributed: bool = False,
) -> float:
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
        distributed (bool): Whether to use distributed training.

    Returns:
        float: Best accuracy achieved during this stage.
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
                wandb.log({_ft_key("ft_frozen", pretrain_epoch, "train_loss"): loss.item()})

        scheduler.step()

        acc = _evaluate_model(model, test_loader, device, distributed)
        if is_main_process():
            wandb.log(
                {
                    _ft_key("ft_frozen", pretrain_epoch, "val_accuracy"): acc,
                    _ft_key("ft_frozen", pretrain_epoch, "epoch"): epoch + 1,
                }
            )
        best_acc = max(best_acc, acc)

    return best_acc


def _train_full(
    model: FineTuneModel,
    train_dataset: torch.utils.data.Dataset,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Any,
    num_workers: int,
    pretrain_epoch: int,
    distributed: bool = False,
) -> float:
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
        distributed (bool): Whether to use distributed training.

    Returns:
        float: Best accuracy achieved during this stage.
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
                wandb.log({_ft_key("ft_full", pretrain_epoch, "train_loss"): loss.item()})

        scheduler.step()

        acc = _evaluate_model(model, test_loader, device, distributed)
        if is_main_process():
            wandb.log(
                {
                    _ft_key("ft_full", pretrain_epoch, "val_accuracy"): acc,
                    _ft_key("ft_full", pretrain_epoch, "epoch"): epoch + 1,
                }
            )
        best_acc = max(best_acc, acc)

    return best_acc


def _evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    distributed: bool = False,
) -> float:
    """
    Evaluates the model on the provided dataloader.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation.
        device (torch.device): Device to perform evaluation on.
        distributed (bool): Whether to aggregate metrics across GPUs.

    Returns:
        float: Accuracy percentage.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Aggregate across all GPUs in distributed mode
    if distributed and dist.is_initialized():
        correct_tensor = torch.tensor(correct, device=device)
        total_tensor = torch.tensor(total, device=device)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        correct = correct_tensor.item()
        total = total_tensor.item()

    return 100 * correct / total


def _get_data_subset(dataset: torch.utils.data.Dataset, subset_ratio: float) -> torch.utils.data.Subset:
    """
    Create a stratified subset of the dataset.

    Args:
        dataset (torch.utils.data.Dataset): Source dataset.
        subset_ratio (float): Fraction of data to retain (0-1).

    Returns:
        torch.utils.data.Subset: Stratified subset of the dataset.
    """
    indices, _ = train_test_split(
        range(len(dataset)),
        train_size=subset_ratio,
        stratify=dataset.targets,
        random_state=42,
    )
    return torch.utils.data.Subset(dataset, indices)
