import copy

import torch
import wandb
from torch import nn
from torchvision import transforms
from sklearn.model_selection import train_test_split
from .data import prepare__ImageNetTrain, prepare__ImageNetTest

from logging import getLogger

log = getLogger(__name__)


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
    config,
    save_path: str | None = None,
) -> tuple[float, float]:
    """
    Two-stage fine-tuning evaluation (does not modify original model).

    Args:
        model (nn.Module): The encoder model (will be deep-copied).
        num_features (int): Output dimension of the encoder.
        device (torch.device): Device to perform training on.
        num_workers (int): Number of subprocesses for data loading.
        config: Configuration object.
        save_path (str | None): Path to save the best model.

    Returns:
        tuple[float, float]: Accuracy after stage 1 (frozen) and stage 2 (full).
    """
    # Deep copy encoder to avoid modifying original SimCLR model
    encoder_copy = copy.deepcopy(model)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_train_loader = prepare__ImageNetTrain(
        preprocess=train_transform, batch_size=config.ft_frozen_batch_size, num_workers=num_workers
    )
    train_dataset_subset = _get_data_subset(full_train_loader.dataset, config.ft_subset_ratio)
    test_loader = prepare__ImageNetTest(
        preprocess=val_transform, batch_size=config.ft_frozen_batch_size, num_workers=num_workers
    )

    ft_model = FineTuneModel(encoder_copy, num_features).to(device)

    log.info("Stage 1: Training classifier with frozen encoder")
    frozen_acc = _train_frozen(ft_model, train_dataset_subset, test_loader, device, config, num_workers)
    log.info(f"Stage 1 complete. Best accuracy: {frozen_acc:.2f}%")

    log.info("Stage 2: Full fine-tuning")
    full_acc = _train_full(ft_model, train_dataset_subset, test_loader, device, config, num_workers)
    log.info(f"Stage 2 complete. Best accuracy: {full_acc:.2f}%")

    if save_path:
        torch.save(ft_model.state_dict(), save_path)
        log.info(f"Model saved to {save_path}")

    return frozen_acc, full_acc


def _train_frozen(
    model: FineTuneModel,
    train_dataset: torch.utils.data.Dataset,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config,
    num_workers: int,
) -> float:
    """
    Train only the classifier head with frozen encoder.

    Returns:
        float: Best accuracy achieved during this stage.
    """
    for param in model.encoder.parameters():
        param.requires_grad = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.ft_frozen_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )

    optimizer = torch.optim.SGD(
        model.classifier.parameters(),
        lr=config.ft_frozen_learning_rate,
        momentum=config.ft_frozen_momentum,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.ft_frozen_epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(config.ft_frozen_epochs):
        model.train()
        model.encoder.eval()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                features = model.encoder(images)
            outputs = model.classifier(features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log({"ft_frozen/train_loss": loss.item()})

        scheduler.step()

        acc = _evaluate_model(model, test_loader, device)
        wandb.log({"ft_frozen/val_accuracy": acc, "ft_frozen/epoch": epoch + 1})
        best_acc = max(best_acc, acc)

    return best_acc


def _train_full(
    model: FineTuneModel,
    train_dataset: torch.utils.data.Dataset,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config,
    num_workers: int,
) -> float:
    """
    Fine-tune the entire model (encoder + classifier).

    Returns:
        float: Best accuracy achieved during this stage.
    """
    for param in model.encoder.parameters():
        param.requires_grad = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.ft_full_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
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

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log({"ft_full/train_loss": loss.item()})

        scheduler.step()

        acc = _evaluate_model(model, test_loader, device)
        wandb.log({"ft_full/val_accuracy": acc, "ft_full/epoch": epoch + 1})
        best_acc = max(best_acc, acc)

    return best_acc


def _evaluate_model(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """
    Evaluates the model on the provided dataloader.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation.
        device (torch.device): Device to perform evaluation on.

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
