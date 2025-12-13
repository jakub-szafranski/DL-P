import torch
import wandb
from torch import nn
from torchvision import transforms
from sklearn.model_selection import train_test_split
from .data import prepare__ImageNetTrain, prepare__ImageNetTest

from logging import getLogger

log = getLogger(__name__)


def linear_evaluation(
    model: nn.Module,
    device: torch.device,
    num_workers: int,
    config,
    save_model: bool,
    save_path: str | None,
) -> float:
    """
    Evaluates the given model using the Linear Evaluation protocol (training a linear classifier on top of frozen encoder).

    Args:
        model (nn.Module): The encoder model.
        config: Configuration object.
        device (torch.device): Device to perform evaluation on.
        num_workers (int): Number of subprocesses to use for data loading.
        save_model (bool): Whether to save the best model.
        save_path (str | None): Path to save the model if save_model is True.

    Returns:
        float: Accuracy of the model on the evaluation dataset.
    """
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

    full_train_loader = prepare__ImageNetTrain(
        preprocess=train_transform, batch_size=config.lin_eval_batch_size, num_workers=num_workers
    )

    train_loader = _get_data_subset(full_train_loader, config.lin_eval_subset_ratio)
    log.debug(f"Linear Evaluation: Using {len(train_loader.dataset)} samples for training.")

    test_loader = prepare__ImageNetTest(preprocess=val_transform, batch_size=config.lin_eval_batch_size, num_workers=num_workers)

    class LinearEvalModel(nn.Module):
        def __init__(self, encoder, num_classes=1000):
            super().__init__()
            self.encoder = encoder
            self.classifier = nn.Linear(2048, num_classes)

        def forward(self, x):
            with torch.no_grad():
                x = self.encoder(x)
            return self.classifier(x)

    # Freeze encoder weights
    for param in model.parameters():
        param.requires_grad = False

    lin_eval_model = LinearEvalModel(model).to(device)
    wandb.watch(lin_eval_model)

    log.debug("Linear Evaluation: Trainable parameters:")
    for name, param in lin_eval_model.named_parameters():
        if param.requires_grad:
            log.debug(f" - {name}: {param.shape}")

    optimizer = torch.optim.SGD(
        lin_eval_model.classifier.parameters(),
        lr=config.lin_eval_learning_rate,
        momentum=config.lin_eval_momentum,
        weight_decay=0,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.lin_eval_epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(config.lin_eval_epochs):
        lin_eval_model.train()
        lin_eval_model.encoder.eval()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = lin_eval_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            wandb.log({"lin_eval/train_loss": loss.item()})

        scheduler.step()

        acc = _evaluate_model(lin_eval_model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            if save_model:
                torch.save(lin_eval_model.state_dict(), save_path)

    return best_acc


def _evaluate_model(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """
    Evaluates the given model on the provided dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to perform evaluation on.

    Returns:
        float: Accuracy of the model on the evaluation dataset.
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

    accuracy = 100 * correct / total
    return accuracy


def _get_data_subset(dataloader: torch.utils.data.DataLoader, subset_ratio: float) -> torch.utils.data.DataLoader:
    """
    Create a class-balanced (stratified) subset of the given DataLoader.

    Args:
        dataloader (torch.utils.data.DataLoader): Source DataLoader.
        subset_ratio (float): Fraction of the dataset to retain (between 0 and 1).

    Returns:
        torch.utils.data.DataLoader: DataLoader for the balanced subset.
    """
    dataset = dataloader.dataset
    indices, _ = train_test_split(range(len(dataset)), train_size=subset_ratio, stratify=dataset.targets, random_state=42)

    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=dataloader.batch_size,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        shuffle=True,
    )
