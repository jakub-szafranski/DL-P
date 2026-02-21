import torch
import torch.distributed as dist
from torchvision import datasets, transforms

DATASET_PATH = "/raid/kszyc/datasets/"

STL10_NUM_CLASSES = 10


class ContrastiveTransformations:
    """Creates two augmented versions of an image for contrastive learning."""

    def __init__(self, base_transforms: transforms.Compose):
        self.base_transforms = base_transforms

    def __call__(self, x):
        return [self.base_transforms(x), self.base_transforms(x)]


class SoftCLRTransformations:
    """
    Creates weak and strong augmented versions for SoftMatch+SimCLR training.

    Returns: [weak_aug, softmatch_aug, simclr_aug]
    """

    def __init__(
        self,
        weak_transforms: transforms.Compose,
        softmatch_transform: transforms.Compose,
        simclr_transform: transforms.Compose,
    ):
        self.weak_transforms = weak_transforms
        self.softmatch_transform = softmatch_transform
        self.simclr_transform = simclr_transform

    def __call__(self, x):
        return [
            self.weak_transforms(x),
            self.softmatch_transform(x),
            self.simclr_transform(x),
        ]


def get_simclr_transforms(img_size: int = 96, s: float = 1.0) -> transforms.Compose:
    """
    Returns SimCLR augmentation pipeline.

    Args:
        img_size (int): Target crop size.
        s (float): Color jitter strength.

    Returns:
        transforms.Compose: Composed transformations.
    """
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    kernel_size = int(0.1 * img_size)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

    return transforms.Compose([
        transforms.RandomResizedCrop(size=img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))],
            p=0.5,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    ])


def get_weak_transforms(img_size: int = 96) -> transforms.Compose:
    """Returns weak augmentation pipeline."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size=img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    ])


def get_softmatch_transforms(img_size: int = 96) -> transforms.Compose:
    """Returns strong augmentation transforms for SoftMatch."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    ])


def get_val_transforms(img_size: int = 96) -> transforms.Compose:
    """Returns validation/test transforms for STL-10."""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    ])


def _make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    distributed: bool = False,
    shuffle: bool = True,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Creates a DataLoader with optional DistributedSampler.

    Args:
        dataset (torch.utils.data.Dataset): Source dataset.
        batch_size (int): Samples per batch.
        num_workers (int): Data loading workers.
        distributed (bool): Use DistributedSampler.
        shuffle (bool): Shuffle data (ignored when distributed).
        drop_last (bool): Drop incomplete last batch.

    Returns:
        torch.utils.data.DataLoader: Configured DataLoader.
    """
    sampler = None
    if distributed and dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        sampler=sampler,
    )


def prepare_stl10_train(
    preprocess: transforms.Compose,
    batch_size: int,
    num_workers: int,
    distributed: bool = False,
    fold: int | None = 0,
) -> torch.utils.data.DataLoader:
    """
    Prepares STL-10 labeled training set loader.

    Args:
        preprocess (transforms.Compose): Preprocessing transforms.
        batch_size (int): Samples per batch.
        num_workers (int): Data loading workers.
        distributed (bool): Use DistributedSampler.
        fold (int | None): Which 1k fold to use (0-9). None uses all 5k labels.

    Returns:
        torch.utils.data.DataLoader: Training DataLoader.
    """
    if fold is not None:
        ds = datasets.STL10(root=DATASET_PATH, split="train", folds=fold, transform=preprocess, download=True)
    else:
        ds = datasets.STL10(root=DATASET_PATH, split="train", transform=preprocess, download=True)
    print(f"Using {len(ds)} labeled samples for training.")
    return _make_loader(ds, batch_size, num_workers, distributed, shuffle=True)


def prepare_stl10_test(
    preprocess: transforms.Compose,
    batch_size: int,
    num_workers: int,
    distributed: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Prepares STL-10 test set loader.

    Args:
        preprocess (transforms.Compose): Preprocessing transforms.
        batch_size (int): Samples per batch.
        num_workers (int): Data loading workers.
        distributed (bool): Use DistributedSampler.

    Returns:
        torch.utils.data.DataLoader: Test DataLoader.
    """
    ds = datasets.STL10(root=DATASET_PATH, split="test", transform=preprocess, download=True)
    return _make_loader(ds, batch_size, num_workers, distributed, shuffle=False)


def prepare_simclr_train_dataset(
    batch_size: int,
    num_workers: int,
    img_size: int = 96,
    distributed: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Prepares STL-10 train+unlabeled loader with SimCLR augmentations.

    Args:
        batch_size (int): Samples per batch.
        num_workers (int): Data loading workers.
        img_size (int): Target image size.
        distributed (bool): Use DistributedSampler.

    Returns:
        torch.utils.data.DataLoader: SimCLR contrastive DataLoader.
    """
    contrastive_transform = ContrastiveTransformations(get_simclr_transforms(img_size=img_size))
    ds = datasets.STL10(root=DATASET_PATH, split="train+unlabeled", transform=contrastive_transform, download=True)
    return _make_loader(ds, batch_size, num_workers, distributed, shuffle=True, drop_last=True)


def prepare_softclr_train_dataset(
    batch_size: int,
    num_workers: int,
    img_size: int = 96,
    distributed: bool = False,
    fold: int | None = 0,
    labeled_batch_size: int | None = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Prepares STL-10 loaders for SoftMatch+SimCLR.

    Unlabeled loader (train+unlabeled split): [weak_aug, softmatch_aug, simclr_aug], label.
    Labeled loader (train split): weak_aug, label.

    Args:
        batch_size (int): Samples per batch for unlabeled loader.
        num_workers (int): Data loading workers.
        img_size (int): Target image size.
        distributed (bool): Use DistributedSampler.
        fold (int | None): Which 1k fold to use (0-9). None uses all 5k labels.

    Returns:
        tuple[DataLoader, DataLoader]: Unlabeled and labeled DataLoaders.
    """
    softclr_transform = SoftCLRTransformations(
        weak_transforms=get_weak_transforms(img_size),
        softmatch_transform=get_softmatch_transforms(img_size),
        simclr_transform=get_simclr_transforms(img_size),
    )
    unlabeled_ds = datasets.STL10(root=DATASET_PATH, split="train+unlabeled", transform=softclr_transform, download=True)
    
    if fold is not None:
        labeled_ds = datasets.STL10(root=DATASET_PATH, split="train", folds=fold, transform=get_weak_transforms(img_size), download=True)
    else:
        labeled_ds = datasets.STL10(root=DATASET_PATH, split="train", transform=get_weak_transforms(img_size), download=True)
    print(f"Using {len(labeled_ds)} labeled samples for training.")

    unlabeled_loader = _make_loader(unlabeled_ds, batch_size, num_workers, distributed, shuffle=True, drop_last=True)
    if labeled_batch_size is None:
        labeled_batch_size = batch_size // 8
    labeled_loader = _make_loader(labeled_ds, labeled_batch_size, num_workers, distributed, shuffle=True, drop_last=True)

    return unlabeled_loader, labeled_loader
