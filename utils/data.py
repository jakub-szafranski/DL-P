import torch
import numpy as np
import torch.distributed as dist
import os
import json
from PIL import Image, ImageFile
from torchvision import transforms
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_PATH = "/raid/kszyc/datasets/"
DATASET_IMAGE_NET_2012_PATH = "{}/{}".format(DATASET_PATH, "ImageNet2012")

with open(DATASET_IMAGE_NET_2012_PATH + "/LOC_synset_mapping.txt") as file:
    IMAGENET_LABELS = [" ".join(line.split(" ")[1:]).replace("\n", "") for line in file]


class ImageNetKaggle(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}

        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)

        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)

        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)

        if split == "train":
            for syn_id in sorted(os.listdir(samples_dir)):
                syn_folder = os.path.join(samples_dir, syn_id)
                if os.path.isdir(syn_folder):
                    target = self.syn_to_class[syn_id]
                    for sample in sorted(os.listdir(syn_folder)):
                        sample_path = os.path.join(syn_folder, sample)
                        self.samples.append(sample_path)
                        self.targets.append(target)
        elif split == "val":
            for entry in sorted(os.listdir(samples_dir)):
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


class ContrastiveTransformations:
    """
    Creates two augmented versions of an image for contrastive learning.
    """

    def __init__(self, base_transforms):
        self.base_transforms = base_transforms

    def __call__(self, x):
        return [self.base_transforms(x), self.base_transforms(x)]


class SoftCLRTransformations:
    """
    Creates weak and strong augmented versions for SoftMatch+SimCLR training.
    Returns: [weak_aug, strong_aug_1, strong_aug_2]
    """

    def __init__(self, weak_transforms, strong_transforms):
        self.weak_transforms = weak_transforms
        self.strong_transforms = strong_transforms

    def __call__(self, x):
        return [
            self.weak_transforms(x),
            self.strong_transforms(x),
            self.strong_transforms(x),
        ]


def get_simclr_transforms(img_size: int, s: float = 1.0) -> torch.nn.Module:
    """
    Returns a composition of data augmentation transformations for SimCLR.

    Args:
        img_size (int): Size to which images will be resized/cropped.
        s (float): Scaling factor for color jittering.

    Returns:
        torch.nn.Module: Composed transformations for SimCLR.
    """
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

    kernel_size = int(0.1 * img_size)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
        ]
    )


def get_weak_transforms(img_size: int) -> torch.nn.Module:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def prepare__ImageNetTrain(
    preprocess: torch.nn.Module,
    batch_size: int,
    num_workers: int,
    distributed: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Prepares ImageNet training dataset loader.

    Args:
        preprocess (torch.nn.Module): Preprocessing transformations.
        batch_size (int): Number of samples per batch (per GPU if distributed).
        num_workers (int): Number of subprocesses for data loading.
        distributed (bool): Whether to use DistributedSampler.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the ImageNet training dataset.
    """
    _set = ImageNetKaggle(DATASET_IMAGE_NET_2012_PATH, "train", transform=preprocess)

    sampler = None
    shuffle = True
    if distributed and dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(_set, shuffle=True)
        shuffle = False

    loader = torch.utils.data.DataLoader(
        _set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    return loader


def prepare__ImageNetTest(
    preprocess: torch.nn.Module,
    batch_size: int,
    num_workers: int,
    distributed: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Prepares ImageNet test dataset loader.

    Args:
        preprocess (torch.nn.Module): Preprocessing transformations.
        batch_size (int): Number of samples per batch (per GPU if distributed).
        num_workers (int): Number of subprocesses for data loading.
        distributed (bool): Whether to use DistributedSampler.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the ImageNet test dataset.
    """
    _set = ImageNetKaggle(DATASET_IMAGE_NET_2012_PATH, "val", transform=preprocess)

    sampler = None
    if distributed and dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(_set, shuffle=False)

    loader = torch.utils.data.DataLoader(
        _set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    return loader


def prepare_simclr_train_dataset(
    batch_size: int,
    num_workers: int,
    img_size: int = 224,
    distributed: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Prepares ImageNet training dataset loader with SimCLR augmentations.

    Args:
        batch_size (int): Number of samples per batch (per GPU if distributed).
        num_workers (int): Number of subprocesses for data loading.
        img_size (int): Size to which images will be resized/cropped.
        distributed (bool): Whether to use DistributedSampler.

    Returns:
        torch.utils.data.DataLoader: DataLoader with SimCLR augmentations.
    """
    contrastive_transform = ContrastiveTransformations(get_simclr_transforms(img_size=img_size))
    _set = ImageNetKaggle(DATASET_IMAGE_NET_2012_PATH, "train", transform=contrastive_transform)

    sampler = None
    shuffle = True
    if distributed and dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(_set, shuffle=True)
        shuffle = False

    loader = torch.utils.data.DataLoader(
        _set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
    )
    return loader


def get_data_subset(dataset: torch.utils.data.Dataset, subset_ratio: float) -> torch.utils.data.Subset:
    """
    Create a stratified subset of the dataset.

    Args:
        dataset (torch.utils.data.Dataset): Source dataset.
        subset_ratio (float): Fraction of data to retain (0-1).

    Returns:
        torch.utils.data.Subset: Stratified subset of the dataset.
    """
    if np.isclose(subset_ratio, 1.0):
        return torch.utils.data.Subset(dataset, range(len(dataset)))
    indices, _ = train_test_split(
        range(len(dataset)),
        train_size=subset_ratio,
        stratify=dataset.targets,
        random_state=42,
    )
    return torch.utils.data.Subset(dataset, indices)


def prepare_softclr_train_dataset(
    batch_size: int,
    num_workers: int,
    img_size: int = 224,
    distributed: bool = False,
    subset_ratio: float = 0.1,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Prepares ImageNet training dataset loaders for SoftMatch+SimCLR.
    
    Full loader returns: [weak_aug, strong_aug_1, strong_aug_2], label
    Subset loader returns: weak_aug, label

    Args:
        batch_size (int): Number of samples per batch (per GPU if distributed).
        num_workers (int): Number of subprocesses for data loading.
        img_size (int): Size to which images will be resized/cropped.
        distributed (bool): Whether to use DistributedSampler.
        subset_ratio (float): Fraction of data for the labeled subset loader.

    Returns:
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: Unlabeled and labeled subset DataLoaders.
    """
    weak_transform = get_weak_transforms(img_size=img_size)
    strong_transform = get_simclr_transforms(img_size=img_size)
    softclr_transform = SoftCLRTransformations(weak_transform, strong_transform)
    
    _unlabeled_set = ImageNetKaggle(DATASET_IMAGE_NET_2012_PATH, "train", transform=softclr_transform)
    
    _labeled_set = ImageNetKaggle(DATASET_IMAGE_NET_2012_PATH, "train", transform=weak_transform)
    _labeled_subset = get_data_subset(_labeled_set, subset_ratio)

    sampler = None
    subset_sampler = None
    shuffle = True
    if distributed and dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(_unlabeled_set, shuffle=True)
        subset_sampler = torch.utils.data.distributed.DistributedSampler(_labeled_subset, shuffle=True)
        shuffle = False

    loader = torch.utils.data.DataLoader(
        _unlabeled_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
    )
    
    subset_loader = torch.utils.data.DataLoader(
        _labeled_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=subset_sampler,
    )
    
    return loader, subset_loader

