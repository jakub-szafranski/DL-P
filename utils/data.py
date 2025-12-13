import torch
import os
import json
from PIL import Image, ImageFile
from torchvision import transforms

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


def prepare__ImageNetTrain(preprocess: torch.nn.Module, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    """
    Prepares ImageNet training dataset loader.

    Args:
        preprocess (torch.nn.Module): Preprocessing transformations to apply to the images.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the ImageNet training dataset.
    """
    _set = ImageNetKaggle(DATASET_IMAGE_NET_2012_PATH, "train", transform=preprocess)
    loader = torch.utils.data.DataLoader(
        _set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def prepare__ImageNetTest(preprocess: torch.nn.Module, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    """
    Prepares ImageNet test dataset loader.

    Args:
        preprocess (torch.nn.Module): Preprocessing transformations to apply to the images.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
    Returns:
        torch.utils.data.DataLoader: DataLoader for the ImageNet test dataset.
    """
    _set = ImageNetKaggle(DATASET_IMAGE_NET_2012_PATH, "val", transform=preprocess)
    loader = torch.utils.data.DataLoader(
        _set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def prepare_simclr_train_dataset(batch_size: int, num_workers: int, img_size: int = 224) -> torch.utils.data.DataLoader:
    """
    Prepares ImageNet training dataset loader with SimCLR augmentations.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        img_size (int): Size to which images will be resized/cropped.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the ImageNet training dataset with SimCLR augmentations.
    """
    contrastive_transform = ContrastiveTransformations(get_simclr_transforms(img_size=img_size))
    _set = ImageNetKaggle(DATASET_IMAGE_NET_2012_PATH, "train", transform=contrastive_transform)
    loader = torch.utils.data.DataLoader(
        _set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader
