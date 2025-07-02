import os
from torch.utils.data import random_split, Subset
from torchvision import datasets, transforms


def get_datasets(dataset_path, subset=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    total = len(full_dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

    if subset:
        train_ds = Subset(train_ds, range(1000))
        val_ds = Subset(val_ds, range(200))

    return train_ds, val_ds, test_ds
